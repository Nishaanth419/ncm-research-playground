# Implementation Log

## Milestone 1: Inspect repository and machine environment (2026-11-03), done

**Commands executed**
- Machine inspection: `uname`, `sw_vers`, `sysctl`, `system_profiler`.
- `git clone https://github.com/HazyResearch/ncm research-ncm` (HEAD `c702742`).
- Read the README, `requirements.txt`, `evaluate.py`, `scripts/eval_shard.py`, `config/eval/{tinystories,xsum}.yaml`, `models/next_concept.py::generate`, `models/quant.py` (index ↔ latent functions), `models/vqvae.py`, `tokenization/vqvae_tokenizer.py`, `evaluation/gen_ppl.py`.
- `uv venv --python 3.12 .venv` and installed the official requirements (torch 2.15.0, transformers 5.22.1).
- Downloaded the `tinystories/` and `xsum/` checkpoints and the `xsum/` test set at pinned revisions; `diff` of shipped vs. repo configs showed them identical.

**Outcome**
- Assessment written to `docs/ASSESSMENT.md`.

## Milestone 2: Install NCM and run real TinyStories inference/evaluation (2026-11-03), done

**Decisions (user)**
- Prompt-driven modes: XSum and WMT14 (TinyStories is unconditional).
- Device: CPU now; an MPS shim may come later, only once verified against CPU output.
- Run the official eval on CPU here.

**Done**
- `experiments/smoke_tinystories.py`: device-parameterised mirror of the unconditional path of `eval_shard.py`.
  - CPU run: 4 coherent samples, codes `(4, 641)`, batch latency 0.77 s.
  - Log: `results/logs/smoke_tinystories_cpu.log`.
- `experiments/run_official.py`: runs official scripts **unmodified** via `runpy`, remapping `.cuda()` / `device="cuda"` / `map_location="cuda"` to `--device`.
- Downloaded the `wmt14/` checkpoint and test set (same pinned revisions); its config is identical to the repo's.
- **Official TinyStories evaluation** (`eval_shard.py` + `eval_merge.py`; 5 seeds × 256 samples; GPT-2-large judge) on CPU:

  | seed | genPPL | entropy |
  |---|---|---|
  | 42 | 15.88 | 3.9639 |
  | 0 | 16.90 | 3.9763 |
  | 1 | 15.74 | 3.9669 |
  | 2 | 16.56 | 3.9632 |
  | 3 | 16.10 | 3.9671 |
  | **mean** | **16.23 ± 0.48** | **3.9674** |

  - Paper, Table 8: genPPL **15.41**, entropy **3.98**. We are within 2 seed-std; the device and library versions differ (CPU fp32, torch 2.15, transformers 5.22).
  - Generation took 4 min 24 s; scoring took 18 min 37 s, including the GPT-2-large download.
  - Logs: `results/official_eval/tinystories/{shard_0,merge}.log`.
- XSum smoke test (official shard, `--n 4`): on-topic summaries. Log: `results/official_eval/smoke/xsum_n4.log`.
- WMT14 smoke test (official shard, `--n 4`): plausible translations with visible repetition ("coup d coup coup"). Log: `results/official_eval/smoke/wmt14_n4.log`.

**Errors**
- MPS: `Adaptive pool MPS: input sizes must be divisible by output sizes` at `research-ncm/models/quant.py:703`. `PYTORCH_ENABLE_MPS_FALLBACK=1` does not help. Log: `results/logs/smoke_tinystories_mps.log`.
- `hf download --include ... --revision <sha>` failed for the model repo; `snapshot_download` worked.

**Unresolved**
- NFE accounting: the paper's Table 8 lists NFE 12 for TinyStories. The probe counts 11 transformer forward passes; level 0 is drawn from the checkpoint's unigram, not from the transformer. The UI will report "11 transformer passes + level-0 unigram draw" rather than a single number.
- Full XSum and WMT14 metric evaluations not run (11,334 XSum docs on CPU is hours); smoke tests only.

## Milestone 3: Understand the generation pipeline (2026-11-03), done

- `experiments/probe_levels.py` wraps instance methods `net._sample_level` and `vq.get_next_autoregressive_input` around one real `generate` call. Log: `results/logs/probe_levels.log`.
- **Verified**
  - 11 level predictions, each a single parallel forward producing `[B, levels[k]]` codes.
  - Level-0 codes + captured codes are exactly equal to `generate()`'s output.
  - The official `idx_to_zhats` `z_hats` list aliases one tensor, while the captured clones are distinct.
- **Capture semantics:** the latent seen at level k is the transformer's input, i.e. cumulative `z_hat` through level k-1. Cosine to the true final latent (`vq.idx_to_z_q`) rises from 0.28 (level 1 input) to 0.90 (level 11 input).
- Per-level timings in that log were taken while the eval ran concurrently and are not benchmark numbers.

**Next: Milestone 4.** Build `backend/app/services/trace_collector.py` (test-first) from the probe, with the JSON trace schema: codes, per-level timing, latent scalars, codebook stats, and official per-level decodes.

## Milestone 4: Instrument real intermediate representations (2026-11-03), done

**Files**
- `backend/pyproject.toml`: pytest config; `model` marker for tests that need checkpoints.
- `backend/app/core/research.py`: `load_bundle(task, device)` mirrors `eval_shard.py` loading; `missing_files(task)` checks availability; `NCMBundle.resolve()` reads `generator.pt:inference.*` references.
- `backend/app/schemas/trace.py`: Pydantic `Trace`, `LevelTrace`, `CodebookStats`, `LatentSummary` (ints, floats and strings only).
- `backend/app/services/trace_collector.py`: `TraceCollector` context manager, `codebook_stats`, `latent_summary`.
- `backend/tests/test_trace_collector.py`: 10 unit tests with a fake model (no checkpoints).
- `backend/tests/test_trace_real_model.py`: real TinyStories trace on CPU.

**How the collector works**
- It wraps instance methods `net._sample_level` and `vq.get_next_autoregressive_input` and restores them on exit, including on exceptions.
- `level_started` / `level_completed` events fire from those real calls, with device-synchronised timing.
- With CFG, the doubled batch is sliced back to B.
- Latent statistics use the first `latent_samples` (default 8) samples, to bound memory (a full batch of latents is ~280 MB).
- Codebook statistics cover the whole batch.
- Codes and per-level decodes are kept for the first `max_detail_samples` (default 8) samples.

**Verified on the real model** (4 samples, CPU)
- 11 transformer passes; codes identical to `generate()`; 23 events.
- Trace JSON is 37 KB.
- Per-level time grows with level length (21 ms at 8 codes → 120 ms at 128 codes, batch 4).
- Unique codes grow from 32 to 312 and entropy from 5.0 to 8.0 bits.
- Input-latent cosine to the final latent goes from 0.33 to 0.89.

**Tests**
- `cd backend && ../.venv/bin/pytest -q`: 11 passed.

**Next: Milestone 5.** FastAPI `NCMRunner` (lazy-load bundles, keep them in memory; unconditional TinyStories plus prompt-driven XSum/WMT14 using `eval_shard.py`'s prefix and `pick_tokens` final decoding) and `/api/generate`, `/api/models`.

## Milestone 5: FastAPI model runner (2026-11-03), done

**Files**
- `backend/app/main.py`: `create_app(runner=None, settings=None)`.
  - Error mapping: `ModelNotFound` → 404, `InvalidRequest` → 422, `ModelUnavailable` → 503.
  - CORS for the Vite dev server; `/api/health`.
- `backend/app/core/config.py`: `NCM_DEVICE` (default `cpu`), `NCM_CORS_ORIGINS`, `NCM_LOG_LEVEL`.
- `backend/app/core/official.py`: compiles `pick_tokens` **verbatim from** `research-ncm/scripts/eval_shard.py` (via `ast`), so the final-decode penalties are the official code, not a copy.
- `backend/app/services/ncm_runner.py`: `NCMRunner`.
  - Registry: `ncm-tinystories`, `ncm-xsum`, `ncm-wmt14`.
  - Bundles load lazily on first use and stay in memory; one load lock and one inference lock.
  - Input handling follows `eval_shard.py`:
    - level-0 unigram draw for the unconditional model;
    - for conditional models, a text prompt is tokenized with the config's `pre_tokenizer.model_id`, then truncated/padded to `prefix_len`;
    - `example_index` uses the official test-set item as-is and returns its reference.
  - Returns outputs, resolved sampling settings, prompt token count and truncation flag, device-synchronised timing (load / generate / decode), and the optional trace.
- `backend/app/schemas/generation.py`: `ModelInfo`, `GenerateRequest` (validated ranges; prompt XOR example_index), `GenerateResponse`, `Timing`, `SamplingParams`.
- `backend/app/routes/{models,generate}.py`: `GET /api/models`, `POST /api/generate` (sync handler, runs in a worker thread).
- Trace collector fix: XSum (`query_level0`) and WMT14 (`predict_level0`) sample level 0 inside `generate` before any level hook.
  - The collector now records that as a transformer level 0, timed from entering the collector (so it includes prefix encoding).
  - No `level_started` event is emitted for it, since no real hook precedes it.

**Verification**
- `pytest -m "not model"`: 24 passed (API contracts with a fake runner, official-helper tests, trace tests).
- `pytest -m model`: 7 passed. **The runner's outputs equal the official `eval_shard.py` outputs exactly:**
  - TinyStories, seed 42, 4 samples;
  - XSum test items 0–3;
  - WMT14 test items 0–3.
  - The references are in `results/official_eval/smoke/*_n4.pkl`, regenerated with `experiments/run_official.py`.
- Live server (`uvicorn --factory app.main:create_app`):
  - WMT14 "Ich trinke gern Kaffee am Morgen." → "I like like coffee in the morning." (load 5.75 s, generate 0.38 s, decode 0.02 s);
  - a TinyStories prompt is rejected with 422.

**Notes**
- XSum's level schedule starts at 4 codes (`[4, 6, 10, 14, ...]`), so its level 0 is 4 transformer-predicted codes.
- Per-level decodes in the trace use the official `decode_multiscales` (argmax). Final outputs use `pick_tokens`, which applies penalties for XSum. The UI must label these separately.

**Next: Milestone 6.** WebSocket streaming of real `generation.*` events (`WS /api/stream/{id}`), with experiment records (`POST /api/experiments`, `GET /api/experiments/{id}[/trace|/metrics]`) in SQLite.

## Milestone 6: WebSocket streaming and experiment records (2026-11-03), done

**Files**
- `backend/app/services/experiment_store.py`: `ExperimentStore`, stdlib `sqlite3` with JSON columns.
  - Columns: request, outputs, timing, trace, metrics, events.
  - Default database `results/playground.sqlite` (gitignored); override with `NCM_DB_PATH`.
- `backend/app/services/event_hub.py`: thread → asyncio fan-out.
  - A late subscriber gets the full backlog, then live events, with no gaps (copy and registration happen under one lock).
- `backend/app/services/experiment_service.py`: `ExperimentService`.
  - A single worker thread, so requests queue (inference is serialised anyway).
  - Event order: `generation.started` → (`model.loading`/`model.loaded` if the model is cold) → real `generation.level_*` from `TraceCollector` → `generation.completed` | `generation.failed`.
  - Each event carries `experiment_id`, `seq` and `t` (seconds since start).
  - `compute_metrics`: timing, passes, per-level seconds/unique/entropy, output words/chars, sampling, prompt tokens.
- `backend/app/routes/experiments.py`:
  - `POST /api/experiments` (201, returns `stream_url`)
  - `GET /api/experiments` (newest first, limit/offset)
  - `GET /api/experiments/{id}`
  - `GET /api/experiments/{id}/trace`
  - `GET /api/experiments/{id}/metrics`
  - `WS /api/stream/{id}`: live, or replayed from the database with `replay: true` when the run is not in memory; close code 4404 for an unknown id.
- `NCMRunner.generate(req, on_event=None)` and `bundle(...)` now emit model-load events and pass `on_event` to `TraceCollector`.
- `backend/tests/conftest.py`: every test uses a temporary database.

**Verification**
- `pytest -q`: 42 passed.
  - New: store round trip and reopen; hub replay plus live ordering across threads; experiment API (create, stream, fetch, metrics, failure path, database replay, 404/4404, validation).
  - Real-model stream (TinyStories seed 42): 27 events in order; final outputs equal the official `eval_shard.py` outputs; the second run emits no `model.loading`.
- Live uvicorn + `websockets` client, XSum test item 1:
  - 21 events; load 6.06 s; 9 transformer passes at ~0.093 s each (roughly flat: the 512-token article prefix dominates attention cost); generate 0.857 s.
  - Output identical to the official item-1 summary.

**Next: Milestone 7.** React + Vite + TypeScript + Tailwind + shadcn/ui workspace shell (sidebar, Overview, Playground), using `/api/models`, `/api/experiments` and the WebSocket.

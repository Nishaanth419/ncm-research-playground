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

## Milestone 7: React research workspace (2026-11-03), done

**Stack**
- Vite 8, React 19, TypeScript 7, Tailwind 4 (`@tailwindcss/vite`).
- shadcn/ui (base-nova style, Base UI primitives), Zustand 5, react-router 7, Recharts 3 (installed for M8–10), Lucide.
- shadcn's CLI wired `cn` to a new npm package `cn` (it ships a compiler and a `bin`). I replaced it with the conventional local `clsx` + `tailwind-merge` helper in `src/lib/utils.ts` and uninstalled the package.

**Files** (`frontend/src`)
- `lib/types.ts` (mirrors the backend Pydantic schemas), `lib/api.ts` (fetch client with error detail, `streamExperiment` WebSocket), `lib/format.ts`.
- `store/models.ts`: models + health, retried every 3 s while the backend is unreachable.
- `store/playground.ts`: form, run state machine (idle → submitting → queued → loading-model → running → completed | failed), per-level state driven only by received events. Run state is cleared when switching models.
- `components/common.tsx` (PageHeader, Panel, StatTile, EmptyState, StatusDot, KV), `components/layout/Sidebar.tsx`.
- `pages/Overview.tsx`:
  - checkpoint availability, experiment counts, latency by model (per run and per sample);
  - latest experiments;
  - a labelled empty state for benchmarks;
  - an "ad-hoc runs: not a benchmark" caveat.
- `pages/Playground.tsx` + `features/playground/{ParamControl,LevelStrip,EventLog}.tsx`:
  - model selector; free text vs. official test example for XSum/WMT14 (no prompt for TinyStories);
  - sampling overrides with reset-to-config-default; seed and sample count;
  - live level strip (cell width ∝ codes per level, one cell per parallel step);
  - outputs, reference, timing and the raw event stream.
- `pages/Settings.tsx`; `pages/Placeholder.tsx` for Concept Explorer / Timeline (M8), Comparison (M9), Benchmarks / History (M10), each saying it is not built yet.
- Theme: an obsidian dark palette in `index.css`; amber = NCM (`--ncm`), sky = autoregressive (`--ar`, reserved for M9); Geist + Geist Mono.
- `vite.config.ts`: `@` alias; proxies `/api` and the `/api/stream` WebSocket to `NCM_BACKEND_URL` (default `http://127.0.0.1:8000`).

**Backend change**
- `compute_metrics` now stores `reference` (official test-set reference), so the Playground can show it from the experiment record. The real-model test asserts it for XSum and WMT14.

**Verification**
- `tsc -b`: clean. `vite build`: OK. `oxlint`: only the generated shadcn fast-refresh warnings.
- In the in-app browser, against the live backend:
  - TinyStories, seed 42, 4 samples: outputs **identical** to the official `eval_shard.py` reference; 27 events; the strip fills 12/12.
  - XSum, official example 0: model-load events shown live; 9/9 levels; summary matches the official item 0; truncation notice shown (article > 512 tokens).
- Mobile (375 × 812): Overview, Playground, Settings and placeholders have no horizontal overflow (`scrollWidth == innerWidth`), after fixing grid-item `min-width`, the latency table and the KV wrapping.
- Backend: `pytest -q`, 42 passed.

**Run locally**
```bash
cd backend && ../.venv/bin/uvicorn --factory app.main:create_app --port 8000
cd frontend && npm run dev   # http://localhost:5173
```

**Next: Milestone 8.** Concept Explorer (level navigation, code-index heatmap, codebook distribution, latent stats, per-level decodes labelled separately from the final penalised decode) and Generation Timeline, both from recorded traces.

## Milestone 8: Concept hierarchy visualisation and Generation Timeline (2026-11-03), done

**Files** (`frontend/src`)
- `lib/trace.ts`
  - `timelineSegments(events)`: lays out load / level / decode segments from recorded event timestamps only.
    - A unigram level 0 is a zero-width marker.
    - A transformer level 0 (no started event) is placed using its measured `seconds`.
    - A failed run marks its open level as incomplete.
  - `codeCounts`, `codeColor` (categorical golden-angle hue: same id gives the same colour; hue carries no similarity meaning).
- `lib/trace.test.ts`: 5 vitest tests (`npm test`).
- `hooks/useExperimentTrace.ts`: `?id=`, defaulting to the newest completed traced experiment; loads the record (with events) and the trace. `components/ExperimentPicker.tsx`.
- `features/concepts/`
  - `HierarchyGrid`: one full-width row per level, cells = that level's codes; colour by code id or by frequency across stored samples.
  - `LevelDetail`: step type, logits shape, wall time, codebook usage over the whole batch, input-latent stats, top-codes bar chart, raw code indices, decoded text after the level.
  - `LevelSeries`: utilisation, entropy and latent cosine across levels.
  - `TextLadder`: all per-level decodes plus the final output.
- `features/timeline/TimelineBar.tsx`, `pages/Timeline.tsx`: time axis (checkpoint load toggle), stage table, stage inspector with a deep link `/concepts?id=…&level=k`, time-per-level chart.
- `pages/ConceptExplorer.tsx`: sample stepper, colour-mode toggle, `?level=` deep link.
- Playground "Outputs" panel links to `concepts →` / `timeline →` for the finished run.

**Labelling (no fabricated intermediates)**
- Per-level text is labelled as official `decode_multiscales` (decoder argmax on the cumulative latent after level k), distinct from the final output.
- For XSum the page states that the final output uses the official `pick_tokens` repetition penalties.
- Latent statistics state their sample count (first `latent_samples`); codebook statistics state they are batch-wide.

**Fixes**
- Grids without a column template below `xl` sized their implicit column to max-content (a 3005 px panel): added `grid-cols-1` (`minmax(0,1fr)`) to every responsive grid.
- **Test isolation bug:** module-scoped fixtures in `test_ncm_runner_real.py` built their app before the function-scoped autouse fixture set `NCM_DB_PATH`, so two real-model test runs had been written to `results/playground.sqlite`.
  - `conftest.py` now sets the variable in `pytest_configure`; verified the row count stays at 4 across a full run.
  - The two test-created rows (`552dba8d…`, `aaff517f…`) were deleted.

- Lint cleanup: per-experiment view state is reset by keying `ConceptBody` by experiment id (no setState-in-effect); `useExperimentTrace` keys fetched results by id and derives `loading`. `oxlint` reports 0 warnings outside the generated shadcn files.

**Verification**
- `tsc -b` clean; `vitest` 5 passed; `vite build` OK; backend `pytest -q` 42 passed.
- In-app browser:
  - Concept Explorer on the TinyStories seed-42 run (12 rows, 641 codes; level 6 via `?level=6` shows logits `[4, 52, 4096]`, 190/4096 unique, 7.52 bits, cos 0.605).
  - Timeline on the XSum run (load 6.41 s hidden by default; 9 level steps of ~0.2 s; decode 32.8 ms; stage inspector and deep link).
  - Both fit 375 px with no horizontal overflow.

**Next: Milestone 9.** Autoregressive baseline runner and the Model Comparison page.

## Milestone 9: Autoregressive baseline and Model Comparison (2026-11-03), done

**Baseline (user choice):** `roneneldan/TinyStories-33M` @ `2ad0a164221b7c4d21cac7c46aec74f6f98dbfc8`.
- GPT-Neo causal LM: 4 layers, hidden size 768, GPT-2 BPE vocabulary of 50,257, 68.5M parameters including embeddings.
- Trained on the same TinyStories corpus as NCM-TinyStories, in an independent run. Downloaded once into the HF cache (291 MB `pytorch_model.bin`) and loaded with `local_files_only=True`.
- Loading reports `attn.attention.bias/masked_bias` as UNEXPECTED keys: these are legacy causal-mask buffers in the 2023 checkpoint, not weights.

**Backend**
- `services/ar_runner.py` `AutoregressiveRunner`:
  - unconditional from BOS (= EOS 50256, mirroring NCM starting from EOS); up to 128 new tokens; stops at EOS; KV cache on.
  - Real `generation.step_completed` events come from a transformers streamer (one `put` per decoding step).
  - Records time to first token, and own-tokenizer entropy as defined in `eval_merge.py`.
- `services/registry.py` `ModelRegistry`: dispatches by model id; `/api/models` lists NCM and AR models together.
- `services/quality.py` `QualityScorer`: calls the official `research-ncm/evaluation/gen_ppl.py::generation_perplexity` with GPT-2-large (the configs' `eval.ref_model`).
  - `corpus_ppl` = the official `mean_ppl`.
  - Entropy is also reported over the judge's GPT-2 tokens, for cross-tokenizer comparison.
  - Empty outputs are not scored.
- `POST /api/experiments/{id}/quality` scores the stored outputs and saves the result under `metrics.quality`.
- `services/memory.py`: CUDA reports peak bytes; MPS reports driver allocation (labelled not-a-peak); CPU reports `null` with the reason.
- New request fields:
  - `tag` (comparison group);
  - `warmup` (one untimed pass with the same batch, then reseed; tests show outputs are unchanged, including the official NCM seed-42 stories).
- New response/metrics fields: `output_tokens`, `tokenizer`, throughput (tokens/s in the model's own tokenizer, words/s), `first_output_s`, `output_entropy_own`, `memory`, `warmup`, `tag`.
  - NCM's `first_output_s` = complete text, since text exists only after the final level and decode.

**Frontend**
- `pages/Comparison.tsx` + `store/comparison.ts`:
  - shared seed, batch and sampling; runs NCM then the baseline back to back on the single worker, each with `warmup: true`;
  - live level strip vs. token step bar;
  - GPT-2-large scoring of both;
  - grouped measurement table (only like-for-like rows are colour-coded as better; others greyed);
  - side-by-side outputs with per-sample PPL.
- `lib/compare.ts` (`summarize`, `caveats`) + 6 vitest tests. Caveats are generated from the actual metadata: parameter counts, tokenizers, training data, decoding-setting or batch mismatch, length policy, the meaning of "first output", warm-up / single run / device, and the note that published H100 speedups are not comparable.
- Playground accepts the baseline (`StepProgress` instead of the level strip; no concept/timeline links, since the baseline has no trace).

**Measured on this machine** (CPU, 4 samples, seed 0, T=1, no top-k/top-p, warm-up on; one run each, not a benchmark)

| | ncm-tinystories | ar-tinystories-33m |
|---|---|---|
| params | 125.0M (generator 88.6M + VQ-VAE 36.4M) | 68.5M |
| sequential forward passes | 11 | 128 |
| generate | 797 ms | 1.58 s |
| first output | 816 ms (complete text) | 11.6 ms (first token) |
| words/s | 500 | 259 |
| gen-PPL (GPT-2-large) | 13.48 | 9.74 |
| entropy (judge tokens) | 4.056 | 3.993 |

- Without warm-up, the first baseline call after loading measured 2.65 s generate and 974 ms to first token. This is why the comparison always warms up.
- A repeated warm baseline run took 1.24 s generate vs. 1.58 s here: single-run variance, to be handled with medians in Milestone 10.

**Verification**
- Backend `pytest -q`: 52 passed (adds registry/quality API contracts, AR generation/streaming/seed/warm-up, quality scorer, NCM warm-up equality).
- Frontend: `tsc` clean, vitest 11 passed, build OK, oxlint 0 warnings outside the generated shadcn files.
- In-app browser: comparison run end to end on a freshly restarted backend; Comparison and Playground fit 375 px.
- Not browser-verified: running the baseline from the Playground page (covered by API tests only).

**Next: Milestone 10.** Benchmark workflow (fixed suite, warm-ups, repeated measurements, median/p90/p95, hardware and config capture, SQLite, CSV export) and Experiment History (reopen and replay).

## Milestone 10: Benchmarking and experiment history (2026-11-03), done

**Backend**
- `schemas/benchmark.py` `BenchmarkRequest`: 1–4 models, 1–4 batch sizes (1–16), repeats 1–100, warm-ups 0–5, base seed, sampling overrides.
- `services/benchmark.py`
  - **Workflow:** for each (model, batch): `warmup_runs` untimed generations, then `repeats` timed runs with seeds `seed+i`. No trace is collected, so instrumentation decodes are not timed.
    - Runs on the **same single worker as experiments**, so nothing else uses the device during a benchmark.
  - **Statistics:** `percentile` (linear interpolation, numpy default); `describe` (n, median, p90, p95, mean, sample std, min, max); `summarize` per group: total, generate, first output, per-sample latency, words/s, tokens/s (own tokenizer), passes, mean output tokens, peak memory where the device reports it.
  - **Recorded context:** `hardware_info` (platform, CPU brand via `sysctl`, core count, RAM, Python/torch/transformers versions, torch threads, device, CUDA device, research-ncm commit, checkpoint revision), plus model metadata from the registry at start.
  - **Storage:** SQLite tables `benchmarks` and `benchmark_runs` in the same database file; live `progress` (phase, model, batch, done/total).
- **Endpoints**
  - `POST /api/benchmarks` (201): validates models; rejects conditional models with 422, since the workload has no prompts.
  - `GET /api/benchmarks`, `GET /api/benchmarks/{id}` (config, hardware, runs, summary).
  - `GET /api/benchmarks/{id}/runs.csv`, `/summary.csv`.
  - `GET /api/experiments.csv`: request, sampling, timings, passes, token/word totals, gen-PPL, outputs.

**Frontend**
- `pages/Benchmarks.tsx`:
  - config form (unconditional models, batch chips, repeats, warm-ups, seed, generation count); past-benchmark picker; live progress (1 s polling).
  - results table (median, p90, p95, per sample, first output, words/s, passes); median ratio between the two families per batch size.
  - charts with bar = median and whisker = p95 (Recharts `ErrorBar`); hardware and configuration panels; CSV downloads; reading caveats.
  - `lib/bench.ts` (`chartRows`, `medianRatio`) + 2 vitest tests.
- `pages/History.tsx`:
  - search (prompt / output / tag / id), model and status filters, CSV export.
  - Detail view: request, timings, gen-PPL; links to Concept Explorer / Timeline for traced NCM runs.
  - **Re-run** submits the identical request as a new experiment (a new measurement).
  - **Replay** (`features/history/Replay.tsx`) plays back the recorded events at their recorded offsets (0.25×/1×/4×); labelled as playback, nothing is recomputed.
- Overview "Recent benchmark results" now lists the latest completed benchmarks (median and p95 per group).
- `Placeholder.tsx` removed: every section is implemented.
- `color-scheme: dark`, so native scrollbars match the theme.

**Benchmark on this machine** (`58a26ec7…`; Apple M5 Pro, CPU, 6 torch threads; 2 warm-ups + 10 timed runs per cell; checkpoint-default sampling; exported to `results/benchmarks/58a26ec7…-{summary,runs}.csv`)

| model | batch | total median | p90 | p95 | per sample | first output | words/s | passes |
|---|---|---|---|---|---|---|---|---|
| ncm-tinystories | 1 | 307.5 ms | 353.9 ms | 399.3 ms | 307.5 ms | 307.5 ms (full text) | 340.6 | 11 |
| ncm-tinystories | 4 | 733.7 ms | 772.6 ms | 788.8 ms | 183.4 ms | 733.7 ms (full text) | 571.1 | 11 |
| ar-tinystories-33m | 1 | 436.9 ms | 480.0 ms | 491.3 ms | 436.9 ms | 3.9 ms (1st token) | 239.5 | 128 |
| ar-tinystories-33m | 4 | 1313.0 ms | 1371.4 ms | 1430.3 ms | 328.2 ms | 10.2 ms (1st token) | 304.7 | 128 |

- The baseline's median total is 1.42× NCM's at batch 1 and 1.79× at batch 4.
- Quality is not part of the benchmark; see Model Comparison: gen-PPL 13.48 (NCM) vs 9.74 (baseline) on one 4-sample run.
- Not comparable with the paper's H100 numbers against diffusion models.

**Verification**
- Backend `pytest -q`: 65 passed.
  - percentile/describe; workflow (warm-ups excluded, seeds, summary, hardware keys); CSVs; failure path; validation; experiments CSV.
  - Real 2-model benchmark through the default app.
- Frontend: `tsc` clean; vitest 13 passed; build OK; oxlint 0 warnings outside the generated shadcn files.
- In-app browser:
  - 48-generation benchmark run end to end from the page; table, charts and ratios render.
  - History: filter, open the XSum run, replay all 21 events (9/9 levels), Re-run, which produced identical outputs with a new timing (1.83 s vs 1.94 s).
  - Benchmarks, History and Overview have no horizontal overflow at 375 px.

**Next: Milestone 11.** Test, document and polish: top-level README, final pass over error states, a full regression run.

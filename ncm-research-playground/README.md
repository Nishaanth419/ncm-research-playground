# NCM Research Playground

An interactive workspace for **Next-Concept Modeling** ([Kumbong et al., 2026, arXiv:2610.14162](https://arxiv.org/abs/2610.14162)). It runs the official [HazyResearch/ncm](https://github.com/HazyResearch/ncm) checkpoints, captures what the model really does at every coarse-to-fine level, and benchmarks it against an autoregressive baseline.

Everything shown in the app comes from real execution: real checkpoints, real generation events, real timings. Where the model does not expose something, the UI says so rather than inventing it.

## What you can do

| Section | What it shows |
|---|---|
| **Playground** | Run NCM TinyStories (unconditional), XSum (article → summary) or WMT14 (German → English), or the TinyStories-33M baseline. Live level-by-level progress from WebSocket events. |
| **Concept Explorer** | The discrete code hierarchy of a run: one row per level, colour by code id or frequency, per-level codebook statistics, input-latent statistics, and the official per-level decodes. |
| **Generation Timeline** | The recorded execution on one time axis (load → levels → decode), stage inspector, time per level. |
| **Model Comparison** | NCM vs. TinyStories-33M, back to back with the same seed, batch and sampling, warmed up, then scored with the official gen-PPL judge (GPT-2-large). Lists every way the comparison is not like-for-like. |
| **Benchmarks** | Warm-ups plus repeated timed runs per (model, batch size): median, p90, p95, recorded hardware and config, charts, CSV export. |
| **Experiment History** | Every run in SQLite: search and filter, reopen, replay the recorded events, re-run the same request, CSV export. |

## Results on this machine

Apple M5 Pro, **CPU** (Apple GPU (MPS) is not usable, see [Limitations](#limitations)). Benchmark `58a26ec7`: 2 warm-ups + 10 timed runs per cell, checkpoint-default sampling (T=1, no top-k/top-p). Raw data is in `results/benchmarks/`.

| model | batch | total median | p95 | per sample | first output | words/s | sequential passes |
|---|---|---|---|---|---|---|---|
| ncm-tinystories | 1 | 307.5 ms | 399.3 ms | 307.5 ms | 307.5 ms (complete text) | 340.6 | 11 |
| ncm-tinystories | 4 | 733.7 ms | 788.8 ms | 183.4 ms | 733.7 ms (complete text) | 571.1 | 11 |
| ar-tinystories-33m | 1 | 436.9 ms | 491.3 ms | 436.9 ms | 3.9 ms (first token) | 239.5 | 128 |
| ar-tinystories-33m | 4 | 1313.0 ms | 1430.3 ms | 328.2 ms | 10.2 ms (first token) | 304.7 | 128 |

Quality, from one comparison run (4 samples, seed 0): gen-PPL under GPT-2-large is **13.48** for NCM and **9.74** for the baseline (lower is more fluent). Entropy over GPT-2 tokens is 4.056 and 3.993.

The official TinyStories evaluation reproduced on CPU (5 seeds × 256 samples, unmodified `eval_shard.py` + `eval_merge.py`) gives **gen-PPL 16.23 ± 0.48, entropy 3.967**. The paper (Table 8) reports 15.41 / 3.98.

**Reading these numbers**
- The models differ in size (125M including the VQ-VAE vs. 68.5M), tokenizer (8192-BPE vs. GPT-2 BPE) and training run.
- NCM always fills a 128-token canvas; the baseline stops at EOS.
- "First output" means the first token for the baseline and the complete text for NCM.
- The paper's speedups were measured on an H100 against diffusion models; these CPU numbers against an autoregressive model are a different experiment.

## Architecture

```
ncm-research-playground/
├─ research-ncm/          official HazyResearch/ncm, cloned at a pinned commit, never modified (gitignored)
│  ├─ artifacts/          downloaded checkpoints (tinystories/, xsum/, wmt14/)
│  └─ data/               downloaded test sets (xsum/, wmt14/)
├─ backend/               FastAPI + PyTorch
│  ├─ app/core/           research.py (official loading path), official.py (pick_tokens read from eval_shard.py), config.py
│  ├─ app/services/       ncm_runner, ar_runner, registry, trace_collector, quality, experiment_*, event_hub, benchmark, memory
│  ├─ app/routes/         models, generate, experiments (+ WebSocket), benchmarks
│  ├─ app/schemas/        Pydantic request/response/trace models
│  └─ tests/              unit, API-contract and real-model integration tests
├─ frontend/              React 19 + Vite + TypeScript + Tailwind 4 + shadcn/ui + Zustand + Recharts
├─ experiments/           standalone scripts: smoke test, official-script CPU launcher, level probe
├─ results/               logs, official eval outputs, benchmark CSVs, playground.sqlite (gitignored)
└─ docs/                  ASSESSMENT.md (architecture + compatibility), IMPLEMENTATION_LOG.md (per milestone)
```

**How faithfulness is kept**
- **Loading** mirrors `scripts/eval_shard.py`: the shipped config, `VQVAETokenizer.from_config`, `NextConceptTransformer`, strict state dict.
- **Inputs** are built as in the official script: the level-0 unigram draw for TinyStories, prefix truncation and padding for XSum/WMT14, official test items via `example_index`.
- **Final decoding** runs `pick_tokens`, taken verbatim from `eval_shard.py` by parsing its source.
- **Tests** check that the backend's outputs are **identical** to the official script's for TinyStories (seed 42), XSum and WMT14 (test items 0–3).
- **Tracing** temporarily wraps two instance methods the generate loop calls once per level (`get_next_autoregressive_input`, `_sample_level`) and restores them afterwards. Events and timings come from those real calls.
  - The official `idx_to_zhats` returns aliased latents, so the collector clones them.
- **Per-level text** is the official `decode_multiscales` (decoder argmax on the cumulative latent after level k). It is labelled separately from the final output, which uses repetition penalties for XSum.
- **Quality** uses the official `evaluation/gen_ppl.py::generation_perplexity` with GPT-2-large.

## Setup

Tested with macOS 27 (arm64), Python 3.12, Node 26, `uv`. A CUDA GPU is optional; CPU is the default.

```bash
cd ncm-research-playground
git clone https://github.com/HazyResearch/ncm research-ncm
git -C research-ncm checkout c70274273f489abc73832b2b98584a8483b6b203
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r backend/requirements.txt
```

Download the checkpoints and test sets at the pinned revisions (about 2.2 GB of checkpoints plus 20 MB of test sets):

```bash
cd research-ncm
../.venv/bin/python -c "from huggingface_hub import snapshot_download as d; d('hazyresearch/ncm-checkpoints', revision='d342ce8aaea58830eef3b8ad473664673e41f692', allow_patterns=['tinystories/*','xsum/*','wmt14/*'], local_dir='artifacts'); d('hazyresearch/ncm-datasets', repo_type='dataset', revision='6580ba41a4b0a976dde876fbe84d25327617b0e6', allow_patterns=['xsum/*','wmt14/*'], local_dir='data')"
cd ..
```

The baseline (291 MB) and the quality judge (GPT-2-large, about 3.1 GB) go into the Hugging Face cache:

```bash
.venv/bin/python -c "from huggingface_hub import snapshot_download as d; d('roneneldan/TinyStories-33M', revision='2ad0a164221b7c4d21cac7c46aec74f6f98dbfc8'); d('gpt2-large')"
```

Frontend:

```bash
cd frontend && npm install && cd ..
```

Security note: the NCM checkpoints are Python pickles, loaded by the official code with `torch.load(weights_only=False)`. They come from the official repository at pinned revisions.

## Run

```bash
cd backend && ../.venv/bin/uvicorn --factory app.main:create_app --host 127.0.0.1 --port 8000
```

```bash
cd frontend && npm run dev
```

Open http://localhost:5173. Models load on first use and stay in memory.

**Server environment variables**

| var | default | |
|---|---|---|
| `NCM_DEVICE` | `cpu` | `cuda` works if available; `mps` fails, see below |
| `NCM_DB_PATH` | `results/playground.sqlite` | experiments and benchmarks |
| `NCM_RESEARCH_ROOT` | `research-ncm/` | official implementation |
| `NCM_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | |
| `NCM_BACKEND_URL` (frontend) | `http://127.0.0.1:8000` | Vite proxy target for `/api` and the WebSocket |

## Tests

```bash
cd backend && ../.venv/bin/pytest -q               # all: unit, API contracts, real-model integration
cd backend && ../.venv/bin/pytest -q -m "not model" # without checkpoints
cd frontend && npm test && npx tsc -b && npm run build
```

Real-model tests skip themselves when checkpoints are missing. Some compare against the official reference outputs in `results/official_eval/smoke/*_n4.pkl`, which you can regenerate with:

```bash
.venv/bin/python experiments/run_official.py scripts/eval_shard.py --config config/eval/tinystories.yaml --checkpoint-dir artifacts/tinystories --n 4 --bs 4 --out "$PWD/results/official_eval/smoke/tinystories_n4.pkl"
```

The official evaluation scripts hard-code CUDA. `experiments/run_official.py` runs them **unmodified** on CPU by remapping `cuda` device requests.

## API

| method | path | |
|---|---|---|
| GET | `/api/health`, `/api/models` | status; models with availability, levels, tokenizer, parameters once loaded |
| POST | `/api/generate` | synchronous generation (returns outputs, timing, trace) |
| POST | `/api/experiments` | queued generation, recorded in SQLite; returns `stream_url` |
| WS | `/api/stream/{id}` | `generation.started`, `model.loading/loaded`, `generation.level_started/level_completed` (NCM), `generation.step_completed` (AR), `generation.completed/failed`; replayed from the database after completion |
| GET | `/api/experiments`, `/api/experiments/{id}`, `/trace`, `/metrics`, `/api/experiments.csv` | records, trace, metrics, CSV export |
| POST | `/api/experiments/{id}/quality` | official gen-PPL (GPT-2-large) for that run's outputs |
| POST/GET | `/api/benchmarks`, `/api/benchmarks/{id}`, `/runs.csv`, `/summary.csv` | benchmark workflow and exports |

Interactive docs: http://127.0.0.1:8000/docs.

## Limitations

- **Apple GPU (MPS):** NCM fails in `research-ncm/models/quant.py` (adaptive pooling with non-divisible sizes is not implemented in PyTorch MPS), so everything runs on CPU. A verified application-side shim is possible but not implemented.
- **Only TinyStories is unconditional;** it cannot take a prompt. Prompts work for XSum and WMT14. Benchmarks use the unconditional workload only.
- **Full XSum/WMT14 metric evaluations** (11,334 / 3,003 items) were not run on CPU; only 4-item smoke tests that match the official outputs exactly.
- **No partial sentences exist while a level is being predicted:** a level is one parallel step. The timeline shows one block per level.
- **Latent statistics** use the first 8 samples of a batch, to bound memory. Codebook statistics cover the whole batch.
- **Device memory** is not measured on CPU (PyTorch has no per-request peak there); the UI shows "n/a" and says why.
- **NFE:** the paper lists 12 for TinyStories; the app counts 11 transformer passes plus the level-0 unigram draw.
- **Single worker:** generations and benchmarks run one at a time. Runs interrupted by a server restart are marked failed at startup.

See `docs/ASSESSMENT.md` for the architecture study and `docs/IMPLEMENTATION_LOG.md` for every milestone's commands, results and issues.

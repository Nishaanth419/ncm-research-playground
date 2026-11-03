# NCM vs. Autoregressive LLM: Speed–Quality Trade-off on a T4

Date: 2026-11-03
Status: Draft for review

## Goal

Measure, on identical consumer-grade hardware (free Colab/Kaggle T4), how
Next-Concept Modeling (NCM, arXiv:2610.14162) compares with a same-size
autoregressive (AR) LLM on the speed–quality trade-off for open-ended text
generation, and turn the results into a LinkedIn post with honest caveats.

The paper compares NCM mainly against diffusion LMs and reports no matched-size
AR baseline for unconditional generation. This experiment fills that gap.

## Non-goals

- Training any model. Released checkpoints only.
- Reproducing the paper's H100 wall-clock numbers.
- Conditional tasks (XSum, WMT14) and Sudoku. TinyStories is optional extra
  qualitative material only if time remains.

## Models

| Role | Model | Size | Source |
|---|---|---|---|
| NCM | `owt` checkpoint | ~100M generator | `hazyresearch/ncm-checkpoints` (HF), code `HazyResearch/ncm` |
| AR baseline | GPT-2 small | 124M | `gpt2` (HF) |
| Judge | GPT-2-large | 774M | `gpt2-large` (HF), same judge as the paper |

Known asymmetry to disclose: GPT-2 small saw far more training tokens than
NCM-owt.

## Experiments

### E1: Speed–quality frontier (headline)

- Output length: 1024 tokens.
- NCM sweep: temperature, plus per-level refinement steps / guidance if the
  `config/eval/owt.yaml` exposes them. Only parameters that exist are swept.
- GPT-2 sweep: temperature × top-p, each with KV cache on (primary) and off
  (reference).
- 128 samples per setting (reduced to 64 if a full run exceeds ~2 h), 3 seeds.

### E2: Length scaling

- Lengths 128, 256, 512, 1024.
- One default setting per model (NCM paper default; GPT-2 temperature 1.0,
  top-p 1.0, KV cache on).

### Metrics (per sample, aggregated per setting)

- Wall-clock seconds per sample (GPU-synchronised; model load excluded)
- Tokens per second
- Model forward passes (NFE)
- Generative perplexity under GPT-2-large
- Mean unigram entropy (degeneracy guard)

### Fairness controls

- Same GPU, same batch size for both models (start at 16, drop to 8/4 on OOM;
  recorded per row).
- Same precision where possible (fp16; fallback recorded per row).
- 2 warm-up batches before timing.
- Same judge model and tokenisation of text for scoring.

## Repository layout

```
NCM/
├─ ncm_vs_llm.ipynb      # orchestrates everything top to bottom on a T4
├─ src/
│  ├─ setup_env.py       # install deps, clone HazyResearch/ncm, download owt checkpoint
│  ├─ gen_ncm.py         # NCM batch generation -> GenResult
│  ├─ gen_ar.py          # GPT-2 batch generation (KV cache on/off) -> GenResult
│  ├─ score.py           # GPT-2-large perplexity + unigram entropy
│  ├─ timing.py          # CUDA-synced timer, warm-up helper
│  ├─ runlog.py          # append rows to results/runs.csv, skip finished settings
│  └─ plots.py           # frontier + length-scaling charts from runs.csv
├─ tests/                # CPU-only unit tests with tiny models
├─ results/              # runs.csv, samples.jsonl
└─ post/
   ├─ figures/           # PNG at 1200×1200 and 1200×627
   └─ linkedin_draft.md
```

### Interfaces

- `GenResult`: `texts: list[str]`, `seconds: float`, `nfe: int`,
  `tokens_generated: int`, `config: dict`.
- `score(texts) -> list[{ppl: float, entropy: float}]`.
- `runlog.append(row)` writes immediately; `runlog.done(key)` lets the notebook
  skip completed `(model, setting, length, seed)` keys after a disconnect.
- `plots.py` reads only `runs.csv`, so charts can be regenerated without
  re-running generation.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| NCM code assumes bf16 or kernels unsupported on T4 | Smoke-test cell first; fall back fp16 → fp32; record precision. |
| No Python-level sampling API (only `evaluate.py`) | Import its model/sampler classes directly; if impractical, time the CLI and subtract separately measured load time. |
| No steps-per-level knob | Sweep available knobs only; state reduced NCM frontier in the post. |
| OOM at 1024 tokens on 16 GB | Reduce batch size equally for both models. |
| Colab disconnects | Incremental CSV + skip-completed logic; target ≤2 h per full run. |
| Async CUDA skews timing | `torch.cuda.synchronize()` around timed regions, warm-ups, exclude load. |

## Validation (before trusting any number)

1. GPT-2 small at temperature 1.0 yields a GPT-2-large perplexity in line with
   published figures; otherwise the scorer is wrong.
2. NCM at paper defaults yields perplexity and entropy close to the paper's
   OWT numbers; otherwise our NCM integration is wrong.
3. Settings whose mean entropy is below 4.0 are flagged as degenerate and
   excluded from "wins" on the chart (shown greyed out).
4. Unit tests for `score.py` (entropy on known strings, perplexity with a tiny
   model) and `timing.py` pass on CPU.

Only settings passing these checks, averaged over 3 seeds with spread shown,
are reported as findings.

## Deliverables

1. Frontier chart: perplexity vs. seconds per sample, one line per model.
2. Length-scaling chart: seconds per sample vs. output length.
3. Side-by-side sample table at matched quality.
4. `post/linkedin_draft.md`: ~200 words, numbers filled from `runs.csv`, with
   caveats (T4 not H100, training-data asymmetry, ~100M scale only, the
   paper's results not yet independently replicated).

## Conventions

- Git branch names and commit messages contain no tool or assistant names, and
  commits carry no co-author or attribution trailers.

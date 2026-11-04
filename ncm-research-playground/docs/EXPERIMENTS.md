# Experiments: NCM vs. a traditional language model

Everything below was run on one machine (Apple M5 Pro, CPU) with the official code and checkpoints. Raw outputs, logs and CSVs are in `results/`.

## The paper

**Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation.** Hermann Kumbong, Ishan S. Khare, Iddah Mlauzi, Andrew Woen, Charles Ding, E. Kelly Buchanan, Simran Arora, Azalia Mirhoseini, Christopher Ré (Stanford). 2026.

| | |
|---|---|
| Paper | https://arxiv.org/abs/2610.14162 |
| Official code | https://github.com/HazyResearch/ncm (used at commit `c70274273f489abc73832b2b98584a8483b6b203`) |
| Checkpoints and data | https://huggingface.co/collections/hazyresearch/ncm |
| Blog posts | https://hazyresearch.stanford.edu/blog/2026-10-12-ncm, https://scalingintelligence.stanford.edu/blogs/ncm/ |

**Idea in one paragraph.** A multiscale residual VQ-VAE compresses a passage into a stack of levels of discrete codes: 1 code for the whole passage at the top, up to one code per position at the bottom. A transformer predicts that stack coarse to fine, all codes of a level in one forward pass. A 128-token TinyStories story therefore takes 11 transformer passes, where an autoregressive model takes up to 128.

## The question

How does NCM compare with a traditional (autoregressive) language model of the same domain on speed and text quality, on ordinary hardware? And is what the paper reports reproducible from the released code?

## Models

| Role | Model | Source (pinned) | Size |
|---|---|---|---|
| NCM, stories | `ncm-tinystories` | `hazyresearch/ncm-checkpoints` @ `d342ce8aaea58830eef3b8ad473664673e41f692`, folder `tinystories/` | 125.0M (generator 88.6M + VQ-VAE 36.4M) |
| NCM, summarisation | `ncm-xsum` | same repo, `xsum/` | |
| NCM, translation | `ncm-wmt14` | same repo, `wmt14/` | |
| Traditional LM baseline | TinyStories-33M | `roneneldan/TinyStories-33M` @ `2ad0a164221b7c4d21cac7c46aec74f6f98dbfc8` | 68.5M |
| Quality judge | GPT-2-large | `gpt2-large` | 774M |

Both TinyStories models were trained on the same dataset, which is what makes them a fair pair. They still differ in size, tokenizer (NCM: 8192-entry BPE; baseline: GPT-2 BPE) and training run.

## Datasets

| Dataset | What it is | Used for | Size used | Link |
|---|---|---|---|---|
| **TinyStories** | Short, simple synthetic children's stories (Eldan & Li, 2023) | Training data of both story models. Generation is unconditional: no input, so no test items are needed; the models' own samples are scored. | 5 seeds × 256 samples (official eval); 4 samples (comparison); 10 runs × batch 1 and 4 per model (benchmark) | https://huggingface.co/datasets/roneneldan/TinyStories |
| **XSum** | BBC news articles with one-sentence summaries | Test set, NCM summarisation (article → summary) | Test set has 11,334 pairs; items 0–3 run here | https://huggingface.co/datasets/hazyresearch/ncm-datasets/tree/main/xsum (pinned @ `6580ba41a4b0a976dde876fbe84d25327617b0e6`) |
| **WMT14 German→English** | News sentence pairs | Test set, NCM translation | Test set has 3,003 pairs; items 0–3 run here | https://huggingface.co/datasets/hazyresearch/ncm-datasets/tree/main/wmt14 (same pin) |

Input lengths come from the official configs: XSum uses up to 512 article tokens (GPT-2 tokenizer) and produces a 48-token summary; WMT14 uses 96 German tokens and produces 64 English tokens.

## Hardware and software

Apple M5 Pro, **CPU only**, fp32, 6 torch threads. Python 3.12, torch 2.15.0, transformers 5.22.1. The Apple GPU (MPS) cannot run NCM: PyTorch's MPS backend does not implement the adaptive pooling NCM uses (`research-ncm/models/quant.py`).

## Experiment 1: reproduce the paper's TinyStories result

**Method.** The official evaluation, unmodified (`scripts/eval_shard.py` + `scripts/eval_merge.py`): 5 seeds × 256 unconditional samples, scored by **generative perplexity (gen-PPL)** under GPT-2-large (lower = more fluent) and **unigram entropy** (diversity; too low means repetitive). The scripts hard-code CUDA, so `experiments/run_official.py` runs them on CPU by remapping device requests only.

| seed | gen-PPL | entropy |
|---|---|---|
| 42 | 15.88 | 3.9639 |
| 0 | 16.90 | 3.9763 |
| 1 | 15.74 | 3.9669 |
| 2 | 16.56 | 3.9632 |
| 3 | 16.10 | 3.9671 |
| **mean** | **16.23 ± 0.48** | **3.967** |
| paper (Table 8) | 15.41 | 3.98 |

**Result.** Reproduced within 2 seed standard deviations; entropy matches. The gap is plausibly device and library versions (CPU fp32, newer torch and transformers). Generation took 4 min 24 s, scoring 18 min 37 s. Logs: `results/official_eval/tinystories/`.

## Experiment 2: does the app run the model faithfully?

**Method.** The app's backend generates through its own code path, so its outputs were compared with the official script's outputs for the same inputs and seeds.

**Result.** Outputs are **identical** for TinyStories (seed 42, 4 samples), XSum test items 0–3 and WMT14 test items 0–3. This is enforced by automated tests (`backend/tests/`). References: `results/official_eval/smoke/*_n4.pkl`.

Example (WMT14, free text): "Ich trinke gern Kaffee am Morgen." → "I like like coffee in the morning." The XSum and WMT14 checkpoints work but show visible repetition in some outputs.

## Experiment 3: what happens at each level?

**Method.** `experiments/probe_levels.py` wraps two methods the official generate loop calls once per level and records the codes, timing and the text the decoder reads after each level (official `decode_multiscales`).

**Result.** TinyStories uses 12 levels with `[1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128]` codes, each from its own 4096-entry codebook. Level 0 is drawn from a unigram stored in the checkpoint, so there are **11 transformer passes** (the paper's table counts 12 NFE). Early levels decode to word salad with the right characters and topic; the story becomes coherent only in the last levels. Log: `results/logs/probe_levels.log`. The app's **How NCM works** page animates this from a real run.

## Experiment 4: head-to-head quality (NCM vs. TinyStories-33M)

**Method.** Same seed (0), batch (4 samples), sampling (temperature 1, no top-k/top-p), both warmed up, back to back; gen-PPL under GPT-2-large. One run, not a benchmark.

| | NCM | TinyStories-33M |
|---|---|---|
| sequential forward passes | 11 | 128 |
| generate time (4 samples) | 797 ms | 1.58 s |
| gen-PPL (lower = more fluent) | **13.48** | **9.74** |
| entropy over GPT-2 tokens | 4.056 | 3.993 |

**Result.** The autoregressive baseline writes more fluent text; NCM is faster and slightly more diverse.

## Experiment 5: speed benchmark

**Method.** Benchmark `58a26ec7`: for each model and batch size, 2 untimed warm-ups, then 10 timed runs (seeds 0–9), checkpoint-default sampling. Medians, p90, p95. Raw data: `results/benchmarks/58a26ec74500472a85196c789f305bfa-{runs,summary}.csv`.

| model | batch | total median | p95 | per sample | first output | words/s |
|---|---|---|---|---|---|---|
| NCM | 1 | 307.5 ms | 399.3 ms | 307.5 ms | 307.5 ms (whole text) | 340.6 |
| NCM | 4 | 733.7 ms | 788.8 ms | 183.4 ms | 733.7 ms (whole text) | 571.1 |
| TinyStories-33M | 1 | 436.9 ms | 491.3 ms | 436.9 ms | 3.9 ms (first token) | 239.5 |
| TinyStories-33M | 4 | 1313.0 ms | 1430.3 ms | 328.2 ms | 10.2 ms (first token) | 304.7 |

**Result.** NCM finishes a whole story **1.42× faster at batch 1 and 1.79× faster at batch 4**. The baseline shows its first token almost immediately, but NCM has nothing to show until the whole text is done.

## Findings

1. **The paper's TinyStories quality reproduces** on a laptop CPU: gen-PPL 16.23 vs. 15.41 reported, same entropy.
2. **NCM is faster end to end**: 1.4–1.8× on CPU against an autoregressive model of the same domain, because it needs 11 sequential passes instead of 128. The advantage grows with batch size.
3. **The autoregressive model writes more fluent text** (gen-PPL 9.74 vs. 13.48) and streams its first token in milliseconds.
4. **Coarse to fine is real and visible**: the passage is planned at the top levels and filled in at the bottom ones, and every intermediate level can be decoded.

## What these numbers are not

- The paper's speedups (9–31×) were measured on an H100 GPU against **diffusion** models. These are CPU numbers against an **autoregressive** model: a different experiment, not a contradiction.
- The models differ in size (125M vs. 68.5M), tokenizer and training run.
- NCM always fills the 128-token canvas; the baseline may stop early at end-of-text.
- The quality comparison is one 4-sample run; the benchmark (10 runs per cell) measures speed only.
- Full XSum (11,334) and WMT14 (3,003) evaluations were not run on CPU; only 4-item exact-match checks.

## Reproduce

Setup is in the root `README.md`. Then, from `ncm-research-playground/`:

```bash
# Experiment 1: official TinyStories eval on CPU (about 25 minutes; downloads GPT-2-large)
O=$PWD/results/official_eval/tinystories
.venv/bin/python experiments/run_official.py scripts/eval_shard.py --config config/eval/tinystories.yaml --checkpoint-dir artifacts/tinystories --out $O/shard_0.pkl
.venv/bin/python experiments/run_official.py scripts/eval_merge.py --config config/eval/tinystories.yaml --glob "$O/shard_*.pkl"

# Experiment 2: official outputs for 4 items, then the exact-match tests
O=$PWD/results/official_eval/smoke
.venv/bin/python experiments/run_official.py scripts/eval_shard.py --config config/eval/xsum.yaml --checkpoint-dir artifacts/xsum --data-dir data/xsum --n 4 --bs 4 --out $O/xsum_n4.pkl
(cd backend && ../.venv/bin/pytest -q -m model)

# Experiment 3: per-level probe
.venv/bin/python experiments/probe_levels.py
```

The comparison and the benchmark are run from the app (`./run.sh`, then **Model Comparison** and **Benchmarks**), which also exports them as CSV.

## Cite

```bibtex
@misc{kumbong2026nextconcept,
  title={Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation},
  author={Hermann Kumbong and Ishan S. Khare and Iddah Mlauzi and Andrew Woen and Charles Ding and E. Kelly Buchanan and Simran Arora and Azalia Mirhoseini and Christopher Ré},
  year={2026},
  eprint={2610.14162},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2610.14162},
}
```

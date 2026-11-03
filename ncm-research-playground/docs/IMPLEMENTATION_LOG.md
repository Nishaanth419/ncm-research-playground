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

## Milestone 2: Install NCM and run real TinyStories inference/evaluation, in progress

**Done**
- `experiments/smoke_tinystories.py`: device-parameterised mirror of the unconditional path of `eval_shard.py`.
- CPU run succeeded: 4 coherent samples, codes `(4, 641)`, batch latency 0.77 s, official per-level decodes printed. Log: `results/logs/smoke_tinystories_cpu.log`.

**Errors**
- MPS: `Adaptive pool MPS: input sizes must be divisible by output sizes` at `research-ncm/models/quant.py:703`. `PYTORCH_ENABLE_MPS_FALLBACK=1` does not help. Log: `results/logs/smoke_tinystories_mps.log`.
- `hf download --include ... --revision <sha>` failed for the model repo; `snapshot_download` worked.

**Unresolved**
- Official evaluation not yet run: `evaluate.py` and `eval_shard.py` require CUDA. Waiting on decision 3 in `docs/ASSESSMENT.md`.
- XSum inference not yet tried.

**Next**
- Resolve the three decisions in `docs/ASSESSMENT.md`, then run the CPU-port TinyStories evaluation and an XSum smoke test.

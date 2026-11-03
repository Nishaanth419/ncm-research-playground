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

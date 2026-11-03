# NCM Research Playground: Assessment (Milestone 1)

Date: 2026-11-03. Everything below was checked against the checked-out code,
not taken from the paper or README alone.

- Official repo: `HazyResearch/ncm` @ `c70274273f489abc73832b2b98584a8483b6b203` (2026-10-14), cloned to `research-ncm/` and left unmodified.
- Checkpoints: `hazyresearch/ncm-checkpoints` @ `d342ce8aaea58830eef3b8ad473664673e41f692` (`tinystories/`, `xsum/`).
- Datasets: `hazyresearch/ncm-datasets` @ `6580ba41a4b0a976dde876fbe84d25327617b0e6` (`xsum/` test set only).
- The shipped `config.yaml` of both checkpoints is byte-identical to `research-ncm/config/eval/<task>.yaml` (verified with `diff`).

## 1. NCM architecture as implemented

**Two frozen stages per released model**

1. **Tokenizer = BPE pre-tokenizer + multiscale residual VQ-VAE** (`tokenization/vqvae_tokenizer.py`, `models/vqvae.py`, `models/quant.py::MultiscaleResidualVectorQuantizer`).
   - Text → BPE ids (TinyStories: 8192-vocab BPE, 128 tokens).
   - Encoder → continuous latent `z` of shape `[B, 128, 384]`.
   - The quantizer represents `z` as a **residual hierarchy of 12 levels** with lengths
     `[1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128]` (TinyStories), each with its own 4096-entry codebook.
   - Level *i*'s codes are embedded, upsampled to length 128 (`_up_to_full`), passed through a `Phi` residual conv (`_apply_quant_resi`), and **added** into the running latent `z_hat`.
   - One sample = 641 codes (sum of level lengths).
   - The decoder maps `z_hat` back to BPE logits; argmax gives the tokens.
2. **Generator = `NextConceptTransformer`** (`models/next_concept.py`, 88.6M params for TinyStories; VQ-VAE 36.4M).

**Generation (`NextConceptTransformer.generate`)**

- **Level 0** is not predicted by the transformer for TinyStories (`predict_level0: false`). It is drawn from a unigram stored in the checkpoint (`generator.pt → inference.level0_unigram`), exactly as `scripts/eval_shard.py` does.
- For each subsequent level `k = 1 … 11` (**11 sequential transformer forward passes**):
  1. Embed level `k-1`'s codes.
  2. Add them into `z_hat` (`vq.get_next_autoregressive_input`).
  3. Area-downsample `z_hat` to level `k`'s length; this is the transformer's input.
  4. **One forward pass predicts all `levels[k]` codes of level `k` in parallel** (`_sample_level`, with temperature / top-k / top-p).
- Levels are sequential; positions within a level are parallel. Optional KV cache for the prefix.
- Final text: `vq.idx_to_zhats(codes)` → last `z_level` → `vqvae.decoder` → argmax (or the sequential `pick_tokens` repetition penalty, which the XSum config enables) → BPE decode, cut at EOS.

**Conditioning**

| Checkpoint | Prefix | Output | Notes |
|---|---|---|---|
| tinystories / owt / lm1b | 1 token (EOS) | 128 / 1024 / 128 tokens | **Unconditional: no user prompt possible** |
| xsum | 512 article tokens (GPT-2 tokenizer) | 48-token summary | frequency + adjacency penalties in final decode |
| wmt14 | 96 German tokens | 64 English tokens | |
| sudoku | board | 89 tokens | |

**Officially supported intermediate outputs**

- `VQVAETokenizer.decode_multiscales(idx)` decodes the text implied by the cumulative latent after **each** level (the README's coarse-to-fine GIF).
- These are real model-derived decodes, not fabricated partial sentences. They are *not* the final text truncated: level *k* is decoded at length `levels[k]` from the partial latent.

**Implementation gotcha found while reading**

`MultiscaleResidualVectorQuantizer.idx_to_zhats` appends the **same** `z_hat` tensor to `z_hats` after in-place `add_`, so every element of the returned `z_hats` list aliases the final latent. The `z_levels` list is fine (new tensors from `F.interpolate`, except the last). Instrumentation must clone per level and not trust `z_hats[i]`.

## 2. Compatibility of this machine

| Item | Value |
|---|---|
| OS | macOS 27.0.1 (Darwin 27.0.0, arm64) |
| CPU / GPU | Apple M5 Pro, 20-core GPU, Metal 4; **no CUDA** |
| RAM | 24 GB unified |
| Disk free | 664 GB |
| Python | system 3.15 (too new for wheels); project venv uses **3.12.13** via `uv` |
| PyTorch | 2.15.0, MPS available |
| Node | 26.4.0 |

**Findings**

- **The official entry points do not run here unmodified.**
  - `evaluate.py` exits with `no GPU visible` unless `--gpus` is passed.
  - `scripts/eval_shard.py` hard-codes `.cuda()` throughout.
  - The model code itself is device-agnostic.
- **CPU: works.** Real TinyStories sampling with the official model code, loaded the way `eval_shard.py` loads it: a batch of 4 in **0.77 s** after warm-up. Samples are coherent stories. Log: `results/logs/smoke_tinystories_cpu.log`.
- **MPS: fails** in `quant.py:703` with `Adaptive pool MPS: input sizes must be divisible by output sizes` ([pytorch#96056](https://github.com/pytorch/pytorch/issues/96056)).
  - The cause is area interpolation from 128 to non-divisors (24, 40, 52, …).
  - `PYTORCH_ENABLE_MPS_FALLBACK=1` does not help (the op exists, the size case is unimplemented).
  - Log: `results/logs/smoke_tinystories_mps.log`.
- **Official TinyStories evaluation** = 5 seeds × 256 samples + GPT-2-large gen-ppl + entropy.
  - It is feasible on CPU via a device-parameterised port of `eval_shard.py` + `eval_merge.py`. Estimated ~5 min generation plus GPT-2-large scoring (~3.1 GB download).
  - The H100 timings in the paper cannot be reproduced here and will not be compared directly.

## 3. Installation plan (executed so far)

```bash
cd ncm-research-playground
git clone https://github.com/HazyResearch/ncm research-ncm   # pinned SHA above
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r research-ncm/requirements.txt
cd research-ncm
../.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('hazyresearch/ncm-checkpoints', revision='d342ce8aaea58830eef3b8ad473664673e41f692', allow_patterns=['tinystories/*','xsum/*'], local_dir='artifacts')"
../.venv/bin/hf download hazyresearch/ncm-datasets --repo-type dataset --revision 6580ba41a4b0a976dde876fbe84d25327617b0e6 --include "xsum/*" --local-dir data
```

Note: `hf download --include` with a pinned `--revision` failed for the model repo; `snapshot_download` with the same revision worked.

Security note: the checkpoints and BPE files are Python pickles (`torch.load(weights_only=False)` in the official code). They come from the official repo the project is built on, at a pinned revision.

## 4. Verified commands: smallest practical experiment

```bash
cd ncm-research-playground
.venv/bin/python experiments/smoke_tinystories.py --device cpu --n 4
```

Prints load time, parameter counts, the level schedule, code tensor shape `(4, 641)`, timed batch latency, four final stories, and the official per-level decodes for sample 0.

## 5. Instrumentation strategy (proposed, not yet built)

The official code stays unmodified. Tracing is done from the application side.

**What to capture, and how**

- **Per-level codes and timing:** wrap the *instance* methods `net._sample_level` and `vq.get_next_autoregressive_input` for the duration of one `generate` call.
  - Each `_sample_level` return is one level's code tensor `[B, levels[k]]`.
  - Device-synchronised timestamps around each call give per-level wall time.
  - The level-0 unigram draw is recorded before the call.
- **Per-level latent summaries** from the `get_next_autoregressive_input` wrapper, using cloned `z_hat` (see the aliasing gotcha). Stored as scalars and small vectors only:
  - `‖z_hat‖`
  - `‖Δz‖` contributed by the level
  - cosine similarity of the level's latent to the final latent
- **Per-level text:** `tk.decode_multiscales(codes)`, the official function.
- **Codebook statistics per level**, computed from the code tensor slices:
  - unique codes used / 4096
  - usage entropy
  - top-k code histogram (k ≤ 64)
- **Storage:** JSON with ints/floats only. Full code index arrays stay small (641 ints per sample) and are stored. No float tensors are serialised.

**Streaming**

- Generation runs in a worker thread.
- The wrappers push `generation.level_started` / `generation.level_completed` events onto an asyncio queue as the real calls happen.
- The WebSocket relays them.
- Because a level is a single forward pass, there are no within-level progress events; within-level parallelism is shown as one block per level.

**What cannot be extracted, and the labelled alternative**

- Per-position "partial sentences while a level is being generated" do not exist: a level is produced in one parallel step.
- The UI will show only per-level decodes, labelled "decoded from cumulative latent after level k (official `decode_multiscales`)".

## Decisions needed before Milestone 2 continues

1. **Prompts:** TinyStories is unconditional. The Playground's "prompt" input is only meaningful for XSum (article → summary) and WMT14 (German → English). For TinyStories the controls are seed / temperature / top-k / top-p and batch size.
2. **Device:** run on CPU (works now, unmodified), or add an application-side MPS shim that replaces non-divisible `F.interpolate(mode="area")` with an exact averaging-matrix equivalent. The shim would be verified against CPU outputs code-for-code before use, and is never applied to `research-ncm/`.
3. **Official eval:** run the CPU port of the TinyStories eval (downloads GPT-2-large, ~3.1 GB) now, or defer it to a cloud GPU where `evaluate.py` runs unmodified.

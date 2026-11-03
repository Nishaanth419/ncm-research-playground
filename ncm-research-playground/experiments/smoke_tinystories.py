"""Minimal real NCM TinyStories sampling on a non-CUDA device.

Mirrors the unconditional path of research-ncm/scripts/eval_shard.py
(same loading, level-0 unigram draw, net.generate, final decode) with the
device as a parameter, because the official script hard-codes .cuda().
Also prints the official per-level decodes (VQVAETokenizer.decode_multiscales).
"""
import argparse
import os
import sys
import time

import torch
from omegaconf import OmegaConf

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "research-ncm")
sys.path.insert(0, ROOT)
from models.next_concept import NextConceptTransformer  # noqa: E402
from tokenization.vqvae_tokenizer import VQVAETokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
dev = a.device

ck = os.path.join(ROOT, "artifacts", "tinystories")
CFG = OmegaConf.load(os.path.join(ROOT, "config", "eval", "tinystories.yaml"))
G = CFG.generation
CFG.tokenizer.vqvae.checkpoint_path = os.path.join(ck, CFG.checkpoints.tokenizer)
CFG.tokenizer.pre_tokenizer.vocab_file_path = os.path.join(ck, CFG.checkpoints.bpe_vocab)
CFG.tokenizer.pre_tokenizer.merges_file_path = os.path.join(ck, CFG.checkpoints.bpe_merges)

t0 = time.perf_counter()
tk = VQVAETokenizer.from_config(CFG.tokenizer)
tk.vqvae = tk.vqvae.to(dev).eval()
net = NextConceptTransformer(vqvae_local=tk.vqvae, config=CFG.model).to(dev)
sd = torch.load(os.path.join(ck, CFG.checkpoints.generator), map_location="cpu", weights_only=False)
net.load_state_dict(sd["model"], strict=True)
net.eval()
path, ref = str(G.level0_unigram).split(":", 1)
L0_P = sd
for k in ref.split("."):
    L0_P = L0_P[k]
L0_P = L0_P.double().float()
EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
print(f"loaded on {dev} in {time.perf_counter() - t0:.1f}s; "
      f"generator params={sum(p.numel() for p in net.parameters()) / 1e6:.1f}M; "
      f"levels={list(tk.vqvae.vq.levels)}")


def sync():
    if dev == "mps":
        torch.mps.synchronize()
    elif dev.startswith("cuda"):
        torch.cuda.synchronize()


torch.manual_seed(a.seed)
B, L0 = a.n, tk.vqvae.vq.levels[0]
l0 = torch.multinomial(L0_P, B * L0, replacement=True).view(B, L0).to(dev)
x = torch.full((B, 1), EOS, device=dev)
with torch.no_grad():
    for i in range(2):  # warm-up + timed
        sync(); t = time.perf_counter()
        g = net.generate(l0, x, temperature=float(G.temperature), top_k=int(G.top_k),
                         top_p=float(G.top_p), cfg_scale=float(G.cfg_scale), cfg_null_eos=EOS)
        _, zl = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
        ids = tk.vqvae.decoder(zl[-1]).argmax(-1).tolist()
        sync(); dt = time.perf_counter() - t
    print(f"codes shape={tuple(g.shape)} (sum of levels={sum(tk.vqvae.vq.levels)}); "
          f"batch of {B} in {dt * 1000:.0f} ms ({'timed' if i else 'warm-up'})")
    levels = tk.decode_multiscales(g, skip_special_tokens=True)


def dec(r):
    if EOS in r:
        r = r[:r.index(EOS)]
    return tk.pre_tokenizer.decode([t for t in r if t not in (EOS, PAD)], skip_special_tokens=True)


for b in range(B):
    print(f"\n=== sample {b} final ===\n{dec(ids[b])}")
print("\n=== sample 0, official per-level decodes (decode_multiscales) ===")
for li, (n, txt) in enumerate(zip(tk.vqvae.vq.levels, levels[0])):
    print(f"[L{li:02d} n={n:3d}] {txt[:160]!r}")

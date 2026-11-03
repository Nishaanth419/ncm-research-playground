"""Milestone 3 probe: verify the per-level instrumentation strategy against real NCM.

Wraps the *instance* methods `net._sample_level` and
`vq.get_next_autoregressive_input` for one `generate` call (official code is
not edited) and checks that:
  1. `_sample_level` is called once per generated level (len(levels) - 1),
  2. each call returns codes shaped [B, levels[k]],
  3. level-0 codes + captured per-level codes == generate()'s output exactly,
  4. the z_hat latents returned by official `idx_to_zhats` alias each other
     (so instrumentation must clone), while captured clones differ per level.
Prints per-level wall time (device-synchronised).
"""
import os
import sys
import time

import torch
from omegaconf import OmegaConf

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "research-ncm")
sys.path.insert(0, ROOT)
from models.next_concept import NextConceptTransformer  # noqa: E402
from tokenization.vqvae_tokenizer import VQVAETokenizer  # noqa: E402

dev = "cpu"
ck = os.path.join(ROOT, "artifacts", "tinystories")
CFG = OmegaConf.load(os.path.join(ROOT, "config", "eval", "tinystories.yaml"))
CFG.tokenizer.vqvae.checkpoint_path = os.path.join(ck, CFG.checkpoints.tokenizer)
CFG.tokenizer.pre_tokenizer.vocab_file_path = os.path.join(ck, CFG.checkpoints.bpe_vocab)
CFG.tokenizer.pre_tokenizer.merges_file_path = os.path.join(ck, CFG.checkpoints.bpe_merges)
tk = VQVAETokenizer.from_config(CFG.tokenizer)
tk.vqvae = tk.vqvae.to(dev).eval()
net = NextConceptTransformer(vqvae_local=tk.vqvae, config=CFG.model).to(dev)
sd = torch.load(os.path.join(ck, CFG.checkpoints.generator), map_location="cpu", weights_only=False)
net.load_state_dict(sd["model"], strict=True)
net.eval()
L0_P = sd["inference"]["level0_unigram"].double().float()
EOS = tk.pre_tokenizer.eos_token_id
vq = tk.vqvae.vq
levels = list(vq.levels)

events = []
orig_sample, orig_next = net._sample_level, vq.get_next_autoregressive_input


def sample_level(logits, *a, **k):
    out = orig_sample(logits, *a, **k)
    events.append(("codes", time.perf_counter(), out.clone(), tuple(logits.shape)))
    return out


def next_input(level_idx, z_hat, z_q_level):
    events.append(("level_start", time.perf_counter(), level_idx, None))
    z_hat, x = orig_next(level_idx, z_hat, z_q_level)
    events.append(("latent", time.perf_counter(), z_hat.detach().clone(), tuple(x.shape)))
    return z_hat, x


net._sample_level, vq.get_next_autoregressive_input = sample_level, next_input
torch.manual_seed(0)
B = 2
l0 = torch.multinomial(L0_P, B * levels[0], replacement=True).view(B, levels[0])
with torch.no_grad():
    g = net.generate(l0, torch.full((B, 1), EOS), temperature=1.0, top_k=0, top_p=1.0,
                     cfg_scale=0.0, cfg_null_eos=EOS)
net._sample_level, vq.get_next_autoregressive_input = orig_sample, orig_next

codes = [e for e in events if e[0] == "codes"]
starts = [e for e in events if e[0] == "level_start"]
latents = [e for e in events if e[0] == "latent"]
assert len(codes) == len(levels) - 1, (len(codes), len(levels))
for k, e in enumerate(codes, start=1):
    assert e[2].shape == (B, levels[k]), (k, e[2].shape)
assert torch.equal(torch.cat([l0] + [e[2] for e in codes], dim=1), g)
print(f"OK: {len(codes)} parallel level predictions; captured codes == generate() output {tuple(g.shape)}")

# The latent captured at level k is the transformer's *input* for level k:
# z_hat after adding codes of levels 0..k-1. The true final latent adds level 11.
print(f"{'lvl':>3} {'len':>4} {'logits':>16} {'ms':>7} {'|z_in|':>9} {'cos(final)':>10} {'uniq/4096':>9}")
with torch.no_grad():
    z_final = vq.idx_to_z_q(g)
for k, (s, c, z) in enumerate(zip(starts, codes, latents), start=1):
    cos = torch.nn.functional.cosine_similarity(z[2].flatten(1), z_final.flatten(1)).mean().item()
    print(f"{k:>3} {levels[k]:>4} {str(c[3]):>16} {(c[1] - s[1]) * 1000:7.1f} "
          f"{z[2].norm(dim=-1).mean().item():9.3f} {cos:10.4f} {c[2].unique().numel():>9}")

with torch.no_grad():
    z_hats, _ = vq.idx_to_zhats(g, return_zlevels=True)
aliased = all(z.data_ptr() == z_hats[-1].data_ptr() for z in z_hats)
distinct = not torch.equal(latents[0][2], latents[-1][2])
print(f"official idx_to_zhats z_hats alias one tensor: {aliased}; captured clones distinct: {distinct}")

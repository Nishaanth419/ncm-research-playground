"""Capture real per-level information from one NCM `generate` call.

The official code is not modified. For the duration of a `with TraceCollector(...)`
block the *instance* methods `net._sample_level` and
`vq.get_next_autoregressive_input` are wrapped, which the official generate loop
calls once per level in that order:

    get_next_autoregressive_input(k, ...)   -> level k+1 starts (input latent built)
    _sample_level(logits, ...)              -> level k+1's codes, all positions at once

Events are emitted as those calls actually happen. Latents are cloned because
the official quantizer mutates `z_hat` in place.
"""
import math
import time
from typing import Any, Callable, Optional

import torch

from app.schemas.trace import CodeCount, CodebookStats, LatentSummary, LevelTrace, Trace

EventFn = Callable[[dict[str, Any]], None]


def device_sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def codebook_stats(codes: torch.Tensor, vocab_size: int, top_k: int = 32) -> CodebookStats:
    flat = codes.reshape(-1).to("cpu", torch.long)
    counts = torch.bincount(flat, minlength=vocab_size).double()
    n = int(flat.numel())
    p = counts[counts > 0] / max(n, 1)
    entropy = float(-(p * p.log2()).sum()) if n else 0.0
    vals, idx = counts.topk(min(top_k, int((counts > 0).sum())))
    return CodebookStats(
        vocab_size=vocab_size, n_codes=n, unique=int((counts > 0).sum()),
        utilization=float((counts > 0).sum()) / vocab_size, entropy_bits=abs(entropy),
        perplexity=2.0 ** abs(entropy),
        top=[CodeCount(code=int(i), count=int(v)) for v, i in zip(vals, idx)])


def latent_summary(z: torch.Tensor, prev: torch.Tensor, final: torch.Tensor) -> LatentSummary:
    z, prev, final = (t.detach().float().cpu() for t in (z, prev, final))
    cos = torch.nn.functional.cosine_similarity(z.flatten(1), final.flatten(1), dim=1)
    return LatentSummary(
        norm_mean=float(z.norm(dim=-1).mean()), delta_norm_mean=float((z - prev).norm(dim=-1).mean()),
        cos_to_final=float(cos.mean()), n_samples=int(z.shape[0]))


class _Level:
    def __init__(self, level: int, n_codes: int, source: str):
        self.level, self.n_codes, self.source = level, n_codes, source
        self.codes: Optional[torch.Tensor] = None
        self.logits_shape: Optional[list[int]] = None
        self.t_start: Optional[float] = None
        self.seconds: Optional[float] = None
        self.latent: Optional[torch.Tensor] = None  # first `latent_samples` rows, on CPU
        self.latent_summary: Optional[LatentSummary] = None


class TraceCollector:
    def __init__(self, net: Any, vq: Any, batch_size: int, device: str = "cpu",
                 on_event: Optional[EventFn] = None, latent_samples: int = 8):
        self.net, self.vq, self.B, self.device = net, vq, batch_size, device
        self.on_event, self.latent_samples = on_event, latent_samples
        self.levels: list[_Level] = []
        self._orig: dict[str, Any] = {}

    # context management: patch instance attributes only, always restore
    def __enter__(self) -> "TraceCollector":
        device_sync(self.device)
        self._t_enter = time.perf_counter()
        self._orig = {"sample": self.net._sample_level, "next": self.vq.get_next_autoregressive_input}
        self.net._sample_level = self._wrap_sample(self._orig["sample"])
        self.vq.get_next_autoregressive_input = self._wrap_next(self._orig["next"])
        return self

    def __exit__(self, *exc: Any) -> None:
        for obj, name in ((self.net, "_sample_level"), (self.vq, "get_next_autoregressive_input")):
            if name in vars(obj):
                delattr(obj, name)

    def _emit(self, **e: Any) -> None:
        if self.on_event is not None:
            self.on_event(e)

    def record_level0(self, codes: torch.Tensor, source: str = "unigram") -> None:
        lv = _Level(0, int(codes.shape[1]), source)
        lv.codes = codes[: self.B].detach().cpu().clone()
        self.levels.append(lv)
        self._emit(type="generation.level_completed", level=0, n_codes=lv.n_codes, seconds=None, source=source)

    def _wrap_next(self, fn: Callable) -> Callable:
        def wrapped(level_idx: int, z_hat: torch.Tensor, z_q_level: torch.Tensor):
            k = level_idx + 1
            device_sync(self.device)
            lv = _Level(k, int(self.vq.levels[k]), "transformer")
            lv.t_start = time.perf_counter()
            self.levels.append(lv)
            self._emit(type="generation.level_started", level=k, n_codes=lv.n_codes)
            z_hat, x = fn(level_idx, z_hat, z_q_level)
            lv.latent = z_hat[: min(self.B, self.latent_samples)].detach().float().cpu().clone()
            return z_hat, x
        return wrapped

    def _wrap_sample(self, fn: Callable) -> Callable:
        def wrapped(logits: torch.Tensor, *a: Any, **k: Any):
            out = fn(logits, *a, **k)
            device_sync(self.device)
            if not self.levels:
                # level 0 predicted by the transformer before the level loop
                # (predict_level0 / query_level0); timed from entering the
                # collector, so it includes prefix encoding. No started event:
                # there is no hook before it.
                lv = _Level(0, int(out.shape[1]), "transformer")
                lv.t_start = self._t_enter
                self.levels.append(lv)
            lv = self.levels[-1]
            lv.seconds = time.perf_counter() - lv.t_start if lv.t_start is not None else None
            lv.codes = out[: self.B].detach().cpu().clone()
            lv.logits_shape = [min(int(logits.shape[0]), self.B), *map(int, logits.shape[1:])]
            self._emit(type="generation.level_completed", level=lv.level, n_codes=lv.n_codes,
                       seconds=lv.seconds, source=lv.source)
            return out
        return wrapped

    def codes_tensor(self) -> torch.Tensor:
        return torch.cat([lv.codes for lv in self.levels], dim=1)

    def build(self, *, task: str, final_latent: torch.Tensor, decoded_levels: list[list[str]],
              final_texts: list[str], total_seconds: float, vocab_sizes: list[int],
              device: Optional[str] = None, max_detail_samples: int = 8) -> Trace:
        """decoded_levels: per sample, one string per level (decode_multiscales output)."""
        final = final_latent[: min(self.B, self.latent_samples)]
        prev = torch.zeros_like(final, device="cpu", dtype=torch.float32)
        out: list[LevelTrace] = []
        for lv in self.levels:
            summary = None
            if lv.latent is not None:
                summary = latent_summary(lv.latent, prev, final)
                prev = lv.latent
            out.append(LevelTrace(
                level=lv.level, n_codes=lv.n_codes, source=lv.source, logits_shape=lv.logits_shape,
                seconds=lv.seconds, stats=codebook_stats(lv.codes, vocab_sizes[lv.level]),
                latent_in=summary, codes=lv.codes[:max_detail_samples].tolist(),
                decoded=[d[lv.level] for d in decoded_levels[:max_detail_samples]]))
        return Trace(
            task=task, device=device or self.device, batch_size=self.B,
            levels_schedule=[lv.n_codes for lv in self.levels],
            transformer_passes=sum(lv.source == "transformer" for lv in self.levels),
            total_seconds=total_seconds, latent_samples=int(final.shape[0]),
            max_detail_samples=max_detail_samples, levels=out, final_texts=final_texts)

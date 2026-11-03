"""Autoregressive baseline: released TinyStories-33M (GPT-Neo), unconditional.

Generation starts from the BOS (= EOS, 50256) token, like NCM-TinyStories
starts from EOS, and samples up to `max_new_tokens` new tokens, stopping at EOS.
Each decoding step is reported through a transformers streamer, which
`generate()` calls once per step, so `generation.step_completed` events are real.
"""
import logging
import math
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional

import torch

from app.schemas.generation import GenerateRequest, GenerateResponse, ModelInfo, SamplingParams, Timing
from app.services.errors import InvalidRequest, ModelNotFound, ModelUnavailable
from app.services.memory import read_memory, reset_peak
from app.services.trace_collector import EventFn, device_sync

log = logging.getLogger("ncm.ar")


@dataclass(frozen=True)
class ARSpec:
    id: str
    hf_id: str
    revision: str
    task: str
    description: str
    training_data: str
    max_new_tokens: int


AR_MODELS = {s.id: s for s in [
    ARSpec("ar-tinystories-33m", "roneneldan/TinyStories-33M", "2ad0a164221b7c4d21cac7c46aec74f6f98dbfc8",
           "tinystories", "TinyStories-33M (GPT-Neo, 4 layers, causal LM): token-by-token baseline, unconditional from BOS.",
           "TinyStories (same corpus as NCM-TinyStories; independent training run)", 128),
]}


def unigram_entropy(ids: list[int]) -> float:
    """Same definition as eval_merge.py `entropy` (natural log)."""
    c, m = Counter(ids), len(ids)
    return -sum(v / m * math.log(v / m) for v in c.values()) if m else 0.0


class _StepStreamer:
    """transformers BaseStreamer protocol: put() once with the prompt, then once per step."""

    def __init__(self, device: str, on_step):
        self.device, self.on_step, self.seen_prompt, self.step = device, on_step, False, 0

    def put(self, value: torch.Tensor) -> None:
        if not self.seen_prompt:
            self.seen_prompt = True
            return
        self.step += 1
        self.on_step(self.step)

    def end(self) -> None:
        pass


class AutoregressiveRunner:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._models: dict[str, tuple[Any, Any]] = {}
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    def owns(self, model_id: str) -> bool:
        return model_id in AR_MODELS

    def _cached(self, spec: ARSpec) -> bool:
        from huggingface_hub import try_to_load_from_cache
        return all(isinstance(try_to_load_from_cache(spec.hf_id, f, revision=spec.revision), str)
                   for f in ("config.json", "pytorch_model.bin", "tokenizer.json"))

    def list_models(self) -> list[ModelInfo]:
        out = []
        for spec in AR_MODELS.values():
            loaded = self._models.get(spec.id)
            cached = self._cached(spec)
            out.append(ModelInfo(
                id=spec.id, family="autoregressive", task=spec.task, conditional=False, prompt_kind="none",
                description=spec.description, available=cached,
                missing_files=[] if cached else [f"{spec.hf_id}@{spec.revision[:7]} (not in the local HF cache)"],
                loaded=loaded is not None,
                parameters={"model": sum(p.numel() for p in loaded[0].parameters())} if loaded else None,
                levels=[], output_tokens=spec.max_new_tokens, prefix_tokens=1, n_examples=0,
                defaults=SamplingParams(temperature=1.0, top_k=0, top_p=1.0, cfg_scale=0.0),
                tokenizer="gpt2-bpe (50257)", training_data=spec.training_data,
                notes=[f"hf: {spec.hf_id} @ {spec.revision[:7]}", f"max {spec.max_new_tokens} new tokens, stops at EOS"]))
        return out

    def _load(self, spec: ARSpec, on_event: Optional[EventFn]) -> tuple[Any, Any, float]:
        with self._load_lock:
            if spec.id in self._models:
                return (*self._models[spec.id], 0.0)
            if not self._cached(spec):
                raise ModelUnavailable(f"{spec.id}: {spec.hf_id}@{spec.revision[:7]} is not in the local HF cache")
            if on_event:
                on_event({"type": "model.loading", "model": spec.id, "device": self.device})
            from transformers import AutoModelForCausalLM, AutoTokenizer
            t = time.perf_counter()
            try:
                tok = AutoTokenizer.from_pretrained(spec.hf_id, revision=spec.revision, local_files_only=True)
                model = AutoModelForCausalLM.from_pretrained(spec.hf_id, revision=spec.revision,
                                                             local_files_only=True).to(self.device).eval()
            except Exception as e:
                log.exception("load failed model=%s", spec.id)
                raise ModelUnavailable(f"{spec.id}: failed to load: {e}") from e
            dt = time.perf_counter() - t
            self._models[spec.id] = (model, tok)
            log.info("loaded model=%s device=%s seconds=%.2f", spec.id, self.device, dt)
            if on_event:
                on_event({"type": "model.loaded", "model": spec.id, "seconds": dt})
            return model, tok, dt

    def generate(self, req: GenerateRequest, on_event: Optional[EventFn] = None) -> GenerateResponse:
        if req.model not in AR_MODELS:
            raise ModelNotFound(f"unknown model {req.model!r}")
        spec = AR_MODELS[req.model]
        if req.prompt is not None or req.example_index is not None:
            raise InvalidRequest(f"{spec.id} is run unconditionally here (to match NCM-TinyStories); no prompt")
        o = req.sampling
        sp = SamplingParams(temperature=1.0 if o.temperature is None else o.temperature,
                            top_k=0 if o.top_k is None else o.top_k,
                            top_p=1.0 if o.top_p is None else o.top_p, cfg_scale=0.0)
        if o.cfg_scale:
            raise InvalidRequest("cfg_scale is not applicable to the autoregressive baseline")
        model, tok, load_s = self._load(spec, on_event)
        B, eos = req.num_samples, int(tok.eos_token_id)

        with self._infer_lock:
            x = torch.full((B, 1), eos, dtype=torch.long, device=self.device)
            gen_kw = dict(attention_mask=torch.ones_like(x), do_sample=sp.temperature > 0,
                          temperature=sp.temperature if sp.temperature > 0 else None,
                          top_k=sp.top_k if sp.temperature > 0 else None,
                          top_p=sp.top_p if sp.temperature > 0 else None,
                          max_new_tokens=spec.max_new_tokens, eos_token_id=eos, pad_token_id=eos, use_cache=True)
            if req.warmup:  # untimed pass with the same shapes; reseeded below so outputs are unchanged
                torch.manual_seed(req.seed)
                with torch.no_grad():
                    model.generate(x, **gen_kw)
                device_sync(self.device)
            reset_peak(self.device)
            torch.manual_seed(req.seed)
            step_times: list[float] = []
            device_sync(self.device)
            t0 = time.perf_counter()

            def on_step(step: int) -> None:
                step_times.append(time.perf_counter())
                if on_event:
                    on_event({"type": "generation.step_completed", "step": step, "n_tokens": B})

            with torch.no_grad():
                out = model.generate(x, **gen_kw, streamer=_StepStreamer(self.device, on_step))
            device_sync(self.device)
            t1 = time.perf_counter()
            new = out[:, 1:].tolist()
            cut = [r[:r.index(eos)] if eos in r else r for r in new]
            outputs = [tok.decode(r, skip_special_tokens=True) for r in cut]
            t2 = time.perf_counter()
            memory = read_memory(self.device)

        timing = Timing(load_s=load_s, generate_s=t1 - t0, decode_s=t2 - t1, total_s=t2 - t0,
                        first_output_s=(step_times[0] - t0) if step_times else None)
        log.info("generate model=%s B=%d seed=%d steps=%d generate_s=%.3f", spec.id, B, req.seed, len(step_times),
                 timing.generate_s)
        return GenerateResponse(
            id=uuid.uuid4().hex, model=spec.id, task=spec.task, device=self.device, seed=req.seed, sampling=sp,
            prompt_tokens=None, prompt_truncated=False, reference=None, outputs=outputs,
            transformer_passes=len(step_times), levels=[], timing=timing, trace=None,
            output_tokens=[len(r) for r in cut], tokenizer="gpt2-bpe (50257)",
            output_entropy_own=[unigram_entropy(tok.encode(t)) for t in outputs], memory=memory)

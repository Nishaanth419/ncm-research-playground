"""Run released NCM checkpoints the way research-ncm/scripts/eval_shard.py does.

- Unconditional (TinyStories): level-0 codes drawn from the checkpoint's
  unigram, prefix = one EOS token.
- Conditional (XSum, WMT14): prefix = input ids truncated / padded to
  `model.prefix_len`; text inputs are tokenized with the config's
  `pre_tokenizer.model_id` like the script; dataset items are used as-is.
- Final text: `pick_tokens` (taken verbatim from eval_shard.py) on the decoder
  logits of the final latent, cut at EOS, special tokens removed.

Bundles are loaded lazily on first use and kept in memory. One inference runs
at a time (a single lock), since the models share one device.
"""
import logging
import pickle
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import torch

from app.core.official import official_pick_tokens, parse_adjacency
from app.core.research import RESEARCH_ROOT, NCMBundle, config_path, load_bundle, missing_files
from app.schemas.generation import GenerateRequest, GenerateResponse, ModelInfo, SamplingParams, Timing
from app.services.errors import InvalidRequest, ModelNotFound, ModelUnavailable
from app.services.trace_collector import TraceCollector, device_sync

log = logging.getLogger("ncm.runner")


@dataclass(frozen=True)
class ModelSpec:
    id: str
    task: str
    prompt_kind: str
    description: str


NCM_MODELS = {
    s.id: s for s in [
        ModelSpec("ncm-tinystories", "tinystories", "none",
                  "NCM TinyStories (unconditional, 128 BPE tokens). No prompt: the official model starts from EOS."),
        ModelSpec("ncm-xsum", "xsum", "article",
                  "NCM XSum: news article (first 512 GPT-2 tokens) -> 48-token one-sentence summary."),
        ModelSpec("ncm-wmt14", "wmt14", "german",
                  "NCM WMT14 De->En: German sentence (first 96 Qwen3 tokens) -> 64-token English translation."),
    ]
}


class NCMRunner:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._bundles: dict[str, NCMBundle] = {}
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._src_tok: dict[str, Any] = {}
        self._examples: dict[str, tuple[list, list]] = {}

    # ---- registry -------------------------------------------------------
    def _spec(self, model_id: str) -> ModelSpec:
        if model_id not in NCM_MODELS:
            raise ModelNotFound(f"unknown model {model_id!r}; available: {sorted(NCM_MODELS)}")
        return NCM_MODELS[model_id]

    def list_models(self) -> list[ModelInfo]:
        from omegaconf import OmegaConf
        out = []
        for spec in NCM_MODELS.values():
            cfg = OmegaConf.load(config_path(spec.task))
            missing = missing_files(spec.task)
            b = self._bundles.get(spec.id)
            params = None
            if b is not None:
                params = {"generator": sum(p.numel() for p in b.net.parameters()),
                          "tokenizer_vqvae": sum(p.numel() for p in b.tokenizer.vqvae.parameters())}
            out.append(ModelInfo(
                id=spec.id, family="ncm", task=spec.task, conditional=spec.prompt_kind != "none",
                prompt_kind=spec.prompt_kind, description=spec.description, available=not missing,
                missing_files=missing, loaded=b is not None, parameters=params,
                levels=list(cfg.tokenizer.vqvae.vector_quantizer_config.levels.manual_levels),
                output_tokens=int(cfg.tokenizer.vqvae.context_length), prefix_tokens=int(cfg.model.prefix_len),
                n_examples=self._n_examples(spec.task, cfg), defaults=self._defaults(cfg)))
        return out

    @staticmethod
    def _defaults(cfg) -> SamplingParams:
        G = cfg.generation  # same fallbacks as eval_shard.py's `pick(...)`
        return SamplingParams(temperature=float(G.get("temperature", 1.0)), top_k=int(G.get("top_k", 1)),
                              top_p=float(G.get("top_p", 0.0)), cfg_scale=float(G.get("cfg_scale", 0.0)))

    def _n_examples(self, task: str, cfg) -> int:
        if not cfg.data or not cfg.data.get("test"):
            return 0
        path = RESEARCH_ROOT / "data" / task / cfg.data.test
        return len(self._load_examples(task, cfg)[0]) if path.exists() else 0

    def _load_examples(self, task: str, cfg) -> tuple[list, list]:
        if task not in self._examples:
            d = RESEARCH_ROOT / "data" / task
            # official test-set pickles from hazyresearch/ncm-datasets (pinned revision)
            items = pickle.load(open(d / cfg.data.test, "rb"))
            refs = pickle.load(open(d / cfg.data.test_refs, "rb"))
            self._examples[task] = (items, refs)
        return self._examples[task]

    def bundle(self, model_id: str) -> tuple[NCMBundle, float]:
        spec = self._spec(model_id)
        with self._load_lock:
            if model_id in self._bundles:
                return self._bundles[model_id], 0.0
            missing = missing_files(spec.task)
            if missing:
                raise ModelUnavailable(f"{model_id}: checkpoint files missing: {missing}")
            t = time.perf_counter()
            try:
                b = load_bundle(spec.task, self.device)
            except Exception as e:  # surfaced as 503 with the reason
                log.exception("load failed model=%s", model_id)
                raise ModelUnavailable(f"{model_id}: failed to load: {e}") from e
            dt = time.perf_counter() - t
            self._bundles[model_id] = b
            log.info("loaded model=%s device=%s seconds=%.2f", model_id, self.device, dt)
            return b, dt

    # ---- generation -----------------------------------------------------
    def _prefix(self, b: NCMBundle, spec: ModelSpec, req: GenerateRequest) -> tuple[torch.Tensor, int, bool, Optional[str]]:
        P = int(b.cfg.model.prefix_len)
        pad = b.tokenizer.pre_tokenizer.pad_token_id
        reference = None
        if req.example_index is not None:
            items, refs = self._load_examples(spec.task, b.cfg)
            if req.example_index >= len(items):
                raise InvalidRequest(f"example_index must be < {len(items)}")
            item, ref = items[req.example_index], refs[req.example_index]
            reference = ref if isinstance(ref, str) else self._dec(b, ref)
        else:
            if not req.prompt or not req.prompt.strip():
                raise InvalidRequest(f"{spec.id} needs a prompt ({spec.prompt_kind}) or an example_index")
            item = req.prompt
        if isinstance(item, str):
            mid = str(b.cfg.tokenizer.pre_tokenizer.model_id)
            if mid not in self._src_tok:
                from transformers import AutoTokenizer
                self._src_tok[mid] = AutoTokenizer.from_pretrained(mid)
            ids = self._src_tok[mid](item)["input_ids"]
        else:
            ids = list(item)
        prefix = (ids[:P] + [pad] * P)[:P]  # eval_shard.py `prefix(item)`
        x = torch.tensor([prefix] * req.num_samples, device=self.device)
        return x, len(ids), len(ids) > P, reference

    @staticmethod
    def _dec(b: NCMBundle, ids: list[int]) -> str:
        # eval_shard.py `dec`
        tk = b.tokenizer
        eos, pad = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
        ids = list(ids)
        if bool(b.cfg.generation.get("cut_at_eos", True)) and eos in ids:
            ids = ids[:ids.index(eos)]
        return tk.pre_tokenizer.decode([t for t in ids if t not in (eos, pad)], skip_special_tokens=True)

    def _picker(self, b: NCMBundle):
        G = b.cfg.generation
        allow = b.resolve(G.get("frequency_allowance"))
        return official_pick_tokens(
            freq=float(G.get("frequency_penalty", 0.0)), mu=float(G.get("frequency_deficit", 0.0)),
            adj=parse_adjacency(str(G.get("adjacency_penalty", "") or "")),
            allow=allow.to(self.device).double() if allow is not None else None)

    def generate(self, req: GenerateRequest) -> GenerateResponse:
        spec = self._spec(req.model)
        conditional = spec.prompt_kind != "none"
        if not conditional and (req.prompt is not None or req.example_index is not None):
            raise InvalidRequest(f"{spec.id} is unconditional and takes no prompt or example_index")
        b, load_s = self.bundle(req.model)
        d = self._defaults(b.cfg)
        o = req.sampling
        sp = SamplingParams(
            temperature=d.temperature if o.temperature is None else o.temperature,
            top_k=d.top_k if o.top_k is None else o.top_k,
            top_p=d.top_p if o.top_p is None else o.top_p,
            cfg_scale=d.cfg_scale if o.cfg_scale is None else o.cfg_scale)
        if not conditional and sp.cfg_scale > 0:
            raise InvalidRequest("cfg_scale needs a conditional model")

        tk, net, vq, B = b.tokenizer, b.net, b.vq, req.num_samples
        eos = tk.pre_tokenizer.eos_token_id
        with self._infer_lock:
            torch.manual_seed(req.seed)
            if conditional:
                x, n_prompt, truncated, reference = self._prefix(b, spec, req)
                l0 = None
            else:
                x, n_prompt, truncated, reference = torch.full((B, 1), eos, device=self.device), None, False, None
                L0_P = b.resolve(b.cfg.generation.get("level0_unigram"))
                l0 = (torch.multinomial(L0_P.float().cpu(), B * b.levels[0], replacement=True)
                      .view(B, b.levels[0]).to(self.device) if L0_P is not None else None)
            tc = TraceCollector(net, vq, batch_size=B, device=self.device)
            with torch.no_grad():
                device_sync(self.device)
                t0 = time.perf_counter()
                with tc:
                    if l0 is not None:
                        tc.record_level0(l0, source="unigram")
                    g = net.generate(l0, x, temperature=sp.temperature, top_k=sp.top_k, top_p=sp.top_p,
                                     cfg_scale=sp.cfg_scale, cfg_null_eos=eos)
                device_sync(self.device)
                t1 = time.perf_counter()
                _, zl = vq.idx_to_zhats(g, return_zlevels=True)
                ids = self._picker(b)(tk.vqvae.decoder(zl[-1])).tolist()
                outputs = [self._dec(b, r) for r in ids]
                device_sync(self.device)
                t2 = time.perf_counter()
                trace = None
                if req.include_trace:
                    trace = tc.build(task=spec.task, final_latent=vq.idx_to_z_q(g),
                                     decoded_levels=tk.decode_multiscales(g, skip_special_tokens=True),
                                     final_texts=outputs, total_seconds=t2 - t0, vocab_sizes=b.vocab_sizes)
        timing = Timing(load_s=load_s, generate_s=t1 - t0, decode_s=t2 - t1, total_s=t2 - t0)
        log.info("generate model=%s B=%d seed=%d generate_s=%.3f decode_s=%.3f", spec.id, B, req.seed,
                 timing.generate_s, timing.decode_s)
        return GenerateResponse(
            id=uuid.uuid4().hex, model=spec.id, task=spec.task, device=self.device, seed=req.seed,
            sampling=sp, prompt_tokens=n_prompt, prompt_truncated=truncated, reference=reference,
            outputs=outputs, transformer_passes=sum(lv.source == "transformer" for lv in tc.levels),
            levels=b.levels, timing=timing, trace=trace)

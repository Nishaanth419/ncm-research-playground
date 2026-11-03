"""Generation quality with the official NCM protocol's judge.

Generative perplexity uses the official, importable
`research-ncm/evaluation/gen_ppl.py::generation_perplexity` with GPT-2-large
(the `eval.ref_model` of the shipped configs). `corpus_ppl` is its
`mean_ppl` (exp of total NLL / total tokens over the scored texts).

Entropy: eval_merge.py measures unigram entropy over each model's *own*
tokens, which is not comparable across tokenizers. For cross-model comparison
we also report entropy over the judge's GPT-2 tokens of the same text.
Empty outputs are not scored (they would get a meaningless perplexity of 1).
"""
import threading
from typing import Any, Optional

import torch

from app.core.research import _import_research
from app.services.ar_runner import unigram_entropy


class QualityScorer:
    judge = "gpt2-large"

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._lock = threading.Lock()
        self._model: Optional[Any] = None
        self._tok: Optional[Any] = None

    def _load(self) -> None:
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.judge)
            self._model = AutoModelForCausalLM.from_pretrained(self.judge).to(self.device).eval()

    def score(self, texts: list[str]) -> dict[str, Any]:
        _import_research()
        from evaluation.gen_ppl import generation_perplexity

        idx = [i for i, t in enumerate(texts) if t.strip()]
        with self._lock:
            self._load()
            per: list[Optional[dict[str, float]]] = [None] * len(texts)
            corpus = None
            if idx:
                with torch.no_grad():
                    r = generation_perplexity(self._model, self._tok, [""] * len(idx), [texts[i] for i in idx],
                                              device=self.device, batch_size=8)
                corpus = float(r.mean_ppl)
                for j, i in enumerate(idx):
                    ids = self._tok.encode(texts[i])
                    per[i] = {"ppl": float(r.ppl[j]), "judge_tokens": int(r.token_count[j]),
                              "entropy_judge_tokens": unigram_entropy(ids)}
        return {"judge": self.judge, "protocol": "research-ncm evaluation/gen_ppl.py generation_perplexity",
                "corpus_ppl": corpus, "n_scored": len(idx), "per_sample": per}

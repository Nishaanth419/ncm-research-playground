"""Load released NCM checkpoints through the official, unmodified research code.

Loading mirrors research-ncm/scripts/eval_shard.py: the shipped eval config,
VQVAETokenizer.from_config, NextConceptTransformer(vqvae_local=..., config=...),
strict state-dict loading, and the generator.pt `inference` entries.
"""
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from omegaconf import DictConfig, OmegaConf

RESEARCH_ROOT = Path(os.environ.get(
    "NCM_RESEARCH_ROOT", Path(__file__).resolve().parents[3] / "research-ncm"))
TASKS = ("tinystories", "xsum", "wmt14")


def _import_research() -> None:
    if str(RESEARCH_ROOT) not in sys.path:
        sys.path.insert(0, str(RESEARCH_ROOT))


def checkpoint_dir(task: str) -> Path:
    return RESEARCH_ROOT / "artifacts" / task


def config_path(task: str) -> Path:
    return RESEARCH_ROOT / "config" / "eval" / f"{task}.yaml"


def missing_files(task: str) -> list[str]:
    """Files the task needs that are not on disk (empty list = available)."""
    cfg = OmegaConf.load(config_path(task))
    ck = checkpoint_dir(task)
    return [str(ck / name) for name in cfg.checkpoints.values() if name and not (ck / name).exists()]


@dataclass
class NCMBundle:
    task: str
    device: str
    cfg: DictConfig
    tokenizer: Any  # tokenization.vqvae_tokenizer.VQVAETokenizer
    net: Any  # models.next_concept.NextConceptTransformer
    inference: dict  # generator.pt["inference"], e.g. level0_unigram, token_unigram

    @property
    def vq(self) -> Any:
        return self.tokenizer.vqvae.vq

    @property
    def levels(self) -> list[int]:
        return list(self.vq.levels)

    @property
    def vocab_sizes(self) -> list[int]:
        return list(self.cfg.tokenizer.vqvae.vqvae_vocab_size)

    def resolve(self, spec: Optional[str]) -> Optional[torch.Tensor]:
        """Resolve a config reference like 'generator.pt:inference.level0_unigram'."""
        if not spec:
            return None
        path, ref = str(spec).split(":", 1)
        if path != str(self.cfg.checkpoints.generator):
            raise ValueError(f"unsupported reference {spec!r}")
        obj: Any = {"inference": self.inference}
        for k in ref.split("."):
            obj = obj[k]
        return obj


def load_bundle(task: str, device: str = "cpu") -> NCMBundle:
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")
    missing = missing_files(task)
    if missing:
        raise FileNotFoundError(f"checkpoint files missing for {task}: {missing}")
    _import_research()
    from models.next_concept import NextConceptTransformer
    from tokenization.vqvae_tokenizer import VQVAETokenizer

    cfg = OmegaConf.load(config_path(task))
    ck = checkpoint_dir(task)
    cfg.tokenizer.vqvae.checkpoint_path = str(ck / cfg.checkpoints.tokenizer)
    for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
        if cfg.checkpoints.get(name):
            cfg.tokenizer.pre_tokenizer[key] = str(ck / cfg.checkpoints[name])
    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    tk.vqvae = tk.vqvae.to(device).eval()
    net = NextConceptTransformer(vqvae_local=tk.vqvae, config=cfg.model).to(device)
    sd = torch.load(ck / cfg.checkpoints.generator, map_location="cpu", weights_only=False)
    net.load_state_dict(sd["model"], strict=True)
    net.eval()
    return NCMBundle(task=task, device=device, cfg=cfg, tokenizer=tk, net=net, inference=sd.get("inference", {}))

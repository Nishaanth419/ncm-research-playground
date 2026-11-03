"""Reuse helpers defined inside the official eval script without copying them.

`research-ncm/scripts/eval_shard.py` defines `pick_tokens` (final-decode
repetition penalties) at module level next to code that runs on import, so it
cannot be imported. We parse the file, take that function's source verbatim and
compile it against a namespace holding the settings it reads.
"""
import ast
from functools import lru_cache
from typing import Callable, Optional

import torch

from app.core.research import RESEARCH_ROOT

EVAL_SHARD = RESEARCH_ROOT / "scripts" / "eval_shard.py"


@lru_cache(maxsize=None)
def _function_source(name: str) -> str:
    src = EVAL_SHARD.read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    raise LookupError(f"{name} not found in {EVAL_SHARD}")


def official_pick_tokens(freq: float, mu: float, adj: Optional[list[float]],
                         allow: Optional[torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
    """`pick_tokens(logits)` from eval_shard.py, bound to the given settings
    (names match the script's globals: freq, mu, ADJ, ALLOW)."""
    ns = {"torch": torch, "freq": freq, "mu": mu, "ADJ": adj, "ALLOW": allow}
    exec(compile(_function_source("pick_tokens"), str(EVAL_SHARD), "exec"), ns)
    return ns["pick_tokens"]


def parse_adjacency(spec: str) -> Optional[list[float]]:
    """Same parsing as eval_shard.py: `ADJ = [float(v) for v in adj.split(",")] if adj else None`."""
    return [float(v) for v in spec.split(",")] if spec else None

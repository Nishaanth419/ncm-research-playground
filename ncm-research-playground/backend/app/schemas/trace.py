"""Trace schemas: lightweight, JSON-safe summaries of one NCM generation.

Only ints, floats and strings are stored; no raw float tensors.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class CodeCount(BaseModel):
    code: int
    count: int


class CodebookStats(BaseModel):
    vocab_size: int
    n_codes: int = Field(description="codes emitted at this level across the whole batch")
    unique: int
    utilization: float = Field(description="unique / vocab_size")
    entropy_bits: float = Field(description="entropy of the empirical code distribution")
    perplexity: float = Field(description="2 ** entropy_bits")
    top: list[CodeCount]


class LatentSummary(BaseModel):
    """Statistics of the cumulative latent z_hat that is the transformer's input
    for this level (codes of all earlier levels summed)."""
    norm_mean: float = Field(description="mean L2 norm per position")
    delta_norm_mean: float = Field(description="mean L2 norm per position of the change since the previous level's input")
    cos_to_final: float = Field(description="cosine similarity to the final latent (all levels), per sample, averaged")
    n_samples: int


class LevelTrace(BaseModel):
    level: int
    n_codes: int = Field(description="codes per sample at this level, all predicted in one parallel step")
    source: Literal["unigram", "transformer", "given"]
    logits_shape: Optional[list[int]] = None
    seconds: Optional[float] = Field(None, description="device-synchronised wall time of this level's step")
    stats: CodebookStats
    latent_in: Optional[LatentSummary] = None
    codes: list[list[int]] = Field(description="code indices for the first max_detail_samples samples")
    decoded: list[str] = Field(
        default_factory=list,
        description="official VQVAETokenizer.decode_multiscales text after this level, first max_detail_samples samples")


class Trace(BaseModel):
    task: str
    device: str
    batch_size: int
    levels_schedule: list[int]
    transformer_passes: int
    total_seconds: float
    latent_samples: int
    max_detail_samples: int
    levels: list[LevelTrace]
    final_texts: list[str]

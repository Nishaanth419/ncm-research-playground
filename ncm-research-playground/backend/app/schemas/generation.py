from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from app.schemas.trace import Trace


class SamplingParams(BaseModel):
    temperature: float = Field(ge=0.0, le=5.0, description="0 = greedy")
    top_k: int = Field(ge=0, description="0 = disabled")
    top_p: float = Field(ge=0.0, le=1.0)
    cfg_scale: float = Field(ge=0.0, le=10.0, description="classifier-free guidance; conditional models only")


class SamplingOverrides(BaseModel):
    """Any field left out uses the checkpoint's shipped config value."""
    temperature: Optional[float] = Field(None, ge=0.0, le=5.0)
    top_k: Optional[int] = Field(None, ge=0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    cfg_scale: Optional[float] = Field(None, ge=0.0, le=10.0)


class ModelInfo(BaseModel):
    id: str
    family: Literal["ncm", "autoregressive"]
    task: str
    conditional: bool
    prompt_kind: Literal["none", "article", "german"]
    description: str
    available: bool
    missing_files: list[str]
    loaded: bool
    parameters: Optional[dict[str, int]] = None
    levels: list[int]
    output_tokens: int
    prefix_tokens: int
    n_examples: int = Field(description="official test-set items addressable by example_index")
    defaults: SamplingParams
    tokenizer: str = ""
    training_data: str = ""
    notes: list[str] = Field(default_factory=list)


class GenerateRequest(BaseModel):
    model: str
    prompt: Optional[str] = Field(None, max_length=50_000)
    example_index: Optional[int] = Field(None, ge=0, description="use this official test-set item as input")
    num_samples: int = Field(1, ge=1, le=16)
    seed: int = Field(0, ge=0, le=2**31 - 1)
    sampling: SamplingOverrides = Field(default_factory=SamplingOverrides)
    include_trace: bool = True
    tag: Optional[str] = Field(None, max_length=64, description="free label, e.g. a comparison group id")
    warmup: bool = Field(False, description="run one untimed pass with the same batch first (outputs are unchanged)")

    @model_validator(mode="after")
    def _one_input(self) -> "GenerateRequest":
        if self.prompt is not None and self.example_index is not None:
            raise ValueError("give either prompt or example_index, not both")
        return self


class Timing(BaseModel):
    load_s: float = Field(description="checkpoint loading triggered by this request (0 if already loaded)")
    generate_s: float = Field(description="net.generate, device-synchronised")
    decode_s: float = Field(description="latent -> tokens -> text")
    total_s: float = Field(description="generate_s + decode_s (excludes load_s)")
    first_output_s: Optional[float] = Field(
        None, description="autoregressive: first generated token; NCM: complete text (only exists after the final level and decode)")


class MemoryStats(BaseModel):
    device: str
    peak_bytes: Optional[int] = None
    note: str = ""


class GenerateResponse(BaseModel):
    id: str
    model: str
    task: str
    device: str
    seed: int
    sampling: SamplingParams
    prompt_tokens: Optional[int]
    prompt_truncated: bool
    reference: Optional[str]
    outputs: list[str]
    transformer_passes: int
    levels: list[int]
    timing: Timing
    trace: Optional[Trace]
    output_tokens: list[int] = Field(default_factory=list, description="generated tokens per sample, in the model's own tokenizer, up to EOS")
    tokenizer: str = ""
    output_entropy_own: list[float] = Field(
        default_factory=list, description="unigram entropy (nats) over the model's own tokens of each output, as in eval_merge.py")
    memory: Optional[MemoryStats] = None

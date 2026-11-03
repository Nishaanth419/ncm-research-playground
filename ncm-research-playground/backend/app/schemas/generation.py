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


class GenerateRequest(BaseModel):
    model: str
    prompt: Optional[str] = Field(None, max_length=50_000)
    example_index: Optional[int] = Field(None, ge=0, description="use this official test-set item as input")
    num_samples: int = Field(1, ge=1, le=16)
    seed: int = Field(0, ge=0, le=2**31 - 1)
    sampling: SamplingOverrides = Field(default_factory=SamplingOverrides)
    include_trace: bool = True

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

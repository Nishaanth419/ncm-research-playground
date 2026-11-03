from pydantic import BaseModel, Field

from app.schemas.generation import SamplingOverrides


class BenchmarkRequest(BaseModel):
    name: str = Field("", max_length=80)
    models: list[str] = Field(min_length=1, max_length=4)
    batch_sizes: list[int] = Field([1, 4], min_length=1, max_length=4)
    repeats: int = Field(10, ge=1, le=100, description="timed runs per (model, batch size)")
    warmup_runs: int = Field(2, ge=0, le=5, description="untimed runs per (model, batch size) before timing")
    seed: int = Field(0, ge=0, le=2**31 - 1, description="timed run i uses seed + i")
    sampling: SamplingOverrides = Field(default_factory=SamplingOverrides)

    def model_post_init(self, _: object) -> None:
        if any(not 1 <= b <= 16 for b in self.batch_sizes):
            raise ValueError("batch sizes must be between 1 and 16")

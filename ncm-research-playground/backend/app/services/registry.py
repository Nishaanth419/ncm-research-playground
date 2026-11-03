"""Dispatch requests to the runner that owns the model id."""
from typing import Any, Optional

from app.schemas.generation import GenerateResponse, ModelInfo
from app.services.errors import ModelNotFound
from app.services.trace_collector import EventFn


class ModelRegistry:
    def __init__(self, runners: list[Any]):
        self.runners = runners

    def list_models(self) -> list[ModelInfo]:
        return [m for r in self.runners for m in r.list_models()]

    def runner_for(self, model_id: str) -> Any:
        for r in self.runners:
            if r.owns(model_id):
                return r
        known = sorted(m.id for m in self.list_models())
        raise ModelNotFound(f"unknown model {model_id!r}; available: {known}")

    def generate(self, req: Any, on_event: Optional[EventFn] = None) -> GenerateResponse:
        return self.runner_for(req.model).generate(req, on_event=on_event)

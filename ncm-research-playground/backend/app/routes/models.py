from fastapi import APIRouter, Request

from app.schemas.generation import ModelInfo

router = APIRouter()


@router.get("/models", response_model=list[ModelInfo])
def list_models(request: Request) -> list[ModelInfo]:
    return request.app.state.runner.list_models()

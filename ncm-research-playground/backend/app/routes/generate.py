from fastapi import APIRouter, Request

from app.schemas.generation import GenerateRequest, GenerateResponse

router = APIRouter()


# Plain `def`: FastAPI runs it in a worker thread, so CPU-bound inference
# does not block the event loop.
@router.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest, request: Request) -> GenerateResponse:
    return request.app.state.runner.generate(req)

import logging
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import Settings
from app.routes import generate, models
from app.services.errors import InvalidRequest, ModelNotFound, ModelUnavailable

log = logging.getLogger("ncm.api")

_STATUS = {ModelNotFound: 404, InvalidRequest: 422, ModelUnavailable: 503}


def create_app(runner: Optional[Any] = None, settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings()
    logging.basicConfig(level=settings.log_level,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    app = FastAPI(title="NCM Research Playground", version="0.1.0")
    if runner is None:
        from app.services.ncm_runner import NCMRunner
        runner = NCMRunner(device=settings.device)
    app.state.runner = runner
    app.state.settings = settings
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                       allow_methods=["*"], allow_headers=["*"])

    for exc_type, status in _STATUS.items():
        def handler(request: Request, exc: Exception, status: int = status) -> JSONResponse:
            log.warning("%s %s -> %d: %s", request.method, request.url.path, status, exc)
            return JSONResponse(status_code=status, content={"detail": str(exc)})
        app.add_exception_handler(exc_type, handler)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "device": settings.device}

    app.include_router(models.router, prefix="/api")
    app.include_router(generate.router, prefix="/api")
    return app

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import Settings
from app.routes import experiments, generate, models
from app.services.errors import InvalidRequest, ModelNotFound, ModelUnavailable
from app.services.event_hub import EventHub
from app.services.experiment_service import ExperimentService
from app.services.experiment_store import ExperimentStore

log = logging.getLogger("ncm.api")

_STATUS = {ModelNotFound: 404, InvalidRequest: 422, ModelUnavailable: 503}


def create_app(runner: Optional[Any] = None, settings: Optional[Settings] = None,
               scorer: Optional[Any] = None) -> FastAPI:
    settings = settings or Settings()
    logging.basicConfig(level=settings.log_level,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if runner is None:
        from app.services.ar_runner import AutoregressiveRunner
        from app.services.ncm_runner import NCMRunner
        from app.services.registry import ModelRegistry
        runner = ModelRegistry([NCMRunner(device=settings.device), AutoregressiveRunner(device=settings.device)])
    if scorer is None:
        from app.services.quality import QualityScorer
        scorer = QualityScorer(device=settings.device)
    store, hub = ExperimentStore(settings.db_path), EventHub()
    service = ExperimentService(runner, store, hub)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        service.shutdown()

    app = FastAPI(title="NCM Research Playground", version="0.1.0", lifespan=lifespan)
    app.state.runner, app.state.settings, app.state.scorer = runner, settings, scorer
    app.state.store, app.state.hub, app.state.experiments = store, hub, service
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
    app.include_router(experiments.router, prefix="/api")
    return app

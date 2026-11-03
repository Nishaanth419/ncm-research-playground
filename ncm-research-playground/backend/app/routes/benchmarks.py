from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response

from app.schemas.benchmark import BenchmarkRequest
from app.services.benchmark import runs_csv, summary_csv

router = APIRouter()


def _get(request: Request, bid: str) -> dict[str, Any]:
    b = request.app.state.bench_store.get(bid)
    if b is None:
        raise HTTPException(404, f"benchmark {bid} not found")
    return b


def _csv(body: str, name: str) -> Response:
    return Response(body, media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@router.post("/benchmarks", status_code=201)
def create_benchmark(req: BenchmarkRequest, request: Request) -> dict[str, str]:
    infos = {m.id: m for m in request.app.state.runner.list_models()}
    unknown = [m for m in req.models if infos and m not in infos]
    if unknown:
        raise HTTPException(422, f"unknown model(s): {unknown}")
    conditional = [m for m in req.models if m in infos and infos[m].conditional]
    if conditional:
        raise HTTPException(422, f"benchmarks use an unconditional workload; {conditional} need a prompt")
    return {"id": request.app.state.benchmarks.create(req), "status": "queued"}


@router.get("/benchmarks")
def list_benchmarks(request: Request, limit: int = Query(50, ge=1, le=500)) -> list[dict[str, Any]]:
    return request.app.state.bench_store.list(limit)


@router.get("/benchmarks/{bid}")
def get_benchmark(bid: str, request: Request) -> dict[str, Any]:
    return _get(request, bid)


@router.get("/benchmarks/{bid}/runs.csv")
def benchmark_runs_csv(bid: str, request: Request) -> Response:
    return _csv(runs_csv(_get(request, bid)["runs"]), f"benchmark-{bid[:8]}-runs.csv")


@router.get("/benchmarks/{bid}/summary.csv")
def benchmark_summary_csv(bid: str, request: Request) -> Response:
    b = _get(request, bid)
    return _csv(summary_csv(b["summary"], b["hardware"]), f"benchmark-{bid[:8]}-summary.csv")

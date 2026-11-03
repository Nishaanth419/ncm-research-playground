import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket
from fastapi.responses import Response

from app.schemas.generation import GenerateRequest

router = APIRouter()


def _record(request: Request, eid: str) -> dict[str, Any]:
    rec = request.app.state.store.get(eid)
    if rec is None:
        raise HTTPException(404, f"experiment {eid} not found")
    return rec


@router.post("/experiments", status_code=201)
def create_experiment(req: GenerateRequest, request: Request) -> dict[str, str]:
    eid = request.app.state.experiments.create(req)
    return {"id": eid, "status": "queued", "stream_url": f"/api/stream/{eid}"}


@router.get("/experiments")
def list_experiments(request: Request, limit: int = Query(50, ge=1, le=500),
                     offset: int = Query(0, ge=0)) -> list[dict[str, Any]]:
    return request.app.state.store.list(limit=limit, offset=offset)


EXPORT_COLS = ("id", "created_at", "finished_at", "model", "status", "error", "tag", "num_samples", "seed",
               "prompt", "example_index", "temperature", "top_k", "top_p", "cfg_scale", "load_s", "generate_s",
               "decode_s", "total_s", "first_output_s", "transformer_passes", "output_tokens_total", "output_words_total",
               "gen_ppl", "outputs")


@router.get("/experiments.csv")
def export_experiments(request: Request, limit: int = Query(5000, ge=1, le=100_000)) -> Response:
    import csv
    import io
    import json
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(EXPORT_COLS))
    w.writeheader()
    store = request.app.state.store
    for row in store.list(limit=limit):
        rec = store.get(row["id"])
        req, t, m = rec["request"], rec["timing"] or {}, rec["metrics"] or {}
        sp = m.get("sampling") or {}
        w.writerow({
            "id": rec["id"], "created_at": rec["created_at"], "finished_at": rec["finished_at"], "model": rec["model"],
            "status": rec["status"], "error": rec["error"], "tag": req.get("tag"), "num_samples": req.get("num_samples"),
            "seed": req.get("seed"), "prompt": req.get("prompt"), "example_index": req.get("example_index"),
            "temperature": sp.get("temperature"), "top_k": sp.get("top_k"), "top_p": sp.get("top_p"),
            "cfg_scale": sp.get("cfg_scale"), "load_s": t.get("load_s"), "generate_s": t.get("generate_s"),
            "decode_s": t.get("decode_s"), "total_s": t.get("total_s"), "first_output_s": t.get("first_output_s"),
            "transformer_passes": m.get("transformer_passes"),
            "output_tokens_total": sum(m.get("output_tokens") or []) if m else None,
            "output_words_total": sum(m.get("output_words") or []) if m else None,
            "gen_ppl": (m.get("quality") or {}).get("corpus_ppl"),
            "outputs": json.dumps(rec["outputs"]) if rec["outputs"] is not None else None})
    return Response(buf.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="experiments.csv"'})


@router.get("/experiments/{eid}")
def get_experiment(eid: str, request: Request) -> dict[str, Any]:
    return _record(request, eid)


@router.get("/experiments/{eid}/trace")
def get_trace(eid: str, request: Request) -> dict[str, Any]:
    _record(request, eid)
    trace = request.app.state.store.get_field(eid, "trace")
    if trace is None:
        raise HTTPException(404, f"experiment {eid} has no trace (not finished, failed, or include_trace=false)")
    return trace


@router.get("/experiments/{eid}/metrics")
def get_metrics(eid: str, request: Request) -> dict[str, Any]:
    rec = _record(request, eid)
    if rec["metrics"] is None:
        raise HTTPException(404, f"experiment {eid} has no metrics (status {rec['status']})")
    return rec["metrics"]


@router.post("/experiments/{eid}/quality")
def score_quality(eid: str, request: Request) -> dict[str, Any]:
    """Score a completed experiment's outputs with the official gen-PPL judge; stored in metrics.quality."""
    rec = _record(request, eid)
    if rec["status"] != "completed" or not rec["outputs"]:
        raise HTTPException(409, f"experiment {eid} has no outputs to score (status {rec['status']})")
    q = request.app.state.scorer.score(rec["outputs"])
    metrics = dict(rec["metrics"] or {})
    metrics["quality"] = q
    request.app.state.store.update(eid, metrics=metrics)
    return q


@router.websocket("/stream/{eid}")
async def stream(ws: WebSocket, eid: str) -> None:
    await ws.accept()
    hub, store = ws.app.state.hub, ws.app.state.store
    if hub.known(eid):
        async for ev in hub.subscribe(eid, asyncio.get_running_loop()):
            await ws.send_json(ev)
    else:
        rec = store.get(eid)
        if rec is None:
            await ws.close(code=4404, reason="experiment not found")
            return
        # finished before this process started (or forgotten): replay recorded events
        for ev in rec["events"] or []:
            await ws.send_json({**ev, "replay": True})
    await ws.close(code=1000)

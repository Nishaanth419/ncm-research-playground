"""Experiment + WebSocket contract tests with a fake runner that emits events
the way the real one does (via on_event)."""
import time

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.core.config import Settings
from app.main import create_app
from app.schemas.generation import GenerateResponse, SamplingParams, Timing
from app.services.errors import InvalidRequest


class EmittingRunner:
    def list_models(self):
        return []

    def generate(self, req, on_event=None):
        if req.seed == 13:
            raise InvalidRequest("bad seed")
        for k, n in enumerate([1, 8]):
            if k:
                on_event({"type": "generation.level_started", "level": k, "n_codes": n})
            on_event({"type": "generation.level_completed", "level": k, "n_codes": n, "seconds": 0.01, "source": "x"})
        return GenerateResponse(
            id="r", model=req.model, task="tinystories", device="cpu", seed=req.seed,
            sampling=SamplingParams(temperature=1, top_k=0, top_p=1, cfg_scale=0), outputs=["one two"],
            prompt_tokens=None, prompt_truncated=False, reference=None, transformer_passes=1, levels=[1, 8],
            timing=Timing(load_s=0, generate_s=0.02, decode_s=0.01, total_s=0.03), trace=None)


@pytest.fixture
def client(tmp_path):
    app = create_app(runner=EmittingRunner(), settings=Settings(db_path=str(tmp_path / "t.sqlite")))
    with TestClient(app) as c:
        yield c


def wait_done(client, eid):
    for _ in range(100):
        rec = client.get(f"/api/experiments/{eid}").json()
        if rec["status"] in ("completed", "failed"):
            return rec
        time.sleep(0.02)
    raise AssertionError("experiment did not finish")


def test_create_stream_and_fetch(client):
    r = client.post("/api/experiments", json={"model": "ncm-tinystories", "seed": 4})
    assert r.status_code == 201
    eid = r.json()["id"]
    assert r.json()["stream_url"] == f"/api/stream/{eid}"
    with client.websocket_connect(f"/api/stream/{eid}") as ws:
        events = []
        while True:
            try:
                events.append(ws.receive_json())
            except WebSocketDisconnect:
                break
    types = [e["type"] for e in events]
    assert types == ["generation.started", "generation.level_completed", "generation.level_started",
                     "generation.level_completed", "generation.completed"]
    assert [e["seq"] for e in events] == list(range(5))
    assert all(e["experiment_id"] == eid and e["t"] >= 0 for e in events)
    rec = wait_done(client, eid)
    assert rec["status"] == "completed" and rec["outputs"] == ["one two"] and rec["request"]["seed"] == 4
    m = client.get(f"/api/experiments/{eid}/metrics").json()
    assert m["timing"]["total_s"] == pytest.approx(0.03) and m["output_words"] == [2]
    assert client.get(f"/api/experiments/{eid}/trace").status_code == 404  # runner returned no trace
    assert client.get("/api/experiments").json()[0]["id"] == eid


def test_stream_after_completion_replays_from_db(client):
    eid = client.post("/api/experiments", json={"model": "ncm-tinystories"}).json()["id"]
    wait_done(client, eid)
    client.app.state.hub.forget(eid)  # simulate a restart: only the DB knows it
    with client.websocket_connect(f"/api/stream/{eid}") as ws:
        first = ws.receive_json()
    assert first["type"] == "generation.started" and first["replay"] is True


def test_failure_is_streamed_and_recorded(client):
    eid = client.post("/api/experiments", json={"model": "ncm-tinystories", "seed": 13}).json()["id"]
    rec = wait_done(client, eid)
    assert rec["status"] == "failed" and "bad seed" in rec["error"]
    assert rec["events"][-1]["type"] == "generation.failed"


def test_unknown_ids(client):
    assert client.get("/api/experiments/nope").status_code == 404
    with pytest.raises(WebSocketDisconnect) as e:
        with client.websocket_connect("/api/stream/nope") as ws:
            ws.receive_json()
    assert e.value.code == 4404


def test_create_validates_body(client):
    assert client.post("/api/experiments", json={"model": "m", "num_samples": 99}).status_code == 422

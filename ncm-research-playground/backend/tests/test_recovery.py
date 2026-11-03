"""Runs left queued/running by a previous server process are marked failed at startup."""
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.core.config import Settings
from app.main import create_app
from app.schemas.benchmark import BenchmarkRequest
from app.services.benchmark import BenchmarkStore
from app.services.experiment_store import ExperimentStore


class NoRunner:
    def list_models(self):
        return []


def test_interrupted_experiments_and_benchmarks_are_failed_on_startup(tmp_path):
    db = tmp_path / "r.sqlite"
    s = ExperimentStore(db)
    s.create("queued1", model="m", request={"model": "m"})
    s.create("running1", model="m", request={"model": "m"})
    s.update("running1", status="running", events=[{"type": "generation.started", "seq": 0, "t": 0.0}])
    s.create("done1", model="m", request={"model": "m"})
    s.update("done1", status="completed", outputs=["x"], events=[{"type": "generation.completed", "seq": 0, "t": 0.1}])
    b = BenchmarkStore(db)
    b.create("bench1", BenchmarkRequest(models=["m"]))
    b.update("bench1", status="running")

    with TestClient(create_app(runner=NoRunner(), settings=Settings(db_path=str(db)), scorer=object())) as c:
        for eid in ("queued1", "running1"):
            rec = c.get(f"/api/experiments/{eid}").json()
            assert rec["status"] == "failed" and "restart" in rec["error"]
            last = rec["events"][-1]
            assert last["type"] == "generation.failed" and last["recovered"] is True
        assert c.get("/api/experiments/done1").json()["status"] == "completed"
        bench = c.get("/api/benchmarks/bench1").json()
        assert bench["status"] == "failed" and "restart" in bench["error"]

        # a client streaming the interrupted run gets the terminal event instead of waiting forever
        events = []
        with c.websocket_connect("/api/stream/running1") as ws:
            while True:
                try:
                    events.append(ws.receive_json())
                except WebSocketDisconnect:
                    break
        assert [e["type"] for e in events] == ["generation.started", "generation.failed"]
        assert events[-1]["seq"] == 1

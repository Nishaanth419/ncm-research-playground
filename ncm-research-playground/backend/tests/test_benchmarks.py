"""Benchmark statistics, workflow and API contracts with a fake runner."""
import csv
import io
import time

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.schemas.generation import GenerateResponse, MemoryStats, SamplingParams, Timing
from app.services.benchmark import describe, percentile

SP = SamplingParams(temperature=1, top_k=0, top_p=1, cfg_scale=0)


def test_percentile_linear_interpolation():
    xs = [1.0, 2.0, 3.0, 4.0]
    assert percentile(xs, 50) == pytest.approx(2.5)
    assert percentile(xs, 90) == pytest.approx(3.7)
    assert percentile([5.0], 95) == 5.0


def test_describe():
    d = describe([0.3, 0.1, 0.2])
    assert d == {"n": 3, "median": pytest.approx(0.2), "p90": pytest.approx(0.28), "p95": pytest.approx(0.29),
                 "mean": pytest.approx(0.2), "std": pytest.approx(0.1), "min": 0.1, "max": 0.3}
    assert describe([])["n"] == 0 and describe([])["median"] is None


class CountingRunner:
    def __init__(self, fail_model=None):
        self.calls, self.fail_model = [], fail_model

    def list_models(self):
        return []

    def generate(self, req, on_event=None):
        self.calls.append(req)
        if req.model == self.fail_model:
            raise RuntimeError("boom")
        dt = 0.1 * req.num_samples
        return GenerateResponse(
            id="r", model=req.model, task="tinystories", device="cpu", seed=req.seed, sampling=SP,
            outputs=["a b c"] * req.num_samples, prompt_tokens=None, prompt_truncated=False, reference=None,
            transformer_passes=11, levels=[], trace=None,
            timing=Timing(load_s=0, generate_s=dt, decode_s=0.01, total_s=dt + 0.01, first_output_s=dt + 0.01),
            output_tokens=[10] * req.num_samples, tokenizer="t", output_entropy_own=[1.0] * req.num_samples,
            memory=MemoryStats(device="cpu", peak_bytes=None, note="n/a"))


def make(tmp_path, runner):
    app = create_app(runner=runner, settings=Settings(db_path=str(tmp_path / "b.sqlite")), scorer=object())
    return TestClient(app)


def wait(c, bid):
    for _ in range(200):
        b = c.get(f"/api/benchmarks/{bid}").json()
        if b["status"] in ("completed", "failed"):
            return b
        time.sleep(0.02)
    raise AssertionError("benchmark did not finish")


CFG = {"name": "t", "models": ["m1", "m2"], "batch_sizes": [1, 4], "repeats": 3, "warmup_runs": 2, "seed": 100}


def test_benchmark_runs_warmups_then_repeats_and_summarises(tmp_path):
    runner = CountingRunner()
    with make(tmp_path, runner) as c:
        r = c.post("/api/benchmarks", json=CFG)
        assert r.status_code == 201
        b = wait(c, r.json()["id"])
    assert b["status"] == "completed"
    assert len(runner.calls) == 2 * 2 * (2 + 3)
    assert all(not q.include_trace for q in runner.calls)
    assert len(b["runs"]) == 2 * 2 * 3  # warm-ups are not recorded
    timed = [q.seed for q in runner.calls if q.model == "m1" and q.num_samples == 1][2:]
    assert timed == [100, 101, 102]
    s = {(x["model"], x["batch_size"]): x for x in b["summary"]}
    assert s[("m1", 4)]["generate_s"]["median"] == pytest.approx(0.4)
    assert s[("m1", 4)]["words_per_s"]["median"] == pytest.approx(12 / 0.41)
    assert s[("m1", 4)]["passes"] == 11
    hw = b["hardware"]
    assert {"python", "torch", "platform", "cpu_count", "device", "ncm_commit"} <= set(hw)


def test_benchmark_csv_exports(tmp_path):
    with make(tmp_path, CountingRunner()) as c:
        bid = c.post("/api/benchmarks", json={**CFG, "models": ["m1"], "batch_sizes": [2]}).json()["id"]
        wait(c, bid)
        r = c.get(f"/api/benchmarks/{bid}/runs.csv")
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
        rows = list(csv.DictReader(io.StringIO(r.text)))
        assert len(rows) == 3 and rows[0]["model"] == "m1" and float(rows[0]["generate_s"]) == pytest.approx(0.2)
        s = list(csv.DictReader(io.StringIO(c.get(f"/api/benchmarks/{bid}/summary.csv").text)))
        assert s[0]["generate_s_median"] and s[0]["n"] == "3"
        assert c.get("/api/benchmarks").json()[0]["id"] == bid


def test_benchmark_failure_is_recorded(tmp_path):
    with make(tmp_path, CountingRunner(fail_model="m2")) as c:
        bid = c.post("/api/benchmarks", json=CFG).json()["id"]
        b = wait(c, bid)
    assert b["status"] == "failed" and "boom" in b["error"]


@pytest.mark.parametrize("bad", [
    {**CFG, "repeats": 0}, {**CFG, "repeats": 101}, {**CFG, "models": []}, {**CFG, "batch_sizes": [0]},
    {**CFG, "batch_sizes": [1, 2, 4, 8, 16]}, {**CFG, "warmup_runs": 9},
])
def test_benchmark_validation(tmp_path, bad):
    with make(tmp_path, CountingRunner()) as c:
        assert c.post("/api/benchmarks", json=bad).status_code == 422


def test_experiments_csv_export(tmp_path):
    with make(tmp_path, CountingRunner()) as c:
        eid = c.post("/api/experiments", json={"model": "m1", "num_samples": 2}).json()["id"]
        for _ in range(100):
            if c.get(f"/api/experiments/{eid}").json()["status"] == "completed":
                break
            time.sleep(0.02)
        rows = list(csv.DictReader(io.StringIO(c.get("/api/experiments.csv").text)))
        assert rows[0]["id"] == eid and rows[0]["model"] == "m1" and rows[0]["num_samples"] == "2"
        assert float(rows[0]["total_s"]) == pytest.approx(0.21)

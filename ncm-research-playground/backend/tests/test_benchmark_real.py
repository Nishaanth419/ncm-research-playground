"""A tiny real benchmark through the default app (NCM + TinyStories-33M on CPU)."""
import time

import pytest
from fastapi.testclient import TestClient
from huggingface_hub import try_to_load_from_cache

from app.core.research import missing_files
from app.main import create_app
from app.services.ar_runner import AR_MODELS

pytestmark = pytest.mark.model


def test_small_real_benchmark():
    spec = AR_MODELS["ar-tinystories-33m"]
    if missing_files("tinystories") or not isinstance(
            try_to_load_from_cache(spec.hf_id, "config.json", revision=spec.revision), str):
        pytest.skip("checkpoints missing")
    with TestClient(create_app()) as c:
        bid = c.post("/api/benchmarks", json={"models": ["ncm-tinystories", "ar-tinystories-33m"],
                                              "batch_sizes": [1], "repeats": 2, "warmup_runs": 1}).json()["id"]
        for _ in range(600):
            b = c.get(f"/api/benchmarks/{bid}").json()
            if b["status"] in ("completed", "failed"):
                break
            time.sleep(0.1)
    assert b["status"] == "completed", b.get("error")
    s = {g["model"]: g for g in b["summary"]}
    assert s["ncm-tinystories"]["passes"] == 11 and s["ncm-tinystories"]["n"] == 2
    assert 1 <= s["ar-tinystories-33m"]["passes"] <= 128
    assert all(g["generate_s"]["median"] > 0 for g in b["summary"])
    assert b["hardware"]["device"] == "cpu" and b["hardware"]["ncm_commit"].startswith("c702742")
    assert b["models"]["ar-tinystories-33m"]["family"] == "autoregressive"

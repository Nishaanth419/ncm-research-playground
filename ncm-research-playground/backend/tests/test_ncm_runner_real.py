"""Integration tests: NCMRunner (through the API) reproduces the official
eval_shard.py outputs exactly. Reference pickles are produced by

    python experiments/run_official.py scripts/eval_shard.py --config config/eval/<task>.yaml \
        --checkpoint-dir artifacts/<task> [--data-dir data/<task>] --n 4 --bs 4 \
        --out results/official_eval/smoke/<task>_n4.pkl
"""
import pickle
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.research import missing_files
from app.main import create_app
from app.services.ncm_runner import NCMRunner

pytestmark = pytest.mark.model
SMOKE = Path(__file__).resolve().parents[2] / "results" / "official_eval" / "smoke"


def official(task):
    p = SMOKE / f"{task}_n4.pkl"
    if missing_files(task) or not p.exists():
        pytest.skip(f"{task} checkpoint or official reference {p.name} missing")
    return pickle.load(open(p, "rb"))


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(runner=NCMRunner("cpu")))


def test_models_endpoint_reports_real_checkpoints(client):
    models = {m["id"]: m for m in client.get("/api/models").json()}
    assert set(models) == {"ncm-tinystories", "ncm-xsum", "ncm-wmt14"}
    ts = models["ncm-tinystories"]
    assert ts["conditional"] is False and ts["levels"][0] == 1 and ts["output_tokens"] == 128
    assert models["ncm-xsum"]["prefix_tokens"] == 512 and models["ncm-xsum"]["n_examples"] == 11334


def test_tinystories_matches_official_seed42(client):
    ref = official("tinystories")
    r = client.post("/api/generate", json={"model": "ncm-tinystories", "num_samples": 4, "seed": 42})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["outputs"] == ref["hyps"][:4]
    assert body["transformer_passes"] == 11
    tr = body["trace"]
    assert [lv["source"] for lv in tr["levels"]][:2] == ["unigram", "transformer"]
    assert tr["final_texts"] == body["outputs"]


@pytest.mark.parametrize("task", ["xsum", "wmt14"])
def test_conditional_examples_match_official(client, task):
    ref = official(task)
    for i in range(4):
        r = client.post("/api/generate", json={"model": f"ncm-{task}", "example_index": i, "include_trace": i == 0})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["outputs"][0] == ref["hyps"][i], (task, i)
        assert body["reference"] == ref["refs"][i]
    tr = client.post("/api/generate", json={"model": f"ncm-{task}", "example_index": 0}).json()["trace"]
    assert tr["levels"][0]["source"] == "transformer"
    assert tr["transformer_passes"] == len(tr["levels"])


def test_wmt14_free_text_prompt(client):
    if missing_files("wmt14"):
        pytest.skip("wmt14 missing")
    r = client.post("/api/generate", json={"model": "ncm-wmt14", "prompt": "Das Haus ist klein.",
                                           "include_trace": False})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["outputs"][0].strip() and body["prompt_tokens"] > 0 and body["prompt_truncated"] is False


def test_prompt_rejected_for_unconditional(client):
    r = client.post("/api/generate", json={"model": "ncm-tinystories", "prompt": "Once upon"})
    assert r.status_code == 422

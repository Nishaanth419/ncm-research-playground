"""Registry dispatch and the quality endpoint, with fakes (no checkpoints)."""
import time

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.schemas.generation import GenerateResponse, ModelInfo, SamplingParams, Timing
from app.services.errors import ModelNotFound
from app.services.registry import ModelRegistry

SP = SamplingParams(temperature=1, top_k=0, top_p=1, cfg_scale=0)


def info(mid, family):
    return ModelInfo(id=mid, family=family, task="tinystories", conditional=False, prompt_kind="none",
                     description="", available=True, missing_files=[], loaded=False, levels=[], output_tokens=128,
                     prefix_tokens=0, n_examples=0, defaults=SP)


class Fake:
    def __init__(self, mid, family, outputs):
        self.mid, self.family, self.outputs = mid, family, outputs

    def owns(self, model_id):
        return model_id == self.mid

    def list_models(self):
        return [info(self.mid, self.family)]

    def generate(self, req, on_event=None):
        return GenerateResponse(
            id="r", model=self.mid, task="tinystories", device="cpu", seed=req.seed, sampling=SP,
            outputs=self.outputs, prompt_tokens=None, prompt_truncated=False, reference=None,
            transformer_passes=3, levels=[], timing=Timing(load_s=0, generate_s=0.5, decode_s=0.01, total_s=0.51),
            trace=None, output_tokens=[len(o.split()) for o in self.outputs], tokenizer="fake",
            output_entropy_own=[0.0 for _ in self.outputs])


class FakeScorer:
    judge = "fake-judge"

    def score(self, texts):
        return {"judge": self.judge, "corpus_ppl": 12.5, "n_scored": len([t for t in texts if t.strip()]),
                "per_sample": [{"ppl": 10.0, "judge_tokens": 3, "entropy_judge_tokens": 1.0} for _ in texts]}


def test_registry_dispatches_and_merges():
    reg = ModelRegistry([Fake("a", "ncm", ["x"]), Fake("b", "autoregressive", ["y"])])
    assert [m.id for m in reg.list_models()] == ["a", "b"]

    class R:
        model, seed, num_samples = "b", 0, 1
    assert reg.generate(R()).outputs == ["y"]
    R.model = "zzz"
    with pytest.raises(ModelNotFound):
        reg.generate(R())


@pytest.fixture
def client(tmp_path):
    reg = ModelRegistry([Fake("ncm-x", "ncm", ["one two three", ""]), Fake("ar-x", "autoregressive", ["a b"])])
    app = create_app(runner=reg, settings=Settings(db_path=str(tmp_path / "q.sqlite")), scorer=FakeScorer())
    with TestClient(app) as c:
        yield c


def wait(c, eid):
    for _ in range(100):
        r = c.get(f"/api/experiments/{eid}").json()
        if r["status"] in ("completed", "failed"):
            return r
        time.sleep(0.02)
    raise AssertionError


def test_quality_endpoint_scores_and_persists(client):
    eid = client.post("/api/experiments", json={"model": "ncm-x", "tag": "cmp-1"}).json()["id"]
    rec = wait(client, eid)
    assert rec["request"]["tag"] == "cmp-1"
    assert rec["metrics"]["output_tokens"] == [3, 0] and rec["metrics"]["tokenizer"] == "fake"
    assert rec["metrics"]["throughput_tokens_per_s"] == pytest.approx(3 / 0.5)
    q = client.post(f"/api/experiments/{eid}/quality").json()
    assert q["corpus_ppl"] == 12.5 and q["judge"] == "fake-judge"
    assert client.get(f"/api/experiments/{eid}/metrics").json()["quality"]["corpus_ppl"] == 12.5


def test_quality_requires_completed_experiment(client):
    assert client.post("/api/experiments/nope/quality").status_code == 404


def test_tag_is_bounded(client):
    assert client.post("/api/experiments", json={"model": "ncm-x", "tag": "x" * 100}).status_code == 422

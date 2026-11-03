"""API contract tests with a fake runner (no checkpoints needed)."""
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas.generation import GenerateRequest, GenerateResponse, ModelInfo, SamplingParams, Timing
from app.services.errors import InvalidRequest, ModelNotFound, ModelUnavailable


class FakeRunner:
    def __init__(self):
        self.calls = []

    def list_models(self):
        return [ModelInfo(id="ncm-tinystories", family="ncm", task="tinystories", conditional=False,
                          prompt_kind="none", description="d", available=True, missing_files=[], loaded=False,
                          levels=[1, 8], output_tokens=128, prefix_tokens=1, n_examples=0,
                          defaults=SamplingParams(temperature=1.0, top_k=0, top_p=1.0, cfg_scale=0.0))]

    def generate(self, req: GenerateRequest) -> GenerateResponse:
        self.calls.append(req)
        if req.model == "missing":
            raise ModelNotFound("no such model")
        if req.model == "ncm-xsum":
            raise ModelUnavailable("checkpoint files missing")
        if req.prompt:
            raise InvalidRequest("ncm-tinystories is unconditional and takes no prompt")
        return GenerateResponse(
            id="x", model=req.model, task="tinystories", device="cpu", seed=req.seed,
            sampling=SamplingParams(temperature=1.0, top_k=0, top_p=1.0, cfg_scale=0.0),
            outputs=["hello"] * req.num_samples, prompt_tokens=None, prompt_truncated=False,
            reference=None, transformer_passes=11, levels=[1, 8],
            timing=Timing(load_s=0.0, generate_s=0.1, decode_s=0.01, total_s=0.11), trace=None)


@pytest.fixture
def client():
    app = create_app(runner=FakeRunner())
    return TestClient(app)


def test_models_lists_availability(client):
    r = client.get("/api/models")
    assert r.status_code == 200
    m = r.json()[0]
    assert m["id"] == "ncm-tinystories" and m["available"] is True and m["conditional"] is False


def test_generate_ok(client):
    r = client.post("/api/generate", json={"model": "ncm-tinystories", "num_samples": 2, "seed": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["outputs"] == ["hello", "hello"] and body["seed"] == 3 and body["timing"]["total_s"] > 0


@pytest.mark.parametrize("payload", [
    {"model": "ncm-tinystories", "num_samples": 0},
    {"model": "ncm-tinystories", "num_samples": 17},
    {"model": "ncm-tinystories", "sampling": {"temperature": -1}},
    {"model": "ncm-tinystories", "sampling": {"top_p": 1.5}},
    {"model": "ncm-tinystories", "prompt": "a", "example_index": 0},
    {"num_samples": 1},
])
def test_generate_rejects_invalid_payloads(client, payload):
    assert client.post("/api/generate", json=payload).status_code == 422


def test_error_mapping(client):
    assert client.post("/api/generate", json={"model": "missing"}).status_code == 404
    r = client.post("/api/generate", json={"model": "ncm-xsum", "prompt": "x"})
    assert r.status_code == 503 and "missing" in r.json()["detail"]
    r = client.post("/api/generate", json={"model": "ncm-tinystories", "prompt": "hi"})
    assert r.status_code == 422 and "unconditional" in r.json()["detail"]


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"

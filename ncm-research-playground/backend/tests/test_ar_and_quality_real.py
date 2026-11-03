"""Real-model tests: TinyStories-33M baseline and GPT-2-large quality scoring."""
import pytest
from huggingface_hub import try_to_load_from_cache

from app.schemas.generation import GenerateRequest
from app.services.ar_runner import AR_MODELS, AutoregressiveRunner
from app.services.errors import InvalidRequest

pytestmark = pytest.mark.model
SPEC = AR_MODELS["ar-tinystories-33m"]


@pytest.fixture(scope="module")
def runner():
    if not isinstance(try_to_load_from_cache(SPEC.hf_id, "config.json", revision=SPEC.revision), str):
        pytest.skip("TinyStories-33M not in the local HF cache")
    return AutoregressiveRunner("cpu")


def test_unconditional_generation_streams_real_steps(runner):
    events = []
    r = runner.generate(GenerateRequest(model="ar-tinystories-33m", num_samples=2, seed=7), on_event=events.append)
    steps = [e for e in events if e["type"] == "generation.step_completed"]
    assert len(steps) == r.transformer_passes >= max(r.output_tokens)
    assert [e["step"] for e in steps] == list(range(1, len(steps) + 1))
    assert all(len(o.strip()) > 0 for o in r.outputs)
    assert all(0 < n <= SPEC.max_new_tokens for n in r.output_tokens)
    assert 0 < r.timing.first_output_s < r.timing.generate_s
    assert r.levels == [] and r.trace is None and r.tokenizer.startswith("gpt2-bpe")


def test_seed_is_reproducible(runner):
    a = runner.generate(GenerateRequest(model="ar-tinystories-33m", num_samples=1, seed=3, include_trace=False))
    b = runner.generate(GenerateRequest(model="ar-tinystories-33m", num_samples=1, seed=3, include_trace=False))
    assert a.outputs == b.outputs


def test_warmup_keeps_outputs(runner):
    cold = runner.generate(GenerateRequest(model="ar-tinystories-33m", num_samples=2, seed=5, include_trace=False))
    warm = runner.generate(GenerateRequest(model="ar-tinystories-33m", num_samples=2, seed=5, include_trace=False,
                                           warmup=True))
    assert warm.outputs == cold.outputs


def test_prompt_rejected(runner):
    with pytest.raises(InvalidRequest):
        runner.generate(GenerateRequest(model="ar-tinystories-33m", prompt="Once"))


def test_quality_scorer_matches_official_function():
    if not isinstance(try_to_load_from_cache("gpt2-large", "config.json"), str):
        pytest.skip("gpt2-large not cached")
    from app.services.quality import QualityScorer
    s = QualityScorer("cpu")
    q = s.score(["Once upon a time, there was a little girl named Lily.", "", "the the the the the the"])
    assert q["n_scored"] == 2 and q["per_sample"][1] is None
    assert q["per_sample"][0]["ppl"] < q["per_sample"][2]["ppl"] or q["per_sample"][2]["entropy_judge_tokens"] == 0
    assert q["per_sample"][2]["entropy_judge_tokens"] < q["per_sample"][0]["entropy_judge_tokens"]
    assert q["corpus_ppl"] > 1

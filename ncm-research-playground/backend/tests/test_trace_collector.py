"""Unit tests for trace processing, using a fake model that mimics the call
pattern of NextConceptTransformer.generate (no checkpoints needed)."""
import math

import pytest
import torch

from app.services.trace_collector import TraceCollector, codebook_stats, latent_summary

LEVELS = [1, 2, 4]
VOCAB = 8
D = 3


class FakeVQ:
    levels = LEVELS

    def get_next_autoregressive_input(self, level_idx, z_hat, z_q_level):
        z_hat.add_(1.0)  # in-place, like the real quantizer
        return z_hat, z_hat[:, : LEVELS[level_idx + 1]]


class FakeNet:
    """Calls vq.get_next_autoregressive_input then _sample_level once per level,
    in the same order as the official generate loop."""

    def __init__(self, vq, cfg_double=False, fail_at=None):
        self.vq, self.cfg_double, self.fail_at = vq, cfg_double, fail_at

    def _sample_level(self, logits, temperature, top_k, top_p):
        return logits.argmax(-1)

    def generate(self, idx):
        B = idx.shape[0] * (2 if self.cfg_double else 1)
        z_hat = torch.zeros(B, LEVELS[-1], D)
        out = [idx.repeat(2, 1) if self.cfg_double else idx]
        for k in range(len(LEVELS) - 1):
            if k == self.fail_at:
                raise RuntimeError("boom")
            z_hat, _ = self.vq.get_next_autoregressive_input(k, z_hat, None)
            logits = torch.zeros(B, LEVELS[k + 1], VOCAB)
            logits[..., k + 1] = 1.0
            out.append(self._sample_level(logits, 1.0, 0, 1.0))
        g = torch.cat(out, dim=1)
        return g[: idx.shape[0]]


def run(cfg_double=False, events=None):
    vq = FakeVQ()
    net = FakeNet(vq, cfg_double=cfg_double)
    l0 = torch.zeros(2, 1, dtype=torch.long)
    with TraceCollector(net, vq, batch_size=2, on_event=events.append if events is not None else None) as tc:
        tc.record_level0(l0, source="unigram")
        g = net.generate(l0)
    return tc, g


def test_codebook_stats_uniform():
    s = codebook_stats(torch.tensor([[0, 1, 2, 3]]), vocab_size=8, top_k=2)
    assert s.unique == 4 and s.n_codes == 4
    assert s.utilization == pytest.approx(0.5)
    assert s.entropy_bits == pytest.approx(2.0)
    assert s.perplexity == pytest.approx(4.0)
    assert len(s.top) == 2


def test_codebook_stats_constant():
    s = codebook_stats(torch.tensor([[5, 5, 5]]), vocab_size=8)
    assert s.entropy_bits == 0.0 and s.top[0].code == 5 and s.top[0].count == 3


def test_latent_summary_cosine():
    z = torch.ones(2, 4, 3)
    s = latent_summary(z, prev=torch.zeros_like(z), final=2 * z)
    assert s.cos_to_final == pytest.approx(1.0)
    assert s.norm_mean == pytest.approx(math.sqrt(3))
    assert s.delta_norm_mean == pytest.approx(math.sqrt(3))


def test_one_record_per_generated_level_with_parallel_shapes():
    tc, g = run()
    lv = tc.levels
    assert [r.level for r in lv] == [0, 1, 2]
    assert [r.n_codes for r in lv] == LEVELS
    assert lv[0].source == "unigram" and lv[0].logits_shape is None and lv[0].seconds is None
    assert all(r.source == "transformer" for r in lv[1:])
    assert lv[2].logits_shape == [2, 4, VOCAB]
    assert all(r.seconds is not None and r.seconds >= 0 for r in lv[1:])


def test_captured_codes_equal_generate_output():
    tc, g = run()
    assert torch.equal(tc.codes_tensor(), g)


def test_latents_are_cloned_not_aliased():
    tc, _ = run()
    tr = tc.build(task="fake", final_latent=torch.ones(2, LEVELS[-1], D), decoded_levels=[["", "", ""]] * 2,
                  final_texts=["", ""], total_seconds=0.0, vocab_sizes=[VOCAB] * 3)
    # the fake adds 1.0 in place per level; aliasing would make every norm equal
    norms = [r.latent_in.norm_mean for r in tr.levels[1:]]
    assert norms[0] < norms[1]
    assert tr.levels[0].latent_in is None


def test_cfg_doubled_batch_is_sliced():
    tc, g = run(cfg_double=True)
    assert tc.codes_tensor().shape == g.shape == (2, sum(LEVELS))


def test_events_are_emitted_in_execution_order():
    events = []
    run(events=events)
    types = [(e["type"], e["level"]) for e in events]
    assert types == [
        ("generation.level_completed", 0),
        ("generation.level_started", 1), ("generation.level_completed", 1),
        ("generation.level_started", 2), ("generation.level_completed", 2),
    ]
    assert events[-1]["n_codes"] == 4 and "seconds" in events[-1]


class FakeNetPredictsL0(FakeNet):
    """Conditional checkpoints sample level 0 inside generate before the level loop."""

    def generate(self, prefix):
        B = prefix.shape[0]
        l0 = self._sample_level(torch.zeros(B, LEVELS[0], VOCAB), 1.0, 0, 1.0)
        return super().generate(l0)


def test_level0_predicted_inside_generate():
    vq = FakeVQ()
    net = FakeNetPredictsL0(vq)
    events = []
    with TraceCollector(net, vq, batch_size=2, on_event=events.append) as tc:
        g = net.generate(torch.zeros(2, 5))
    assert [r.level for r in tc.levels] == [0, 1, 2]
    assert tc.levels[0].source == "transformer" and tc.levels[0].seconds is not None
    assert torch.equal(tc.codes_tensor(), g)
    assert events[0]["type"] == "generation.level_completed" and events[0]["level"] == 0


def test_methods_restored_even_on_failure():
    vq = FakeVQ()
    net = FakeNet(vq, fail_at=1)
    orig_s, orig_n = net._sample_level, vq.get_next_autoregressive_input
    with pytest.raises(RuntimeError):
        with TraceCollector(net, vq, batch_size=2):
            net.generate(torch.zeros(2, 1, dtype=torch.long))
    assert net._sample_level == orig_s and vq.get_next_autoregressive_input == orig_n
    assert "_sample_level" not in vars(net)


def test_build_trace_is_json_serialisable_and_bounded():
    tc, g = run()
    final = torch.ones(2, LEVELS[-1], D) * 3
    decoded = [["a", "b", "c"], ["d", "e", "f"]]
    tr = tc.build(task="fake", device="cpu", final_latent=final, decoded_levels=decoded,
                  final_texts=["x", "y"], total_seconds=0.5, vocab_sizes=[VOCAB] * 3, max_detail_samples=1)
    js = tr.model_dump_json()
    assert '"generation.level' not in js
    assert len(tr.levels[2].codes) == 1  # detail kept for 1 sample only
    assert tr.levels[2].decoded == ["c"]
    assert tr.levels[2].stats.n_codes == 2 * 4  # stats over the whole batch
    assert tr.transformer_passes == 2
    assert tr.levels[1].latent_in.cos_to_final == pytest.approx(1.0)

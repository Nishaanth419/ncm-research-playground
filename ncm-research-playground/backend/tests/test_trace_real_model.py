"""Integration test: trace a real TinyStories NCM generation on CPU."""
import pytest
import torch

from app.core.research import load_bundle, missing_files
from app.services.trace_collector import TraceCollector

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def bundle():
    if missing_files("tinystories"):
        pytest.skip("tinystories checkpoint not downloaded")
    return load_bundle("tinystories", "cpu")


def test_trace_matches_real_generation(bundle):
    b, B = bundle, 2
    eos = b.tokenizer.pre_tokenizer.eos_token_id
    torch.manual_seed(0)
    l0 = torch.multinomial(b.resolve(b.cfg.generation.level0_unigram).float(), B * b.levels[0],
                           replacement=True).view(B, b.levels[0])
    events = []
    with TraceCollector(b.net, b.vq, batch_size=B, on_event=events.append) as tc:
        tc.record_level0(l0, source="unigram")
        g = b.net.generate(l0, torch.full((B, 1), eos), temperature=1.0, top_k=0, top_p=1.0,
                           cfg_scale=0.0, cfg_null_eos=eos)
    assert "_sample_level" not in vars(b.net)
    assert torch.equal(tc.codes_tensor(), g)

    with torch.no_grad():
        final = b.vq.idx_to_z_q(g)
        decoded = b.tokenizer.decode_multiscales(g, skip_special_tokens=True)
    tr = tc.build(task="tinystories", final_latent=final, decoded_levels=decoded,
                  final_texts=[d[-1] for d in decoded], total_seconds=0.0, vocab_sizes=b.vocab_sizes)

    assert tr.levels_schedule == b.levels == [1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128]
    assert tr.transformer_passes == 11
    assert [lv.logits_shape for lv in tr.levels[1:]] == [[B, n, 4096] for n in b.levels[1:]]
    assert len(events) == 1 + 2 * 11
    cos = [lv.latent_in.cos_to_final for lv in tr.levels[1:]]
    assert cos == sorted(cos) and cos[-1] < 1.0  # input to the last level still lacks its codes
    assert all(len(lv.decoded) == B for lv in tr.levels)
    assert len(tr.levels[-1].decoded[0]) > len(tr.levels[1].decoded[0])
    assert len(tr.model_dump_json()) < 200_000

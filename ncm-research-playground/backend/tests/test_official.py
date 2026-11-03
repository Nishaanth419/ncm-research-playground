import torch

from app.core.official import official_pick_tokens, parse_adjacency


def test_pick_tokens_without_penalties_is_argmax():
    logits = torch.randn(2, 5, 7)
    pick = official_pick_tokens(freq=0.0, mu=0.0, adj=None, allow=None)
    assert torch.equal(pick(logits), logits.argmax(-1))


def test_pick_tokens_frequency_penalty_discourages_repeats():
    logits = torch.zeros(1, 3, 4)
    logits[..., 2] = 1.0  # argmax would repeat code 2 three times
    pick = official_pick_tokens(freq=5.0, mu=0.0, adj=None, allow=None)
    out = pick(logits)[0].tolist()
    assert out[0] == 2 and out[1] != 2


def test_parse_adjacency_matches_official_format():
    assert parse_adjacency("5,4,0.5") == [5.0, 4.0, 0.5]
    assert parse_adjacency("") is None

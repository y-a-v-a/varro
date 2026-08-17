"""tokenizer.py — embedding-cost arithmetic and a miniature BPE round-trip."""

import pytest

from tokenizer import SPECIAL, embedding_cost, morphology, train_one, verify_roundtrip

SENTENCES = [
    "Gallia est omnis diuisa in partes tres.",
    "Quarum unam incolunt Belgae, aliam Aquitani.",
    "Arma uirumque cano Troiae qui primus ab oris.",
    "Amo amas amat amamus amatis amant amare.",
    "Ceterum censeo Carthaginem esse delendam.",
] * 4


def test_embedding_cost_arithmetic():
    cost = embedding_cost(vocab_size=8000, d_model=256, n_layer=6)
    assert cost["body"] == 12 * 6 * 256 ** 2
    assert cost["embed"] == 8000 * 256
    assert cost["total"] == cost["body"] + cost["embed"]
    assert cost["embed_share"] == pytest.approx(cost["embed"] / cost["total"])


@pytest.fixture(scope="module")
def tiny_tokenizer():
    records = [{"text": s} for s in SENTENCES]
    return train_one(records, vocab_size=300), records


def test_special_tokens_reserved(tiny_tokenizer):
    tok, _ = tiny_tokenizer
    for token in SPECIAL:
        assert tok.token_to_id(token) is not None, token


def test_roundtrip_is_exact(tiny_tokenizer):
    tok, records = tiny_tokenizer
    assert verify_roundtrip(tok, records) == 0


def test_morphology_report_shape(tiny_tokenizer):
    tok, _ = tiny_tokenizer
    report = morphology(tok)
    assert report["paradigm_total"] == 12
    assert 0 <= report["paradigm_whole"] <= report["paradigm_total"]
    assert 0.0 < report["stem_spread"] <= 1.0
    assert report["example"]  # how `amabamus` divides

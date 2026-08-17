"""evaluate.py — n-gram baseline maths and the memorisation instrument."""

import math

from evaluate import NgramBaseline, build_haystack, memorisation, perplexity

WORDS = ("arma uirumque cano troiae qui primus ab oris italiam fato profugus "
         "lauiniaque uenit litora multum ille et terris iactatus alto").split()


def test_ngram_logprob_matches_hand_computation():
    model = NgramBaseline(vocab_size=10, lambda_bigram=0.7)
    model.train([[1, 2, 3, 2]])
    lp, n = model.logprob([1, 2])
    denominator = 4 + 10
    p1 = (1 + 1) / denominator
    p2 = 0.7 * 1.0 + 0.3 * ((2 + 1) / denominator)  # bigram(1->2) is certain
    assert n == 2
    assert abs(lp - (math.log(p1) + math.log(p2))) < 1e-12


def test_ngram_unseen_token_smoothed_not_zero():
    model = NgramBaseline(vocab_size=10)
    model.train([[1, 2]])
    lp, _ = model.logprob([9])
    assert lp == math.log(1 / 12)  # add-1 over total+vocab


def test_perplexity():
    assert perplexity(0.0, 0) is None
    assert abs(perplexity(-2 * math.log(4.0), 2) - 4.0) < 1e-12


def test_memorised_text_scores_one():
    haystack = build_haystack([{"text": " ".join(WORDS)}])
    result = memorisation(" ".join(WORDS[2:16]), haystack)
    assert result["windows"] == 3
    assert result["memorised"] == 3
    assert result["fraction"] == 1.0
    assert result["longest_verbatim_words"] == 14


def test_novel_text_scores_zero():
    haystack = build_haystack([{"text": " ".join(WORDS)}])
    novel = " ".join(f"nouum{i}" for i in range(15))
    result = memorisation(novel, haystack)
    assert result["memorised"] == 0
    assert result["fraction"] == 0.0
    assert result["longest_verbatim_words"] == 0


def test_text_shorter_than_window_is_unscored():
    assert memorisation("breue dictum", "quodlibet")["fraction"] is None


def test_haystack_normalises_internal_whitespace():
    haystack = build_haystack([{"text": "arma  uirumque\ncano"}])
    assert "arma uirumque cano" in haystack

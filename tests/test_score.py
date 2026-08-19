"""score.py — window coverage. Every token scored once, none scored cold."""

import math

import pytest

from score import perplexity, window_plan


def covered(n, context, stride):
    """Target indices scored, in order, across the whole plan."""
    out = []
    for start, end, first_target in window_plan(n, context, stride):
        out.extend(range(max(first_target, start + 1), end))
    return out


@pytest.mark.parametrize("n,context,stride", [
    (10, 512, 256), (512, 512, 256), (513, 512, 256), (1000, 512, 256),
    (5000, 512, 128), (5000, 512, 511), (2, 512, 256), (77, 8, 3),
])
def test_every_token_scored_exactly_once(n, context, stride):
    """Missing a token understates its cost; double-counting weights it twice."""
    assert covered(n, context, stride) == list(range(1, n))


@pytest.mark.parametrize("n,context,stride", [
    (1000, 512, 256), (5000, 512, 128), (77, 8, 3),
])
def test_no_token_is_scored_without_left_context(n, context, stride):
    """A target at the very start of a window would be predicted from nothing.

    Only the true first token of the text may sit at position 1 of its window;
    everything else must carry at least `context - stride` tokens of history.
    """
    for start, end, first_target in window_plan(n, context, stride):
        if start == 0:
            continue
        assert first_target - start >= context - stride


def test_short_text_needs_no_window():
    assert window_plan(1, 512, 256) == []
    assert window_plan(0, 512, 256) == []


def test_single_window_when_text_fits():
    assert window_plan(300, 512, 256) == [(0, 300, 1)]


def test_windows_never_exceed_the_context():
    for start, end, _ in window_plan(5000, 512, 256):
        assert end - start <= 512


def test_perplexity_is_the_exponential_of_mean_negative_logprob():
    assert perplexity([]) is None
    assert abs(perplexity([math.log(0.5)] * 4) - 2.0) < 1e-12
    assert abs(perplexity([math.log(0.25), math.log(0.25)]) - 4.0) < 1e-12


def test_a_confident_model_scores_lower_than_an_uncertain_one():
    assert perplexity([math.log(0.9)] * 10) < perplexity([math.log(0.1)] * 10)

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


# --- two-tailed classification ---------------------------------------------
#
# Thresholds calibrated on the 2026-08-19 audit of the 79M-word corpus. The
# numbers in these tests are measurements, not fixtures: if the defaults move,
# they should move because a later audit said so.

from score import DEFAULT_FLOOR_VARIETY, classify  # noqa: E402


def test_ordinary_latin_passes():
    """Real held-out Latin scored 41.4; the corpus median is 24.9."""
    assert classify(24.9, 0.395) == "ok"
    assert classify(41.4, 0.50) == "ok"
    assert classify(61.4, 0.40) == "ok"      # the corpus p99


def test_foreign_text_is_flagged_surprising():
    """English measured 495.8, gibberish 3406.5, a Lorem ipsum stub 212.3."""
    for ppl in (212.3, 495.8, 3406.5):
        assert classify(ppl, 0.70) == "surprising"


def test_degenerate_documents_are_flagged():
    """Navigation indexes, dash placeholders, and a repeated EDCS field."""
    assert classify(1.06, 0.006) == "degenerate"   # Legio XXII, 20k words
    assert classify(1.06, 0.010) == "degenerate"   # InterNexii link list
    assert classify(1.13, 0.022) == "degenerate"   # Nepos dash placeholder


def test_formulaic_latin_is_not_condemned_for_being_predictable():
    """The reason the floor test needs a variety term at all.

    Epigraphy, liturgy and legal boilerplate are legitimately repetitive. The
    least varied genuine document measured 0.129, six times the threshold.
    """
    assert classify(2.5, 0.129) == "ok"
    assert classify(1.5, 0.30) == "ok"
    assert DEFAULT_FLOOR_VARIETY < 0.129


def test_low_perplexity_alone_is_not_enough():
    assert classify(1.0, 0.40) == "ok"
    assert classify(1.0, 0.01) == "degenerate"


def test_missing_variety_never_triggers_the_floor():
    """A document too short to measure must not be condemned by default."""
    assert classify(1.0, None) == "ok"


def test_unscoreable_document_is_not_a_finding():
    assert classify(None, None) == "ok"
    assert classify(None, 0.001) == "ok"


def test_thresholds_are_overridable():
    assert classify(80.0, 0.4, ceiling=100.0) == "ok"
    assert classify(80.0, 0.4, ceiling=50.0) == "surprising"
    assert classify(2.0, 0.20, floor=5.0, floor_variety=0.25) == "degenerate"


def test_surprising_wins_over_degenerate():
    """They cannot both apply, but the order must be defined, not accidental."""
    assert classify(500.0, 0.001) == "surprising"

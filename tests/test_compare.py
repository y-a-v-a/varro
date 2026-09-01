"""compare.py — the guard against tabulating incomparable perplexities."""

import compare

FINGERPRINT = {"track": "A", "records": 10_192, "words": 78_975_089,
               "group_by": "author", "seed": 20260815}
OLDER = {**FINGERPRINT, "records": 909, "words": 17_256_637}
TOKENIZER = {"vocab": 4000, "sha256": "9f1c0a2b3d4e5f60"}
RETRAINED = {"vocab": 4000, "sha256": "0000111122223333"}


def run(fingerprint=FINGERPRINT, tokenizer=TOKENIZER):
    report = {"run": "latin-2", "parameters": 15_900_000}
    if fingerprint is not None:
        report["corpus"] = fingerprint
    if tokenizer is not None:
        report["tokenizer"] = tokenizer
    return report


def floor_report(fingerprint=FINGERPRINT, tokenizer=TOKENIZER):
    report = {"baselines": {"overall_perplexity": 195.2}}
    if fingerprint is not None:
        report["corpus"] = fingerprint
    if tokenizer is not None:
        report["tokenizer"] = tokenizer
    return report


def test_same_build_is_comparable():
    assert compare.comparability(run(), 195.2, floor_report()) == "ok"


def test_corpus_that_grew_under_a_fixed_name_is_flagged():
    assert compare.comparability(run(OLDER), 195.2, floor_report()) == "stale"


def test_report_without_a_fingerprint_is_never_assumed_comparable():
    """Every checkpoint trained before 2026-08-18 is in this state."""
    assert compare.comparability(run(None), 195.2, floor_report()) == "unverified"


def test_floor_without_a_fingerprint_is_also_unverified():
    assert compare.comparability(run(), 195.2, floor_report(None)) == "unverified"


def test_absent_floor_is_reported_as_such_not_as_drift():
    assert compare.comparability(run(), None, None) == "no-floor"


def test_retrained_tokenizer_over_the_same_corpus_is_flagged():
    """Measured: the classical floor moved 158.8 -> 149.8 on a corpus that
    changed by three words, because the 4k vocabulary had been retrained."""
    assert compare.comparability(run(tokenizer=RETRAINED), 195.2,
                                 floor_report()) == "stale"


def test_missing_tokenizer_stamp_is_unverified_not_comparable():
    assert compare.comparability(run(tokenizer=None), 195.2,
                                 floor_report()) == "unverified"
    assert compare.comparability(run(), 195.2,
                                 floor_report(tokenizer=None)) == "unverified"


def test_every_status_has_a_mark_and_an_explanation():
    for status in ("ok", "stale", "unverified", "no-floor"):
        assert status in compare.MARKS
    assert set(compare.EXPLANATIONS) == set(compare.MARKS) - {"ok"}

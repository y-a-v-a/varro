"""compare.py — the guard against tabulating incomparable perplexities."""

import compare

FINGERPRINT = {"track": "A", "records": 10_192, "words": 78_975_089,
               "group_by": "author", "seed": 20260815}
OLDER = {**FINGERPRINT, "records": 909, "words": 17_256_637}


def run(fingerprint=FINGERPRINT):
    report = {"run": "latin-2", "parameters": 15_900_000}
    if fingerprint is not None:
        report["corpus"] = fingerprint
    return report


def test_same_build_is_comparable():
    assert compare.comparability(run(), 195.2, FINGERPRINT) == "ok"


def test_corpus_that_grew_under_a_fixed_name_is_flagged():
    assert compare.comparability(run(OLDER), 195.2, FINGERPRINT) == "stale"


def test_report_without_a_fingerprint_is_never_assumed_comparable():
    """Every checkpoint trained before 2026-08-18 is in this state."""
    assert compare.comparability(run(None), 195.2, FINGERPRINT) == "unverified"


def test_floor_without_a_fingerprint_is_also_unverified():
    assert compare.comparability(run(), 195.2, None) == "unverified"


def test_absent_floor_is_reported_as_such_not_as_drift():
    assert compare.comparability(run(), None, None) == "no-floor"


def test_every_status_has_a_mark_and_an_explanation():
    for status in ("ok", "stale", "unverified", "no-floor"):
        assert status in compare.MARKS
    assert set(compare.EXPLANATIONS) == set(compare.MARKS) - {"ok"}

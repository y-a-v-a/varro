"""train.py — the guard that stops a stale bigram floor being quoted.

Skipped where MLX is absent: train.py imports it at module level, and the rest
of the suite deliberately runs without it.
"""

import json

import pytest

pytest.importorskip("mlx.core", reason="train.py requires MLX")

import train  # noqa: E402


SUMMARY = {"track": "A", "records": 909, "words": 17_256_637,
           "train_words": 16_401_214, "validation_words": 855_423,
           "group_by": "author", "seed": 20260815, "period_filter": None,
           "sources": {"perseus-canonical-latinLit": 284}}


@pytest.fixture
def paths(tmp_path, monkeypatch):
    processed, reports = tmp_path / "processed", tmp_path / "reports"
    processed.mkdir()
    reports.mkdir()
    monkeypatch.setattr(train, "PROCESSED", processed)
    monkeypatch.setattr(train, "REPORTS", reports)
    return processed, reports


def write_summary(processed, corpus="corpus", **overrides):
    (processed / f"{corpus}.summary.json").write_text(
        json.dumps({**SUMMARY, **overrides}), encoding="utf-8")


def write_baseline(reports, corpus="corpus", fingerprint=SUMMARY,
                   perplexity=195.2, corpus_name="corpus"):
    payload = {"baselines": {"overall_perplexity": perplexity},
               "corpus": {k: fingerprint.get(k) for k in train.FINGERPRINT_KEYS}}
    if corpus_name is not None:
        payload["corpus_name"] = corpus_name
    (reports / f"baselines-{corpus}.json").write_text(json.dumps(payload),
                                                      encoding="utf-8")


def test_fingerprint_drops_fields_that_do_not_identify_a_build(paths):
    processed, _ = paths
    write_summary(processed)
    assert train.corpus_fingerprint("corpus") == {
        k: SUMMARY[k] for k in train.FINGERPRINT_KEYS}
    assert "sources" not in train.corpus_fingerprint("corpus")


def test_no_baseline_is_not_a_stale_baseline(paths):
    processed, _ = paths
    write_summary(processed)
    assert train.read_floor("corpus") == (None, "")


def test_matching_fingerprint_is_quotable(paths):
    processed, reports = paths
    write_summary(processed)
    write_baseline(reports)
    assert train.read_floor("corpus") == (195.2, "")


def test_grown_corpus_makes_the_floor_unquotable(paths):
    """The case this exists for: a source lands, the name does not change."""
    processed, reports = paths
    write_summary(processed, words=78_975_089, train_words=75_043_813,
                  records=10_192)
    write_baseline(reports)
    floor, reason = train.read_floor("corpus")
    assert floor == 195.2
    assert "words: 17256637 -> 78975089" in reason
    assert "records" in reason and "train_words" in reason


def test_regrouped_split_makes_the_floor_unquotable(paths):
    """Same text, different held-out authors — the perplexities do not compare."""
    processed, reports = paths
    write_summary(processed, group_by="work")
    _, reason = train.read_floor("corpus")
    assert reason == ""  # no baseline yet
    write_baseline(reports)
    assert "group_by: 'author' -> 'work'" in train.read_floor("corpus")[1]


def test_baseline_stamped_with_another_corpus_is_rejected(paths):
    """The bug as it was: baselines-classical.json stamped with the full corpus."""
    processed, reports = paths
    write_summary(processed, corpus="classical", records=203,
                  words=4_539_515, period_filter=["CLASSICAL"])
    write_baseline(reports, corpus="classical", perplexity=158.8,
                   corpus_name="corpus")
    floor, reason = train.read_floor("classical")
    assert floor == 158.8
    assert "corpus_name='corpus'" in reason


def test_unstamped_legacy_report_falls_back_to_fingerprint_drift(paths):
    """Reports written before corpus_name existed are still checked."""
    processed, reports = paths
    write_summary(processed, corpus="classical", records=203)
    write_baseline(reports, corpus="classical", corpus_name=None)
    _, reason = train.read_floor("classical")
    assert "records: 909 -> 203" in reason

"""bundle.py — HF bundle assembly: guards, fingerprints, substitution."""

import json

import pytest

import bundle
from bundle import (build_values, default_period_tag, fmt_m, guard_track,
                    matched_floor, memo_table, render_card, source_table)


def train_report(track="A", **overrides):
    report = {
        "config": {"vocab": 4000, "dim": 288, "layers": 6, "heads": 8,
                   "context": 512, "steps": 9249, "batch": 32, "lr": 3e-4,
                   "warmup": 300, "weight_decay": 0.1, "seed": 1,
                   "period_tags": True, "config": "configs/x.yaml"},
        "corpus": {"track": track, "records": 261, "words": 5_055_913,
                   "period_filter": ["CLASSICAL"]},
        "corpus_name": "classical",
        "tokenizer": {"sha256": "f64b7246a477af75", "vocab": 4000},
        "parameters": 7_278_912,
        "train_tokens": 10_671_909,
        "epochs": 14.2,
        "final": {"val_perplexity": 70.1, "elapsed_s": 11013.0},
        "generated": "2026-09-01T08:45:22+00:00",
    }
    report.update(overrides)
    return report


def memo_report():
    return {
        "memorization": {"memorised": 0, "longest_verbatim_words": 0},
        "curve": {"3": {"text": 0.178, "control": 0.07},
                  "12": {"text": 0.0, "control": 0.012}},
        "tokenizer": {"sha256": "f64b7246a477af75", "vocab": 4000},
    }


def test_track_b_is_refused():
    with pytest.raises(SystemExit, match="Track 'B'"):
        guard_track(train_report(track="B"), "latin-lii-latinitas")


def test_track_a_passes():
    guard_track(train_report(), "latin-vii")


def test_missing_track_is_refused_not_assumed():
    report = train_report()
    del report["corpus"]["track"]
    with pytest.raises(SystemExit):
        guard_track(report, "x")


def test_floor_refused_on_tokenizer_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(bundle, "REPORTS", tmp_path)
    report = train_report()
    baseline = {"corpus": report["corpus"],
                "tokenizer": {"sha256": "someotherhash00", "vocab": 4000},
                "baselines": {"overall_perplexity": 185.7}}
    (tmp_path / "baselines-classical.json").write_text(json.dumps(baseline))
    with pytest.raises(SystemExit, match="tokenizer"):
        matched_floor(report, "classical")


def test_floor_quoted_when_fingerprints_match(tmp_path, monkeypatch):
    monkeypatch.setattr(bundle, "REPORTS", tmp_path)
    report = train_report()
    baseline = {"corpus": report["corpus"], "tokenizer": report["tokenizer"],
                "baselines": {"overall_perplexity": 185.7}}
    (tmp_path / "baselines-classical.json").write_text(json.dumps(baseline))
    assert matched_floor(report, "classical") == 185.7


def values():
    sources = [{"name": "Perseus", "share": 0.873, "license": "CC-BY-SA-4.0"}]
    return build_values("latin-vii", train_report(), 185.7, memo_report(),
                        sources, "CC-BY-SA-4.0", "CLASSICAL")


def test_ratio_is_computed_not_transcribed():
    assert values()["ratio"] == "2.65"


def test_tokens_seen_derived_from_recipe():
    # 9249 steps x 32 batch x 512 context
    assert values()["tokens_seen_m"] == "151.5M"


def test_unmapped_licence_is_an_error_not_a_default():
    with pytest.raises(SystemExit, match="HF licence"):
        build_values("x", train_report(), 185.7, memo_report(),
                     [], "CC-BY-NC-SA-4.0", "CLASSICAL")


def test_period_tag_single_period():
    assert default_period_tag(train_report()) == "CLASSICAL"


def test_period_tag_none_for_mixed_corpus():
    report = train_report()
    report["corpus"]["period_filter"] = ["CLASSICAL", "LATE"]
    assert default_period_tag(report) is None


def test_period_tag_none_when_tags_disabled():
    report = train_report()
    report["config"]["period_tags"] = False
    assert default_period_tag(report) is None


def test_missing_placeholder_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setattr(bundle, "MODELCARDS", tmp_path)
    (tmp_path / "x.md").write_text("ppl is ${ppl}, and ${no_such_value}")
    with pytest.raises(SystemExit, match="no_such_value"):
        render_card("x", {"ppl": "70.1"})


def test_bibtex_braces_survive_substitution(tmp_path, monkeypatch):
    monkeypatch.setattr(bundle, "MODELCARDS", tmp_path)
    (tmp_path / "x.md").write_text("@software{claudius_2026,\n} ${ppl}")
    assert "@software{claudius_2026," in render_card("x", {"ppl": "70.1"})


def test_fmt_m():
    assert fmt_m(151_552_000) == "151.6M"
    assert fmt_m(1_000_000) == "1M"
    assert fmt_m(999) == "999"


def test_source_table_renders_public_domain_readably():
    rows = [{"name": "A", "share": 0.5, "license": "public-domain"}]
    assert source_table(rows) == "| A | 50.0% | public domain |"


def test_memo_table_sorts_windows_numerically():
    table = memo_table({"12": {"text": 0.0, "control": 0.012},
                        "3": {"text": 0.178, "control": 0.07}})
    assert table.index("| 3 |") < table.index("| 12 |")

"""fetch.py — manifest validation and track/tier selection (no network)."""

from fetch import selected, validate


def source(**overrides):
    base = {"id": "src-a", "name": "Source A", "url": "https://example.org",
            "method": "git", "license": "CC-BY-SA-4.0", "tier": "green",
            "text_provenance": "transcribed", "include": True}
    return {**base, **overrides}


def manifest(*sources):
    return {"sources": list(sources)}


def test_valid_manifest_passes():
    assert validate(manifest(source())) == []


def test_empty_manifest_fails():
    assert validate({}) == ["manifest has no `sources` list"]


def test_missing_field_reported():
    src = source()
    del src["license"]
    assert any("missing required field `license`" in e
               for e in validate(manifest(src)))


def test_duplicate_id_reported():
    errors = validate(manifest(source(), source()))
    assert any("duplicate id" in e for e in errors)


def test_bad_tier_and_method_reported():
    errors = validate(manifest(source(tier="purple", method="carrier-pigeon")))
    assert any("tier must be one of" in e for e in errors)
    assert any("method must be one of" in e for e in errors)


def test_unknown_tier_cannot_be_included():
    errors = validate(manifest(source(tier="unknown", restriction="unclear")))
    assert any("tier `unknown` with include:true" in e for e in errors)


def test_amber_requires_restriction():
    errors = validate(manifest(source(tier="amber")))
    assert any("requires a `restriction`" in e for e in errors)
    ok = validate(manifest(source(tier="amber", restriction="non-commercial")))
    assert not ok


def test_selected_track_a_excludes_amber():
    m = manifest(source(),
                 source(id="src-b", tier="amber", restriction="nc"))
    assert [s["id"] for s in selected(m, None, None, False, "A")] == ["src-a"]
    assert [s["id"] for s in selected(m, None, None, False, "B")] == \
        ["src-a", "src-b"]


def test_selected_only_bypasses_track_filter():
    # An explicitly named source is returned even off-track, so the caller's
    # track guard can refuse it by name instead of skipping it silently.
    m = manifest(source(), source(id="src-b", tier="amber", restriction="nc"))
    assert [s["id"] for s in selected(m, "src-b", None, False, "A")] == ["src-b"]


def test_selected_honours_include_flag():
    m = manifest(source(include=False))
    assert selected(m, None, None, False, "A") == []
    assert [s["id"] for s in selected(m, None, None, True, "A")] == ["src-a"]


def test_selected_tier_filter():
    m = manifest(source(), source(id="src-b", tier="amber", restriction="nc"))
    assert [s["id"] for s in selected(m, None, "amber", False, "B")] == ["src-b"]


# --- text provenance --------------------------------------------------------

def test_text_provenance_is_required():
    """No default. A source that does not say how its text was produced must
    fail rather than be assumed human-transcribed — that assumption is exactly
    the one that would let model-corrected text in unnoticed."""
    entry = source()
    del entry["text_provenance"]
    problems = validate(manifest(entry))
    assert any("text_provenance is required" in p for p in problems)


def test_unknown_text_provenance_class_is_rejected():
    problems = validate(manifest(source(text_provenance="probably-fine")))
    assert any("text_provenance must be one of" in p for p in problems)


def test_every_valid_class_passes():
    for cls in ("transcribed", "ocr", "model-corrected", "unknown"):
        assert not [p for p in validate(manifest(source(text_provenance=cls)))
                    if "text_provenance" in p]


def test_the_shipped_manifest_declares_provenance_for_every_source():
    import yaml
    with open("manifest/sources.yaml", encoding="utf-8") as fh:
        declared = yaml.safe_load(fh)["sources"]
    missing = [s["id"] for s in declared if not s.get("text_provenance")]
    assert not missing, f"no text_provenance: {missing}"


def test_csel_is_labelled_ocr_because_it_is():
    """The corpus already contains OCR-derived text; the label says so."""
    import yaml
    with open("manifest/sources.yaml", encoding="utf-8") as fh:
        by_id = {s["id"]: s for s in yaml.safe_load(fh)["sources"]}
    assert by_id["opengreekandlatin-csel"]["text_provenance"] == "ocr"
    assert by_id["perseus-canonical-latinLit"]["text_provenance"] == "transcribed"


# --- Corpus Corporum: one work per request, slowly --------------------------
#
# There is no bulk export, so the corpus arrives a work at a time from
# download.php?type=file-pos-xml&idno=<id>. The properties that matter are
# that it fetches the right subset, waits between requests, and can be
# interrupted — a fourteen-hour job will be.

import re  # noqa: E402

from fetch import (CC_BACKOFF_SECONDS, _cc_selection,  # noqa: E402
                   fetch_corpus_corporum)

CC_ROWS = """,work_id,title_cc,series_title,word_count,prelemmatized
0,10,De facto Ungarie,Itinera,3340,True
1,20,Epistolae,"Patrologia Latina, vol. 144",231050,True
2,30,Carmina,Musisque Deoque 2007,5000,True
3,40,Aliud,digilibLT,7000,True
4,50,Summa,Corpus Thomisticum,900,True
5,60,Adversus Simoniacos,Itinera,248673,
"""


def _meta(tmp_path, body=CC_ROWS):
    p = tmp_path / "cc_metadata.csv"
    p.write_text(body, encoding="utf-8")
    return p


def test_selection_excludes_what_the_corpus_already_holds(tmp_path):
    src = {"exclude_series": [r"Patrologia\s+Latina", "digilibLT"]}
    ids = [r["work_id"] for r in _cc_selection(src, _meta(tmp_path))]
    assert ids == ["10", "30", "50"]


def test_selection_without_exclusions_takes_everything_available(tmp_path):
    ids = [r["work_id"] for r in _cc_selection({}, _meta(tmp_path))]
    assert ids == ["10", "20", "30", "40", "50"]


def test_works_with_no_lemmatised_copy_are_never_requested(tmp_path):
    """`type=file-pos-xml` serves only lemmatised works, and the metadata says
    which those are. Work 60 would earn a soft 404 and a wasted 20s wait."""
    assert "60" not in [r["work_id"] for r in _cc_selection({}, _meta(tmp_path))]


def test_patrologia_pattern_tolerates_the_volume_suffix(tmp_path):
    """The series is per volume — `Patrologia Latina, vol. 144` — so an exact
    match would exclude nothing at all."""
    src = {"exclude_series": [r"Patrologia\s+Latina"]}
    assert "20" not in [r["work_id"] for r in _cc_selection(src, _meta(tmp_path))]


def _run(tmp_path, monkeypatch, delay=0):
    """Run the fetcher with the network and the clock replaced."""
    import fetch
    calls, sleeps = [], []
    def fake_download(url, target):
        calls.append(url)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"PK\x03\x04zipdata")
    monkeypatch.setattr(fetch, "_download", fake_download)
    monkeypatch.setattr(fetch.time, "sleep", lambda s: sleeps.append(s))
    dest = tmp_path / "raw"
    (dest / "_metadata").mkdir(parents=True, exist_ok=True)
    _meta(dest / "_metadata")
    src = {"id": "corpus-corporum", "delay_seconds": delay,
           "exclude_series": [r"Patrologia\s+Latina", "digilibLT"],
           "id_source": "https://example.invalid/cc.csv",
           "url_template": "https://mlat.uzh.ch/php_modules/download.php"
                           "?type=file-pos-xml&idno={idno}"}
    return fetch_corpus_corporum(src, dest), calls, sleeps, dest


def test_fetches_only_the_selected_works(tmp_path, monkeypatch):
    result, calls, _, dest = _run(tmp_path, monkeypatch)
    assert result["works_selected"] == 3
    assert result["works_fetched"] == 3
    assert [c.rsplit("=", 1)[-1] for c in calls] == ["10", "30", "50"]
    assert (dest / "works" / "10.zip").exists()


def test_a_second_run_fetches_nothing(tmp_path, monkeypatch):
    """Resumability is the point: this job will be interrupted."""
    _run(tmp_path, monkeypatch)
    result, calls, _, _ = _run(tmp_path, monkeypatch)
    assert calls == []
    assert result["works_cached"] == 3 and result["works_fetched"] == 0


def test_it_waits_between_requests_but_not_before_the_first(tmp_path, monkeypatch):
    _, calls, sleeps, _ = _run(tmp_path, monkeypatch, delay=20)
    assert len(calls) == 3
    assert sleeps == [20, 20], "one wait between each pair, none leading"


def test_a_failing_work_is_counted_and_does_not_stop_the_run(tmp_path, monkeypatch):
    import fetch
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    seen = []
    def flaky(url, target):
        seen.append(url)
        if url.endswith("30"):
            raise OSError("503 from the server")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"PK\x03\x04ok")
    monkeypatch.setattr(fetch, "_download", flaky)
    dest = tmp_path / "raw"
    (dest / "_metadata").mkdir(parents=True)
    _meta(dest / "_metadata")
    result = fetch_corpus_corporum(
        {"id": "corpus-corporum", "delay_seconds": 0,
         "exclude_series": [r"Patrologia\s+Latina", "digilibLT"],
         "id_source": "x", "url_template": "https://h/?idno={idno}"}, dest)
    assert result["works_failed"] == 1
    assert result["works_fetched"] == 2          # 50 still fetched after 30 failed
    assert not (dest / "works" / "30.zip").exists()   # no truncated file left


def test_a_soft_404_is_not_stored_as_a_corpus_file(tmp_path, monkeypatch):
    """The endpoint answers a missing work with HTTP 200 and 21 bytes of
    `File not found. Sorry`. Stored, that is a file every reader downstream
    reports as empty while the fetch reports success."""
    import fetch
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    def soft_404(url, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        if url.endswith("30"):
            target.write_bytes(b"File not found. Sorry")
        else:
            target.write_bytes(b"PK\x03\x04zipdata")
    monkeypatch.setattr(fetch, "_download", soft_404)
    dest = tmp_path / "raw"
    (dest / "_metadata").mkdir(parents=True)
    _meta(dest / "_metadata")
    result = fetch_corpus_corporum(
        {"id": "corpus-corporum", "delay_seconds": 0,
         "exclude_series": [r"Patrologia\s+Latina", "digilibLT"],
         "id_source": "x", "url_template": "https://h/?idno={idno}"}, dest)
    assert result["works_unavailable"] == 1
    assert result["works_fetched"] == 2
    assert not (dest / "works" / "30.zip").exists()


def test_a_soft_404_is_not_retried(tmp_path, monkeypatch):
    """Retrying it three times with a backoff wastes three minutes to be told
    the same thing. Only real faults get the retry."""
    import fetch
    sleeps = []
    monkeypatch.setattr(fetch.time, "sleep", lambda s: sleeps.append(s))
    calls = []
    def soft_404(url, target):
        calls.append(url)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"File not found. Sorry")
    monkeypatch.setattr(fetch, "_download", soft_404)
    dest = tmp_path / "raw"
    (dest / "_metadata").mkdir(parents=True)
    _meta(dest / "_metadata")
    fetch_corpus_corporum(
        {"id": "corpus-corporum", "delay_seconds": 0,
         "exclude_series": [r"Patrologia\s+Latina", "digilibLT"],
         "id_source": "x", "url_template": "https://h/?idno={idno}"}, dest)
    assert len(calls) == 3, "one request per work, not one per retry"
    assert CC_BACKOFF_SECONDS not in sleeps

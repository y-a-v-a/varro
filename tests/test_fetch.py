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

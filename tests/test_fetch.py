"""fetch.py — manifest validation and track/tier selection (no network)."""

from fetch import selected, validate


def source(**overrides):
    base = {"id": "src-a", "name": "Source A", "url": "https://example.org",
            "method": "git", "license": "CC-BY-SA-4.0", "tier": "green",
            "include": True}
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

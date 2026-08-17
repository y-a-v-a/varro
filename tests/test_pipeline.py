"""pipeline.py — fingerprint chaining: what invalidates a stage."""

import json

import pipeline
from pipeline import Stage


def write_lock(root, pin="4620cf8a", fetched="2026-08-17T13:08:13+00:00",
               generated="2026-08-17T13:08:13+00:00", **extra):
    (root / "manifest").mkdir(exist_ok=True)
    (root / "manifest" / "fetch.lock.json").write_text(json.dumps({
        "generated": generated,
        "sources": {
            "perseus-canonical-latinLit": {
                "resolved_pin": pin, "license": "CC-BY-SA-4.0", "tier": "green",
                "method": "git", "url": "https://github.com/PerseusDL/x",
                "n_files": 1361, "bytes": 174_711_532,
                "fetched": fetched, "name": "Perseus", **extra,
            },
        },
    }), encoding="utf-8")


def make_stage(root, name="stage", config=("config.yaml",)):
    (root / "script.py").write_text("print('hi')\n", encoding="utf-8")
    for rel in config:
        (root / rel).write_text("k: 1\n", encoding="utf-8")
    return Stage(name, "script.py", list(config))


def test_fingerprint_stable_for_unchanged_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    stage = make_stage(tmp_path)
    assert stage.fingerprint("") == stage.fingerprint("")


def test_fingerprint_changes_when_config_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    stage = make_stage(tmp_path)
    before = stage.fingerprint("")
    (tmp_path / "config.yaml").write_text("k: 2\n", encoding="utf-8")
    assert stage.fingerprint("") != before


def test_fingerprint_changes_when_script_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    stage = make_stage(tmp_path)
    before = stage.fingerprint("")
    (tmp_path / "script.py").write_text("print('bye')\n", encoding="utf-8")
    assert stage.fingerprint("") != before


def test_fingerprint_chains_from_previous_stage(tmp_path, monkeypatch):
    # Upstream change propagates: same files, different previous fingerprint.
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    stage = make_stage(tmp_path)
    assert stage.fingerprint("aaaa") != stage.fingerprint("bbbb")


def test_missing_config_file_still_fingerprints(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    stage = make_stage(tmp_path, config=("missing.yaml",))
    (tmp_path / "missing.yaml").unlink()
    assert stage.fingerprint("")  # placeholder hash, no crash


# --- the lockfile projection ------------------------------------------------
#
# fetch.py rewrites the lockfile on every run. Hashing its bytes marked
# extract — and by chaining, the whole corpus build — stale after every fetch.

def test_refetch_with_unchanged_pins_does_not_invalidate(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    write_lock(tmp_path)
    before = pipeline.lockfile_identity()
    write_lock(tmp_path, fetched="2026-08-18T09:00:00+00:00",
               generated="2026-08-18T09:00:00+00:00")
    assert pipeline.lockfile_identity() == before


def test_moved_pin_does_invalidate(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    write_lock(tmp_path)
    before = pipeline.lockfile_identity()
    write_lock(tmp_path, pin="deadbeef")
    assert pipeline.lockfile_identity() != before


def test_relicensed_source_invalidates(tmp_path, monkeypatch):
    """Licence and tier decide whether a source may enter the corpus at all."""
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    write_lock(tmp_path)
    before = pipeline.lockfile_identity()
    write_lock(tmp_path, tier="amber")
    assert pipeline.lockfile_identity() != before


def test_changed_download_checksum_invalidates(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    write_lock(tmp_path, files=[{"name": "dump.bz2", "sha256": "aaa", "bytes": 1}])
    before = pipeline.lockfile_identity()
    write_lock(tmp_path, files=[{"name": "dump.bz2", "sha256": "bbb", "bytes": 1}])
    assert pipeline.lockfile_identity() != before


def test_missing_and_corrupt_lockfiles_are_distinct_and_not_silent(
        tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    assert pipeline.lockfile_identity() == "<missing>"
    (tmp_path / "manifest").mkdir(exist_ok=True)
    (tmp_path / "manifest" / "fetch.lock.json").write_text("{ not json",
                                                           encoding="utf-8")
    corrupt = pipeline.lockfile_identity()
    assert corrupt.startswith("<unreadable")
    write_lock(tmp_path)
    assert pipeline.lockfile_identity() not in (corrupt, "<missing>")


def test_derived_input_feeds_the_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    box = {"value": "one"}
    stage = Stage("extract", "script.py", [], derived=lambda: box["value"])
    (tmp_path / "script.py").write_text("print('hi')\n", encoding="utf-8")
    before = stage.fingerprint("")
    box["value"] = "two"
    assert stage.fingerprint("") != before


def test_derived_is_evaluated_at_fingerprint_time_not_construction(
        tmp_path, monkeypatch):
    """fetch rewrites the lockfile mid-run; a cached value would miss it."""
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    (tmp_path / "script.py").write_text("print('hi')\n", encoding="utf-8")
    calls = []
    stage = Stage("extract", "script.py", [],
                  derived=lambda: calls.append(1) or "x")
    stage.fingerprint("")
    stage.fingerprint("")
    assert len(calls) == 2

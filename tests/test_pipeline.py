"""pipeline.py — fingerprint chaining: what invalidates a stage."""

import pipeline
from pipeline import Stage


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

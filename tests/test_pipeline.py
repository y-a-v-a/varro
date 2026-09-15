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


# --- partitioned stages -----------------------------------------------------
#
# Extract has one part per source. The point is that adding the tenth source
# does not re-extract the nine already there, so the cost of growing the corpus
# stays constant instead of rising with it.

from pipeline import Part  # noqa: E402


def partitioned(root, names, values):
    (root / "script.py").write_text("print('hi')\n", encoding="utf-8")
    return Stage("extract", "script.py", [],
                 parts=[Part(n, ["--source", n], lambda n=n: values[n])
                        for n in names])


def test_one_source_moving_leaves_the_others_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    values = {"a": "pin-a", "b": "pin-b", "c": "pin-c"}
    stage = partitioned(tmp_path, ["a", "b", "c"], values)
    before = {p.name: stage.part_fingerprint("", p) for p in stage.parts}
    values["b"] = "pin-b2"
    after = {p.name: stage.part_fingerprint("", p) for p in stage.parts}
    assert after["b"] != before["b"]
    assert after["a"] == before["a"] and after["c"] == before["c"]


def test_adding_a_source_does_not_disturb_existing_ones(tmp_path, monkeypatch):
    """The property that makes the corpus cheap to grow."""
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    values = {"a": "pin-a", "b": "pin-b", "d": "pin-d"}
    before_stage = partitioned(tmp_path, ["a", "b"], values)
    before = {p.name: before_stage.part_fingerprint("", p) for p in before_stage.parts}
    after_stage = partitioned(tmp_path, ["a", "b", "d"], values)
    after = {p.name: after_stage.part_fingerprint("", p) for p in after_stage.parts}
    assert after["a"] == before["a"] and after["b"] == before["b"]
    assert "d" in after


def test_any_part_moving_invalidates_the_stage_and_so_everything_after(
        tmp_path, monkeypatch):
    """Downstream must still rebuild: one new source changes the corpus."""
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    values = {"a": "pin-a", "b": "pin-b"}
    stage = partitioned(tmp_path, ["a", "b"], values)
    before = stage.fingerprint("")
    values["b"] = "pin-b2"
    assert stage.fingerprint("") != before


def test_editing_the_extractor_invalidates_every_part(tmp_path, monkeypatch):
    """Shared code is shared: a change to the script re-runs all sources."""
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    values = {"a": "pin-a", "b": "pin-b"}
    stage = partitioned(tmp_path, ["a", "b"], values)
    before = {p.name: stage.part_fingerprint("", p) for p in stage.parts}
    (tmp_path / "script.py").write_text("print('changed')\n", encoding="utf-8")
    after = {p.name: stage.part_fingerprint("", p) for p in stage.parts}
    assert all(after[k] != before[k] for k in before)


def test_upstream_change_propagates_into_every_part(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    values = {"a": "pin-a"}
    stage = partitioned(tmp_path, ["a"], values)
    assert stage.part_fingerprint("upstream-1", stage.parts[0]) \
        != stage.part_fingerprint("upstream-2", stage.parts[0])


def test_lockfile_identity_can_be_scoped_to_one_source(tmp_path, monkeypatch):
    """Scoping is what lets one source be invalidated without the rest."""
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    (tmp_path / "manifest").mkdir(exist_ok=True)
    (tmp_path / "manifest" / "fetch.lock.json").write_text(json.dumps({
        "sources": {
            "alpha": {"resolved_pin": "aaa", "tier": "green"},
            "beta": {"resolved_pin": "bbb", "tier": "green"},
        }}), encoding="utf-8")
    alpha = pipeline.lockfile_identity("alpha")
    assert "alpha" in alpha and "beta" not in alpha
    assert pipeline.lockfile_identity() != alpha        # whole file differs
    assert pipeline.lockfile_identity("beta") != alpha
    # A source absent from the lockfile still fingerprints, distinctly.
    assert pipeline.lockfile_identity("gamma") not in (alpha, "")


def test_tokenizer_identity_reports_absence_rather_than_a_constant(
        tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    assert pipeline.tokenizer_identity() == "<missing>"
    d = tmp_path / "data" / "processed" / "tokenizers"
    d.mkdir(parents=True)
    (d / "latin-bpe-4000.json").write_text('{"merges": ["a b"]}', encoding="utf-8")
    first = pipeline.tokenizer_identity()
    assert first != "<missing>"
    (d / "latin-bpe-4000.json").write_text('{"merges": ["c d"]}', encoding="utf-8")
    assert pipeline.tokenizer_identity() != first


def test_the_floors_are_a_stage_and_come_after_the_tokenizer():
    """Both were outside the pipeline until 2026-08-19 and had to be remembered."""
    names = [s.name for s in pipeline.build_stages()]
    assert "tokenizer" in names and "baselines" in names
    assert names.index("build") < names.index("tokenizer") < names.index("baselines")


def test_a_declared_but_unfetched_source_is_not_extractable(tmp_path, monkeypatch):
    """A `manual` source sits in the lockfile from the moment it is declared,
    with present:false and no files. Once its format gained a reader the
    pipeline tried to extract it and failed the entire run — which is how
    Gutenberg broke the first eight-source build."""
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    (tmp_path / "manifest").mkdir(exist_ok=True)
    (tmp_path / "manifest" / "fetch.lock.json").write_text(json.dumps({
        "sources": {
            "fetched-one": {"n_files": 12, "tier": "green"},
            "manual-one": {"n_files": 0, "present": False, "tier": "green"},
            "broken-one": {"n_files": 5, "error": "clone failed", "tier": "green"},
        }}), encoding="utf-8")
    monkeypatch.setattr(pipeline, "extractable_sources",
                        pipeline.extractable_sources)
    import sys
    sys.modules.setdefault("extract", None)
    # Exercise the filter directly rather than importing extract's manifest.
    fetched = json.loads((tmp_path / "manifest" / "fetch.lock.json")
                         .read_text())["sources"]
    declared = ["fetched-one", "manual-one", "broken-one"]
    usable = [s for s in declared
              if s in fetched and "error" not in fetched[s]
              and (fetched[s].get("n_files") or 0) > 0]
    assert usable == ["fetched-one"]


# --- a fingerprint must describe the inputs the stage actually read ---------

def test_a_stage_edited_while_running_is_not_recorded_as_fresh(
        tmp_path, monkeypatch):
    """Recording after the run stamps output with code that never made it.

    Hit for real: extract.py was edited while the extract stage was working,
    the stage recorded the new source, and the pipeline then reported extract
    as up to date over an interim built by the old code. Fingerprinting at the
    moment the stage starts is what makes the record describe the run.
    """
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    stage = make_stage(tmp_path)

    at_start = stage.fingerprint("")
    # The stage is running; someone commits an edit to its script.
    (tmp_path / "script.py").write_text("print('edited mid-run')\n",
                                        encoding="utf-8")
    at_end = stage.fingerprint("")

    assert at_start != at_end, "the edit must be visible to the fingerprint"
    # The pipeline records at_start, so the next run sees the stage as stale
    # and rebuilds it with the edited code.
    assert stage.fingerprint("") != at_start


# --- running a stage's invocations ------------------------------------------
#
# extract has one part per source, each reading data/raw/<source>/ and writing
# data/interim/<source>.jsonl. No part touches another's input or output, which
# is the same independence that already lets them be invalidated separately —
# so they can overlap, and extraction is XML parsing, which is where the idle
# cores are. build's four invocations look similar and are not: each reads
# every interim file into memory, so running them together multiplies the peak.

import time

import pytest


SLEEPER = """\
import sys, time
name = sys.argv[1]
time.sleep(0.4)
open(name + ".done", "w").write("done")
print("worked on " + name)
if name == "boom":
    print("something went wrong", file=sys.stderr)
    sys.exit(3)
"""


def sleeper_stage(root, names, concurrent):
    (root / "script.py").write_text(SLEEPER, encoding="utf-8")
    return Stage("s", "script.py", [], argv=[[n] for n in names],
                 concurrent=concurrent)


def test_a_concurrent_stage_overlaps_its_invocations(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    stage = sleeper_stage(tmp_path, ["a", "b", "c", "d"], concurrent=True)
    started = time.perf_counter()
    stage.run(jobs=4)
    elapsed = time.perf_counter() - started
    assert elapsed < 4 * 0.4, f"ran in {elapsed:.2f}s — no overlap"
    assert all((tmp_path / f"{n}.done").exists() for n in "abcd")


def test_a_stage_that_does_not_declare_it_stays_sequential(tmp_path, monkeypatch):
    """build reads every interim file per invocation; four at once is four copies."""
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    stage = sleeper_stage(tmp_path, ["a", "b", "c"], concurrent=False)
    started = time.perf_counter()
    stage.run(jobs=4)
    assert time.perf_counter() - started >= 3 * 0.4


def test_jobs_one_runs_a_concurrent_stage_in_order(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    stage = sleeper_stage(tmp_path, ["a", "b", "c"], concurrent=True)
    started = time.perf_counter()
    stage.run(jobs=1)
    assert time.perf_counter() - started >= 3 * 0.4


def test_a_failing_invocation_fails_the_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    stage = sleeper_stage(tmp_path, ["a", "boom"], concurrent=True)
    with pytest.raises(RuntimeError) as excinfo:
        stage.run(jobs=4)
    assert "boom" in str(excinfo.value) and "3" in str(excinfo.value)


def test_the_other_sources_are_still_extracted_when_one_fails(tmp_path, monkeypatch):
    """No sense abandoning eight sources because the ninth has a bad file."""
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    stage = sleeper_stage(tmp_path, ["boom", "a", "b"], concurrent=True)
    with pytest.raises(RuntimeError):
        stage.run(jobs=4)
    assert (tmp_path / "a.done").exists() and (tmp_path / "b.done").exists()


def test_each_invocations_output_is_kept_with_its_own_command(tmp_path, monkeypatch,
                                                              capsys):
    """Interleaved lines from four extractors are unreadable; grouped ones are not."""
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    sleeper_stage(tmp_path, ["a", "b"], concurrent=True).run(jobs=2)
    out = capsys.readouterr().out
    for name in ("a", "b"):
        header = out.index(f"$ script.py {name}")
        body = out.index(f"worked on {name}")
        assert body > header, "output printed before the command that produced it"


def test_a_failure_is_marked_on_the_command_that_produced_it(tmp_path, monkeypatch,
                                                             capsys):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    with pytest.raises(RuntimeError):
        sleeper_stage(tmp_path, ["a", "boom"], concurrent=True).run(jobs=2)
    out = capsys.readouterr().out
    assert "$ script.py boom   <-- EXIT 3" in out
    assert "something went wrong" in out
    assert "EXIT" not in out.split("$ script.py a")[1].split("$ script.py boom")[0]


def test_a_single_invocation_is_never_worth_capturing(tmp_path, monkeypatch, capsys):
    """One extractor should stream as it goes, the way it always did."""
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    sleeper_stage(tmp_path, ["a"], concurrent=True).run(jobs=4)
    assert "at a time)" not in capsys.readouterr().out


def test_extract_is_the_only_stage_declared_concurrent(monkeypatch):
    """build must not be: each of its invocations holds the whole corpus."""
    monkeypatch.setattr(pipeline, "extractable_sources", lambda: ["one", "two"])
    declared = {s.name for s in pipeline.build_stages() if s.concurrent}
    assert declared == {"extract"}

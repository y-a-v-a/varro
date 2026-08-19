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
    tokenizers = processed / "tokenizers"
    processed.mkdir()
    reports.mkdir()
    tokenizers.mkdir()
    monkeypatch.setattr(train, "PROCESSED", processed)
    monkeypatch.setattr(train, "REPORTS", reports)
    monkeypatch.setattr(train, "TOKENIZERS", tokenizers)
    (tokenizers / "latin-bpe-4000.json").write_text('{"merges": []}',
                                                    encoding="utf-8")
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
    assert train.read_floor("corpus", 4000) == (None, "")


def test_matching_fingerprint_is_quotable(paths):
    processed, reports = paths
    write_summary(processed)
    write_baseline(reports)
    assert train.read_floor("corpus", 4000) == (195.2, "")


def test_grown_corpus_makes_the_floor_unquotable(paths):
    """The case this exists for: a source lands, the name does not change."""
    processed, reports = paths
    write_summary(processed, words=78_975_089, train_words=75_043_813,
                  records=10_192)
    write_baseline(reports)
    floor, reason = train.read_floor("corpus", 4000)
    assert floor == 195.2
    assert "words: 17256637 -> 78975089" in reason
    assert "records" in reason and "train_words" in reason


def test_regrouped_split_makes_the_floor_unquotable(paths):
    """Same text, different held-out authors — the perplexities do not compare."""
    processed, reports = paths
    write_summary(processed, group_by="work")
    _, reason = train.read_floor("corpus", 4000)
    assert reason == ""  # no baseline yet
    write_baseline(reports)
    assert "group_by: 'author' -> 'work'" in train.read_floor("corpus", 4000)[1]


def test_baseline_stamped_with_another_corpus_is_rejected(paths):
    """The bug as it was: baselines-classical.json stamped with the full corpus."""
    processed, reports = paths
    write_summary(processed, corpus="classical", records=203,
                  words=4_539_515, period_filter=["CLASSICAL"])
    write_baseline(reports, corpus="classical", perplexity=158.8,
                   corpus_name="corpus")
    floor, reason = train.read_floor("classical", 4000)
    assert floor == 158.8
    assert "corpus_name='corpus'" in reason


def test_retrained_tokenizer_makes_the_floor_unquotable(paths):
    """The case the corpus fingerprint cannot see.

    Between the 2026-08-17 and 2026-08-18 builds the classical corpus moved by
    three words while its floor moved 158.8 -> 149.8, because the 4k vocabulary
    had been retrained on a corpus 4.6x larger. Corpus identical, floor not.
    """
    processed, reports = paths
    write_summary(processed)
    write_baseline(reports)
    stale_tok = {"vocab": 4000, "sha256": "0123456789abcdef"}
    report = json.loads((reports / "baselines-corpus.json").read_text())
    report["tokenizer"] = stale_tok
    (reports / "baselines-corpus.json").write_text(json.dumps(report),
                                                   encoding="utf-8")
    _, reason = train.read_floor("corpus", 4000)
    assert "tokenizer:" in reason
    assert "0123456789abcdef" in reason


def test_matching_tokenizer_stamp_stays_quotable(paths):
    processed, reports = paths
    write_summary(processed)
    write_baseline(reports)
    report = json.loads((reports / "baselines-corpus.json").read_text())
    report["tokenizer"] = train.tokenizer_fingerprint(4000)
    (reports / "baselines-corpus.json").write_text(json.dumps(report),
                                                   encoding="utf-8")
    assert train.read_floor("corpus", 4000) == (195.2, "")


def test_absent_tokenizer_file_fingerprints_as_none_not_a_crash(paths):
    _, _ = paths
    assert train.tokenizer_fingerprint(64000) == {"vocab": 64000,
                                                  "sha256": None}


def test_unstamped_legacy_report_falls_back_to_fingerprint_drift(paths):
    """Reports written before corpus_name existed are still checked."""
    processed, reports = paths
    write_summary(processed, corpus="classical", records=203)
    write_baseline(reports, corpus="classical", corpus_name=None)
    _, reason = train.read_floor("classical", 4000)
    assert "records: 909 -> 203" in reason


# --- reporting blocks -------------------------------------------------------

def block_sizes(steps, eval_every):
    """Sizes of the blocks the training loop averages train loss over."""
    sizes, since = [], 0
    for step in range(steps):
        since += 1
        if train.is_report_step(step, steps, eval_every):
            sizes.append(since)
            since = 0
    return sizes


def test_blocks_tile_the_run_exactly():
    for steps, every in ((5000, 250), (74700, 500), (3500, 250), (7, 3), (1, 500)):
        assert sum(block_sizes(steps, every)) == steps


def test_final_block_is_short_when_steps_is_not_a_multiple():
    """The case that corrupted LATIN-XXV's reported train/val gap.

    74,700 steps at eval_every 500 leaves a 200-step tail. Dividing that tail
    by 500 scaled the final train loss to 0.4x — 3.12 was reported as 1.25,
    and the train/val gap as 7.25x where the truth was 1.12x.
    """
    sizes = block_sizes(74_700, 500)
    assert sizes[-1] == 200
    assert all(s == 500 for s in sizes[:-1])


def test_exact_multiple_has_no_short_tail():
    """Why every run before LATIN-XXV escaped this: 5000/250, 16000/250, 1750/250."""
    assert set(block_sizes(5000, 250)) == {250}
    assert set(block_sizes(1750, 250)) == {250}


# --- recipes ----------------------------------------------------------------
#
# A recipe states EPOCHS, never steps. A step count means something different
# every time the corpus changes — 74,700 steps was four passes over 153M
# tokens, and on a corpus half again as large it is 2.7 passes with nothing
# to say the model is now undertrained.

def test_steps_are_derived_from_epochs_and_the_corpus_in_hand():
    assert train.steps_for_epochs(4.0, 152_985_050, 16, 512) == 74_700
    assert train.steps_for_epochs(3.0, 9_562_431, 16, 512) == 3_502


def test_the_same_recipe_asks_for_more_steps_on_a_bigger_corpus():
    """The property that makes a recipe survive corpus growth."""
    small = train.steps_for_epochs(4.0, 100_000_000, 16, 512)
    large = train.steps_for_epochs(4.0, 200_000_000, 16, 512)
    assert large == 2 * small


def test_a_bigger_batch_needs_proportionally_fewer_steps():
    assert train.steps_for_epochs(4.0, 152_985_050, 32, 512) == \
        train.steps_for_epochs(4.0, 152_985_050, 16, 512) // 2


def test_epochs_never_round_down_to_nothing():
    assert train.steps_for_epochs(0.0001, 1000, 16, 512) == 1


def recipe(tmp_path, body):
    (tmp_path / "r.yaml").write_text(body, encoding="utf-8")
    return str(tmp_path / "r.yaml")


RECIPE = """
name: demo
defaults:
  dim: 512
  heads: 8
  batch: 16
phases:
  - name: phase-one
    corpus: corpus
    epochs: 4.0
    lr: 2.5e-4
  - name: phase-two
    corpus: classical
    init_from: phase-one
    epochs: 3.0
    batch: 8
"""


def test_defaults_merge_into_each_phase(tmp_path):
    settings, name = train.load_recipe(recipe(tmp_path, RECIPE), "phase-one")
    assert name == "phase-one"
    assert settings["dim"] == 512 and settings["heads"] == 8
    assert settings["corpus"] == "corpus" and settings["epochs"] == 4.0


def test_a_phase_overrides_the_defaults_it_disagrees_with(tmp_path):
    settings, _ = train.load_recipe(recipe(tmp_path, RECIPE), "phase-two")
    assert settings["batch"] == 8        # phase wins
    assert settings["dim"] == 512        # default carries through
    assert settings["init_from"] == "phase-one"


def test_first_phase_is_the_default(tmp_path):
    _, name = train.load_recipe(recipe(tmp_path, RECIPE), None)
    assert name == "phase-one"


def test_unknown_phase_names_the_alternatives(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        train.load_recipe(recipe(tmp_path, RECIPE), "phase-three")
    assert "phase-one" in str(excinfo.value) and "phase-two" in str(excinfo.value)


def test_a_recipe_with_no_phases_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        train.load_recipe(recipe(tmp_path, "name: empty\nphases: []\n"), None)


def test_missing_recipe_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        train.load_recipe(str(tmp_path / "absent.yaml"), None)


def test_shipped_recipes_reproduce_the_runs_that_were_actually_made():
    """configs/ is the record of what produced the checkpoints in reports/."""
    settings, name = train.load_recipe("latin-xxv", "latin-xxv-latinitas")
    assert name == "latin-xxv-latinitas"
    assert settings["dim"] == 512 and settings["layers"] == 8
    assert settings["batch"] == 16 and settings["epochs"] == 4.0
    assert train.steps_for_epochs(settings["epochs"], 152_985_050,
                                  settings["batch"], settings["context"]) == 74_700

    second, _ = train.load_recipe("latin-xxv", "latin-xxv")
    assert second["init_from"] == "latin-xxv-latinitas"
    assert second["corpus"] == "classical"

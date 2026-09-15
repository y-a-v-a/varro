"""tokenizer.py — embedding-cost arithmetic, a miniature BPE round-trip, and
the report the sweep never used to write.
"""

import json
import pathlib

import pytest

import tokenizer as T
from tokenizer import (SPECIAL, embedding_cost, morphology, train_one,
                       verify_roundtrip)

SENTENCES = [
    "Gallia est omnis diuisa in partes tres.",
    "Quarum unam incolunt Belgae, aliam Aquitani.",
    "Arma uirumque cano Troiae qui primus ab oris.",
    "Amo amas amat amamus amatis amant amare.",
    "Ceterum censeo Carthaginem esse delendam.",
] * 4


def test_embedding_cost_arithmetic():
    cost = embedding_cost(vocab_size=8000, d_model=256, n_layer=6)
    assert cost["body"] == 12 * 6 * 256 ** 2
    assert cost["embed"] == 8000 * 256
    assert cost["total"] == cost["body"] + cost["embed"]
    assert cost["embed_share"] == pytest.approx(cost["embed"] / cost["total"])


@pytest.fixture(scope="module")
def tiny_tokenizer():
    records = [{"text": s} for s in SENTENCES]
    return train_one(records, vocab_size=300), records


def test_special_tokens_reserved(tiny_tokenizer):
    tok, _ = tiny_tokenizer
    for token in SPECIAL:
        assert tok.token_to_id(token) is not None, token


def test_roundtrip_is_exact(tiny_tokenizer):
    tok, records = tiny_tokenizer
    assert verify_roundtrip(tok, records) == 0


def test_morphology_report_shape(tiny_tokenizer):
    tok, _ = tiny_tokenizer
    report = morphology(tok)
    assert report["paradigm_total"] == 12
    assert 0 <= report["paradigm_whole"] <= report["paradigm_total"]
    assert 0.0 < report["stem_spread"] <= 1.0
    assert report["example"]  # how `amabamus` divides


# ---------------------------------------------------------------------------
# write_report
#
# The sweep printed a table and kept nothing. tokens-per-word is what the whole
# parameter ladder is computed against — 20 tokens per parameter is a statement
# about tokens — and the project ran on an ASSUMED 1.8 for months because no
# artifact held the measured figure.
# ---------------------------------------------------------------------------

class Args:
    d_model = 256
    n_layer = 6


def rows():
    return [{
        "vocab": 4000,
        "vocab_used": 4000,
        "sha256": "24d1515e65807859",
        "roundtrip_failures": 0,
        "full": {"tokens_per_word": 1.97, "chars_per_token": 3.4,
                 "words": 90_000_000, "tokens": 177_300_000},
        "classical": {"tokens_per_word": 2.221, "chars_per_token": 3.12,
                      "words": 4_805_330, "tokens": 10_671_431},
        "morphology": {"paradigm_whole": 3, "paradigm_total": 12,
                       "paradigm_openers": 4, "stem_spread": 0.42,
                       "example": "am · ab · amus"},
        "embedding_cost": embedding_cost(4000, 256, 6),
    }]


@pytest.fixture
def reports(tmp_path, monkeypatch):
    d = tmp_path / "reports"
    monkeypatch.setattr(T, "REPORTS", d)
    monkeypatch.setattr(T, "corpus_fingerprint", lambda corpus: {"records": 10782})
    return d


def test_the_sweep_is_written_where_every_other_stage_writes(reports):
    path = T.write_report(rows(), Args(), corpus_shape={"words": 90_000_000})
    assert path == reports / "tokenizer.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["schema_version"] == T.SCHEMA_VERSION
    assert report["corpus"] == {"records": 10782}
    assert report["embedding_cost_model"] == {"d_model": 256, "n_layer": 6}


def test_every_vocabulary_is_named_by_the_file_it_wrote(reports):
    """Without the digest the report cannot be tied to a tokenizer on disk,
    which is the only thing that makes it checkable later."""
    report = json.loads(
        T.write_report(rows(), Args(), corpus_shape={}).read_text(encoding="utf-8"))
    entry, = report["vocabularies"]
    assert entry["sha256"] == "24d1515e65807859"
    assert entry["vocab_used"] == 4000


def test_the_measured_fertility_survives_into_the_file(reports):
    """The number the parameter ladder is computed against."""
    report = json.loads(
        T.write_report(rows(), Args(), corpus_shape={}).read_text(encoding="utf-8"))
    entry, = report["vocabularies"]
    assert entry["classical"]["tokens_per_word"] == 2.221
    assert entry["classical"]["tokens"] == 10_671_431


def test_report_diffs_as_measurements_and_not_as_key_order(reports):
    """Written with sort_keys, so re-running the sweep produces a diff a person
    can read: what moved, not where the serialiser happened to put it."""
    first = T.write_report(rows(), Args(), corpus_shape={}).read_text(encoding="utf-8")
    second = T.write_report(rows(), Args(), corpus_shape={}).read_text(encoding="utf-8")

    # Identical but for the timestamp, which is the only thing that may move
    # between two runs measuring the same corpus.
    drop = lambda text: [ln for ln in text.splitlines() if '"generated"' not in ln]
    assert drop(first) == drop(second)

    top = [k for k in json.loads(first)]
    assert top == sorted(top), f"keys are not sorted: {top}"


# ---------------------------------------------------------------------------
# held_out_ids
#
# build.py runs once per corpus and each run picks its own validation split, so
# a work held out of one corpus can sit in another's training text. The
# tokenizer trains on corpus.train and every model uses the result, so without
# this it fits merges on text those models are scored against.
# ---------------------------------------------------------------------------

@pytest.fixture
def processed(tmp_path, monkeypatch):
    d = tmp_path / "processed"
    d.mkdir()
    monkeypatch.setattr(T, "PROCESSED", d)
    return d


def jsonl(path, ids):
    path.write_text("".join(json.dumps({"id": i, "text": i, "n_words": 1}) + "\n"
                            for i in ids), encoding="utf-8")


def test_every_corpus_contributes_not_just_this_one(processed):
    """The whole point: corpus-b's held-out works are the bulk of the overlap."""
    jsonl(processed / "corpus.validation.jsonl", ["a"])
    jsonl(processed / "classical.validation.jsonl", ["b"])
    jsonl(processed / "corpus-b.validation.jsonl", ["c"])
    jsonl(processed / "classical-b.validation.jsonl", ["d"])
    assert T.held_out_ids() == {"a", "b", "c", "d"}


def test_training_splits_are_not_mistaken_for_validation_ones(processed):
    """Only *.validation.jsonl. Globbing wider would withhold the corpus."""
    jsonl(processed / "corpus.validation.jsonl", ["held"])
    jsonl(processed / "corpus.train.jsonl", ["trained"])
    jsonl(processed / "dialogue.jsonl", ["mined"])
    assert T.held_out_ids() == {"held"}


def test_no_built_corpus_withholds_nothing_rather_than_failing(processed):
    """A first-ever run has no validation file to read yet."""
    assert T.held_out_ids() == set()


def test_the_fitting_split_is_what_the_sweep_trains_on():
    """Static, because running the sweep means training six tokenizers.

    The measurements deliberately stay on the whole train split — they describe
    the corpus train.py will tokenise — so only the train_one call moves.
    """
    text = (pathlib.Path(__file__).resolve().parent.parent
            / "src" / "tokenizer.py").read_text(encoding="utf-8")
    assert "tok = train_one(fitting, vocab_size)" in text
    assert "full = fertility(tok, train)" in text

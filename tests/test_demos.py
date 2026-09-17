"""Tests for the demos.

Scope: everything here is the pure part -- gap parsing, word regrouping,
orthography folding, trial construction, token alignment, the corruptions the
minimal-pair benchmark applies, the shuffles the word-order ladder applies.
The model itself is not loaded, both
because MLX is not available everywhere the tests run and because what could go
wrong in a demo is the plumbing around the model, not the matrix multiply.

The one thing that IS about the model is covered indirectly: the batching in
score_readings was verified against the unbatched path by hand and produced
identical numbers to two decimal places on the Caesar sentence.
"""

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "demos"))
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Gap parsing
# ---------------------------------------------------------------------------

def _split(text):
    from lacuna import split_gaps
    return split_gaps(text)


@pytest.mark.parametrize("marker", ["[...]", "[---]", "[----]", "[?]", "[??]"])
def test_every_documented_marker_is_recognised(marker):
    parts, n = _split(f"alpha {marker} omega")
    assert n == 1
    assert [p.strip() for p in parts] == ["alpha", "omega"]


def test_multiple_gaps_give_one_more_segment_than_gaps():
    parts, n = _split("a [...] b [---] c")
    assert n == 2
    assert len(parts) == 3


def test_text_without_a_gap_reports_none():
    _, n = _split("Gallia est omnis divisa in partes tres")
    assert n == 0


def test_a_bare_bracket_is_not_a_gap():
    # Editors bracket supplied letters constantly: `pat[res]` is a restored
    # word, not a hole to fill, and must not be treated as one.
    _, n = _split("pat[res] conscripti")
    assert n == 0


# ---------------------------------------------------------------------------
# Orthography -- the fold that made the demos usable on real text
# ---------------------------------------------------------------------------

def test_v_is_folded_to_u_because_the_corpus_has_no_v():
    from _common import to_corpus_orthography
    folded, changes = to_corpus_orthography("divisa virtute")
    assert folded == "diuisa uirtute"
    assert changes == {"v -> u": 2}


def test_j_is_folded_to_i():
    from _common import to_corpus_orthography
    folded, changes = to_corpus_orthography("Jupiter jam")
    assert folded == "Iupiter iam"
    assert changes == {"j -> i": 2}


def test_already_folded_text_reports_no_change():
    from _common import to_corpus_orthography
    folded, changes = to_corpus_orthography("diuisa uirtute")
    assert folded == "diuisa uirtute"
    assert changes == {}


def test_folding_would_destroy_a_gap_marker():
    """The reason lacuna.py splits before it folds.

    src/normalize.py strips square brackets as editorial marks. If the demo
    folded the whole line first there would be no gap left to fill, and the
    text would silently become a continuous -- and wrong -- sentence.
    """
    from _common import to_corpus_orthography
    folded, _ = to_corpus_orthography("alpha [...] omega")
    assert "[" not in folded and "]" not in folded

    # Split first, and both halves survive intact.
    parts, n = _split("alpha [...] omega")
    assert n == 1
    assert [to_corpus_orthography(p)[0] for p in parts] == ["alpha", "omega"]


# ---------------------------------------------------------------------------
# Regrouping token surprisal onto words
# ---------------------------------------------------------------------------

class FakeTok:
    """Decodes the way a byte-level BPE does: the space rides on the token
    that follows it, which is what makes a leading space a word boundary."""

    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, ids):
        return "".join(self.pieces[i] for i in ids)


def test_word_surprisal_is_the_sum_of_its_tokens():
    from surprisal import to_words
    pieces = ["<bos>", " di", "u", "isa", " in"]
    tok = FakeTok(pieces)
    ids = [0, 1, 2, 3, 4]
    # 1 nat each, so each token is 1/ln2 bits.
    logprobs = [None, -1.0, -1.0, -1.0, -1.0]
    words = to_words(tok, ids, logprobs, prefix_len=1)
    assert [w.text for w in words] == ["diuisa", "in"]
    assert words[0].tokens == 3
    assert words[0].bits == pytest.approx(3 / math.log(2))
    assert words[1].tokens == 1


def test_tokens_before_the_prefix_are_not_scored():
    from surprisal import to_words
    tok = FakeTok(["<bos>", "<CLASSICAL>", " arma"])
    words = to_words(tok, [0, 1, 2], [None, -9.0, -1.0], prefix_len=2)
    assert [w.text for w in words] == ["arma"]


def test_unscored_tokens_are_skipped_not_counted_as_certain():
    """A None logprob means "not measured", which is not the same as "cost
    nothing" -- counting it as zero would quietly make a word look ordinary."""
    from surprisal import to_words
    tok = FakeTok(["<bos>", " alpha", " beta"])
    words = to_words(tok, [0, 1, 2], [None, None, -1.0], prefix_len=1)
    assert [w.text for w in words] == ["beta"]


def test_bands_are_ordered_and_total():
    from surprisal import BANDS, band
    thresholds = [t for t, _, _ in BANDS]
    assert thresholds == sorted(thresholds)
    for bits in (0.0, 11.9, 12.0, 23.0, 31.9, 1e12):
        assert band(bits) is not None


# ---------------------------------------------------------------------------
# Benchmark trial construction
# ---------------------------------------------------------------------------

def _write_split(tmp_path, records):
    import json
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    path = processed / "toy.validation.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    return tmp_path


def test_trials_hide_a_real_word_and_keep_both_contexts(tmp_path, monkeypatch):
    import lacuna
    words = " ".join(f"uerbum{chr(97+i%26)}{chr(97+i//26)}" for i in range(120))
    root = _write_split(tmp_path, [{"id": "toy/1", "text": words}])
    monkeypatch.setattr(lacuna, "ROOT", root)

    trials = lacuna.draw_trials("toy", 3, seed=1, span=10, min_len=4)
    assert trials
    for trial in trials:
        assert trial["answer"] in words.split()
        assert len(trial["left"].split()) == 10
        assert len(trial["right"].split()) == 10
        # The hidden word must not still be sitting in the context it is
        # supposed to be recovered from.
        assert trial["answer"] not in trial["left"].split()
        assert trial["answer"] not in trial["right"].split()


def test_short_documents_are_skipped_rather_than_yielding_a_ragged_trial(
        tmp_path, monkeypatch):
    import lacuna
    root = _write_split(tmp_path, [{"id": "toy/1", "text": "breuis est"}])
    monkeypatch.setattr(lacuna, "ROOT", root)
    assert lacuna.draw_trials("toy", 3, seed=1, span=10, min_len=4) == []


def test_a_missing_split_is_a_clear_exit_not_a_traceback(tmp_path, monkeypatch):
    import lacuna
    monkeypatch.setattr(lacuna, "ROOT", tmp_path)
    with pytest.raises(SystemExit):
        lacuna.draw_trials("nope", 1, seed=1, span=5, min_len=4)


def test_special_token_leakage_is_filtered_from_candidates():
    from lacuna import SPECIAL
    assert SPECIAL.search("<|endoftext|>")
    assert SPECIAL.search("<CLASSICAL>")
    assert not SPECIAL.search("partes")


# ---------------------------------------------------------------------------
# Candidate cleaning
# ---------------------------------------------------------------------------

def test_punctuation_variants_collapse_to_one_candidate():
    """The reason edge punctuation is trimmed at all.

    A 24-sample pool for `Gallia est omnis diuisa in [...] tres` came back
    holding duas, `duas,`, `duas:` and `duas.` as four separate candidates --
    four slots spent on one word, with `partes` crowded out of the pool.
    """
    from lacuna import clean_candidate
    variants = ["duas", "duas,", "duas:", "duas.", "(duas)", "duas!"]
    assert {clean_candidate(v, 1) for v in variants} == {"duas"}


def test_a_candidate_shorter_than_the_gap_is_rejected():
    from lacuna import clean_candidate
    assert clean_candidate("partes", 2) is None
    assert clean_candidate("partes tres", 2) == "partes tres"


def test_a_candidate_that_is_only_punctuation_is_rejected():
    from lacuna import clean_candidate
    assert clean_candidate(",", 1) is None
    assert clean_candidate("...", 1) is None


def test_special_tokens_never_become_candidates():
    from lacuna import clean_candidate
    assert clean_candidate("<|endoftext|>", 1) is None
    assert clean_candidate("<CLASSICAL>", 1) is None


def test_interior_punctuation_survives_a_multi_word_fill():
    # Only the edges are trimmed: a comma between two words of a two-word
    # supplement is part of the reading, not decoration around it.
    from lacuna import clean_candidate
    assert clean_candidate("partes, tres", 2) == "partes, tres"


# ---------------------------------------------------------------------------
# Era estimation
# ---------------------------------------------------------------------------

def test_the_best_era_is_the_highest_mean_logprob_not_the_first_key():
    from aetas import verdict
    scores = {
        "CLASSICAL": {"mean_logprob": -3.20, "perplexity": 24.5},
        "LATE": {"mean_logprob": -3.10, "perplexity": 22.2},
        "MEDIEVAL": {"mean_logprob": -3.40, "perplexity": 30.0},
        "NEOLATIN": {"mean_logprob": -3.50, "perplexity": 33.1},
    }
    best, second, margin = verdict(scores)
    assert best == "LATE"
    assert second == "CLASSICAL"
    assert margin == pytest.approx(0.10 / math.log(2))


def test_margin_bands_are_ordered_and_cover_everything():
    from aetas import BAND_NAMES, band_of
    assert band_of(0.0) == BAND_NAMES[0]
    assert band_of(1e6) == BAND_NAMES[-1]
    seen = [band_of(x) for x in (0.0, 0.003, 0.008, 0.02)]
    assert seen == list(BAND_NAMES)


def test_a_missing_calibration_file_is_absence_not_a_crash():
    import aetas
    assert aetas.load_calibration("no-such-model-anywhere") is None


# ---------------------------------------------------------------------------
# Minimal pairs
# ---------------------------------------------------------------------------

def test_align_finds_the_shared_ends():
    from paria import align
    assert align([1, 2, 3, 4], [1, 2, 9, 4]) == (2, 1)
    assert align([1, 2, 3], [1, 2, 3]) == (3, 0)
    assert align([1, 2, 3], [7, 8]) == (0, 0)


def test_align_never_lets_the_suffix_overlap_the_prefix():
    # `a b c` against `a c`: the shared `a` is the prefix and the shared `c`
    # the suffix, and the middle is `b` against nothing -- not a suffix of 2
    # that would swallow the prefix.
    from paria import align
    assert align([1, 2, 3], [1, 3]) == (1, 1)
    assert align([1, 1], [1]) == (1, 0)


def test_every_suite_pair_is_a_minimal_pair():
    """Each pair must differ, and differ in one place, or it tests nothing."""
    from paria import SUITE
    for pair in SUITE:
        assert pair.a != pair.b, pair
        assert pair.expected in ("a", None), pair
        a, b = pair.a.split(), pair.b.split()
        # One word changed, or the same words in another order.
        if sorted(a) == sorted(b):
            assert pair.expected is None, f"reordering has no right answer: {pair}"
        else:
            assert sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b)) <= 2, pair


def test_suite_is_already_in_corpus_orthography():
    """A `v` in a suite sentence would be scored as damage, not as grammar."""
    from paria import SUITE
    for pair in SUITE:
        for text in (pair.a, pair.b):
            assert "v" not in text.lower() and "j" not in text.lower(), text


def test_corrupt_ending_only_returns_attested_words():
    import random
    from paria import corrupt_ending
    attested = {"puella", "puellam", "puellae", "amat", "amant"}
    rng = random.Random(0)
    for _ in range(20):
        out = corrupt_ending("puella", attested, rng)
        assert out in {"puellam", "puellae"}
    assert corrupt_ending("amat", attested, rng) == "amant"
    # No attested alternative: no corruption, rather than a made-up word.
    assert corrupt_ending("miles", attested, rng) is None
    # Not a word at all.
    assert corrupt_ending("A.D.", attested, rng) is None


def test_perturb_keeps_the_words_and_moves_only_what_it_says():
    import random
    from paria import perturb
    left, word, right = ["gallia", "est"], "omnis", ["diuisa", "in", "partes"]
    v = perturb(left, word, right, {"omnis", "omnes"}, random.Random(1))
    assert v["original"] == "gallia est omnis diuisa in partes"
    assert v["swap"] == "gallia est diuisa omnis in partes"
    assert v["ending"] == "gallia est omnes diuisa in partes"
    assert sorted(v["shuffle"].split()) == sorted(v["original"].split())
    assert v["shuffle"] != v["original"]


def test_perturb_declines_a_swap_that_would_change_nothing():
    import random
    from paria import perturb
    v = perturb(["a"], "b", ["b", "c"], set(), random.Random(1))
    assert v["swap"] is None
    v = perturb(["a"], "b", [], set(), random.Random(1))
    assert v["swap"] is None


# ---------------------------------------------------------------------------
# Word order
# ---------------------------------------------------------------------------

def test_shuffles_keep_exactly_the_original_words():
    import random
    from ordo import (shuffle_sentences, shuffle_within_sentences,
                      shuffle_words, reverse_words)
    text = "Gallia est omnis. Diuisa in partes tres! Quarum unam incolunt?"
    for fn in (shuffle_sentences, shuffle_within_sentences, shuffle_words,
               reverse_words):
        out = fn(text, random.Random(3))
        assert sorted(out.split()) == sorted(text.split()), fn.__name__


def test_shuffling_within_sentences_keeps_each_sentence_together():
    import random
    from ordo import sentences, shuffle_within_sentences
    text = "alpha beta gamma. delta epsilon zeta."
    out = shuffle_within_sentences(text, random.Random(5))
    got = [sorted(s.split()) for s in sentences(out)]
    assert got == [sorted("alpha beta gamma.".split()),
                   sorted("delta epsilon zeta.".split())]


def test_sentence_splitting_follows_terminal_punctuation():
    from ordo import sentences
    assert sentences("Quo usque tandem? quam diu etiam! nihil.") == \
        ["Quo usque tandem?", "quam diu etiam!", "nihil."]
    assert sentences("one sentence with no end") == ["one sentence with no end"]


# ---------------------------------------------------------------------------
# Surprisal pieces
# ---------------------------------------------------------------------------

def test_a_word_remembers_the_pieces_it_was_scored_as():
    from surprisal import to_words
    tok = FakeTok(["<bos>", " rn", "erc", "atores", " in"])
    words = to_words(tok, [0, 1, 2, 3, 4], [None, -2.0, -1.0, -0.5, -1.0],
                     prefix_len=1)
    assert words[0].text == "rnercatores"
    assert [p for p, _ in words[0].pieces] == [" rn", "erc", "atores"]
    assert sum(b for _, b in words[0].pieces) == pytest.approx(words[0].bits)

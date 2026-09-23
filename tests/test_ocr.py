"""Tests for the OCR-repair benchmark.

Scope: the arithmetic that decides what a report says -- alignment, the
projection onto true words, the five outcomes, the noise channel and the
channel fitted back from it, the bootstrap. No model is loaded, for the same
reason tests/test_demos.py loads none: what can go wrong here is the
bookkeeping around the model, and a benchmark with wrong bookkeeping reports a
confident number about nothing.
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ocr"))

import channel  # noqa: E402
import metrics  # noqa: E402
import repairers  # noqa: E402

LEX = {"de", "thesauro", "in", "urbem", "uenit", "cum", "eum", "amico", "et",
       "mercatores", "roma"}


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def _reference(a: str, b: str) -> int:
    """The textbook full-matrix Levenshtein, to hold the banded one against."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        row = [i]
        for j, cb in enumerate(b, 1):
            row.append(min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = row
    return prev[-1]


def test_banded_distance_is_exact_not_a_bound():
    rng = random.Random(3)
    for _ in range(200):
        a = "".join(rng.choice("abc ") for _ in range(rng.randrange(0, 40)))
        b = "".join(rng.choice("abc ") for _ in range(rng.randrange(0, 40)))
        assert metrics.align(a, b)[0] == _reference(a, b)


def test_an_alignment_replays_into_the_hypothesis():
    truth, hyp = "mercatores saepe commeant", "rnercatores sae pe comeant"
    distance, ops = metrics.align(truth, hyp)
    rebuilt = "".join(hyp[j] for kind, _, j in ops if kind in "=si")
    assert rebuilt == hyp
    assert sum(kind != "=" for kind, _, _ in ops) == distance


def test_distance_far_beyond_the_first_band_is_still_found():
    # The band starts at 8 and doubles; this needs several doublings.
    assert metrics.align("a" * 100, "b" * 60)[0] == 100


def test_a_two_for_one_confusion_is_one_region_not_two():
    assert metrics.edit_regions("mercatores", "rnercatores") == [("m", "rn")]
    assert metrics.edit_regions("cultu", "cu1tu") == [("l", "1")]


# ---------------------------------------------------------------------------
# Projection and outcomes
# ---------------------------------------------------------------------------

def _outcomes(truth, noisy, repaired):
    return [metrics.outcome(n, r) for n, r in zip(
        metrics.project(truth, noisy), metrics.project(truth, repaired))]


def test_the_five_outcomes():
    truth = "in urbem uenit cum amico"
    noisy = "in urbern uenit eum arnico"
    #        kept  fixed   BROKEN  missed  miscorrected
    fixed = "in urbem ueniit eum amica"
    assert _outcomes(truth, noisy, fixed) == [
        "kept", "fixed", "broken", "missed", "miscorrected"]


def test_a_merge_damages_both_words_and_a_repair_fixes_both():
    truth, noisy = "in urbem uenit", "inurbem uenit"
    standing = metrics.project(truth, noisy)
    assert [s.intact for s in standing] == [False, False, True]
    # Each of the two true words reports the whole run-together token.
    assert [s.text for s in standing[:2]] == ["inurbem", "inurbem"]
    assert _outcomes(truth, noisy, truth) == ["fixed", "fixed", "kept"]


def test_a_split_word_is_not_standing_however_well_its_letters_align():
    truth, noisy = "de thesauro", "de the sauro"
    standing = metrics.project(truth, noisy)
    assert [s.intact for s in standing] == [True, False]
    assert standing[1].text == "the sauro"


def test_a_word_glued_to_a_speck_is_damaged():
    standing = metrics.project("in urbem uenit", "in .urbem uenit")
    assert [s.intact for s in standing] == [True, False, True]


def test_a_vanished_word_is_damaged_and_its_neighbours_are_not():
    standing = metrics.project("in urbem uenit", "in uenit")
    assert [s.intact for s in standing] == [True, False, True]


def test_identity_never_breaks_anything():
    truth = "in urbem uenit cum amico"
    noisy = "in urbern uenit eum arnico"
    c = metrics.tally(truth, noisy, noisy, LEX)
    assert c["broken"] == c["fixed"] == c["miscorrected"] == 0
    assert c["missed"] == 3 and c["kept"] == 2
    assert c["dist_before"] == c["dist_after"]


def test_perfect_repair_scores_perfectly():
    truth = "in urbem uenit cum amico"
    c = metrics.tally(truth, "in urbern uenit eum arnico", truth, LEX)
    s = metrics.summarise(c)
    assert s["recall"] == 1 and s["precision"] == 1
    assert s["harm_per_1000"] == 0 and s["cer_reduction"] == 1


def test_real_word_damage_is_told_from_nonword_damage():
    # cum -> eum is attested Latin; urbem -> urbern is not.
    c = metrics.tally("in urbem uenit cum amico", "in urbern uenit eum amico",
                      "in urbern uenit eum amico", LEX)
    assert c["real:missed"] == 1 and c["nonword:missed"] == 1
    assert c["silent_before"] == 1


def test_laundering_raises_the_silent_count_while_cer_falls():
    """The case the whole outcome table exists for."""
    truth = "roma mercatores uenit"
    noisy = "rorna mercatores uenit"          # a visible non-word
    repaired = "roma mercatores eum"          # fixed one, rewrote a good word
    c = metrics.tally(truth, noisy, repaired, LEX)
    assert c["fixed"] == 1 and c["broken"] == 1
    assert c["silent_before"] == 0 and c["silent_after"] == 1


def test_punctuation_is_compared_but_not_looked_up():
    assert metrics.core("(Tres,") == "tres"
    # `tres,` repaired to `tres` has lost a character and is not standing.
    assert _outcomes("partes tres, quarum", "partes tres, quarum",
                     "partes tres quarum")[1] == "broken"


def test_detection_labels_follow_the_noisy_words():
    labels = metrics.word_labels("in urbem uenit cum amico",
                                 "in urbern uenit eum amico", LEX)
    assert labels == [None, "nonword", None, "real", None]


def test_a_split_gives_a_label_to_every_fragment():
    labels = metrics.word_labels("de thesauro", "de the sauro", LEX)
    assert labels[0] is None and len(labels) == 3
    assert labels[1] is not None and labels[2] is not None


# ---------------------------------------------------------------------------
# Average precision and the bootstrap
# ---------------------------------------------------------------------------

def test_average_precision_of_a_perfect_and_a_useless_detector():
    positive = [True, False, False, True, False, False, False, False]
    perfect = [9, 1, 1, 9, 1, 1, 1, 1]
    assert metrics.average_precision(perfect, positive) == 1.0
    # One score for everything must earn the floor, not a lucky ordering.
    flat = metrics.average_precision([1] * 8, positive)
    assert flat == pytest.approx((1 / 7 + 2 / 8) / 2)
    assert metrics.average_precision(perfect, [False] * 8) is None


def test_bootstrap_brackets_the_point_estimate():
    rng = random.Random(1)
    tallies = [Counter(fixed=rng.randrange(0, 4), missed=rng.randrange(0, 4) + 1)
               for _ in range(80)]
    total = sum(tallies, Counter())
    lo, hi = metrics.bootstrap(tallies, {"recall": metrics.recall}, reps=400)["recall"]
    assert lo < metrics.recall(total) < hi


def test_paired_bootstrap_sees_a_consistent_edge_the_separate_intervals_hide():
    rng = random.Random(2)
    a, b = [], []
    for _ in range(60):
        hard = rng.randrange(1, 12)            # passages differ a lot ...
        b.append(Counter(fixed=hard, missed=12 - hard + 1))
        a.append(Counter(fixed=hard + 1, missed=12 - hard))   # ... a is always +1
    stats = {"recall": metrics.recall}
    lo_a, hi_a = metrics.bootstrap(a, stats, reps=400)["recall"]
    lo_b, hi_b = metrics.bootstrap(b, stats, reps=400)["recall"]
    assert lo_a < hi_b                        # the separate intervals overlap
    lo, _ = metrics.bootstrap(a, stats, reps=400, against=b)["recall"]
    assert lo > 0                             # the paired one does not reach zero


# ---------------------------------------------------------------------------
# The channel
# ---------------------------------------------------------------------------

TEXT = ("gallia est omnis diuisa in partes tres quarum unam incolunt belgae "
        "aliam aquitani tertiam qui ipsorum lingua celtae nostra galli "
        "appellantur hi omnes lingua institutis legibus inter se differunt ") * 6


def test_zero_rate_changes_nothing():
    assert channel.corrupt(TEXT, 0.0, random.Random(1)) == (TEXT, [])


def test_corruption_is_a_function_of_the_seed():
    assert channel.corrupt(TEXT, 0.05, random.Random(9)) == \
        channel.corrupt(TEXT, 0.05, random.Random(9))


def test_the_edit_log_replays_into_the_noisy_text():
    noisy, edits = channel.corrupt(TEXT, 0.08, random.Random(4))
    out, at = [], 0
    for e in edits:
        assert TEXT[e.start:e.end] == e.truth
        out.append(TEXT[at:e.start] + e.noisy)
        at = e.end
    assert "".join(out) + TEXT[at:] == noisy


def test_capitals_and_punctuation_are_never_damaged():
    noisy, _ = channel.corrupt("ROMA, ROMA; ROMA. " * 40, 0.5, random.Random(1))
    assert noisy.replace(" ", "") == ("ROMA,ROMA;ROMA." * 40)


def test_a_lost_space_is_charged_to_both_neighbours():
    edits = [channel.Edit(2, 3, " ", "", "merge")]
    assert channel.classes_by_word("in urbem uenit", edits) == [["merge"], ["merge"], []]


def test_a_substitution_is_charged_to_one_word_only():
    edits = [channel.Edit(7, 8, "m", "rn", "shape")]
    assert channel.classes_by_word("in urbem uenit", edits) == [[], ["shape"], []]


def test_fit_recovers_the_common_confusions_from_pairs_alone():
    pairs = []
    for seed in range(30):
        noisy, _ = channel.corrupt(TEXT, 0.05, random.Random(seed))
        pairs.append((TEXT, noisy))
    fitted = channel.fit(pairs)
    for op in (("m", "rn"), ("e", "c"), ("u", "n"), (" ", ""), ("", " ")):
        assert op in fitted.logp, op
    # A rate, so strictly between never and always.
    assert all(lp < 0 for lp in fitted.logp.values())
    # A rate per appearance of the true side, so a heavily weighted confusion
    # outranks a lightly weighted one whatever the letters' own frequencies.
    assert fitted.logp[("m", "rn")] > fitted.logp[("m", "ni")]


def test_the_table_fingerprint_moves_when_the_table_does(monkeypatch):
    before = channel.table_fingerprint()
    monkeypatch.setattr(channel, "MERGE_FACTOR", channel.MERGE_FACTOR + 0.1)
    assert channel.table_fingerprint() != before


def test_no_repairer_is_shown_the_benchmarks_noise():
    """A repairer learns the channel from dev pairs or not at all."""
    source = (ROOT / "ocr" / "repairers.py").read_text(encoding="utf-8")
    for name in ("CONFUSIONS", "corrupt", "_WEIGHT", "classes_by_word"):
        assert name not in source, name


# ---------------------------------------------------------------------------
# The reference repairers, without a model
# ---------------------------------------------------------------------------

def _unigram():
    u = repairers.Unigram()
    # Roughly the real proportions. They matter: a change has to outweigh the
    # channel's prior against change, about 4 nats, so `cum` beats `eum` only
    # by being some sixty times commoner -- which in Latin it is.
    counts = {"in": 2_000_000, "urbem": 20_000, "uenit": 40_000, "cum": 900_000,
              "eum": 9_000, "amico": 8_000, "de": 800_000, "thesauro": 3_000,
              "mercatores": 2_000, "a": 600_000, "et": 4_000_000}
    pairs = []
    for seed in range(40):
        noisy, _ = channel.corrupt(TEXT, 0.06, random.Random(seed))
        pairs.append((TEXT, noisy))
    u.fit(pairs, counts)
    return u


def test_unigram_mends_a_nonword_and_keeps_edge_punctuation_and_case():
    u = _unigram()
    assert u.repair("(Urbern, uenit") == "(Urbem, uenit"


def test_unigram_rejoins_a_split_and_separates_a_merge():
    u = _unigram()
    assert u.repair("de the sauro uenit") == "de thesauro uenit"
    assert u.repair("inurbem uenit") == "in urbem uenit"


def test_unigram_shows_the_frequency_trap_it_is_there_to_show():
    # `eum` is a word, `cum` is a commoner one, and nothing else is consulted.
    assert _unigram().repair("uenit eum amico") == "uenit cum amico"


def test_identity_is_identity():
    assert repairers.make("identity").repair("rnercatores") == "rnercatores"


def test_edges_and_recasing():
    assert repairers.split_edges('("Roma,') == ('("', "Roma", ",")
    assert repairers.recase("roma", "Rorna") == "Roma"
    assert repairers.recase("roma", "RORNA") == "ROMA"


def test_per_passage_counts_survive_the_report():
    sys.path.insert(0, str(ROOT / "src"))
    import bench
    tallies = [Counter({"fixed": 3, "real:fixed": 1, "class:shape:fixed": 2}),
               Counter({"broken": 1, "kept": 58})]
    back = bench.unpack(bench.pack(tallies))
    assert back[0]["fixed"] == 3 and back[0]["real:fixed"] == 1
    assert back[1]["broken"] == 1 and back[1]["fixed"] == 0
    # Error classes are a Tier 1 luxury and stay out of the packed form.
    assert "class:shape:fixed" not in back[0]


# ---------------------------------------------------------------------------
# The decision rule
# ---------------------------------------------------------------------------

FLAGS = {"known": False, "capital": False, "short": False, "halves_known": False}


def test_the_raw_rule_takes_whatever_scores_highest():
    pool = [("urbern", 0.0, 1), ("urbem", -4.0, 1)]
    assert repairers.decide(repairers.Rule(), pool, [-20.0, -10.0], FLAGS) == 1
    # ... unless the channel's prior against change outweighs the gain.
    assert repairers.decide(repairers.Rule(), pool, [-20.0, -17.0], FLAGS) == 0


def test_a_margin_is_evidence_demanded_not_a_penalty_on_the_winner():
    pool = [("eum", 0.0, 1), ("cum", -4.0, 1)]
    lang = [-12.0, -6.0]                       # a gain of 2 nats after the channel
    assert repairers.decide(repairers.Rule(), pool, lang, FLAGS) == 1
    assert repairers.decide(repairers.Rule(margin=3), pool, lang, FLAGS) == 0
    # `known` applies only when the word on the page is itself a word.
    rule = repairers.Rule(known=3)
    assert repairers.decide(rule, pool, lang, FLAGS) == 1
    assert repairers.decide(rule, pool, lang, {**FLAGS, "known": True}) == 0


def test_weight_sets_the_exchange_rate_between_scorer_and_channel():
    pool = [("eum", 0.0, 1), ("cum", -4.0, 1)]
    assert repairers.decide(repairers.Rule(weight=0.5), pool, [-12.0, -6.0], FLAGS) == 0
    assert repairers.decide(repairers.Rule(weight=2.0), pool, [-12.0, -9.0], FLAGS) == 1


def test_a_join_of_two_known_words_needs_more_than_a_join_of_fragments():
    pool = [("rei", 0.0, 1), ("reipublicae", -5.0, 2)]
    rule = repairers.Rule(join=8)
    assert repairers.decide(rule, pool, [-20.0, -10.0], FLAGS) == 1
    assert repairers.decide(rule, pool, [-20.0, -10.0],
                            {**FLAGS, "halves_known": True}) == 0


def test_numerals_are_never_touched():
    u = _unigram()
    assert u.repair("anno 5. et XL III uenit") == "anno 5. et XL III uenit"
    assert repairers.NUMERAL.match("XLUIII")     # forty-eight, v folded to u
    assert not repairers.NUMERAL.match("uenit")


def test_a_recorded_run_replays_to_the_same_text():
    sys.path.insert(0, str(ROOT / "src"))
    import tune
    u = _unigram()
    noisy = "in urbern uenit eum arnico de the sauro"
    u.trace = []
    repaired = u.repair(noisy)
    passage = {"noisy": noisy, "entries": json.loads(json.dumps(u.trace))}
    assert tune.replay(passage, u.rule) == repaired
    # And under a rule that believes nothing, the text comes back untouched.
    assert tune.replay(passage, repairers.Rule(margin=1e9)) == noisy


# ---------------------------------------------------------------------------
# Tier 2: finding a passage in a page, and reading a transcription
# ---------------------------------------------------------------------------

def test_locate_finds_a_damaged_passage_inside_a_longer_page():
    page = "LIBER PRIMUS 114 in urbern uenit curn amico et dixit ei 1) cf. Verg."
    start, end, distance = metrics.locate("in urbem uenit cum amico", page)
    assert page[start:end] == "in urbern uenit curn amico"
    assert distance == 4


def test_locate_costs_nothing_for_what_surrounds_the_match():
    assert metrics.locate("uenit", "x" * 500 + "uenit" + "y" * 500)[2] == 0


def test_wikitext_is_read_as_the_words_on_the_page():
    import tier2
    text = ('<noinclude><pagequality level="4" user="x" />{{Rh|114|COLUMELLAE|}}'
            "</noinclude>{{Pt|pes|stirpes}} ''insitae'' intra [[biennium|biennium]]"
            "<ref>nota</ref> {{Center|DE VINEIS}}<noinclude>{{ref|a}}</noinclude>")
    assert tier2.clean_wikitext(text).split() == [
        "pes", "insitae", "intra", "biennium", "DE", "VINEIS"]


def test_error_bands_put_a_perfect_read_in_the_control():
    import tier2
    assert tier2.band(0.0) == "0"
    assert tier2.band(0.01) == "<2%"
    assert tier2.band(0.2) == ">6%"


def test_letter_templates_are_letters_and_typography_is_not_damage():
    import tier2
    assert tier2.clean_wikitext("depa{{s}}cere c{{ae}}lum at{{que}}").split() == [
        "depascere", "caelum", "atque"]
    assert tier2.clean_wikitext("uenit{{Pg|153}} {{versus|5}}Roma").split() == ["uenit", "Roma"]
    assert tier2.tidy("conueniat , et con¬ suetudines ( ut aiunt )") == \
        "conueniat, et consuetudines (ut aiunt)"

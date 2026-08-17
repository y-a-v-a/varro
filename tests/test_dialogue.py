"""dialogue.py — turn-pair mining from drama and catechism."""

from dialogue import mine, question_answer, speaker_turns, usable

DRAMA = "\n".join([
    "Euclio",
    "Nam cur me miseram uerberas modo?",
    "Staphyla",
    "Ut misera sis atque ut te dignam mala malam aetatem exigas.",
    "Euclio",
    "Cur ante aedis astas nunc ita?",
    "Staphyla",
    "Quia iussisti me hic stare atque opperiri te diu.",
    "Euclio",
    "Abi intro et occlude ianuam iam nunc.",
    "Staphyla",
    "Occludam et seruabo omnia diligenter intus.",
])


def test_usable_bounds():
    assert usable("tres uerba latina")
    assert not usable("duo tantum")                      # under MIN_WORDS
    assert not usable(" ".join(["uerbum"] * 61))         # over MAX_WORDS
    assert not usable("a b c 12345 67890 !!!! ????")     # too few letters


def test_speaker_turns_pairs_alternating_cast():
    turns = list(speaker_turns({"text": DRAMA}))
    assert len(turns) == 5
    sa, ua, sb, ub = turns[0]
    assert (sa, sb) == ("Euclio", "Staphyla")
    assert ua.startswith("Nam cur") and ub.startswith("Ut misera")


def test_speaker_turns_needs_recurring_cast():
    # Each name appears only once: no cast, no turns.
    text = "\n".join(["Aulus", "Salue amice care.", "Brutus", "Salue et tu quoque."])
    assert list(speaker_turns({"text": text})) == []


def test_question_answer_pairs():
    rec = {"text": "Quid est ueritas ipsa? Veritas est splendor ueri. "
                   "Alia res hic agitur nunc."}
    assert list(question_answer(rec)) == \
        [("Quid est ueritas ipsa?", "Veritas est splendor ueri.")]


def test_question_answer_skips_question_answered_by_question():
    rec = {"text": "Quid est hoc rerum? Quid est illud tandem? Nescio plane amice."}
    pairs = list(question_answer(rec))
    assert ("Quid est hoc rerum?", "Quid est illud tandem?") not in pairs


def test_mine_dedupes_and_labels():
    rec = {"author": "Plautus", "work": "Aulularia", "genre": "DRAMA",
           "period": "CLASSICAL", "source": "s", "license": "l", "text": DRAMA}
    pairs, stats = mine([rec, dict(rec)])  # same record twice: no duplicate pairs
    assert stats["turn"] == 5
    assert all(p["kind"] in ("turn", "qa") for p in pairs)
    assert len(pairs) == len({(p["prompt"][:60], p["response"][:60]) for p in pairs})

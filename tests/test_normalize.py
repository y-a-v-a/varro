"""normalize.py — orthography folding, editorial marks, hyphens, foreign script."""

from collections import Counter

from normalize import normalize, strip_marks, drop_foreign_paragraphs


def test_folds_v_to_u_and_j_to_i_by_default():
    assert normalize("Veni, vidi, Iulius jussit") == "Ueni, uidi, Iulius iussit"


def test_uv_v_folds_j_only():
    assert normalize("vivat jam", uv="v") == "vivat iam"


def test_uv_keep_leaves_both_alone():
    assert normalize("vivat jam", uv="keep") == "vivat jam"


def test_lowercase_option():
    assert normalize("Gallia Est", lowercase=True) == "gallia est"


def test_ligatures_and_long_s_expand():
    assert normalize("æther ſumma Œdipus") == "aether summa Oedipus"


def test_diacritics_stripped_by_default_kept_on_request():
    assert normalize("iússu mālé") == "iussu male"
    assert normalize("iússu", diacritics=True) == "iússu"


def test_editorial_marks_removed_text_kept():
    assert normalize("[uir] †crux† <bona> *uita*") == "uir crux bona uita"


def test_curly_quotes_normalised():
    assert normalize("‘uir’ dixit “ita”") == "'uir' dixit \"ita\""


def test_hyphen_across_line_break_rejoined():
    stats = Counter()
    assert normalize("de-\nturbarunt", stats=stats) == "deturbarunt"
    assert stats["hyphen_rejoined"] == 1


def test_hyphen_before_space_split_by_default_joined_on_request():
    assert normalize("de- turbarunt") == "de turbarunt"
    assert normalize("de- turbarunt", hyphen_space="join") == "deturbarunt"
    assert normalize("de- turbarunt", hyphen_space="keep") == "de- turbarunt"


def test_soft_hyphen_removed():
    assert normalize("uer­bum") == "uerbum"


def test_latin_extended_homoglyphs_mapped():
    assert normalize("Łukasz ømnia") == "Lukasz omnia"


def test_cyrillic_paragraph_dropped_whole():
    stats = Counter()
    text = "Latina uerba hic sunt.\n\nпривет мир как дела"
    assert normalize(text, stats=stats) == "Latina uerba hic sunt."
    assert stats["dropped:foreign_script_paragraphs"] == 1
    assert stats["dropped_foreign_words"] == 4


def test_cyrillic_homoglyph_repaired_inline():
    # A Cyrillic lookalike inside a mostly-Latin paragraph is OCR damage: the
    # homoglyph map repairs it before the foreign-script strip can delete it.
    out = normalize("multa uerba latina sunt in hac sententia longa mаle scripta")
    assert "male" in out


def test_unmappable_foreign_char_stripped_inline():
    # A foreign character with no Latin equivalent is removed; the paragraph
    # around it is kept.
    out = normalize("multa uerba latina sunt in hac sententia longa אmen scripta")
    assert "sententia" in out and "men" in out
    assert "א" not in out


def test_whitespace_collapsed():
    assert normalize("a  b\t c\n\n\n\nd") == "a b c\n\nd"


def test_strip_marks_pure():
    assert strip_marks("ṃạḷẹ mācrōn") == "male macron"


def test_drop_foreign_paragraphs_keeps_latin():
    stats = Counter()
    text = "uerba latina\n\nκαὶ ταῦτα uerba"  # Greek is NOT in FOREIGN_SCRIPT
    assert drop_foreign_paragraphs(text, stats) == text
    assert not stats


# --- parallel normalisation ---------------------------------------------------
#
# normalize() is pure: what one record becomes cannot depend on another. That is
# what makes it safe across processes, and what these tests hold it to — the
# text, and the statistics, which a worker keeps separately and the parent adds
# up. A count is a count whatever order it arrived in; anything order-dependent
# in here would be a bug in its own right.

from collections import Counter

import normalize as N

OPTIONS = {"uv": "u", "diacritics": False, "lowercase": False,
           "hyphen_space": "split"}

CORPUS = [
    "Arma virumque cano, Trojae qui primus ab oris",
    "sicut Solomon de ipsis ait: tollamus inu-\ntilis est nobis",
    "iússu hánc cum mácronibus ēt brevibus ǎ",
    "Præclara æstas œconomia ſanctus",
    "text with a Cyrillic а and о inside words",
    "",
    "una linea",
]


def test_a_pool_normalises_exactly_what_one_process_does():
    assert [t for t, _ in N.normalize_all(CORPUS, OPTIONS, jobs=4)] == \
           [t for t, _ in N.normalize_all(CORPUS, OPTIONS, jobs=1)]


def test_the_statistics_add_up_to_the_same_totals():
    """Counters are merged from workers; a dropped or double-counted one shows here."""
    def totals(jobs):
        return sum((s for _, s in N.normalize_all(CORPUS, OPTIONS, jobs)), Counter())
    assert totals(4) == totals(1)
    assert totals(1)["hyphen_rejoined"] == 1


def test_results_come_back_in_corpus_order():
    """rows[i] is assigned results[i]; a reordering would scramble the corpus."""
    out = [t for t, _ in N.normalize_all(CORPUS, OPTIONS, jobs=4)]
    assert out == [N.normalize_one(t, **OPTIONS)[0] for t in CORPUS]


def test_the_window_boundary_is_not_visible():
    whole = [t for t, _ in N.normalize_all(CORPUS, OPTIONS, 2, window=10**9)]
    split = [t for t, _ in N.normalize_all(CORPUS, OPTIONS, 2, window=10)]
    assert split == whole


def test_normalize_one_keeps_its_own_statistics():
    """A worker cannot share the parent's Counter, so it must not try to."""
    text, stats = N.normalize_one(CORPUS[1], **OPTIONS)
    assert stats["hyphen_rejoined"] == 1
    assert "inutilis" in text


def test_options_reach_the_workers():
    """A pool that silently used defaults would fold u/v when asked not to."""
    lowered = [t for t, _ in N.normalize_all(
        ["Arma Virumque"], {**OPTIONS, "lowercase": True}, jobs=2)]
    assert lowered == ["arma uirumque"]
    kept = [t for t, _ in N.normalize_all(
        ["Arma Virumque"], {**OPTIONS, "uv": "keep"}, jobs=2)]
    assert kept == ["Arma Virumque"]


def test_an_empty_corpus_does_not_start_a_pool():
    assert list(N.normalize_all([], OPTIONS, jobs=4)) == []


# ---------------------------------------------------------------------------
# fold_prompt
#
# The corpus contains no `v` and no `j` at all — 0 of each in 659,309,740
# characters — and until this existed nothing folded the other end. A person
# typing `venit` was asking the model a question in an orthography it had
# never read.
# ---------------------------------------------------------------------------

def test_prompt_is_folded_the_way_the_corpus_was():
    folded, changed = N.fold_prompt("Venit civitas iam Jesus")
    assert folded == "Uenit ciuitas iam Iesus"
    assert changed is True


def test_an_already_classical_prompt_is_left_alone_and_says_so():
    folded, changed = N.fold_prompt("Gallia est omnis diuisa")
    assert folded == "Gallia est omnis diuisa"
    assert changed is False


def test_the_fold_is_idempotent():
    once, _ = N.fold_prompt("Vita brevis, ars longa")
    twice, changed = N.fold_prompt(once)
    assert twice == once
    assert changed is False


def test_prompt_fold_matches_what_the_pipeline_actually_runs():
    """The coupling this rests on, pinned.

    pipeline.py invokes normalize.py with no arguments, so the corpus carries
    normalize()'s DEFAULTS. Those defaults are not recorded per record — the
    stamp is just VERSION — so fold_prompt cannot read back what was used and
    calls normalize() bare instead. If a default moves, the two ends of the
    model part company silently; this is what notices.
    """
    import inspect

    defaults = {name: param.default
                for name, param in inspect.signature(N.normalize).parameters.items()
                if param.default is not inspect.Parameter.empty}
    assert defaults["uv"] == "u"
    assert defaults["diacritics"] is False
    assert defaults["lowercase"] is False
    assert defaults["hyphen_space"] == "split"

    sample = "Venit, civitas iam — Jésus!"
    assert N.fold_prompt(sample)[0] == N.normalize(sample)


def test_diacritics_and_editorial_marks_go_from_a_prompt_too():
    """A prompt copied out of an edition carries the editor's apparatus."""
    folded, changed = N.fold_prompt("iússu [uir] uítae")
    assert "\u00fa" not in folded and "\u00ed" not in folded
    assert changed is True

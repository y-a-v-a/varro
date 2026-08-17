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

"""extract.py — TEI walking, language filters, CoNLL-U parsing, wikitext."""

import textwrap

import pytest

from extract import (CORPUS_TIERS, FORMAT_EXTRACTORS, Stats, _greek_ratio,
                     assemble, drop_english_paragraphs, extract_tei_body,
                     extractable_sources, extractor_for, load_manifest,
                     looks_english, parse_conllu, pl_volume, strip_inline_greek,
                     strip_templates, strip_wikitext, tidy, title_author)


def tei(body: str) -> str:
    return f"<TEI><text><body>{body}</body></text></TEI>"


def extract(tmp_path, body: str, greek: str = "drop"):
    path = tmp_path / "work.xml"
    path.write_text(tei(body), encoding="utf-8")
    stats = Stats()
    text, _ = extract_tei_body(path, stats, greek)
    return text, stats


# --- TEI editorial policy ---------------------------------------------------

def test_note_and_del_dropped_add_kept(tmp_path):
    text, stats = extract(
        tmp_path, "<p>uerba <note>editor talk</note>certa "
                  "<del>sponte</del><add>recte</add> manent</p>")
    assert text == "uerba certa recte manent"
    assert stats["dropped:note"] == 1 and stats["dropped:del"] == 1


def test_choice_prefers_reg_and_corr(tmp_path):
    text, stats = extract(
        tmp_path, "<p><choice><orig>uolgus</orig><reg>uulgus</reg></choice> "
                  "<choice><sic>errat</sic><corr>erat</corr></choice></p>")
    assert text == "uulgus erat"
    assert stats["choice:reg_over_orig"] == 1
    assert stats["choice:corr_over_sic"] == 1


def test_page_break_inside_word_does_not_split_it(tmp_path):
    text, _ = extract(tmp_path, "<p>Athe<pb/>nis fuit</p>")
    assert text == "Athenis fuit"


def test_english_language_attribute_dropped(tmp_path):
    text, stats = extract(
        tmp_path, '<p>manet <foreign xml:lang="eng">the and of this</foreign> hoc</p>')
    assert text == "manet hoc"
    assert stats["dropped:lang:eng"] == 1


def test_greek_policy_drop_and_tag(tmp_path):
    body = '<p>quid sit <foreign xml:lang="grc">περὶ τῆς οὐσίας</foreign> id est</p>'
    dropped, _ = extract(tmp_path, body, greek="drop")
    assert dropped == "quid sit id est"
    tagged, _ = extract(tmp_path, body, greek="tag")
    assert tagged == "quid sit <GRAECUM> περὶ τῆς οὐσίας </GRAECUM> id est"


def test_paragraph_elements_become_paragraph_breaks(tmp_path):
    text, _ = extract(tmp_path, "<p>primus locus</p><p>alter locus</p>")
    assert text == "primus locus\n\nalter locus"


# --- English detection ------------------------------------------------------

def test_looks_english_on_english_not_latin():
    assert looks_english("the introduction and the notes of this edition were")
    assert not looks_english("arma uirumque cano Troiae qui primus ab oris")
    assert not looks_english("the and")  # under 5 tokens


def test_drop_english_paragraphs():
    stats = Stats()
    text = "arma uirumque cano\n\nthe play and the plot of this comedy were praised"
    assert drop_english_paragraphs(text, stats) == "arma uirumque cano"
    assert stats["dropped:english_paragraphs"] == 1


# --- inline Greek -----------------------------------------------------------

def test_strip_inline_greek_drop_tag_keep():
    stats = Stats()
    text = "quid sit περὶ τῆς οὐσίας id est"
    assert strip_inline_greek(text, stats, "keep") == text
    assert tidy(strip_inline_greek(text, stats, "drop")) == "quid sit id est"
    assert strip_inline_greek(text, stats, "tag") == \
        "quid sit <GRAECUM> περὶ τῆς οὐσίας </GRAECUM> id est"


def test_greek_ratio():
    assert _greek_ratio("uerba latina") == 0.0
    assert _greek_ratio("καὶ") == 1.0
    assert _greek_ratio("12345") == 0.0


# --- CoNLL-U ----------------------------------------------------------------

CONLLU = textwrap.dedent("""\
    # docId = phi1234
    # docAuthor = Cicero
    # sent_id = 1
    # text = Prima sententia est.
    1\tPrima\tprimus\tADJ\t_\t_\t_\t_\t_\tref=Paragraphus_1
    2\tsententia\tsententia\tNOUN\t_\t_\t_\t_\t_\t_

    # sent_id = 2
    # text = Secunda sequitur.
    1\tSecunda\tsecundus\tADJ\t_\t_\t_\t_\t_\tref=Paragraphus_1

    # sent_id = 3
    # text = Nouum caput incipit.
    1\tNouum\tnouus\tADJ\t_\t_\t_\t_\t_\tref=Paragraphus_2
    """)


def test_parse_conllu_meta_and_sentences():
    meta, sentences = parse_conllu(iter(CONLLU.splitlines()))
    assert meta == {"docId": "phi1234", "docAuthor": "Cicero"}
    assert sentences == [(1, "Prima sententia est."),
                         (1, "Secunda sequitur."),
                         (2, "Nouum caput incipit.")]


def test_assemble_groups_by_paragraph():
    stats = Stats()
    text = assemble([(1, "Prima."), (1, "Secunda."), (2, "Tertia.")], stats, "drop")
    assert text == "Prima. Secunda.\n\nTertia."


def test_assemble_drops_greek_sentences():
    stats = Stats()
    text = assemble([(1, "Latina manent."), (1, "καὶ ταῦτα πάντα")], stats, "drop")
    assert text == "Latina manent."
    assert stats["greek_passages"] == 1


# --- wikitext ---------------------------------------------------------------

def test_strip_templates_handles_nesting():
    assert strip_templates("a {{outer {{inner}} tail}} b") == "a  b"


def test_strip_wikitext():
    raw = ("{{header|x}}'''Uerba''' [[pagina|prima]] et [[secunda]]\n"
           "== Caput ==\n131.0843C| finis <ref>n</ref>")
    assert " ".join(strip_wikitext(raw).split()) == \
        "Uerba prima et secunda Caput finis"


def test_interwiki_links_are_dropped_not_unwrapped():
    """The leak that put English into a corpus defined by containing none.

    Traced from every Thai character in the built corpus back to *Adeste
    Fideles* and *Jingle Bells*, whose interwiki footers survived as bare text
    because the link rules strip brackets and keep what is inside them.
    """
    raw = ("Adeste fideles\n"
           "[[en:O come all ye faithful]]\n"
           "[[th:ขอเชิญท่านผู้วางใจ]]\n"
           "[[zh-cn:齊來宗主信徒]]\n"
           "[[:fr:Peuple fidèle]]\n")
    assert " ".join(strip_wikitext(raw).split()) == "Adeste fideles"


def test_interwiki_pattern_spares_latin_wikisource_namespaces():
    """Namespaces are capitalised Latin words, language codes are not."""
    raw = "[[Liber:Vergilius]] [[Pagina:Codex.djvu/12]] [[Auctor:Cicero]]"
    kept = " ".join(strip_wikitext(raw).split())
    assert "Liber:Vergilius" in kept
    assert "Auctor:Cicero" in kept


def test_interwiki_pattern_spares_ordinary_wikilinks():
    raw = "uide [[De bello Gallico]] et [[Aeneis|carmen]]"
    assert " ".join(strip_wikitext(raw).split()) == \
        "uide De bello Gallico et carmen"


def test_piped_interwiki_link_is_dropped_whole():
    raw = "Finis [[en:Utopia|the English one]] finis"
    assert " ".join(strip_wikitext(raw).split()) == "Finis finis"


def test_title_author_splits_parenthetical_name():
    assert title_author("Enarrationes in Psalmos (Remigius Antissiodorensis)") == \
        ("Enarrationes in Psalmos", "Remigius Antissiodorensis")


def test_title_author_rejects_editions_and_subpages():
    assert title_author("Opera omnia (ed. Migne)") == ("Opera omnia (ed. Migne)", None)
    assert title_author("Homiliae (Haymo)/4") == ("Homiliae", "Haymo")
    assert title_author("Carmina") == ("Carmina", None)


@pytest.mark.parametrize("title,raw,expected", [
    ("Patrologia Latina/32", "", 32),
    ("Sermones (PL 038)", "", 38),
    ("Sermones", "uide [[Patrologia Latina/45|XLV]]", 45),
    ("Patrologia Latina/300", "", None),   # PL stops at 217
    ("Carmina", "", None),
])
def test_pl_volume(title, raw, expected):
    assert pl_volume(title, raw) == expected


# --- format dispatch --------------------------------------------------------

def test_perseus_and_csel_share_one_extractor():
    """Two sources, one reader — the pattern format dispatch generalises."""
    assert extractor_for("perseus-canonical-latinLit") is \
        extractor_for("opengreekandlatin-csel")


def test_formats_without_a_reader_yet_return_none_rather_than_raising():
    """EpiDoc and plaintext are declared and not yet implemented."""
    assert extractor_for("edh") is None
    assert extractor_for("papyri-idp") is None
    assert extractor_for("latin-library") is None


def test_eval_tier_is_never_extractable(): 
    """The hazard format dispatch introduced, and the guard that closes it.

    The UD treebanks are CoNLL-U, exactly like DigilibLT, so dispatching on
    format alone would extract held-out evaluation data into the training
    corpus — the one mistake that would silently invalidate every number the
    project reports.
    """
    sources = extractable_sources()
    assert sources, "expected some extractable sources"
    assert not any(s.startswith("ud-latin") for s in sources)
    for entry in load_manifest()["sources"]:
        if entry["id"] in sources:
            assert entry["tier"] in CORPUS_TIERS
            assert entry.get("include", True)


def test_excluded_sources_are_not_extractable():
    assert "tesserae" not in extractable_sources()   # include: false


def test_every_registered_format_is_declared_by_some_source():
    """A registry entry no manifest source names is dead code."""
    declared = {e.get("format") for e in load_manifest()["sources"]}
    assert set(FORMAT_EXTRACTORS) <= declared


def test_editorial_dateline_dropped_but_salutation_kept(tmp_path):
    """The bug reading LATIN-XXV's output found, and its boundary.

    Perseus opens each of Cicero's letters with an editor's dateline beside
    the genuine salutation. The first is nineteenth-century scholarship in
    Latin abbreviations; the second is Cicero. 930 datelines were in the
    corpus and the model had learned to write them.
    """
    text, stats = extract(
        tmp_path,
        '<div><label rend="opener">'
        '<seg rend="dateline">Scr. Romae <date when="-0065">m. Quint.</date>.</seg>'
        '<seg rend="salute">CICERO ATTICO salutem</seg>'
        '</label><p>si uales bene est</p></div>')
    assert "Scr." not in text and "Quint" not in text
    assert "CICERO ATTICO salutem" in text
    assert "si uales bene est" in text
    assert stats["dropped:rend:dateline"] == 1


def test_typographic_rend_values_are_untouched(tmp_path):
    """Only editorial rend values are dropped; italics are not editorial."""
    text, _ = extract(
        tmp_path,
        '<p>uerba <seg rend="italics">notata</seg> et '
        '<seg rend="bold">clara</seg> manent</p>')
    assert "notata" in text and "clara" in text

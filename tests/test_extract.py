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


def test_epidoc_sources_share_one_reader():
    """Two sources, one job — the second time format dispatch paid off."""
    assert extractor_for("edh") is extractor_for("papyri-idp")
    assert extractor_for("edh") is not None


def test_plaintext_sources_share_one_reader():
    assert extractor_for("latin-library") is extractor_for("gutenberg-latin")
    assert extractor_for("latin-library") is not None


def test_formats_without_a_reader_yet_return_none_rather_than_raising():
    """What is left are the two sources that cannot be fetched anyway:
    Corpus Corporum's TEI and IntraText's HTML."""
    assert extractor_for("corpus-corporum") is None
    assert extractor_for("intratext") is None


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


# --- EpiDoc -----------------------------------------------------------------
#
# Epigraphic convention differs from literary convention in ways that corrupt
# text if the literary rules are applied. These pin each difference.

from extract import EPIDOC, LATIN_LANGS, epidoc_edition, epidoc_province  # noqa: E402


def epi(tmp_path, body, lang="la"):
    """Walk an EpiDoc edition div under the epigraphic profile."""
    xml = (f'<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body>'
           f'<div type="edition" xml:lang="{lang}">{body}</div>'
           f'</body></text></TEI>')
    path = tmp_path / "i.xml"
    path.write_text(xml, encoding="utf-8")
    import xml.etree.ElementTree as ET
    from extract import _walk, tidy
    root = ET.parse(path).getroot()
    div = epidoc_edition(root)
    stats = Stats()
    if div is None:
        return None, stats
    return " ".join(tidy(_walk(div, stats, "drop", EPIDOC)).split()), stats


def test_line_break_separates_words_on_a_stone(tmp_path):
    """The literary rule elides <lb/> because it falls mid-word in prose.

    On a stone it is a word boundary — measured in EDH, 87.6% of them are.
    Eliding them would weld CAIUS and UIBIUS into one token.
    """
    text, _ = epi(tmp_path, '<ab><lb n="1"/>Caius<lb n="2"/>Uibius</ab>')
    assert text == "Caius Uibius"


def test_break_no_means_the_word_continues(tmp_path):
    text, _ = epi(tmp_path, '<ab><lb n="1"/>Cai<lb n="2" break="no"/>us</ab>')
    assert text == "Caius"


def test_the_head_reading_Text_is_dropped(tmp_path):
    """Every EDH edition div opens with <head>Text</head>."""
    text, _ = epi(tmp_path, '<head>Text</head><ab>Dis Manibus</ab>')
    assert text == "Dis Manibus"


def test_abbreviations_reach_the_corpus_expanded(tmp_path):
    """Off(icina) is stored as Officina, matching what EDCS already supplies."""
    text, _ = epi(tmp_path,
                  '<ab><expan><abbr>Off</abbr><ex>icina</ex></expan></ab>')
    assert text == "Officina"


def test_editorial_restoration_is_kept(tmp_path):
    """Conjecture, but it is the reading the edition asserts; dropping it
    would leave shredded words behind."""
    text, _ = epi(tmp_path, '<ab>Dis <supplied reason="lost">Manibus</supplied></ab>')
    assert text == "Dis Manibus"


def test_lost_text_contributes_nothing(tmp_path):
    text, _ = epi(tmp_path,
                  '<ab>Dis <gap reason="lost" unit="character"/> Manibus</ab>')
    assert text == "Dis Manibus"


def test_apparatus_marks_are_not_text(tmp_path):
    text, _ = epi(tmp_path, '<ab>uixit <am>·</am>annos <surplus>X</surplus> L</ab>')
    assert "·" not in text and "X" not in text
    assert text.startswith("uixit annos")


def test_a_greek_edition_is_not_offered_at_all(tmp_path):
    """Papyri.info is 95% Greek; this filter is what lets one reader serve both."""
    div, _ = epi(tmp_path, "<ab>μῆνιν ἄειδε</ab>", lang="grc")
    assert div is None


def test_both_latin_language_codes_are_accepted():
    """Perseus tags editions lat (639-3); EDH and Papyri.info tag them la."""
    assert {"la", "lat"} <= LATIN_LANGS


def test_latin_edition_found_under_either_code(tmp_path):
    for code in ("la", "lat"):
        text, _ = epi(tmp_path, "<ab>Dis Manibus</ab>", lang=code)
        assert text == "Dis Manibus", code


def test_province_read_from_the_findspot(tmp_path):
    import xml.etree.ElementTree as ET
    xml = ('<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><sourceDesc>'
           '<msDesc><history><origin><origPlace>'
           '<placeName type="provinceItalicRegion">Raetia</placeName>'
           '<placeName>Cambodunum</placeName>'
           '</origPlace></origin></history></msDesc></sourceDesc></teiHeader></TEI>')
    p = tmp_path / "g.xml"; p.write_text(xml, encoding="utf-8")
    assert epidoc_province(ET.parse(p).getroot()) == "Raetia"


def test_province_falls_back_rather_than_failing(tmp_path):
    import xml.etree.ElementTree as ET
    p = tmp_path / "n.xml"
    p.write_text('<TEI xmlns="http://www.tei-c.org/ns/1.0"/>', encoding="utf-8")
    assert epidoc_province(ET.parse(p).getroot()) == "incertae sedis"


# --- plain text -------------------------------------------------------------
#
# feasibility.md §4.4 calls plain text the harder case: a TEI source names its
# apparatus, here the same material is inline and must be recognised by shape.
# Each of these pins one observed pattern, not a general heuristic — a general
# heuristic on undifferentiated text deletes Latin.

from extract import (clean_plaintext, plaintext_header,  # noqa: E402
                     strip_trailing_byline)


def test_header_splits_author_from_work():
    assert plaintext_header("Sallust: Bellum Iugurthinum") == \
        ("Sallust", "Bellum Iugurthinum")


def test_header_without_a_colon_is_a_title_not_a_guess():
    """About half the files carry a bare title or a bare name. Guessing which
    would be worse than leaving the author unset for enrich.py to resolve."""
    assert plaintext_header("DUODECIM TABULARUM LEGES") == \
        (None, "DUODECIM TABULARUM LEGES")
    assert plaintext_header("   ") == (None, None)


def test_navigation_index_is_dropped():
    """The site prints a section index under each title: 1 2 3 ... 114."""
    stats = Stats()
    text = clean_plaintext("1 2 3 4 5 6\nGallia est omnis\n", "latin-library", stats)
    assert "Gallia est omnis" in text
    assert "1 2 3" not in text
    assert stats["dropped:navigation_line"] == 1


def test_section_markers_go_but_editorial_brackets_stay():
    """[ 1 ] is a section number; [et inde] is an editorial insertion that
    normalize.py already counts and strips."""
    stats = Stats()
    text = clean_plaintext("[ 1 ] Falso queritur [et inde] genus\n",
                           "latin-library", stats)
    assert "[ 1 ]" not in text
    assert "[et inde]" in text
    assert stats["dropped:section_marks"] == 1


def test_site_footer_takes_the_whole_line():
    """It reads '  Sallust The Latin Library The Classics Page' — cutting at
    the phrase would leave the author dangling as the last word of the work."""
    stats = Stats()
    text = clean_plaintext("in illo sitae.\n Sallust The Latin Library The Classics Page",
                           "latin-library", stats)
    assert "Latin Library" not in text and "Sallust" not in text
    assert text.strip().endswith("in illo sitae.")


def test_footer_spread_over_separate_lines_is_also_removed():
    """Cicero's files put the byline on its own line above the site names."""
    stats = Stats()
    raw = "successum sit.\n\tCicero\n\tThe Latin Library\n\tThe Classics Page"
    text = clean_plaintext(raw, "latin-library", stats, author="Cicero")
    assert "Cicero" not in text
    assert text.strip().endswith("successum sit.")


def test_trailing_byline_only_strips_an_exact_repeat():
    kept, n = strip_trailing_byline("bellum gessit\nCaesar imperator", "Caesar")
    assert kept.endswith("Caesar imperator") and n == 0
    cut, n = strip_trailing_byline("bellum gessit\nCaesar", "Caesar")
    assert cut.strip() == "bellum gessit" and n == 1


def test_gutenberg_boilerplate_is_cut_at_its_markers():
    stats = Stats()
    raw = ("This ebook is for the use of anyone anywhere...\n"
           "*** START OF THE PROJECT GUTENBERG EBOOK AENEID ***\n"
           "arma uirumque cano\n"
           "*** END OF THE PROJECT GUTENBERG EBOOK AENEID ***\n"
           "Updated editions will replace the previous one.\n")
    text = clean_plaintext(raw, "gutenberg-latin", stats)
    assert "arma uirumque cano" in text
    assert "PROJECT GUTENBERG" not in text and "ebook" not in text
    assert stats["pg_boilerplate_stripped"] == 1


# --- CSEL volumes: separating edited text from editorial apparatus ----------
#
# A quarter of the Volumes/ tree is index locorum, index verborum, sigla and
# corrigenda. Training on it would teach the model page-and-line references,
# and the markup cannot be trusted to identify it: CSEL48 files 68,471 words
# of INDEX VERBORVM under subtype="chapter", while CSEL65 files a real work
# under subtype="MSS".

from extract import (APPARATUS_HEAD, EDITORIAL_SUBTYPES,  # noqa: E402
                     apparatus_features, classify_division,
                     drop_apparatus_paragraphs)

PROSE = ("Chronicorum libri quos Sulpicius Seuerus composuit primum editi "
         "sunt a Flacio Illyrico sub hoc titulo et de his rebus non satis "
         "constat cum plures codices inter se discrepent atque ideo ")
INDEX = ("Vaticanus 496 p. XXV. XXVI. Vaticanus 499 p. V. VI. XXII. XXV 34 "
         "Vaticanus 503 p. LXIX. Vaticanus 512 p. 68 69 Vaticanus 1342 p. 63 ")


def test_features_none_on_empty():
    assert apparatus_features("") is None


def test_features_separate_prose_from_index():
    prose = apparatus_features(PROSE * 3)
    index = apparatus_features(INDEX * 3)
    assert prose.digit < 0.01 and index.digit > 0.20
    assert prose.function > index.function


def test_index_head_beats_prose_markup():
    """The CSEL48 case: 68k words of index declared to be a chapter."""
    verdict, reason = classify_division(PROSE * 5, "chapter",
                                        "III. INDEX VERBORVM")
    assert (verdict, reason) == ("apparatus", "head")


def test_head_match_survives_ocr_damage():
    """TNDEX, INDBX and YERBORVM are all real readings in this repo."""
    for head in ("TNDEX V", "INDBX SRAECVS", "1NDEX AVCTORVM", "INDTCES"):
        assert APPARATUS_HEAD.search(head), head


def test_head_pattern_does_not_fire_inside_a_title():
    assert not APPARATUS_HEAD.search("DE INDICE ANIMAE")
    assert not APPARATUS_HEAD.search("LIBER AD CONSTANTIUM IMPERATOREM")


def test_content_override_catches_unlabelled_index():
    verdict, reason = classify_division(INDEX * 5, "chapter", "")
    assert (verdict, reason) == ("apparatus", "content")


def test_content_override_rescues_a_work_filed_as_mss():
    """The CSEL65 case: Liber ad Constantium, subtype='MSS'."""
    verdict, reason = classify_division(PROSE * 5, "MSS", "III. LIBER AD "
                                        "CONSTANTIUM IMPERATOREM")
    assert (verdict, reason) == ("prose", "content")


def test_verse_is_not_mistaken_for_a_word_list():
    """Poetry drops connectives, so it looks list-like on function words.

    Measured: hymns run 6% function words and a cento 8.6%, against 4.5% for
    the sigla lists the floor exists to catch. A filter that cannot tell a
    hymn from an index is not usable on a volume of Christian poetry.
    """
    hymn = ("Aeterne rerum conditor noctem diemque qui regis "
            "et temporum das tempora ut alleues fastidium ") * 8
    verdict, _ = classify_division(hymn, "poem", "HVMNVS I")
    assert verdict == "prose"


def test_short_division_falls_back_to_markup():
    assert classify_division("breuis oratio est", "work", "") == ("prose", "markup")
    assert classify_division("breuis oratio est", "index", "") == ("apparatus", "markup")


def test_unknown_subtype_without_signal_is_rejected():
    """Neither head nor content decides, and the markup is not a prose claim."""
    neutral = "Roma Carthago Athenae Corinthus Sparta Thebae Mediolanum " * 9
    assert classify_division(neutral, "wibble", "")[0] == "apparatus"


def test_paragraph_pass_keeps_prologue_drops_citation_table():
    """Priscillian's Canones: a prose prologue, then bare scripture refs."""
    stats = Stats()
    text = (PROSE * 3) + "\n\n" + ("Cor. I. 61. 69. Tim. I. 13. 14. Hebr. 18. "
                                   "19. Rom. 3. 114. Cor. II. 4. 13. 14. ") * 4
    out = drop_apparatus_paragraphs(text, stats)
    assert "Sulpicius" in out
    assert "Hebr." not in out
    assert stats["dropped:apparatus_paragraphs"] == 1


def test_paragraph_pass_leaves_clean_prose_untouched():
    stats = Stats()
    text = (PROSE * 3) + "\n\n" + (PROSE * 2)
    assert drop_apparatus_paragraphs(text, stats) == text
    assert stats["dropped:apparatus_paragraphs"] == 0


def test_short_paragraphs_are_never_dropped():
    """Below the profiling floor the rates are noise, and a drop is final."""
    stats = Stats()
    text = "Cor. I. 61. 69. Tim. I. 13."
    assert drop_apparatus_paragraphs(text, stats) == text


def test_editorial_subtypes_are_the_ones_dated_separately():
    assert EDITORIAL_SUBTYPES == {"preface", "prolegomena"}


# --- CSEL volumes: one source, two layouts ----------------------------------

from extract import (LAYOUT_CONTAINMENT, VolumeMeta, _one_name,  # noqa: E402
                     containment, csel_volume_meta, shingle_sample)

CSEL_HEADER = """<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc>
<titleStmt>
  <title>Corpus Scriptorum Ecclesiasticorum Latinorum (CSEL)</title>
  <title ref="http://data.perseus.org/catalog/urn:cts:latinLit:stoa0040a.stoa002"
    >Quaestiones veteris et novi testamenti</title>
  <author ref="urn:cite:perseus:cattg.1784.1">Pseudo-Augustinus</author>
  <editor>Alexander Souter Alexander Souter</editor>
</titleStmt>
<publicationStmt><date>2014</date></publicationStmt>
<sourceDesc><bibl><date>1908</date></bibl></sourceDesc>
</fileDesc></teiHeader><text><body/></text></TEI>"""


def _meta(xml=CSEL_HEADER):
    import xml.etree.ElementTree as ET
    return csel_volume_meta(ET.fromstring(xml))


def test_volume_meta_reads_author_editor_and_titles():
    meta = _meta()
    assert meta.author == "Pseudo-Augustinus"
    assert meta.editor == "Alexander Souter"
    assert meta.titles == {"stoa0040a.stoa002":
                           "Quaestiones veteris et novi testamenti"}
    assert meta.volume_title == "Quaestiones veteris et novi testamenti"


def test_edition_year_is_the_printing_not_the_encoding():
    """Volumes carry two dates: 1908 in print, 2014 keyed in at Leipzig."""
    assert _meta().edition_year == 1908


def test_volume_meta_survives_a_missing_header():
    import xml.etree.ElementTree as ET
    meta = csel_volume_meta(ET.fromstring('<TEI xmlns="http://www.tei-c.org/'
                                          'ns/1.0"><text/></TEI>'))
    assert meta == VolumeMeta(None, None, None, None, {})


def test_editor_name_splits_on_the_repeated_surname():
    assert _one_name("Alexander Souter Alexander Souter") == "Alexander Souter"
    assert _one_name("Franciscus Weihrich Franz Weihrich") == "Franciscus Weihrich"
    assert _one_name("Carolus F. Vrba Karl F. Vrba") == "Carolus F. Vrba"


def test_editor_name_handles_an_odd_length_pair():
    """`von` makes the halves uneven; the surname still marks the split."""
    assert _one_name("Guilelmus Hartelius Wilhelm von Hartel") == \
        "Guilelmus Hartelius"


def test_editor_name_does_not_halve_a_single_name():
    """SAMVEL BRANDT is one man, and halving would publish him as SAMVEL."""
    assert _one_name("SAMVEL BRANDT") == "SAMVEL BRANDT"


def test_shingle_sample_is_stable_across_processes():
    """str.__hash__ is salted per process; a salted corpus is not reproducible.

    The expected value is pinned, so a change of hash primitive fails here
    rather than silently reshuffling which divisions the corpus contains.
    """
    sample = shingle_sample("et in non est ad cum qui quod ut sed de ex si "
                            "enim autem quia per atque sunt esse")
    assert sample == {4729887682581808288}


def test_shingle_sample_ignores_orthography():
    assert shingle_sample("VIVAT REX IVLIVS CAESAR AVGVSTVS") == \
           shingle_sample("uiuat rex iulius caesar augustus")


def test_containment_detects_the_same_text_in_a_second_layout():
    words = ("alpha beta gamma delta epsilon zeta eta theta iota kappa "
             "lambda mu nu xi omicron pi").split()
    text = " ".join(f"{a} sequitur ordine quodam {b}" for a in words for b in words)
    pool = shingle_sample(text)
    assert containment(shingle_sample(text), pool) == 1.0
    assert containment(set(), pool) == 0.0
    other = " ".join(f"{a} plane diuersum carmen {b}" for a in words for b in words)
    assert containment(shingle_sample(other), pool) < LAYOUT_CONTAINMENT

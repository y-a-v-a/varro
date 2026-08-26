#!/usr/bin/env python3
"""Extract plain Latin text + provenance metadata from fetched sources.

Reads data/raw/<source>/, writes data/interim/<source>.jsonl — one JSON record
per work, in the schema documented in README.md.

Extractors are registered by FORMAT, not by source, because sources are added
far more often than formats are:

  tei-capitains    Perseus, CSEL           one walker, two sources
  conllu           DigilibLT               text layer only
  edcs-json        EDCS inscriptions
  mediawiki-xml    Vicifons

  epidoc-xml       EDH, Papyri.info        NOT YET WRITTEN — two sources, one job
  txt              Latin Library, Gutenberg    NOT YET WRITTEN — likewise

A source whose format already has an extractor needs a manifest entry and no
code at all. The record schema, the TEI walker, and the Greek/English filters
are shared by all of them.

Editorial policy applied here (each is a real decision, not a default):

  <note>        dropped entirely — editorial commentary, frequently English
  <choice>      prefers <reg> over <orig>, and <corr> over <sic>
                (regularised and editor-corrected readings win)
  <del>         dropped; <add> kept
  <gap>         emits nothing
  <foreign>     Greek (@xml:lang grc|greek) handled per --greek
  <sp>/<l>/<p>  treated as block boundaries

Language filtering is done on the CTS <ti:edition> element rather than on
filenames: Perseus ships English translations beside the Latin in the same
directory tree, and the TEI header is itself xml:lang="eng".

Usage:
    python src/extract.py --source perseus-canonical-latinLit
    python src/extract.py --source perseus-canonical-latinLit --limit 20 --report
    python src/extract.py --greek tag
"""

import argparse
import json
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from hashlib import blake2b
from pathlib import Path, PurePosixPath
from typing import (Any, Dict, Iterator, List, NamedTuple, Optional, Set,
                    Tuple)

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"

XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
CTS_NS = "{http://chs.harvard.edu/xmlns/cts}"
TEI_NS = "{http://www.tei-c.org/ns/1.0}"

GREEK_LANGS = {"grc", "greek", "gre"}

# Latin under both standards. Perseus tags editions "lat" (ISO 639-3); EDH and
# Papyri.info tag them "la" (ISO 639-1). Testing only for a "lat" prefix threw
# away every EpiDoc edition div as a foreign language — silently, apart from a
# dropped:lang:la counter nobody was reading.
LATIN_LANGS = {"la", "lat", "latin"}

# Elements whose entire subtree is discarded.
DROP_SUBTREE = {
    "note",      # editorial commentary, often English
    "del",       # editorially deleted text
    "gap",       # lacuna marker
    "figure", "figDesc", "graphic",
    "teiHeader", "fileDesc", "encodingDesc", "profileDesc", "revisionDesc",
    "bibl", "biblStruct", "listBibl",
    "app", "rdg", "witDetail", "lem",   # apparatus criticus — copyrightable AND noise
}

# Editorial matter identified by @rend rather than by element name.
#
# Perseus prints a dateline above each of Cicero's letters — "Scr. Romae m.
# Quint. a. 689 (65)" — saying where and when the editor believes it was
# written. It is nineteenth-century scholarship in Latin abbreviations, not
# Cicero, and there are 930 of them. LATIN-XXV learned to generate them, which
# is how they were found: no metric flagged it, reading the samples did.
#
# The sibling <seg rend="salute"> is NOT dropped. "CICERO ATTICO salutem" is
# genuine, it is the form the model reproduces correctly, and dropping it
# would remove the one piece of structure that makes a letter a letter.
DROP_REND = {"dateline"}

# Structural markers carrying no text of their own.
#
# Page/column/line breaks are elided to the empty string rather than a space:
# they occasionally fall *inside* a word (Gellius has "Athenis" split by a <pb/>,
# which a space turns into the nonsense "A the nis"). Where they sit at a word
# boundary the adjacent text/tail already supplies the whitespace, so eliding
# them is safe — measured at 7 mid-word against 1,731 whitespace-preceded.
DROP_EMPTY_JOIN = {"pb", "cb", "lb", "anchor"}
# Milestones genuinely separate content and must not weld tokens together.
DROP_EMPTY_SPACE = {"space"}

# Elements that force a line break in the output.
BLOCK = {"div", "div1", "div2", "div3", "p", "l", "lg", "head", "sp", "speaker",
         "label", "item", "list", "trailer", "opener", "closer", "argument"}

# Elements after which a blank line (paragraph break) is appropriate.
PARA = {"div", "div1", "div2", "p", "lg", "head", "sp"}


# EpiDoc is TEI, but epigraphic convention differs from literary convention in
# ways that silently corrupt text if the literary rules are applied to it.
#
#   <lb/>   In a literary edition a line break usually falls MID-WORD, so it is
#           elided (see DROP_EMPTY_JOIN). On a stone it is a word boundary
#           unless the editor says otherwise — measured in EDH: 12.4% carry
#           break="no", the other 87.6% separate words. Eliding them all would
#           weld "CAIUS" and "UIBIUS" into one token.
#   <head>  Every EDH edition div opens with a <head> reading "Text". Kept, it
#           would put that word at the head of all 79,206 records.
#   <am>    The abbreviation mark itself (a stroke or symbol), not language.
#   <surplus>, <g>, <certainty>  editorial apparatus, not text.
#
# What is deliberately KEPT: <expan>, whose <abbr> and <ex> children together
# spell the expanded word — Off(icina) reaches the corpus as "Officina", which
# is what EDCS already supplies — and <supplied>, the editor's restoration of
# lost letters. Restorations are conjecture, but they are the reading the
# edition asserts, and dropping them would leave shredded words behind.
EPIDOC_DROP = DROP_SUBTREE | {"head", "am", "surplus", "g", "certainty",
                              "milestone", "figure", "orig"}


class Profile(NamedTuple):
    """Which markup conventions a document follows."""
    drop_subtree: frozenset
    join_empty: frozenset      # elided to nothing: the break falls inside a word
    space_empty: frozenset     # elided to a space: the break separates words
    block: frozenset
    para: frozenset


LITERARY = Profile(frozenset(DROP_SUBTREE), frozenset(DROP_EMPTY_JOIN),
                   frozenset(DROP_EMPTY_SPACE), frozenset(BLOCK), frozenset(PARA))
EPIDOC = Profile(frozenset(EPIDOC_DROP), frozenset(),
                 frozenset(DROP_EMPTY_SPACE) | {"lb", "cb", "pb"},
                 frozenset({"ab", "div", "lg", "l"}), frozenset({"div"}))


def localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


# ---------------------------------------------------------------------------
# TEI text walking
# ---------------------------------------------------------------------------

class Stats(Counter):
    """Counts of editorial interventions, so the pipeline reports what it did."""


def _choice(elem: ET.Element, stats: Stats, greek: str,
            profile: Profile = None) -> str:
    """<choice> holds competing readings. Prefer regularised / corrected."""
    children = {localname(c.tag): c for c in elem}
    for preferred, rejected in (("reg", "orig"), ("corr", "sic")):
        if preferred in children:
            if rejected in children:
                stats[f"choice:{preferred}_over_{rejected}"] += 1
            return _walk(children[preferred], stats, greek, profile)
    # No recognised preference — take the first child rather than lose text.
    for child in elem:
        return _walk(child, stats, greek, profile)
    return elem.text or ""


def _walk(elem: ET.Element, stats: Stats, greek: str,
          profile: Profile = None) -> str:
    profile = profile or LITERARY
    tag = localname(elem.tag)

    rend = elem.get("rend")
    if rend in DROP_REND:
        stats[f"dropped:rend:{rend}"] += 1
        return " "

    if tag in profile.drop_subtree:
        stats[f"dropped:{tag}"] += 1
        return ""
    # An explicit break="no" says the word continues across the break, which is
    # the only case where eliding is right in an epigraphic text.
    if tag in ("lb", "cb", "pb") and elem.get("break") == "no":
        return ""
    if tag in profile.join_empty:
        return ""
    if tag in profile.space_empty:
        return " "
    if tag == "milestone":
        # <milestone unit="paragraph"/> is a structural break; others (section,
        # card, line numbering) are reference points inside running text.
        return "\n\n" if elem.get("unit") == "paragraph" else " "
    if tag == "choice":
        return _choice(elem, stats, greek, profile)

    # Language guard on ANY element, not just <foreign>. Perseus marks English
    # editorial front matter with xml:lang="eng" on ordinary structural elements
    # — the Terence editions carry a whole English INTRODUCTION inside an <l> —
    # so keying on the element name would let it through.
    lang = (elem.get(XML_LANG) or elem.get("lang") or "").lower().split("-")[0]
    if lang:
        if lang in GREEK_LANGS:
            stats["greek_passages"] += 1
            if greek == "drop":
                return " "
            if greek == "tag":
                inner = (elem.text or "") + "".join(
                    _walk(c, stats, greek, profile) + (c.tail or "") for c in elem)
                return f" <GRAECUM> {' '.join(inner.split())} </GRAECUM> "
            # greek == "keep": fall through and treat as ordinary text
        elif lang not in LATIN_LANGS:
            stats[f"dropped:lang:{lang}"] += 1
            return " "

    parts: List[str] = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        parts.append(_walk(child, stats, greek, profile))
        if child.tail:
            parts.append(child.tail)

    text = "".join(parts)
    if tag in profile.para:
        return f"\n\n{text}\n\n"
    if tag in profile.block:
        return f"\n{text}\n"
    return text


# English function words with NO collision in Latin. Deliberately excludes
# `an`, `is`, `at`, `sed`, `cum`, `quam` and anything else that is also a Latin
# word — a false positive here silently deletes genuine Latin.
ENGLISH_MARKERS = frozenset("""
the and of which with from this that they their there been were would could
should what when where who whom have has was you your our about into than then
them these those such only other more most who's it's don't upon very these
""".split())


def looks_english(paragraph: str) -> bool:
    tokens = re.findall(r"[A-Za-z']+", paragraph.lower())
    if len(tokens) < 5:
        return False
    hits = sum(1 for tok in tokens if tok in ENGLISH_MARKERS)
    return hits >= 2 and hits / len(tokens) >= 0.02


def drop_english_paragraphs(text: str, stats: Stats) -> str:
    """Remove editorial matter that no markup identifies as English.

    Perseus tags most English front matter with xml:lang="eng", but not all:
    Terence's Hecyra and Phormio carry an English INTRODUCTION in an <l> element
    distinguished only by n="subject_1". Rather than special-case one project's
    convention, test the extracted prose itself.
    """
    kept = []
    for para in text.split("\n\n"):
        if looks_english(para):
            stats["dropped:english_paragraphs"] += 1
            stats["dropped_english_words"] += len(para.split())
            continue
        kept.append(para)
    return "\n\n".join(kept)


GREEK_SCRIPT_RANGE = "Ͱ-Ͽἀ-῿"
# A run of one or more whitespace-separated tokens, each containing Greek script.
GREEK_RUN = re.compile(
    rf"\S*[{GREEK_SCRIPT_RANGE}]\S*(?:[ \t]+\S*[{GREEK_SCRIPT_RANGE}]\S*)*"
)
# Spans the markup pass already tagged; the inline pass must not wrap them again.
GREEK_TAGGED = re.compile(r"(<GRAECUM>.*?</GRAECUM>)", re.S)


def strip_inline_greek(text: str, stats: Stats, greek: str) -> str:
    """Apply the Greek policy to Greek embedded *inside* Latin sentences.

    Markup catches whole Greek passages and the sentence-level ratio test
    catches whole Greek sentences, but neither touches the systematic case:
    grammarians and commentators — Priscian, Charisius, Servius, Macrobius —
    quoting Greek technical vocabulary inline, e.g.
    `quid sit, περὶ τῆς οὐσίας, id est de substantia`.
    Measured at 12,572 tokens (0.19%) of DigilibLT, concentrated in 163 of
    373 works, so it is systematic rather than incidental.
    """
    if greek == "keep":
        return text

    def replace(match: re.Match) -> str:
        stats["greek_inline_runs"] += 1
        if greek == "drop":
            return ""
        return f"<GRAECUM> {match.group(0)} </GRAECUM>"

    # Segments already inside <GRAECUM> markers pass through untouched, so
    # markup-tagged Greek is not wrapped a second time.
    return "".join(
        segment if segment.startswith("<GRAECUM>")
        else GREEK_RUN.sub(replace, segment)
        for segment in GREEK_TAGGED.split(text))


WS_INLINE = re.compile(r"[ \t\r\f\v]+")
WS_BLANK = re.compile(r"\n{3,}")
WS_AROUND = re.compile(r"[ \t]*\n[ \t]*")


def tidy(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = WS_INLINE.sub(" ", text)
    text = WS_AROUND.sub("\n", text)
    text = WS_BLANK.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Perseus CapiTainS / CTS extractor
# ---------------------------------------------------------------------------

def _cts_text(path: Path, tag: str, lang: Optional[str] = None) -> Optional[str]:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return None
    for elem in root.iter(f"{CTS_NS}{tag}"):
        if lang is None or (elem.get(XML_LANG) or "").startswith(lang):
            return (elem.text or "").strip() or None
    for elem in root.iter(f"{CTS_NS}{tag}"):
        return (elem.text or "").strip() or None
    return None


EDITION_YEAR = re.compile(r"\b(1[5-9]\d{2}|20[0-2]\d)\b")
URN_LANG = re.compile(r"-([a-z]{3})\d*$")


def resolve_edition_lang(edition: ET.Element, work_root: ET.Element,
                         urn: str) -> Optional[str]:
    """Language of a CTS edition, by falling back through three signals.

    Only 109 of 325 ti:edition elements in canonical-latinLit carry xml:lang,
    so trusting that attribute alone silently discards two thirds of the corpus.
    ti:work supplies it for most of the rest, and the urn stem (…perseus-lat2)
    is a reliable last resort — every edition urn in this repo ends lat1–lat6.
    """
    for candidate in (edition.get(XML_LANG), work_root.get(XML_LANG)):
        if candidate:
            return candidate.lower()
    match = URN_LANG.search(urn)
    return match.group(1).lower() if match else None


def capitains_works(source_dir: Path) -> Iterator[Dict[str, Any]]:
    """Yield one descriptor per Latin *edition* (not translation) in a CTS repo.

    Shared by Perseus canonical-latinLit and OpenGreekAndLatin csel-dev, which
    use the same CapiTainS layout: data/<textgroup>/<work>/__cts__.xml plus one
    XML file per edition, named for the tail of its urn. CSEL differs only in
    cosmetics — group names carry xml:lang="eng" and urn stems read `opp-lat1`
    rather than `perseus-lat2` — both of which the existing fallbacks absorb.

    CSEL also ships whole-volume files under Volumes/. Those are the printed
    volumes this tree was split out of, so they are mostly duplicates, and are
    not walked here — csel_volume_records picks up the remainder.
    """
    data_dir = source_dir / "data"
    if not data_dir.is_dir():
        raise RuntimeError(f"expected {data_dir} — is the source fetched?")

    for group_cts in sorted(data_dir.glob("*/__cts__.xml")):
        author = _cts_text(group_cts, "groupname", "lat") or \
                 _cts_text(group_cts, "groupname") or group_cts.parent.name

        for work_cts in sorted(group_cts.parent.glob("*/__cts__.xml")):
            try:
                work_root = ET.parse(work_cts).getroot()
            except ET.ParseError:
                continue
            title = _cts_text(work_cts, "title") or work_cts.parent.name

            for edition in work_root.iter(f"{CTS_NS}edition"):
                # ti:edition = original-language text; ti:translation holds the
                # English and is never iterated here. That is the primary
                # language filter; resolve_edition_lang guards against a
                # non-Latin original slipping through.
                urn = edition.get("urn") or ""
                lang = resolve_edition_lang(edition, work_root, urn)
                if lang and not lang.startswith("lat"):
                    continue
                stem = urn.rsplit(":", 1)[-1]
                xml_path = work_cts.parent / f"{stem}.xml"
                if not xml_path.exists():
                    continue

                desc_el = edition.find(f"{CTS_NS}description")
                edition_desc = (desc_el.text or "").strip() if desc_el is not None else None
                year = None
                if edition_desc:
                    found = EDITION_YEAR.findall(edition_desc)
                    if found:
                        year = int(found[-1])

                yield {
                    "path": xml_path,
                    "urn": urn,
                    "author": author,
                    "work": title,
                    "edition": edition_desc,
                    "edition_year": year,
                }


def extract_tei_body(path: Path, stats: Stats, greek: str) -> Tuple[str, Optional[str]]:
    """Return (text, body_lang). Empty text if no <body> found."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        stats["parse_errors"] += 1
        return "", None

    text_el = None
    for elem in root.iter():
        if localname(elem.tag) == "text":
            text_el = elem
            break
    if text_el is None:
        return "", None
    body_lang = (text_el.get(XML_LANG) or "").lower() or None

    body = None
    for elem in text_el.iter():
        if localname(elem.tag) == "body":
            body = elem
            break
    if body is None:
        return "", body_lang

    text = strip_inline_greek(tidy(_walk(body, stats, greek)), stats, greek)
    return drop_english_paragraphs(text, stats), body_lang


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def load_manifest() -> Dict[str, Any]:
    try:
        import yaml
    except ImportError:
        sys.exit("PyYAML required — pip install -r requirements.txt")
    with (ROOT / "manifest" / "sources.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_manifest_entry(source_id: str) -> Dict[str, Any]:
    for src in load_manifest()["sources"]:
        if src["id"] == source_id:
            return src
    sys.exit(f"source `{source_id}` not in manifest")


def load_lock_entry(source_id: str) -> Dict[str, Any]:
    lock = ROOT / "manifest" / "fetch.lock.json"
    if not lock.exists():
        return {}
    with lock.open(encoding="utf-8") as fh:
        return json.load(fh).get("sources", {}).get(source_id, {})


def extract_capitains(source_id: str, greek: str, limit: Optional[int]) -> Tuple[List[Dict], Stats]:
    entry = load_manifest_entry(source_id)
    lock = load_lock_entry(source_id)
    source_dir = RAW / source_id
    stats = Stats()
    records: List[Dict[str, Any]] = []
    seen_works: set = set()

    for i, work in enumerate(capitains_works(source_dir)):
        if limit is not None and i >= limit:
            break
        text, body_lang = extract_tei_body(work["path"], stats, greek)
        stats["works_seen"] += 1

        if not text:
            stats["empty_after_extraction"] += 1
            continue
        # Belt and braces: CTS said this is a Latin edition; confirm the TEI agrees.
        if body_lang and not body_lang.startswith("lat"):
            stats["rejected_body_lang"] += 1
            continue

        n_words = len(text.split())
        stats["words"] += n_words
        # The volume pass below skips anything the CTS tree already carries.
        seen_works.add(".".join(work["urn"].rsplit(":", 1)[-1].split(".")[:2]))
        records.append({
            "id": f"{source_id}/{work['urn'].rsplit(':', 1)[-1]}",
            "text": text,
            "author": work["author"],
            "work": work["work"],
            "approximate_date": None,      # TODO: needs an author-date authority list
            "period": entry.get("period"),  # TODO: per-work, not per-source
            "genre": None,
            "source": source_id,
            "source_url": entry.get("url"),
            "source_version": lock.get("resolved_pin"),
            "license": entry.get("license"),
            "text_provenance": entry.get("text_provenance", "unknown"),
            "edition": work["edition"],
            "edition_year": work["edition_year"],
            "urn": work["urn"],
            "language_variant": "lat",
            "normalization": None,          # set by normalize.py
            "n_words": n_words,
        })
        stats["records"] += 1

    if (source_dir / "Volumes").is_dir():
        cts_shingles: Set[int] = set()
        for rec in records:
            cts_shingles |= shingle_sample(rec["text"])
        records.extend(csel_volume_records(source_id, entry, lock, source_dir,
                                           seen_works, cts_shingles, stats, greek))

    return records, stats


def csel_volume_records(source_id: str, entry: Dict[str, Any],
                        lock: Dict[str, Any], source_dir: Path,
                        seen_works: set, cts_shingles: Set[int], stats: Stats,
                        greek: str) -> List[Dict[str, Any]]:
    """Records for volume prose the per-work CTS tree does not already hold.

    Most of Volumes/ is the source the CTS files were split out of, so nearly
    every work division is skipped here by its own CTS identifier. What is
    left over is the editorial matter — prefaces and prolegomena, which the
    split never produced files for — plus the handful of works it missed.
    """
    authors = csel_group_names(source_dir)
    records: List[Dict[str, Any]] = []
    for div in csel_volume_divisions(source_dir, stats, greek):
        stats["volume_divisions_seen"] += 1
        work_id = div["n"] or ""
        if work_id in seen_works:
            stats["volume_division:already_in_cts"] += 1
            continue
        n_words = len(div["text"].split())
        if n_words < 200:
            stats["volume_division:too_short"] += 1
            continue
        sample = shingle_sample(div["text"])
        if containment(sample, cts_shingles) >= LAYOUT_CONTAINMENT:
            stats["volume_division:already_in_cts_by_text"] += 1
            stats["volume_words_duplicated"] += n_words
            continue
        # Volumes/ also repeats itself: CSEL35 ships as both a directory and a
        # loose file, so its prolegomena would arrive twice. Accepted text
        # joins the pool, which makes the second copy fail the same test.
        cts_shingles |= sample
        editorial = div["subtype"] in EDITORIAL_SUBTYPES
        meta = div["meta"]
        stats["volume_words"] += n_words
        if editorial:
            # Name it for what it is and for what it introduces, so that these
            # words never read as one of the works themselves. The title is
            # constructed rather than quoted, because the head alone is not
            # enough: `Prolegomena.` says nothing about which volume, and a
            # head naming the volume's letters gets pattern-matched into
            # EPISTOLOGRAPHY — a preface ABOUT Augustine's letters is not
            # epistolography. `Praefatio editoris` is the marker genres.yaml
            # keys on; plain `Praefatio in` would collide with Jerome's own
            # prefaces to the biblical books, of which the corpus holds 17.
            head_detail = div["head"].title() if div["head"] else ""
            title = f"Praefatio editoris in {meta.volume_title}"
            if head_detail and not re.match(r"[.\s]*(Praefatio|Prolegomena)[.\s]*$",
                                            head_detail, re.I):
                title = f"{title}: {head_detail}"
        else:
            title = (meta.titles.get(work_id) or div["head"]
                     or div["preceding_head"] or meta.volume_title)
        records.append({
            "id": f"{source_id}/volumes/{div['path'].stem}"
                  f"/{len(records):03d}",
            "text": div["text"],
            # A preface is the editor's own writing. Crediting it to the
            # author of the work it introduces would attribute 447k words of
            # 1890s philology to Lactantius and Cyprian.
            "author": (meta.editor if editorial else
                       authors.get(work_id.split(".")[0]) or meta.author),
            "work": title,
            "approximate_date": None,
            # Editorial matter is Vienna philology of the 1890s, not the late
            # antique text it introduces. Dating it with its volume would put
            # the editors in Augustine's period bucket.
            "period": "NEOLATIN" if editorial else entry.get("period"),
            "genre": "editorial" if editorial else None,
            "source": source_id,
            "source_url": entry.get("url"),
            "source_version": lock.get("resolved_pin"),
            "license": entry.get("license"),
            "text_provenance": entry.get("text_provenance", "unknown"),
            "edition": meta.volume_title,
            "edition_year": meta.edition_year,
            "urn": None,
            "language_variant": "lat",
            "normalization": None,
            "n_words": n_words,
        })
        stats["records"] += 1
        stats["words"] += n_words
    return records


# ---------------------------------------------------------------------------
# CSEL whole-volume files — separating edited text from editorial apparatus
# ---------------------------------------------------------------------------
#
# The CTS tree under data/ holds one file per work. Volumes/ holds the printed
# volumes those were split out of (see Volumes/split_parse.py), and a volume is
# not only its texts: roughly a quarter of it is index locorum, index verborum,
# manuscript sigla, tables of contents and corrigenda. Training on that would
# teach the model page-and-line references, so it has to be filtered out.
#
# The markup alone cannot do it. Both directions fail in practice:
#
#   CSEL48   68,471 words of INDEX VERBORVM filed as <div subtype="chapter">
#   CSEL65   an actual work, Liber ad Constantium, filed as subtype="MSS"
#
# So three signals are combined, strongest first: what a division calls itself
# in its <head>, what its text measurably looks like, and only then what the
# markup claims. Measured over all 579 divisions in the 81 volume files, the
# combination keeps 99.9% of prose words and rejects 100% of apparatus.

# Function words with no numeric or abbreviatory use. An index entry cites; it
# does not connect clauses, so these are what running prose has and lists lack.
LATIN_FUNCTION_WORDS = frozenset("""
et in non est ad cum qui quod ut sed de ex si enim autem quia per atque
sunt esse a ab nec
""".split())

ARABIC_REF = re.compile(r"^[0-9]+[.,;]?$")
ROMAN_REF = re.compile(r"^[IVXLCDM]+[.,;]?$")

# What a division calls itself, when it says so. OCR mangles these headings —
# TNDEX, INDBX, YERBORVM are all real readings in this repo — so the letters
# that scanning confuses (I/T/1, E/B) are spelled as classes rather than
# trusted. Anchored on word boundaries: `INDEX` must not fire inside a title.
APPARATUS_HEAD = re.compile(
    r"\b([IT1]ND[EB]X|[IT1]ND[IT1]C[EB]S|CONSPECTVS|CONSPECTUS|SIGLORVM"
    r"|SIGLORUM|TABVLA|TABULA|ERRATA|CORRIGENDA|ADDENDA|NOTARVM"
    r"|EXPLICATIO\s+SIGNORVM)\b", re.I)

# Subtypes that assert running text. Consulted last, and only when the head is
# silent and the content is unremarkable.
PROSE_SUBTYPES = frozenset({
    "work", "letter", "poem", "book", "chapter", "section", "dictio",
    "dialog", "fragment", "epilog", "retractationes", "preface",
    "prolegomena", "main",
})

# Divisions holding editorial Latin *about* the text rather than the text.
# Kept — it is real prose — but it was written in the 1890s, so labelling it
# LATE with the volume it prefaces would put Vienna philologists in the same
# period bucket as Augustine and quietly corrupt any dating experiment.
EDITORIAL_SUBTYPES = frozenset({"preface", "prolegomena"})

# A paragraph must be this long before its statistics mean anything.
MIN_PROFILE_WORDS = 40


class Apparatus(NamedTuple):
    """How reference-like a passage is. Rates are fractions of all tokens."""
    words: int
    digit: float           # bare arabic numerals: page and line citations
    roman: float           # bare roman numerals: book and volume citations
    function: float        # Latin function words: the signature of syntax
    long_segment: float    # share of tokens in segments of 8+ words


def apparatus_features(text: str) -> Optional[Apparatus]:
    """Measure a passage, or None if it is too short to measure."""
    tokens = text.split()
    if not tokens:
        return None
    n = len(tokens)
    digit = sum(1 for t in tokens if ARABIC_REF.match(t))
    roman = sum(1 for t in tokens if ROMAN_REF.match(t))
    function = sum(1 for t in tokens
                   if t.lower().strip(".,;:") in LATIN_FUNCTION_WORDS)
    segments = [s.split() for s in re.split(r"[.;:]", text)]
    long_tokens = sum(len(s) for s in segments if len(s) >= 8)
    return Apparatus(n, digit / n, roman / n, function / n, long_tokens / n)


def classify_division(text: str, subtype: Optional[str],
                      head: Optional[str]) -> Tuple[str, str]:
    """Return ("prose"|"apparatus", reason) for one division of a volume.

    Thresholds come from measuring every division in the repo. Prose and
    apparatus separate almost perfectly on the digit rate — prose sits at the
    95th percentile on 0.3%, apparatus at the 5th on 6.0% — so the content
    override is set at 10%, an order of magnitude clear of running text, and
    is deliberately not tuned any finer than the two populations warrant.

    The function-word floor of 4.5% is the second override, for apparatus that
    carries no numerals at all, such as a sigla list. It is set below verse:
    hymns run 6% and a cento 8.6%, because poetry drops connectives, and a
    filter that cannot tell a hymn from a word list is not usable here.
    """
    if APPARATUS_HEAD.search(head or ""):
        return "apparatus", "head"

    f = apparatus_features(text)
    if f is None or f.words < MIN_PROFILE_WORDS:
        # Too short to measure: fall through to what the markup claims.
        return ("prose" if subtype in PROSE_SUBTYPES else "apparatus"), "markup"

    if f.digit >= 0.10 or f.function < 0.045:
        return "apparatus", "content"
    # Unambiguous running prose overrides a mistaken subtype, which is how the
    # Liber ad Constantium escapes from subtype="MSS".
    if f.digit < 0.005 and f.function >= 0.15 and f.long_segment >= 0.70:
        return "prose", "content"
    return ("prose" if subtype in PROSE_SUBTYPES else "apparatus"), "markup"


# A division re-judged after paragraph filtering must come down to this, or
# its apparatus was never in separable paragraphs to begin with. Measured: the
# prefaces land at 0.102 and below, the Itinerarium Burdigalense stays at
# 0.139 and Priscillian's canon list at 0.239, because their tables run inside
# the paragraphs rather than as paragraphs of their own.
REJUDGE_DIGIT = 0.12


def usable_prose(text: str, subtype: Optional[str], head: Optional[str],
                 stats: Stats) -> Optional[str]:
    """The usable prose of one volume division, or None if it is apparatus.

    A preface is not uniformly prose or apparatus: it argues about the
    manuscripts for a paragraph and then cites twenty of them by page and
    line. Judging the whole division on its average rejects it — 296,535
    words of preface, of which 72% is discussion — so a division whose markup
    claims prose is given a second hearing on what survives paragraph
    filtering. A division whose apparatus is not paragraph-separable does not
    improve, and is still rejected.
    """
    verdict, reason = classify_division(text, subtype, head)
    if verdict == "prose":
        stats[f"volume_division:prose:{reason}"] += 1
        return drop_apparatus_paragraphs(text, stats)
    if reason != "content" or subtype not in PROSE_SUBTYPES:
        stats[f"volume_division:apparatus:{reason}"] += 1
        return None

    filtered = drop_apparatus_paragraphs(text, stats)
    f = apparatus_features(filtered)
    if f is None or f.words < MIN_PROFILE_WORDS or f.digit >= REJUDGE_DIGIT:
        stats["volume_division:apparatus:content"] += 1
        return None
    stats["volume_division:prose:rejudged"] += 1
    return filtered


def drop_apparatus_paragraphs(text: str, stats: Stats) -> str:
    """Remove reference tables that sit *inside* an otherwise-prose division.

    Whole-division classification is not enough. Priscillian's Canones open
    with a genuine prose prologue and continue as bare scripture citations
    ("Cor. I. 61. 69. Tim. I. 13. 14."), and the Itinerarium Burdigalense is a
    road table with a prose preamble. Both live under one <div subtype="work">.

    The per-paragraph threshold is looser than the per-division one, because a
    short paragraph's rates are noisy and a wrong drop here is unrecoverable.
    """
    kept = []
    for para in text.split("\n\n"):
        f = apparatus_features(para)
        if f is not None and f.words >= MIN_PROFILE_WORDS and \
                (f.digit >= 0.15 or f.function < 0.03):
            stats["dropped:apparatus_paragraphs"] += 1
            stats["dropped_apparatus_words"] += f.words
            continue
        kept.append(para)
    return "\n\n".join(kept)


# --- the same source, shipped twice --------------------------------------
#
# Volumes/ is a re-serialisation of the per-work tree, so a division can be a
# work the CTS files already provided. @n is present on only 274 of the 302
# work divisions, and where present it does not always agree with the CTS
# identifier, so matching on it alone lets 312,686 duplicated words through —
# Ambrosiaster's Quaestiones at 98%, Augustine's Speculum at 100%.
#
# This is not the dedupe stage's job. dedupe.py resolves the same text arriving
# from different SOURCES and needs cross-source scope to do it; this resolves
# one source shipping its own text in two layouts, which is a fact about the
# repository that only its extractor knows.

FOLD = str.maketrans({"v": "u", "j": "i", "V": "u", "J": "i"})
NON_LETTER = re.compile(r"[^a-z\s]+")
SHINGLE = 5
# One shingle in sixteen. The decision threshold is 0.5, nowhere near the
# sampling error, and a full shingle set of the CTS tree is 5.6M hashes.
LAYOUT_SAMPLE = 16
LAYOUT_CONTAINMENT = 0.5


def shingle_sample(text: str) -> Set[int]:
    """A deterministic 1-in-16 sample of the text's folded 5-gram hashes.

    blake2b rather than hash(): str hashing is salted per process, and a
    corpus whose contents depend on the interpreter's startup entropy is not
    reproducible.
    """
    tokens = NON_LETTER.sub(" ", text.lower().translate(FOLD)).split()
    out: Set[int] = set()
    for i in range(len(tokens) - SHINGLE + 1):
        value = int.from_bytes(
            blake2b(" ".join(tokens[i:i + SHINGLE]).encode(),
                    digest_size=8).digest(), "big")
        if value % LAYOUT_SAMPLE == 0:
            out.add(value)
    return out


def containment(sample: Set[int], pool: Set[int]) -> float:
    """Share of this division's shingles that the pool already holds."""
    if not sample:
        return 0.0
    return len(sample & pool) / len(sample)


def csel_group_names(source_dir: Path) -> Dict[str, str]:
    """textgroup id -> author name, read from the CTS tree beside Volumes/."""
    names: Dict[str, str] = {}
    for group_cts in sorted((source_dir / "data").glob("*/__cts__.xml")):
        name = _cts_text(group_cts, "groupname", "lat") or \
               _cts_text(group_cts, "groupname")
        if name:
            names[group_cts.parent.name] = name
    return names


class VolumeMeta(NamedTuple):
    author: Optional[str]
    editor: Optional[str]         # who wrote the preface, not who wrote the text
    edition_year: Optional[int]
    volume_title: Optional[str]
    titles: Dict[str, str]        # CTS work id -> title of that work


def _one_name(raw: str) -> Optional[str]:
    """CSEL headers name each editor twice, in Latin and then in German.

    `Alexander Souter Alexander Souter`, `Franciscus Weihrich Franz Weihrich`,
    `Guilelmus Hartelius Wilhelm von Hartel` — one editor each. The Latin form
    comes first and is the one that belongs beside Latin text.

    Splitting at the halfway point would be wrong twice over: `von` makes the
    Hartel entries odd-length, and it would cut the single name `SAMVEL
    BRANDT` in half. The reliable signal is the surname repeating, so the
    split is placed where the two halves end in the same name.
    """
    tokens = " ".join(raw.split()).split()
    for cut in range(1, len(tokens)):
        left, right = tokens[:cut], tokens[cut:]
        if left[-1][:4].lower() == right[-1][:4].lower():
            return " ".join(left)
    return " ".join(tokens) or None


def csel_volume_meta(root: ET.Element) -> VolumeMeta:
    """Author, printed year and per-work titles from a volume's teiHeader.

    The header is far richer than the divisions are: it names the author once,
    dates the printed edition, and — through <title ref="...urn:cts:...">  —
    titles each work the volume contains. The divisions themselves carry only
    @n, so this map is the only way to name them.
    """
    header = root.find(f".//{TEI_NS}teiHeader")
    if header is None:
        return VolumeMeta(None, None, None, None, {})

    author = editor = None
    for el in header.iter(f"{TEI_NS}author"):
        author = " ".join("".join(el.itertext()).split()) or None
        if author:
            break
    for el in header.iter(f"{TEI_NS}editor"):
        editor = _one_name("".join(el.itertext()))
        if editor:
            break

    # Two dates: the year the volume was printed and the year Leipzig encoded
    # it. The older one is the edition; the encoding year is not a text date.
    years = []
    for el in header.iter(f"{TEI_NS}date"):
        for match in re.findall(r"\b(1[5-9][0-9]{2})\b",
                                "".join(el.itertext())):
            years.append(int(match))
    edition_year = min(years) if years else None

    titles: Dict[str, str] = {}
    volume_title = None
    for el in header.iter(f"{TEI_NS}title"):
        text = " ".join("".join(el.itertext()).split())
        if not text or "Corpus Scriptorum" in text:
            continue
        ref = el.get("ref") or ""
        match = re.search(r"urn:cts:latinLit:([^\s/\"]+)", ref)
        if match:
            titles[match.group(1)] = text
        if volume_title is None:
            volume_title = text
    return VolumeMeta(author, editor, edition_year, volume_title, titles)


def csel_volume_divisions(source_dir: Path, stats: Stats,
                          greek: str) -> Iterator[Dict[str, Any]]:
    """Yield the usable prose divisions of the whole-volume files.

    Apparatus is rejected by classify_division; what survives still passes
    through drop_apparatus_paragraphs, because a division can be prose at the
    top and a reference table at the bottom.
    """
    volumes = source_dir / "Volumes"
    for path in sorted(volumes.rglob("CSEL*.xml")):
        if path.suffixes[-2:-1] == [".xml"]:      # CSEL31.xml.bak and friends
            continue
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            stats["parse_errors"] += 1
            continue
        meta = csel_volume_meta(root)
        for edition in [d for d in root.iter(f"{TEI_NS}div")
                        if d.get("type") == "edition"]:
            # A volume sets a work's title on a heading division of its own and
            # leaves the work division untitled, so the last heading seen is
            # carried forward as a fallback name.
            last_head = ""
            for div in edition:
                if div.tag != f"{TEI_NS}div":
                    continue
                head_el = div.find(f"{TEI_NS}head")
                head = " ".join("".join(head_el.itertext()).split()) \
                    if head_el is not None else ""
                raw = tidy(_walk(div, stats, greek, LITERARY))
                text = usable_prose(raw, div.get("subtype"), head, stats)
                if text is None:
                    stats["dropped_apparatus_division_words"] += len(raw.split())
                    if head:
                        last_head = head
                    continue
                text = strip_inline_greek(text, stats, greek)
                text = drop_english_paragraphs(text, stats)
                yield {"path": path, "head": head, "text": text,
                       "subtype": div.get("subtype"), "n": div.get("n"),
                       "preceding_head": last_head, "meta": meta}
                if head:
                    last_head = head


# ---------------------------------------------------------------------------
# DigilibLT — CoNLL-U inside tarballs
# ---------------------------------------------------------------------------

GREEK_SCRIPT = re.compile(r"[Ͱ-Ͽἀ-῿]")
PARAGRAPHUS = re.compile(r"Paragraphus_(\d+)")


def _greek_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return len(GREEK_SCRIPT.findall(text)) / len(letters)


def parse_conllu(lines: Iterator[str]) -> Tuple[Dict[str, str], List[Tuple[Optional[int], str]]]:
    """Return (document metadata, [(paragraph_no, sentence_text)]).

    Only the `# text =` layer is read. The token/lemma/POS/LiLa-URI columns are
    deliberately ignored — see the licence note on this source in
    manifest/sources.yaml.
    """
    meta: Dict[str, str] = {}
    sentences: List[Tuple[Optional[int], str]] = []
    in_sentences = False
    text: Optional[str] = None
    para: Optional[int] = None

    for line in lines:
        line = line.rstrip("\n")
        if line.startswith("#"):
            body = line[1:].strip()
            if body.startswith("sent_id"):
                in_sentences, para = True, None
            elif body.startswith("text ="):
                text = body.split("=", 1)[1].strip()
            elif not in_sentences and "=" in body:
                key, value = body.split("=", 1)
                meta[key.strip()] = value.strip()
            continue
        if not line.strip():
            if text is not None:
                sentences.append((para, text))
                text = None
            continue
        if para is None:
            found = PARAGRAPHUS.search(line)
            if found:
                para = int(found.group(1))

    if text is not None:
        sentences.append((para, text))
    return meta, sentences


def assemble(sentences: List[Tuple[Optional[int], str]], stats: Stats,
             greek: str) -> str:
    """Join sentences into paragraphs, applying the Greek policy per sentence."""
    paragraphs: List[List[str]] = []
    current: Optional[int] = object()  # sentinel distinct from any int or None

    for para, sentence in sentences:
        if _greek_ratio(sentence) > 0.3:
            stats["greek_passages"] += 1
            if greek == "drop":
                continue
            if greek == "tag":
                sentence = f"<GRAECUM> {sentence} </GRAECUM>"
        if para != current or not paragraphs:
            paragraphs.append([])
            current = para
        paragraphs[-1].append(sentence)

    return "\n\n".join(" ".join(p) for p in paragraphs if p)


def extract_conllu(source_id: str, greek: str, limit: Optional[int]) -> Tuple[List[Dict], Stats]:
    import tarfile

    entry = load_manifest_entry(source_id)
    lock = load_lock_entry(source_id)
    stats = Stats()
    records: List[Dict[str, Any]] = []
    seen: set = set()

    tarballs = sorted((RAW / source_id / "conllu").glob("*.tar.gz"))
    if not tarballs:
        raise RuntimeError(f"no conllu tarballs under {RAW / source_id} — fetched?")

    for tarball in tarballs:
        with tarfile.open(tarball, "r:gz") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".conllu"):
                    continue
                if limit is not None and len(records) >= limit:
                    break
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                raw = handle.read().decode("utf-8", errors="replace")
                meta, sentences = parse_conllu(iter(raw.splitlines()))
                stats["works_seen"] += 1

                doc_id = meta.get("docId") or Path(member.name).stem
                if doc_id in seen:          # parts overlap across tarballs
                    stats["duplicate_docids"] += 1
                    continue
                seen.add(doc_id)

                text = strip_inline_greek(
                    tidy(assemble(sentences, stats, greek)), stats, greek)
                text = drop_english_paragraphs(text, stats)
                if not text:
                    stats["empty_after_extraction"] += 1
                    continue

                description = meta.get("description")
                year = None
                if description:
                    found = EDITION_YEAR.findall(description)
                    if found:
                        year = int(found[-1])

                n_words = len(text.split())
                stats["words"] += n_words
                records.append({
                    "id": f"{source_id}/{doc_id}",
                    "text": text,
                    "author": meta.get("docAuthor"),
                    "work": meta.get("docTitle"),
                    "approximate_date": None,
                    "period": entry.get("period"),
                    "genre": None,
                    "source": source_id,
                    "source_url": entry.get("url"),
                    "source_version": lock.get("resolved_pin"),
                    "license": entry.get("license"),
                    "text_provenance": entry.get("text_provenance", "unknown"),
                    "edition": description,
                    "edition_year": year,
                    "urn": meta.get("corpusRef"),
                    "language_variant": "lat",
                    "normalization": None,
                    "n_words": n_words,
                })
                stats["records"] += 1

    return records, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="perseus-canonical-latinLit",
                    choices=extractable_sources(),
                    help="which fetched source to extract")
    ap.add_argument("--greek", default="drop", choices=("drop", "tag", "keep"),
                    help="policy for Greek passages inside Latin texts (default: drop)")
    ap.add_argument("--limit", type=int, help="stop after N works (for testing)")
    ap.add_argument("--report", action="store_true", help="print stats, write nothing")
    args = ap.parse_args()

    extractor = extractor_for(args.source)
    if extractor is None:
        entry = load_manifest_entry(args.source)
        sys.exit(f"no extractor for format `{entry.get('format')}` "
                 f"(source `{args.source}`) — add one to FORMAT_EXTRACTORS")
    records, stats = extractor(args.source, args.greek, args.limit)

    print(f"source        {args.source}")
    print(f"greek policy  {args.greek}")
    print(f"works seen    {stats['works_seen']:,}")
    print(f"records       {stats['records']:,}")
    print(f"words         {stats['words']:,}")
    if stats["records"]:
        print(f"mean words    {stats['words'] // stats['records']:,}")
    print()
    for key in sorted(k for k in stats
                      if k.startswith(("dropped:", "choice:", "volume_division:"))):
        print(f"  {key:<34} {stats[key]:>9,}")
    for key in ("greek_passages", "greek_inline_runs", "dropped_english_words",
                "volume_divisions_seen", "volume_words",
                "dropped_apparatus_division_words", "dropped_apparatus_words",
                "files_repaired", "entities_repaired", "hyphenation_rejoined",
                "files_with_annotation_layer", "annotation_elements_removed",
                "dated_by_source", "no_stated_date", "no_body",
                "parse_errors_after_repair",
                "duplicate_docids", "empty_after_extraction", "rejected_body_lang",
                "author_from_title", "pl_volume_found", "pages_seen", "skipped:namespace",
                "skipped:redirect", "skipped:too_short", "parse_errors"):
        if stats[key]:
            print(f"  {key:<34} {stats[key]:>9,}")

    if args.report:
        print("\n--report: nothing written")
        return 0

    INTERIM.mkdir(parents=True, exist_ok=True)
    out = INTERIM / f"{args.source}.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    size = out.stat().st_size
    print(f"\nwrote {out.relative_to(ROOT)} — {len(records):,} records, {size/1e6:.1f} MB")
    return 0



# ---------------------------------------------------------------------------
# EDCS — Latin inscriptions, JSON
# ---------------------------------------------------------------------------

# Inscriptions are not works. 537,262 of them as individual records would swamp
# a corpus of ~900 documents and make dedupe's all-pairs sweep intractable, so
# they are grouped by province and then split at a word budget. The province is
# the only grouping the dump reliably supports, and it is meaningful: epigraphic
# formulae vary regionally.
EDCS_WORDS_PER_DOC = 20000

# `clean_text_interpretive_word` expands the abbreviations (D M -> Dis Manibus).
# The conservative field preserves them, which is right for epigraphists and
# wrong for a language model: `D M` is two uninformative tokens.
EDCS_TEXT_FIELD = "clean_text_interpretive_word"


def extract_edcs(source_id: str, greek: str, limit: Optional[int]) -> Tuple[List[Dict], Stats]:
    entry = load_manifest_entry(source_id)
    lock = load_lock_entry(source_id)
    stats = Stats()
    records: List[Dict[str, Any]] = []

    files = sorted((RAW / source_id).glob("*.json"))
    if not files:
        raise RuntimeError(f"no JSON under {RAW / source_id} — fetched?")
    with files[0].open(encoding="utf-8") as fh:
        data = json.load(fh)

    by_province: Dict[str, List[str]] = defaultdict(list)
    for item in data:
        stats["inscriptions_seen"] += 1
        text = (item.get(EDCS_TEXT_FIELD) or "").strip()
        if not text:
            stats["empty_inscriptions"] += 1
            continue
        by_province[item.get("province") or "incerta"].append(" ".join(text.split()))

    for province, lines in sorted(by_province.items()):
        chunk: List[str] = []
        words = 0
        part = 0
        for line in lines:
            chunk.append(line)
            words += len(line.split())
            if words >= EDCS_WORDS_PER_DOC:
                part += 1
                records.append(_edcs_record(source_id, entry, lock, province,
                                            part, chunk, stats, greek))
                chunk, words = [], 0
        if chunk:
            part += 1
            records.append(_edcs_record(source_id, entry, lock, province,
                                        part, chunk, stats, greek))
        if limit is not None and len(records) >= limit:
            break

    return records, stats


def _edcs_record(source_id, entry, lock, province, part, lines, stats, greek):
    text = strip_inline_greek("\n".join(lines), stats, greek)
    text = drop_english_paragraphs(tidy(text), stats)
    n_words = len(text.split())
    stats["words"] += n_words
    stats["records"] += 1
    stats["works_seen"] += 1
    return {
        "id": f"{source_id}/{province.replace(' ', '_')}-{part:02d}",
        "text": text,
        "author": None,
        "work": f"Inscriptiones: {province} ({part})",
        "approximate_date": None,
        "period": entry.get("period"),
        "genre": None,
        "source": source_id,
        "source_url": entry.get("url"),
        "source_version": lock.get("resolved_pin"),
        "license": entry.get("license"),
        "text_provenance": entry.get("text_provenance", "unknown"),
        "edition": "EDCS via SDAM cleaned release",
        "edition_year": 2022,
        "urn": None,
        "language_variant": "lat",
        "normalization": None,
        "n_words": n_words,
    }


# ---------------------------------------------------------------------------
# Vicifons — Latin Wikisource, MediaWiki XML dump
# ---------------------------------------------------------------------------

MW_NS = "{http://www.mediawiki.org/xml/export-0.11/}"

# Wikitext is markup for humans editing in a browser, not a document format.
# Stripping order matters: templates and refs must go before link syntax, or a
# link inside a template survives its container's removal.
MW_COMMENT = re.compile(r"<!--.*?-->", re.S)
MW_REF = re.compile(r"<ref[^>]*?/>|<ref.*?</ref>", re.S | re.I)
MW_TAG = re.compile(r"<[^>]+>")
MW_TABLE = re.compile(r"\{\|.*?\|\}", re.S)
MW_FILE = re.compile(r"\[\[(?:File|Image|Fasciculus|Imago|Category|Categoria)"
                     r":[^\]]*?\]\]", re.I | re.S)

# Interwiki language links — `[[en:Jingle Bells]]`, `[[th:...]]`, `[[zh-cn:...]]`
# — sit at the foot of a page and point at the same work on another language's
# wiki. The link rules below would strip only the brackets and keep the text,
# which puts foreign titles into the corpus and, in 125 measured cases, English
# ones: the model's defining claim is that it never sees English. Found by
# tracing every Thai character in the corpus back to *Adeste Fideles* and
# *Jingle Bells*; 620 occurrences across 287 of 9,894 records.
#
# Deliberately narrow. Language codes are lowercase and two or three letters,
# so this cannot touch Latin Wikisource's own namespaces, which are capitalised
# Latin words — Pagina:, Liber:, Auctor:, Categoria:.
MW_INTERWIKI = re.compile(r"\[\[:?[a-z]{2,3}(?:-[a-z]{2,4})*:[^\]]*\]\]")
MW_PIPED = re.compile(r"\[\[[^\]|]*\|([^\]]*)\]\]")
MW_LINK = re.compile(r"\[\[([^\]]*)\]\]")
MW_EXTLINK = re.compile(r"\[https?://\S+\s+([^\]]*)\]")
MW_BARE_EXT = re.compile(r"\[https?://\S+\]")
MW_HEADING = re.compile(r"^={1,6}\s*(.*?)\s*={1,6}$", re.M)
MW_QUOTES = re.compile(r"'{2,5}")
MW_LISTMARK = re.compile(r"^[*#:;]+\s*", re.M)

# Latin Wikisource hosts a great deal of Migne's Patrologia Latina, whose pages
# carry column references (`131.0843C`, `131.0133D|`). They are apparatus, not
# text, and at this volume they would become frequent tokens.
MW_MIGNE_COL = re.compile(r"\b\d{2,3}\.\d{3,4}[A-D]?\|?")

# Titles usually name the author parenthetically — "Enarrationes in Psalmos
# (Remigius Antissiodorensis)" — which is the only author signal the dump gives.
# Without it every page falls to a source prior and the dating authority, which
# works by author, can never fire.
MW_TITLE_AUTHOR = re.compile(r"^(.*?)\s*\(([^)]{3,60})\)\s*$")

# Editions and volume markers that are not authors.
MW_NOT_AUTHOR = re.compile(r"^(ed\.|PL\s|Migne|Tomus|vol\.|liber|pars)", re.I)


def strip_templates(text: str) -> str:
    """Remove {{...}} including nesting, which a regex cannot do reliably."""
    out, depth, i = [], 0, 0
    while i < len(text):
        if text.startswith("{{", i):
            depth += 1
            i += 2
        elif text.startswith("}}", i) and depth:
            depth -= 1
            i += 2
        else:
            if not depth:
                out.append(text[i])
            i += 1
    return "".join(out)


def strip_wikitext(text: str) -> str:
    text = MW_COMMENT.sub(" ", text)
    text = MW_REF.sub(" ", text)
    text = MW_TABLE.sub(" ", text)
    text = strip_templates(text)
    text = MW_FILE.sub(" ", text)
    text = MW_INTERWIKI.sub(" ", text)
    text = MW_PIPED.sub(r"\1", text)
    text = MW_LINK.sub(r"\1", text)
    text = MW_EXTLINK.sub(r"\1", text)
    text = MW_BARE_EXT.sub(" ", text)
    text = MW_TAG.sub(" ", text)
    text = MW_HEADING.sub(r"\n\n\1\n\n", text)
    text = MW_QUOTES.sub("", text)
    text = MW_LISTMARK.sub("", text)
    text = MW_MIGNE_COL.sub(" ", text)
    return text


# Wikisource splits long works across subpages: "Homiliae (Haymo)/4".
MW_SUBPAGE = re.compile(r"/\d+\s*$")

# Migne's Patrologia Latina is arranged CHRONOLOGICALLY by author death, from
# Tertullian to Innocent III, so a volume number is a date. Three conventions
# carry it on Latin Wikisource; the link is by far the most common, because
# works transcribed from PL cite the volume they came from.
PL_TITLE_PATH = re.compile(r"^Patrologia Latina/(\d{1,3})\b")
PL_TITLE_PAREN = re.compile(r"\(PL\s*0*(\d{1,3})\)", re.I)
PL_LINK = re.compile(r"\[\[Patrologia Latina/(\d{1,3})")


def pl_volume(title: str, raw: str) -> Optional[int]:
    for pattern, subject in ((PL_TITLE_PATH, title), (PL_TITLE_PAREN, title),
                             (PL_LINK, raw)):
        match = pattern.search(subject or "")
        if match:
            volume = int(match.group(1))
            if 1 <= volume <= 217:
                return volume
    return None


def title_author(title: str) -> Tuple[str, Optional[str]]:
    """Split "Work (Author)" into its parts, where the parenthetical is a name."""
    title = MW_SUBPAGE.sub("", title or "").strip()
    match = MW_TITLE_AUTHOR.match(title)
    if not match:
        return title, None
    work, candidate = match.group(1).strip(), match.group(2).strip()
    if MW_NOT_AUTHOR.match(candidate) or candidate.isdigit():
        return title, None
    return (work or title), candidate


def extract_vicifons(source_id: str, greek: str, limit: Optional[int]) -> Tuple[List[Dict], Stats]:
    """One record per main-namespace page.

    Only namespace 0 is content. Everything else is machinery — templates, help
    pages, author index stubs, MediaWiki interface strings — and the interface
    strings are in English, so ingesting other namespaces would breach the
    project's central claim, not merely add noise.
    """
    import bz2

    entry = load_manifest_entry(source_id)
    lock = load_lock_entry(source_id)
    stats = Stats()
    records: List[Dict[str, Any]] = []

    dumps = sorted((RAW / source_id).glob("*.xml.bz2"))
    if not dumps:
        raise RuntimeError(f"no dump under {RAW / source_id} — fetched?")

    with bz2.open(dumps[0], "rb") as fh:
        title = ns = None
        for event, elem in ET.iterparse(fh, events=("end",)):
            tag = localname(elem.tag)
            if tag == "title":
                title = (elem.text or "").strip()
            elif tag == "ns":
                ns = (elem.text or "").strip()
            elif tag == "page":
                elem.clear()
                continue
            elif tag == "text":
                stats["pages_seen"] += 1
                raw = elem.text or ""
                elem.clear()
                if ns != "0":
                    stats["skipped:namespace"] += 1
                    continue
                if raw.lstrip().upper().startswith("#REDIRECT") or \
                        raw.lstrip().upper().startswith("#REDIRECTIO"):
                    stats["skipped:redirect"] += 1
                    continue

                text = strip_inline_greek(tidy(strip_wikitext(raw)), stats, greek)
                text = drop_english_paragraphs(text, stats)
                if len(text.split()) < 50:
                    stats["skipped:too_short"] += 1
                    continue

                n_words = len(text.split())
                stats["words"] += n_words
                stats["records"] += 1
                stats["works_seen"] += 1
                work, author = title_author(title or "")
                if author:
                    stats["author_from_title"] += 1
                volume = pl_volume(title or "", raw)
                if volume:
                    stats["pl_volume_found"] += 1
                records.append({
                    "id": f"{source_id}/{(title or 'untitled').replace(' ', '_')[:90]}",
                    "text": text,
                    "author": author,
                    "work": work,
                    "approximate_date": None,
                    "period": entry.get("period"),
                    "genre": None,
                    "source": source_id,
                    "source_url": entry.get("url"),
                    "source_version": lock.get("resolved_pin"),
                    "license": entry.get("license"),
                    "text_provenance": entry.get("text_provenance", "unknown"),
                    "edition": "Latin Wikisource",
                    "edition_year": None,
                    "urn": None,
                    "pl_volume": volume,
                    "language_variant": "lat",
                    "normalization": None,
                    "n_words": n_words,
                })
                if limit is not None and len(records) >= limit:
                    break
    return records, stats


def extract_lasla(source_id: str, greek: str, limit: Optional[int]) -> Tuple[List[Dict], Stats]:
    """LASLA / Opera Latina, from the CIRCSE mirror's CoNLL-U Plus files.

    AMBER — CC BY-NC-SA 4.0, Track B only. Never in a published corpus.

    The parsing is the shared CoNLL-U reader; only the metadata differs. LASLA
    carries no docAuthor or docId, so author and work come from the filename,
    which the mirror names `Author_Work_PartId.conllup` throughout.

    Worth knowing about the text itself: LASLA is a morphological corpus, so its
    `# text =` lines carry no punctuation at all and are already in classical
    u/v orthography. That is not damage — it is the corpus doing its job — but it
    means these records reach normalize.py in a different state from every other
    source, and reach dedupe.py looking more similar to their Perseus twins than
    two editions usually do.
    """
    entry = load_manifest_entry(source_id)
    lock = load_lock_entry(source_id)
    stats = Stats()
    records: List[Dict[str, Any]] = []

    files = sorted((RAW / source_id / "conllup").glob("*.conllup"))
    if not files:
        raise RuntimeError(f"no .conllup files under {RAW / source_id}/conllup — fetched?")

    for path in files:
        if limit is not None and len(records) >= limit:
            break
        stats["works_seen"] += 1
        meta, sentences = parse_conllu(path.read_text(encoding="utf-8",
                                                      errors="replace").splitlines())
        author, work, part = lasla_name(path.stem)

        text = strip_inline_greek(tidy(assemble(sentences, stats, greek)), stats, greek)
        text = drop_english_paragraphs(text, stats)
        if not text:
            stats["empty_after_extraction"] += 1
            continue

        records.append({
            "id": f"{source_id}/{path.stem}",
            "text": text,
            "author": author,
            "work": work,
            "part": part,
            "source": source_id,
            "source_version": lock.get("resolved_pin"),
            "license": entry.get("license"),
            "text_provenance": entry.get("text_provenance", "unknown"),
            "edition": "LASLA / Opera Latina (CIRCSE mirror)",
            "language_variant": "lat-cls",
            "n_words": len(text.split()),
        })
        stats["records"] += 1
        stats["words"] += records[-1]["n_words"]

    return records, stats


# `Caesar_BellumGallicum_CaesBG1` -> Caesar, Bellum Gallicum, CaesBG1.
# The third field is LASLA's own short identifier and is kept as the part, since
# it is what the annotation is cited by.
CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])")


def lasla_name(stem: str) -> Tuple[str, Optional[str], Optional[str]]:
    bits = stem.split("_")
    author = CAMEL.sub(" ", bits[0]) if bits else stem
    work = CAMEL.sub(" ", bits[1]) if len(bits) > 1 else None
    part = bits[2] if len(bits) > 2 else None
    return author, work, part



# EpiDoc inscriptions and papyri: EDH and Papyri.info share this reader.
#
# Records are batched by Roman province rather than one per stone. EDH holds
# 79,206 inscriptions averaging a couple of dozen words; one record each would
# give the corpus 79,206 documents whose validation split is meaningless — they
# are anonymous, so each would be held out as its own group. Batching matches
# what edcs-sdam already does and keeps the two epigraphic sources comparable.
EPIDOC_BATCH_WORDS = 20_000

PROVINCE_TYPES = ("provinceItalicRegion", "province")


def epidoc_edition(root: ET.Element) -> Optional[ET.Element]:
    """The Latin edition div, or None if this document has no Latin text.

    Papyri.info is 95% Greek — measured at 201 Latin edition divs per 4,000
    DDbDP files — so this filter is what makes one reader serve both sources.
    """
    for div in root.iter(f"{TEI_NS}div"):
        if div.get("type") != "edition":
            continue
        lang = (div.get(XML_LANG) or "").lower()
        if lang.startswith("la") and lang != "la-x-none":
            return div
    return None


def epidoc_province(root: ET.Element) -> str:
    for place in root.iter(f"{TEI_NS}placeName"):
        if place.get("type") in PROVINCE_TYPES and (place.text or "").strip():
            return " ".join(place.text.split())
    return "incertae sedis"


def extract_epidoc(source_id: str, greek: str, limit: Optional[int]) -> Tuple[List[Dict], Stats]:
    entry = load_manifest_entry(source_id)
    lock = load_lock_entry(source_id)
    stats = Stats()
    records: List[Dict[str, Any]] = []

    root_dir = RAW / source_id
    if not root_dir.exists():
        raise RuntimeError(f"{root_dir} missing — fetched?")

    # Only the directories holding editions. Papyri.info also ships metadata,
    # bibliography and translations — the translations are modern English and
    # would be a straight contamination of a corpus defined by excluding it.
    subdirs = [d for d in ("inscriptions", "DDbDP", "DCLP") if (root_dir / d).is_dir()]
    search = [root_dir / d for d in subdirs] or [root_dir]

    by_province: Dict[str, List[str]] = defaultdict(list)
    for base in search:
        for path in sorted(base.rglob("*.xml")):
            if limit is not None and stats["works_seen"] >= limit:
                break
            try:
                root = ET.parse(path).getroot()
            except ET.ParseError:
                stats["parse_errors"] += 1
                continue
            stats["files_seen"] += 1
            div = epidoc_edition(root)
            if div is None:
                stats["skipped:not_latin"] += 1
                continue
            stats["works_seen"] += 1
            text = tidy(_walk(div, stats, greek, EPIDOC))
            text = strip_inline_greek(text, stats, greek)
            if not text.strip():
                stats["empty_after_extraction"] += 1
                continue
            by_province[epidoc_province(root)].append(" ".join(text.split()))

    for province in sorted(by_province):
        batch, words, part = [], 0, 0
        for line in by_province[province]:
            batch.append(line)
            words += len(line.split())
            if words >= EPIDOC_BATCH_WORDS:
                part += 1
                records.append(_epidoc_record(source_id, entry, lock, province,
                                              part, batch, stats))
                batch, words = [], 0
        if batch:
            part += 1
            records.append(_epidoc_record(source_id, entry, lock, province,
                                          part, batch, stats))
    return records, stats


def _epidoc_record(source_id, entry, lock, province, part, lines, stats):
    text = "\n".join(lines)
    n_words = len(text.split())
    stats["words"] += n_words
    stats["records"] += 1
    return {
        "id": f"{source_id}/{province.replace(' ', '_')}-{part:02d}",
        "text": text,
        "author": None,
        "work": f"Inscriptiones: {province} ({part})",
        "approximate_date": None,
        "period": entry.get("period"),
        "genre": None,
        "source": source_id,
        "source_url": entry.get("url"),
        "source_version": lock.get("resolved_pin"),
        "license": entry.get("license"),
        "text_provenance": entry.get("text_provenance", "unknown"),
        "edition": entry.get("name"),
        "language_variant": "lat",
        "n_words": n_words,
    }



# Plain text: the Latin Library and Project Gutenberg.
#
# feasibility.md §4.4 calls plain text the harder case, and it is right. A TEI
# source hands you its apparatus in named elements; here the same material —
# running heads, section numbers, navigation, site furniture — is inline and
# has to be recognised by shape. Everything below is a rule about a specific
# observed pattern, not a general heuristic, because a general heuristic on
# undifferentiated text deletes Latin.

# " 1 2 3 4 5 ... 114 " — the section index the site prints under each title.
NAV_LINE = re.compile(r"^[\s\d]+$")
# "[ 1 ]" as a standalone marker. Editorial brackets around WORDS are left for
# normalize.py, which already counts and strips them.
SECTION_MARK = re.compile(r"\[\s*\d+\s*\]")
# The site footer, English, in 2,130 of 2,141 files. The whole LINE goes, not
# just the phrase: it reads "  Sallust The Latin Library The Classics Page", so
# cutting at the phrase leaves the author's name dangling as if it were text.
LL_FOOTER = re.compile(r"\n[^\n]*\b(?:The Latin Library|The Classics Page)\b.*$", re.S)
# Project Gutenberg wraps every text in English boilerplate between these.
PG_START = re.compile(r"^\*\*\*\s*START OF TH(IS|E) PROJECT GUTENBERG.*$", re.M)
PG_END = re.compile(r"^\*\*\*\s*END OF TH(IS|E) PROJECT GUTENBERG.*$", re.M)


def plaintext_header(first: str) -> Tuple[Optional[str], Optional[str]]:
    """Author and work from the Latin Library's first line.

    About half carry "Author: Work"; the rest carry a bare title, or a bare
    author name. Guessing which is which would be worse than admitting the
    ambiguity, so a line without a colon becomes the work and leaves the author
    unset — enrich.py resolves authors from its own authority file anyway.
    """
    line = " ".join(first.split())
    if not line:
        return None, None
    if ": " in line:
        author, _, work = line.partition(": ")
        return author.strip() or None, work.strip() or None
    return None, line


def strip_trailing_byline(text: str, *names: Optional[str]) -> Tuple[str, int]:
    """Drop trailing lines that merely repeat the author or title.

    The Latin Library's footer is three lines — the author, then the two site
    names. Cutting at the site names leaves the author standing alone at the
    end of the text, where it reads as the last word of the work.
    """
    wanted = {" ".join(n.lower().split()) for n in names if n}
    if not wanted:
        return text, 0
    lines = text.rstrip().split("\n")
    removed = 0
    while lines and " ".join(lines[-1].lower().split()) in wanted:
        lines.pop()
        removed += 1
    return "\n".join(lines), removed


def clean_plaintext(raw: str, source_id: str, stats: Stats,
                    author: Optional[str] = None,
                    work: Optional[str] = None) -> str:
    body = raw
    if PG_START.search(body) and PG_END.search(body):
        body = body[PG_START.search(body).end():PG_END.search(body).start()]
        stats["pg_boilerplate_stripped"] += 1

    lines = body.splitlines()
    kept: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if NAV_LINE.match(stripped):
            stats["dropped:navigation_line"] += 1
            continue
        kept.append(stripped)

    text = "\n".join(kept)
    text, marks = SECTION_MARK.subn(" ", text)
    stats["dropped:section_marks"] += marks
    cut = LL_FOOTER.search(text)
    if cut:
        text = text[:cut.start()]
        stats["dropped:site_footer"] += 1
    text, bylines = strip_trailing_byline(text, author, work)
    stats["dropped:trailing_byline"] += bylines
    return text


def extract_plaintext(source_id: str, greek: str, limit: Optional[int]) -> Tuple[List[Dict], Stats]:
    entry = load_manifest_entry(source_id)
    lock = load_lock_entry(source_id)
    stats = Stats()
    records: List[Dict[str, Any]] = []

    root_dir = RAW / source_id
    if not root_dir.exists():
        raise RuntimeError(f"{root_dir} missing — fetched?")
    files = sorted(root_dir.rglob("*.txt"))
    if not files:
        entry_method = entry.get("method")
        raise RuntimeError(
            f"no .txt under {root_dir}. "
            + (f"`{source_id}` is a manual source: download it from "
               f"{entry.get('url')} and place the files there."
               if entry_method == "manual" else "Fetched?"))

    for path in files:
        if limit is not None and len(records) >= limit:
            break
        stats["works_seen"] += 1
        raw = path.read_text(encoding="utf-8", errors="replace")
        first, _, rest = raw.partition("\n")
        author, work = plaintext_header(first)
        # A file in a per-author directory says so more reliably than its
        # header does: cicero/phil10.txt is Cicero whatever line one says.
        if path.parent != root_dir:
            author = author or path.parent.name.replace("_", " ").title()

        text = clean_plaintext(rest, source_id, stats, author, work)
        text = strip_inline_greek(tidy(text), stats, greek)
        text = drop_english_paragraphs(text, stats)
        if len(text.split()) < 20:
            stats["empty_after_extraction"] += 1
            continue

        records.append({
            "id": f"{source_id}/{path.stem}",
            "text": text,
            "author": author,
            "work": work,
            "approximate_date": None,
            "period": entry.get("period"),
            "genre": None,
            "source": source_id,
            "source_url": entry.get("url"),
            "source_version": lock.get("resolved_pin"),
            "license": entry.get("license"),
            "text_provenance": entry.get("text_provenance", "unknown"),
            "edition": entry.get("name"),
            "language_variant": "lat",
            "n_words": len(text.split()),
        })
        stats["records"] += 1
        stats["words"] += records[-1]["n_words"]
    return records, stats



# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
#
# At the foot of the module because a dispatch table can only name functions
# that already exist. The previous version sat above two of its own entries and
# patched them in afterwards by looking them up in globals() by string; putting
# the table last removes the problem rather than working around it.

# Dispatch is by FORMAT, not by source id.
#
# Sources are added far more often than formats are. Perseus and CSEL already
# share one extractor because they share the CapiTainS layout, and the two
# epigraphic sources still to land — EDH and Papyri.info — are both EpiDoc, as
# are Latin Library and Gutenberg both plain text. Keying on the format each
# source already declares in the manifest means a new source of a known format
# needs a manifest entry and no code at all.
#
# Format labels are as precise as the parsing actually requires. `tei-capitains`
# is not `tei-xml`: Corpus Corporum ships TEI with a different structure, and
# pretending otherwise would dispatch it to a parser that cannot read it.

# ---------------------------------------------------------------------------
# Tesserae — .tess, one citation per line
# ---------------------------------------------------------------------------
#
# The Tesserae corpus is a line-per-citation format built for intertextual
# search, not for reading:
#
#     <verg. aen. 1.1>\tArma virumque cano, Troiae qui primus ab oris
#
# Two things make it worth the reader. It is 69% classical by file count, and
# classical is the one part of this corpus that cannot be grown by finding
# more sources — and metadata/metadata.json dates every one of its 748 files
# individually, which is better evidence than the author-level floruits the
# authority file carries.

# A citation tag. Some files lose the opening bracket — `cic. phil. 7.1>` and
# `mac. frag 1>` both occur — so the closing form is matched too, but only
# when it looks like a citation: lowercase, dotted, and containing a digit.
# That last condition is what protects the text, because angle brackets are
# also the editorial mark for a conjectural supplement, and lines beginning
# `pronos> intelligebat` or `quem> adeo` are Latin words, not references.
TESS_TAG = re.compile(r"^\s*<[^>]*>\s*")
TESS_TAG_UNCLOSED = re.compile(r"^\s*(?=[a-z][a-z0-9.\s]*\d)[a-z0-9.\s]{1,30}>\s*")

# ^Z, left at the end of at least one file by whatever wrote it.
TESS_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def strip_tess_tag(line: str) -> str:
    """Remove the leading citation reference from one .tess line."""
    line = TESS_TAG.sub("", line, count=1)
    return TESS_TAG_UNCLOSED.sub("", line, count=1)


def tess_name(stem: str) -> Tuple[str, str]:
    """`ammianus.rerum_gestarum.part.14` -> ("Ammianus", "Rerum Gestarum 14").

    Parts are kept as separate records rather than stitched into one work,
    because metadata.json is keyed by file and stitching would throw away the
    per-file date that is the main reason for reading this source at all.
    """
    bits = stem.split(".")
    author = bits[0].replace("_", " ").title()
    rest = [b for b in bits[1:] if b != "part"]
    work = " ".join(b.replace("_", " ").title() for b in rest) if rest else stem
    return author, work


# Tesserae's `mode`, which is the same distinction the project calls `form`.
TESS_MODE_FORM = {"prose": "PROSE", "verse": "VERSE", "mixed": "PROSIMETRUM"}


def extract_tess(source_id: str, greek: str, limit: Optional[int]) -> Tuple[List[Dict], Stats]:
    entry = load_manifest_entry(source_id)
    lock = load_lock_entry(source_id)
    stats = Stats()
    records: List[Dict[str, Any]] = []

    root_dir = RAW / source_id
    if not root_dir.exists():
        raise RuntimeError(f"{root_dir} missing — fetched?")
    files = sorted(root_dir.rglob("*.tess"))
    if not files:
        raise RuntimeError(f"no .tess under {root_dir} — fetched?")

    meta_path = next(iter(root_dir.rglob("metadata.json")), None)
    meta: Dict[str, Any] = {}
    if meta_path is not None:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        stats["metadata_missing"] += 1

    for path in files:
        if limit is not None and len(records) >= limit:
            break
        stats["works_seen"] += 1
        lines = []
        for raw in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            raw = TESS_CONTROL.sub("", raw)
            if not raw.strip():
                continue
            text = strip_tess_tag(raw).strip()
            if text:
                lines.append(text)

        text = tidy("\n".join(lines))
        text = strip_inline_greek(text, stats, greek)
        text = drop_english_paragraphs(text, stats)
        if len(text.split()) < 20:
            stats["empty_after_extraction"] += 1
            continue

        info = meta.get(path.name, {})
        date = info.get("date")
        try:
            date = int(date)
        except (TypeError, ValueError):
            date = None
            stats["no_stated_date"] += 1
        author, work = tess_name(path.stem)
        records.append({
            "id": f"{source_id}/{path.stem}",
            "text": text,
            "author": author,
            "work": work,
            # enrich ranks this under the curated authorities and records it
            # as `source-metadata`; see Authority.resolve.
            "approximate_date": date,
            "period": entry.get("period"),
            "genre": None,
            "form": TESS_MODE_FORM.get(info.get("mode")),
            "source": source_id,
            "source_url": entry.get("url"),
            "source_version": lock.get("resolved_pin"),
            "license": entry.get("license"),
            "text_provenance": entry.get("text_provenance", "unknown"),
            "edition": entry.get("name"),
            "language_variant": "lat",
            "normalization": None,
            "n_words": len(text.split()),
        })
        stats["records"] += 1
        stats["words"] += records[-1]["n_words"]
        if date is not None:
            stats["dated_by_source"] += 1
    return records, stats


# ---------------------------------------------------------------------------
# Generic TEI XML — Corpus Corporum and anything else shipping plain TEI
# ---------------------------------------------------------------------------
#
# Distinct from tei-capitains, which navigates a CTS tree of __cts__.xml
# catalogues. This reads a directory of TEI files directly, which is the shape
# Corpus Corporum exports and the shape its upstream feeds already publish.
#
# Two repairs are applied that the other TEI readers do not need, both
# measured on real Corpus Corporum files (the Patrologia Latina feed, whose
# filenames literally carry a [CC] prefix):
#
#   entity repair       5 of 21 files in one volume are not well-formed XML
#   de-hyphenation      4.06% of all words are split across a printed line
#
# Together they took one volume from 50,980 words to 218,836 — a 4.3x
# difference between a reader that looks like it works and one that does.

# Character references left unterminated by the OCR pipeline. All four forms
# below are real readings: `&#171 ;` with a space before the semicolon,
# `&gt5` with none at all, `&#0160  .` likewise. An XML parser rejects the
# whole file for any one of them, so without this a quarter of the corpus
# would arrive as a parse_errors counter and nothing else.
# Two shapes, and the order matters. A reference whose semicolon has drifted
# away from it (`&#171 ;`) must have that semicolon CONSUMED, or repairing it
# leaves a stray `;` behind in the text; one with no semicolon at all
# (`&gt5`) just needs one supplied.
DRIFTED_NUMERIC = re.compile(r"&#(x?[0-9A-Fa-f]+)[ \t]+;")
DRIFTED_NAMED = re.compile(r"&(amp|lt|gt|quot|apos)[ \t]+;")
UNTERMINATED_NUMERIC = re.compile(r"&#(x?[0-9A-Fa-f]+)(?![0-9A-Fa-f;])")
UNTERMINATED_NAMED = re.compile(r"&(amp|lt|gt|quot|apos)(?![a-zA-Z;])")

# A word broken across a line of the printed page: `Victo-\nrina`,
# `confla-\ntum`, `sup-\npressit`. Rejoining must happen BEFORE tidy(), which
# collapses the newline to a space and destroys the only evidence that this
# was a line break rather than a real hyphen. A milestone or page break may
# sit in the gap — `qui-\n<milestone/>bus` — but those elide to nothing in
# the walker, so by this point the newline is adjacent.
LINE_HYPHENATION = re.compile(r"(\w)-[ \t]*\n[ \t]*(\w)")

# Corpus Corporum's own exports carry a POS layer in their own namespace, and
# it repeats the text: `<cc:s><cc:w>LIBER</cc:w><cc:w>PRIMUS</cc:w></cc:s>`
# is immediately followed by the running text `LIBER PRIMUS.`. Walking that
# naively yields "LIBER PRIMUS LIBER PRIMUS." — measured at 226,051 words
# against a true 113,622, i.e. every word in the corpus doubled.
#
# The annotation is removed rather than the plain text because the plain text
# is the readable form: it keeps punctuation and capitalisation, which the
# tokenised copy drops.
CC_ANNOTATION_NS = "{http://mlat.uzh.ch/2.0}"

# `Patrologia Latina, vol. 144` in <seriesStmt>, or `[CC]PL10_19_DE.xml` in a
# filename -> the volume number. enrich already interpolates a year from it,
# so recovering it here is worth more than a date from the TEI header — see
# below.
PL_VOLUME_IN_SERIES = re.compile(r"Patrologia\s+Latina,?\s*vol\.?\s*(\d{1,3})", re.I)

# `<author>Petrus Damianus<date>1007-1072</date></author>` — a lifespan, which
# is the only composition evidence these headers carry.
AUTHOR_LIFESPAN = re.compile(r"(-?\d{1,4})\s*[-–]\s*(-?\d{1,4})")
# `\b` after the digits is wrong: the real filenames are `[CC]PL10_19_DE.xml`
# and `_` is a word character, so there is no boundary there. What is meant
# is "not followed by another digit" — volumes run 1 to 217.
PL_VOLUME_IN_NAME = re.compile(r"\bPL(\d{1,3})(?!\d)")


def repair_entities(raw: str) -> Tuple[str, int]:
    """Terminate character references the source left open. Returns (xml, n)."""
    total = 0
    for pattern, form in ((DRIFTED_NUMERIC, "&#{};"), (DRIFTED_NAMED, "&{};"),
                          (UNTERMINATED_NUMERIC, "&#{};"),
                          (UNTERMINATED_NAMED, "&{};")):
        raw, n = pattern.subn(lambda m: form.format(m.group(1)), raw)
        total += n
    return raw, total


def rejoin_hyphenation(text: str) -> Tuple[str, int]:
    """Reunite words split across a printed line. Returns (text, n)."""
    return LINE_HYPHENATION.subn(r"\1\2", text)


def tei_title(root: ET.Element) -> Optional[str]:
    """The work's title: the titleStmt if it has one, else the body's heading.

    Both halves are needed, because the two feeds fail in opposite ways.
    Corpus Corporum's own export names the work properly —
    `<titleStmt><title>Epistolae</title>` — while the Patrologia Latina feed
    on GitHub ships stubs, `<title/>` empty beside a `TO BE ADDED BY LEIPZIG`
    idno, and puts the real title in `<head>` at the top of the text.

    Only titleStmt is consulted, never any <title>: <seriesStmt> carries one
    too ("Patrologia Latina, vol. 144"), and that is the series, not the work.
    """
    for stmt in root.iter(f"{TEI_NS}titleStmt"):
        for el in stmt.iter(f"{TEI_NS}title"):
            text = " ".join("".join(el.itertext()).split())
            if text:
                return text
        break

    body = root.find(f".//{TEI_NS}body")
    if body is not None:
        for head in body.iter(f"{TEI_NS}head"):
            for el in list(head.iter(f"{TEI_NS}title")) + [head]:
                text = " ".join("".join(el.itertext()).split())
                if text:
                    return text
            break
    return None


def strip_cc_annotation(elem: ET.Element) -> int:
    """Remove Corpus Corporum's POS layer, keeping the running text it shadows.

    The tail matters: the plain text sits in the annotation element's tail, so
    removing the element without reattaching its tail deletes the very text
    this is meant to preserve.
    """
    removed = 0
    for child in list(elem):
        if child.tag.startswith(CC_ANNOTATION_NS):
            index = list(elem).index(child)
            tail = child.tail or ""
            elem.remove(child)
            removed += 1
            if tail:
                if index == 0:
                    elem.text = (elem.text or "") + tail
                else:
                    previous = list(elem)[index - 1]
                    previous.tail = (previous.tail or "") + tail
        else:
            removed += strip_cc_annotation(child)
    return removed


def tei_author(root: ET.Element) -> Optional[str]:
    for el in root.iter(f"{TEI_NS}author"):
        # The <date> child is the author's lifespan, not part of their name.
        parts = [el.text or ""] + [(c.tail or "") for c in el]
        text = " ".join(" ".join(parts).split())
        if text:
            return text
    return None


def tei_author_midpoint(root: ET.Element) -> Optional[int]:
    """A composition estimate from the author's lifespan, or None.

    `<author>Petrus Damianus<date>1007-1072</date></author>` gives 1039. That
    is the midpoint of a life, not a date of writing, and it is deliberately
    NOT presented as more than it is: enrich ranks it under every curated
    authority and records it as `source-metadata`. For period bucketing —
    which is what it feeds — a life's midpoint and its writing years fall in
    the same bucket except at a boundary.
    """
    for el in root.iter(f"{TEI_NS}author"):
        for date in el.iter(f"{TEI_NS}date"):
            match = AUTHOR_LIFESPAN.search("".join(date.itertext()))
            if match:
                born, died = int(match.group(1)), int(match.group(2))
                if born <= died:
                    return (born + died) // 2
    return None


def tei_pl_volume(root: ET.Element, filename: str) -> Optional[int]:
    """The Patrologia Latina volume, from <seriesStmt> or from the filename."""
    for el in root.iter(f"{TEI_NS}seriesStmt"):
        match = PL_VOLUME_IN_SERIES.search("".join(el.itertext()))
        if match:
            return int(match.group(1))
    match = PL_VOLUME_IN_NAME.search(filename)
    return int(match.group(1)) if match else None


def tei_body_lang(root: ET.Element) -> Optional[str]:
    """Whatever language the file declares nearest the text, if any."""
    for tag in ("text", "body", "listBibl"):
        el = root.find(f".//{TEI_NS}{tag}")
        if el is not None and el.get(XML_LANG):
            return el.get(XML_LANG).lower()
    return (root.get(XML_LANG) or "").lower() or None


class TeiDocument(NamedTuple):
    name: str            # the member's own filename, which carries the PL volume
    xml: str
    work_id: Optional[str]   # the archive's stem, which keys the metadata table


def cc_work_dates(root_dir: Path) -> Dict[str, int]:
    """work_id -> a year, from the metadata table fetched alongside the works.

    Better evidence than the author element in the TEI, and available where
    that is not: `De facto Ungarie` and `De decem categoriis` both carry an
    author with no parseable lifespan, and the table dates them 1236 and 400.
    The table's own `year` column is free text — `fl.1236`, `c.1175-1253` —
    so the already-parsed not_before/not_after pair is used instead.
    """
    import csv as _csv

    path = root_dir / "_metadata" / "cc_metadata.csv"
    if not path.exists():
        return {}
    dates: Dict[str, int] = {}
    with path.open(encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            bounds = []
            for key in ("not_before", "not_after"):
                try:
                    bounds.append(float(row.get(key) or ""))
                except ValueError:
                    pass
            if bounds and row.get("work_id"):
                dates[row["work_id"]] = int(sum(bounds) / len(bounds))
    return dates


def tei_documents(root_dir: Path) -> Iterator[TeiDocument]:
    """Yield every TEI document under a source directory.

    Zipped members are read in place rather than unpacked, which is how the
    DigilibLT reader already handles its tarballs: Corpus Corporum arrives as
    one zip per work, and unpacking 2,317 of them would double the disk for
    nothing. The name yielded is the member's, not the archive's, because it
    is the member that carries the Patrologia Latina volume.
    """
    import zipfile

    for path in sorted(root_dir.rglob("*")):
        if path.name == "__cts__.xml" or not path.is_file():
            continue
        if path.suffix.lower() == ".xml":
            yield TeiDocument(path.name,
                              path.read_text(encoding="utf-8", errors="replace"),
                              None)
        elif path.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(path) as archive:
                    for member in archive.namelist():
                        if not member.lower().endswith(".xml"):
                            continue
                        with archive.open(member) as handle:
                            yield TeiDocument(
                                PurePosixPath(member).name,
                                handle.read().decode("utf-8", errors="replace"),
                                path.stem)
            except zipfile.BadZipFile:
                continue


def extract_tei_xml(source_id: str, greek: str, limit: Optional[int]) -> Tuple[List[Dict], Stats]:
    entry = load_manifest_entry(source_id)
    lock = load_lock_entry(source_id)
    stats = Stats()
    records: List[Dict[str, Any]] = []

    root_dir = RAW / source_id
    if not root_dir.exists():
        raise RuntimeError(f"{root_dir} missing — fetched?")
    sources = list(tei_documents(root_dir))
    if not sources:
        method = entry.get("method")
        raise RuntimeError(
            f"no .xml or .zip under {root_dir}. "
            + (f"`{source_id}` is a manual source: obtain it from "
               f"{entry.get('url')} and place the files there."
               if method == "manual" else "Fetched?"))

    stated_dates = cc_work_dates(root_dir)
    for name, raw, work_id in sources:
        if limit is not None and len(records) >= limit:
            break
        stats["works_seen"] += 1
        path = PurePosixPath(name)
        try:
            root = ET.fromstring(raw)
        except ET.ParseError:
            raw, repaired = repair_entities(raw)
            if not repaired:
                stats["parse_errors"] += 1
                continue
            try:
                root = ET.fromstring(raw)
            except ET.ParseError:
                stats["parse_errors_after_repair"] += 1
                continue
            stats["files_repaired"] += 1
            stats["entities_repaired"] += repaired

        lang = tei_body_lang(root)
        if lang and lang.split("-")[0] not in LATIN_LANGS:
            stats[f"skipped:lang:{lang}"] += 1
            continue

        body = root.find(f".//{TEI_NS}body")
        if body is None:
            stats["no_body"] += 1
            continue

        stripped = strip_cc_annotation(body)
        if stripped:
            stats["annotation_elements_removed"] += stripped
            stats["files_with_annotation_layer"] += 1

        text, joined = rejoin_hyphenation(_walk(body, stats, greek, LITERARY))
        stats["hyphenation_rejoined"] += joined
        text = strip_inline_greek(tidy(text), stats, greek)
        text = drop_english_paragraphs(text, stats)
        if len(text.split()) < 20:
            stats["empty_after_extraction"] += 1
            continue

        records.append({
            "id": f"{source_id}/{path.stem}",
            "text": text,
            "author": tei_author(root),
            "work": tei_title(root) or path.stem,
            # Deliberately NOT taken from the TEI header. Corpus Corporum's
            # headers date the *encoding* — the Patrologia Latina files say
            # <date>2015</date> — and reading that as a composition date is
            # exactly the error that filed the CSEL prefaces under LATE.
            # enrich resolves this properly from the author, or from the
            # Patrologia Latina volume below.
            # The header's own <date> is the *printing* — these files say
            # 1853, meaning Migne — so it is never read as a composition date;
            # that error is what filed the CSEL prefaces under LATE. The
            # author's lifespan is the one real signal, and enrich ranks it
            # below every curated authority as `source-metadata`.
            "approximate_date": (stated_dates.get(work_id)
                                 if stated_dates.get(work_id) is not None
                                 else tei_author_midpoint(root)),
            "pl_volume": tei_pl_volume(root, path.name),
            "period": entry.get("period"),
            "genre": None,
            "source": source_id,
            "source_url": entry.get("url"),
            "source_version": lock.get("resolved_pin"),
            "license": entry.get("license"),
            "text_provenance": entry.get("text_provenance", "unknown"),
            "edition": entry.get("name"),
            "language_variant": "lat",
            "normalization": None,
            "n_words": len(text.split()),
        })
        stats["records"] += 1
        stats["words"] += records[-1]["n_words"]
    return records, stats

FORMAT_EXTRACTORS = {
    "tei-capitains": extract_capitains,
    "conllu": extract_conllu,
    "edcs-json": extract_edcs,
    "mediawiki-xml": extract_vicifons,
    "lasla-conllup": extract_lasla,
    "epidoc-xml": extract_epidoc,
    "txt": extract_plaintext,
    "tess": extract_tess,
    "tei-xml": extract_tei_xml,
}


def extractor_for(source_id: str):
    """The extractor for a source, or None if its format has none yet."""
    entry = load_manifest_entry(source_id)
    return FORMAT_EXTRACTORS.get(entry.get("format"))


# Tiers that may become corpus text. `eval` is excluded on purpose and the
# exclusion is load-bearing: the UD treebanks are CoNLL-U, exactly like
# DigilibLT, so dispatching on format alone would happily extract the held-out
# evaluation data straight into the training corpus. That is the one mistake
# that would quietly invalidate every number this project reports, and the
# source-keyed table it replaced only avoided it by accident.
CORPUS_TIERS = {"green", "amber"}


def extractable_sources() -> List[str]:
    """Sources that may become corpus text and declare a readable format.

    Manifest order, not alphabetical: the manifest groups sources by role and
    that grouping is more useful to read than the alphabet.
    """
    out = []
    for entry in load_manifest()["sources"]:
        if not entry.get("include", True):
            continue
        if entry.get("tier") not in CORPUS_TIERS:
            continue
        if FORMAT_EXTRACTORS.get(entry.get("format")):
            out.append(entry["id"])
    return out


if __name__ == "__main__":
    sys.exit(main())

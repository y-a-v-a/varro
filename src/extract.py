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
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"

XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
CTS_NS = "{http://chs.harvard.edu/xmlns/cts}"

GREEK_LANGS = {"grc", "greek", "gre"}

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


def localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


# ---------------------------------------------------------------------------
# TEI text walking
# ---------------------------------------------------------------------------

class Stats(Counter):
    """Counts of editorial interventions, so the pipeline reports what it did."""


def _choice(elem: ET.Element, stats: Stats, greek: str) -> str:
    """<choice> holds competing readings. Prefer regularised / corrected."""
    children = {localname(c.tag): c for c in elem}
    for preferred, rejected in (("reg", "orig"), ("corr", "sic")):
        if preferred in children:
            if rejected in children:
                stats[f"choice:{preferred}_over_{rejected}"] += 1
            return _walk(children[preferred], stats, greek)
    # No recognised preference — take the first child rather than lose text.
    for child in elem:
        return _walk(child, stats, greek)
    return elem.text or ""


def _walk(elem: ET.Element, stats: Stats, greek: str) -> str:
    tag = localname(elem.tag)

    rend = elem.get("rend")
    if rend in DROP_REND:
        stats[f"dropped:rend:{rend}"] += 1
        return " "

    if tag in DROP_SUBTREE:
        stats[f"dropped:{tag}"] += 1
        return ""
    if tag in DROP_EMPTY_JOIN:
        return ""
    if tag in DROP_EMPTY_SPACE:
        return " "
    if tag == "milestone":
        # <milestone unit="paragraph"/> is a structural break; others (section,
        # card, line numbering) are reference points inside running text.
        return "\n\n" if elem.get("unit") == "paragraph" else " "
    if tag == "choice":
        return _choice(elem, stats, greek)

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
                    _walk(c, stats, greek) + (c.tail or "") for c in elem)
                return f" <GRAECUM> {' '.join(inner.split())} </GRAECUM> "
            # greek == "keep": fall through and treat as ordinary text
        elif not lang.startswith("lat"):
            stats[f"dropped:lang:{lang}"] += 1
            return " "

    parts: List[str] = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        parts.append(_walk(child, stats, greek))
        if child.tail:
            parts.append(child.tail)

    text = "".join(parts)
    if tag in PARA:
        return f"\n\n{text}\n\n"
    if tag in BLOCK:
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

    CSEL also ships whole-volume files under Volumes/; those duplicate this
    per-work tree and are deliberately ignored.
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

    return records, stats


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
    for key in sorted(k for k in stats if k.startswith(("dropped:", "choice:"))):
        print(f"  {key:<34} {stats[key]:>9,}")
    for key in ("greek_passages", "greek_inline_runs", "dropped_english_words",
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
FORMAT_EXTRACTORS = {
    "tei-capitains": extract_capitains,
    "conllu": extract_conllu,
    "edcs-json": extract_edcs,
    "mediawiki-xml": extract_vicifons,
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

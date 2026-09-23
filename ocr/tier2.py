#!/usr/bin/env python3
"""Tier 2: real scanner output against transcriptions people have checked.

    python ocr/tier2.py scan                # what the dump holds, per scanned book
    python ocr/tier2.py fetch               # the scans' own OCR layers -> data/raw/
    python ocr/tier2.py build               # -> bench-tier2.{dev,test}.jsonl, locked
    python ocr/bench.py run unigram lii --tier 2

WHERE THE PAIRS COME FROM
    Latin Wikisource transcribes books page by page against a scan. Each page
    lives in the `Pagina:` namespace, carries a proofreading status -- 3 is
    "proofread", 4 is "validated" by a second person -- and is tied to one
    page of one DjVu or PDF on Wikimedia Commons. That file carries the text
    layer its scanner produced, Google's or the Internet Archive's, and the
    Commons API returns it page by page.

    So both halves exist already, aligned to the page, and neither is ours:
    the truth is in data/raw/vicifons, fetched and pinned like every other
    source, and the noise is whatever a production OCR engine actually did to
    that page. Nothing is simulated and no OCR engine is run here.

    src/extract.py reads only namespace 0, so `Pagina:` text reached the
    training corpus only where the same work arrived by another road. That is
    checked, not assumed: see CONTAMINATION below.

FROM A PAGE TO PASSAGES
    A page of OCR has things its transcription leaves out -- the running
    head, the page number, footnotes, an apparatus -- so the two are not
    aligned whole. The transcription is cut into passages of sixty words and
    each is LOCATED in the page's OCR by approximate matching
    (metrics.locate), which may begin and end anywhere. What it finds is that
    passage as the scanner read it.

    A passage is kept only if the match is believable. One that differs from
    its OCR by more than MAX_CER has usually not been found at all -- the
    page was a table, the scan was of another edition's pagination, the
    transcriber skipped a column -- and keeping it would score repairers on
    the alignment's mistakes. This biases Tier 2 towards text a scanner read
    tolerably, and the build prints how much was dropped and why, because a
    benchmark that quietly keeps only the easy cases should at least say so.

BOTH SIDES GO THROUGH src/normalize.py
    The model lives in the corpus's spelling: u for v, i for j, no ligatures,
    no diacritics, line-end hyphens rejoined. Truth and OCR are both folded
    before anything is compared, so a repairer is neither blamed nor credited
    for `v`, and what the pipeline already repairs is not counted as the
    model's work.

CONTAMINATION
    Held out here cannot mean held out by author; these are whatever books
    Wikisource volunteers chose. Instead every passage is checked against the
    training split: if any run of SHINGLE consecutive words occurs there, the
    model may have read this text in another edition, and the passage is
    dropped. Counted and reported, per book.

DEV AND TEST ARE DIFFERENT BOOKS.
"""

import argparse
import bz2
import hashlib
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from hashlib import blake2b
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import bench  # noqa: E402
import metrics  # noqa: E402
from normalize import normalize  # noqa: E402

DUMP = ROOT / "data" / "raw" / "vicifons" / "lawikisource-latest-pages-articles.xml.bz2"
RAW = ROOT / "data" / "raw" / "wikisource-scan-ocr"
PAGES = bench.OUT / "tier2-pages.jsonl"
LOCK = HERE / "tier2.lock.json"
API = "https://commons.wikimedia.org/w/api.php"
AGENT = "varro-ocr-eval/0.1 (https://github.com/y-a-v-a/varro; evaluation of OCR repair)"

WIDTH = 60             # words per passage, as in Tier 1
MAX_CER = 0.25         # beyond this the passage was not found, it was guessed
MIN_KNOWN = 0.6        # share of true words the lexicon knows: a language filter
SHINGLE = 8            # consecutive words shared with training = contaminated
MIN_LEVEL = 3
MIN_PAGES = 12         # a book with fewer usable pages is not worth a request
# No book may be more than this many passages. Small on purpose: a model run
# costs two seconds a passage, and a benchmark is worth more for being a
# hundred books wide than for being deep in any one of them.
PER_BOOK = 24

# Bands of measured character error, standing in for Tier 1's injected
# levels. The first is the control: passages the scanner read perfectly, on
# which anything a repairer changes is harm.
BANDS = ((0.0, "0"), (0.02, "<2%"), (0.06, "2-6%"), (1.0, ">6%"))


def band(cer: float) -> str:
    for ceiling, label in BANDS:
        if cer <= ceiling:
            return label
    return BANDS[-1][1]


# ---------------------------------------------------------------------------
# The dump
# ---------------------------------------------------------------------------

def dump_pages() -> Iterator[Tuple[str, str]]:
    """(title, wikitext) for every page of the `Pagina:` namespace."""
    title, ns, buf = "", "", None
    with bz2.open(DUMP, "rt", encoding="utf-8") as fh:
        for line in fh:
            if buf is not None:
                buf.append(line)
                if "</text>" in line:
                    if ns == "104":
                        yield title, html.unescape("".join(buf).split("</text>")[0])
                    buf = None
                continue
            if "<title>" in line:
                title = html.unescape(re.search(r"<title>(.*?)</title>", line).group(1))
            elif "<ns>" in line:
                ns = re.search(r"<ns>(\d+)</ns>", line).group(1)
            elif "<text" in line and ns == "104":
                rest = line.split(">", 1)[1]
                if "</text>" in rest:
                    yield title, html.unescape(rest.split("</text>")[0])
                elif not line.rstrip().endswith("/>"):
                    buf = [rest]


QUALITY = re.compile(r'<pagequality level="(\d)"')
NOINCLUDE = re.compile(r"<noinclude>.*?</noinclude>", re.S)
REF = re.compile(r"<ref[^>]*?/>|<ref[^>]*>.*?</ref>", re.S)
COMMENT = re.compile(r"<!--.*?-->", re.S)
TAG = re.compile(r"</?[a-zA-Z][^>]*>")
LINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]")
INNER_TEMPLATE = re.compile(r"\{\{([^{}]*)\}\}")

# Templates that show their FIRST argument on this page: a word broken across
# two pages is given as {{Pt|what is printed here|the whole word}}.
FIRST_ARG = {"pt", "hws", "hwe", "hyphenated word start", "hyphenated word end",
             "indent", "is"}
# Templates that stand for letters. Vicifons transcribes a long s as {{s}} and
# an abbreviation sign as the letters it abbreviates; read as "nothing", each
# one cut a true word in two and the benchmark counted the scanner's correct
# reading of it as damage.
LETTERS = {"s": "s", "ls": "s", "ae": "ae", "oe": "oe", "et": "et", "um": "um",
           "que": "que", "us": "us", "rum": "rum"}
# Templates that print nothing belonging to the text: line and verse numbers,
# page references, sidenotes, rules, spacing, Greek (which normalize.py would
# strip from both sides anyway).
SILENT = {"ref", "refl", "anchor", "ancora", "nop", "gap", "rule", "regula",
          "clear", "dhr", "page break", "pagequality", "rh", "runninghead", "pg",
          "versus", "vbibliae", "bar", "r", "numvers", "ks§", "ks-", "§", "em",
          "!!", "iwpage", "left sidenote", "right sidenote", "vermilio", "graeca",
          "sidenote", "c1"}


def _template(match: "re.Match[str]") -> str:
    parts = match.group(1).split("|")
    name = parts[0].strip().lower()
    if name in LETTERS:
        return LETTERS[name]
    args = [p for p in parts[1:] if "=" not in p]
    if name in SILENT or not args:
        return " "
    return args[0] if name in FIRST_ARG else args[-1]


# The scanner's own mark for a word broken at a line end, and the space some
# engines (and some printers) put before punctuation. Both are typography.
# Neither is a misreading, no language model is needed to undo them, and
# leaving them in made a quarter of all "damaged" words on real scans
# `conueniat ,` for `conueniat,` -- so both sides are tidied the same way
# before anything is compared, like the orthography.
BROKEN_LINE = re.compile(r"¬\s*")
SPACE_BEFORE = re.compile(r"\s+([,;:.?!)])")
SPACE_AFTER = re.compile(r"([(])\s+")


def tidy(text: str) -> str:
    text = BROKEN_LINE.sub("", text)
    text = SPACE_BEFORE.sub(r"\1", text)
    return SPACE_AFTER.sub(r"\1", text)


def clean_wikitext(text: str) -> str:
    """The words a transcriber typed from the page, without the markup.

    Headers, footers and footnote markers are wrapped in <noinclude> by
    Wikisource convention -- they are on the page but not part of the work --
    and go first. Whatever markup survives this is caught later: a passage
    with a brace or a pipe left in it is dropped rather than trusted.
    """
    text = COMMENT.sub(" ", text)
    text = NOINCLUDE.sub(" ", text)
    text = REF.sub(" ", text)
    for _ in range(6):                      # templates nest
        new = INNER_TEMPLATE.sub(_template, text)
        if new == text:
            break
        text = new
    text = LINK.sub(r"\1", text)
    text = TAG.sub(" ", text)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"^[=]+(.*?)[=]+\s*$", r"\1", text, flags=re.M)
    text = re.sub(r"^[:;*#]+", "", text, flags=re.M)
    text = text.replace("&nbsp;", " ")
    # A table row or a leftover pipe: drop the line, not the page.
    text = "\n".join(l for l in text.split("\n")
                     if not l.lstrip().startswith(("{|", "|", "!", "|}")))
    return text


def cmd_scan(args) -> int:
    books: Dict[str, Counter] = defaultdict(Counter)
    kept = 0
    bench.OUT.mkdir(parents=True, exist_ok=True)
    with PAGES.open("w", encoding="utf-8") as out:
        for title, text in dump_pages():
            m = QUALITY.search(text)
            level = int(m.group(1)) if m else 0
            name, _, number = title.split(":", 1)[1].rpartition("/")
            if not number.isdigit():
                continue
            books[name][level] += 1
            if level >= MIN_LEVEL:
                out.write(json.dumps({"book": name, "page": int(number),
                                      "level": level, "wikitext": text},
                                     ensure_ascii=False) + "\n")
                kept += 1
    ranked = sorted(books.items(), key=lambda kv: -(kv[1][3] + kv[1][4]))
    print(f"{len(books)} scanned books, {kept:,} pages at level {MIN_LEVEL} or above\n")
    print(f"  {'validated':>9} {'proofread':>9}  book")
    for name, c in ranked[:args.top]:
        print(f"  {c[4]:>9} {c[3]:>9}  {name}")
    print(f"\nwrote {PAGES.relative_to(ROOT)}")
    return 0


# ---------------------------------------------------------------------------
# The scans' own OCR
# ---------------------------------------------------------------------------

def raw_path(book: str) -> Path:
    return RAW / (re.sub(r"[^A-Za-z0-9._-]+", "_", book)[:120] + ".json")


def _plain(node: Any) -> Any:
    """The API wraps every map as [{name, value}, ...]; undo that."""
    if isinstance(node, list) and node and all(
            isinstance(x, dict) and set(x) == {"name", "value"} for x in node):
        return {x["name"]: _plain(x["value"]) for x in node}
    return node


def fetch_layer(book: str) -> Optional[Dict[str, Any]]:
    query = urllib.parse.urlencode({
        "action": "query", "titles": f"File:{book}", "prop": "imageinfo",
        "iiprop": "sha1|size|mime|metadata|timestamp", "format": "json"})
    request = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = json.load(response)
    page = next(iter(data["query"]["pages"].values()))
    info = (page.get("imageinfo") or [None])[0]
    if not info:
        return None
    meta = _plain(info.get("metadata") or [])
    text = meta.get("text") if isinstance(meta, dict) else None
    if isinstance(text, list):
        text = dict(enumerate(text))
    if not text:
        return None
    return {"book": book, "sha1": info.get("sha1"), "mime": info.get("mime"),
            "uploaded": info.get("timestamp"),
            "pages": {str(k): v for k, v in text.items() if isinstance(v, str)}}


def cmd_fetch(args) -> int:
    """One request a book, a pause between them, nothing fetched twice.

    Written under data/raw/ and never edited there, like every fetched
    source: the OCR is evidence, and cleaning it up in place would destroy
    the thing being measured.
    """
    counts: Counter = Counter()
    with PAGES.open(encoding="utf-8") as fh:
        for line in fh:
            counts[json.loads(line)["book"]] += 1
    RAW.mkdir(parents=True, exist_ok=True)
    wanted = [b for b, n in counts.most_common() if n >= MIN_PAGES]
    print(f"{len(wanted)} books with at least {MIN_PAGES} checked pages")
    for k, book in enumerate(wanted, 1):
        path = raw_path(book)
        if path.exists():
            continue
        try:
            layer = fetch_layer(book)
        except Exception as error:          # a missing file must not stop the rest
            print(f"  [{k}/{len(wanted)}] {book}: {error}")
            time.sleep(2)
            continue
        if layer is None:
            print(f"  [{k}/{len(wanted)}] {book}: no text layer")
            path.write_text(json.dumps({"book": book, "pages": {}}), encoding="utf-8")
        else:
            path.write_text(json.dumps(layer, ensure_ascii=False), encoding="utf-8")
            print(f"  [{k}/{len(wanted)}] {book}: {len(layer['pages'])} pages of OCR")
        time.sleep(1.5)
    return 0


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

LEFTOVER = re.compile(r"[{}|<>=_\\]")


def passages_of(truth_page: str, ocr_page: str, lex: set
                ) -> Tuple[List[Dict[str, Any]], Counter]:
    why: Counter = Counter()
    found: List[Dict[str, Any]] = []
    words = truth_page.split()
    for slot in range(len(words) // WIDTH):
        chunk = words[slot * WIDTH:(slot + 1) * WIDTH]
        truth = " ".join(chunk)
        if LEFTOVER.search(truth):
            why["markup left in the transcription"] += 1
            continue
        if not bench.latinate(chunk):
            why["not running prose"] += 1
            continue
        known = sum(1 for w in chunk if metrics.core(w) in lex) / len(chunk)
        if known < MIN_KNOWN:
            why["not Latin enough"] += 1
            continue
        # Look where the passage ought to be, give or take a third of the
        # page: headers and footnotes shift things, but not by more than
        # that, and the search is three times quicker for not reading it all.
        centre = (slot + 0.5) * WIDTH / max(1, len(words)) * len(ocr_page)
        reach = len(ocr_page) // 3 + 2 * len(truth)
        lo, hi = max(0, int(centre - reach)), min(len(ocr_page), int(centre + reach))
        start, end, distance = metrics.locate(truth, ocr_page[lo:hi])
        start, end = start + lo, end + lo
        cer = distance / len(truth)
        if cer > MAX_CER:
            why["not found in the OCR"] += 1
            continue
        noisy = " ".join(ocr_page[start:end].split())
        found.append({"slot": slot, "truth": truth, "noisy": noisy,
                      "cer": metrics.align(truth, noisy)[0] / len(truth)})
    return found, why


def contaminated(passages: List[Dict[str, Any]], corpus: str) -> set:
    """Ids of passages sharing a run of SHINGLE words with the training split.

    The shingles of the BENCHMARK are hashed into a set, which is small, and
    the training split is streamed past it once -- the other way round needs
    a set of a hundred million entries.
    """
    def keys(words: List[str]) -> Iterator[int]:
        for k in range(len(words) - SHINGLE + 1):
            yield hash(" ".join(words[k:k + SHINGLE]))

    def plain(text: str) -> List[str]:
        return [w for w in (metrics.core(w) for w in text.split()) if w]

    owner: Dict[int, List[str]] = defaultdict(list)
    for p in passages:
        for key in keys(plain(p["truth"])):
            owner[key].append(p["id"])
    hit: set = set()
    path = bench.PROCESSED / f"{corpus}.train.jsonl"
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            for key in keys(plain(json.loads(line)["text"])):
                if key in owner:
                    hit.update(owner[key])
    return hit


def cmd_build(args) -> int:
    lex = set(bench.lexicon(args.corpus))
    by_book: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with PAGES.open(encoding="utf-8") as fh:
        for line in fh:
            page = json.loads(line)
            by_book[page["book"]].append(page)

    passages: List[Dict[str, Any]] = []
    why: Counter = Counter()
    sources: Dict[str, Any] = {}
    for book, pages in sorted(by_book.items()):
        path = raw_path(book)
        if not path.exists():
            continue
        layer = json.loads(path.read_text(encoding="utf-8"))
        if not layer.get("pages"):
            continue
        took = 0
        # Validated pages first, then proofread ones, each in a seeded order;
        # and no further than four times the cap (contamination can take most
        # of a book), because locating a passage is
        # the slow step and a book can only contribute PER_BOOK of them.
        for page in sorted(pages, key=lambda p: (-p["level"], bench._order(
                args.seed, f"{book}/{p['page']}"))):
            if took >= 4 * PER_BOOK:
                break
            ocr = layer["pages"].get(str(page["page"] - 1))
            if not ocr or not ocr.strip():
                why["page has no OCR"] += 1
                continue
            truth_page = tidy(normalize(clean_wikitext(page["wikitext"])))
            ocr_page = " ".join(tidy(normalize(ocr)).split())
            got, reasons = passages_of(" ".join(truth_page.split()), ocr_page, lex)
            why.update(reasons)
            for g in got:
                g.update({"id": f"{book}/{page['page']}#{g.pop('slot')}",
                          "doc": book, "author": book, "quality": page["level"],
                          "source": "wikisource-scan-ocr", "license": "CC-BY-SA-4.0",
                          "period": None, "classes": None})
                passages.append(g)
                took += 1
        if took:
            sources[book] = {"sha1": layer.get("sha1"), "uploaded": layer.get("uploaded")}
        print(f"  {took:>5} passages  {book}", flush=True)

    print(f"\n{len(passages):,} passages located; checking them against "
          f"{args.corpus}.train ...", flush=True)
    dirty = contaminated(passages, args.corpus)
    per_book = Counter(p["doc"] for p in passages if p["id"] in dirty)
    passages = [p for p in passages if p["id"] not in dirty]
    why[f"shares {SHINGLE} words running with the training split"] = len(dirty)

    # Cap each book, by a seeded order, so one long validated book is not the
    # benchmark. Then books -- not passages -- are dealt to dev and test.
    passages.sort(key=lambda p: bench._order(args.seed, p["id"]))
    seen: Counter = Counter()
    capped = []
    for p in passages:
        if seen[p["doc"]] < PER_BOOK:
            seen[p["doc"]] += 1
            capped.append(p)
    why["over the per-book cap"] = len(passages) - len(capped)

    splits: Dict[str, List[Dict[str, Any]]] = {"dev": [], "test": []}
    for p in capped:
        p["split"] = "dev" if bench._order(args.seed, p["doc"]) % 3 == 0 else "test"
        p["level"] = band(p["cer"])
        splits[p["split"]].append(p)

    hashes = {}
    for split, rows in splits.items():
        body = "".join(json.dumps(p, ensure_ascii=False, sort_keys=True) + "\n"
                       for p in rows)
        hashes[split] = hashlib.sha256(body.encode()).hexdigest()[:16]
        (bench.OUT / f"bench-tier2.{split}.jsonl").write_text(body, encoding="utf-8")

    print("\n  dropped, and why:")
    for reason, n in why.most_common():
        print(f"    {n:>6,}  {reason}")
    if per_book:
        print("  contaminated, by book: " +
              ", ".join(f"{b[:30]} {n}" for b, n in per_book.most_common(8)))
    for split, rows in splits.items():
        chars = sum(len(p["truth"]) for p in rows) or 1
        cer = sum(p["cer"] * len(p["truth"]) for p in rows) / chars
        print(f"\n  {split}: {len(rows)} passages from {len({p['doc'] for p in rows})} "
              f"books, CER {cer:.2%}, sha256 {hashes[split]}")
        for label, n in Counter(p["level"] for p in rows).most_common():
            print(f"      {label:>5}  {n}")

    lock = {"corpus": args.corpus, "seed": args.seed, "width": WIDTH,
            "max_cer": MAX_CER, "shingle": SHINGLE, "min_level": MIN_LEVEL,
            "per_book": PER_BOOK, "sha256": hashes,
            "passages": {s: len(r) for s, r in splits.items()},
            "dropped": dict(why), "scans": sources,
            "dump": hashlib.sha256(str(DUMP.stat().st_size).encode()).hexdigest()[:16]}
    if LOCK.exists() and not args.relock:
        old = json.loads(LOCK.read_text(encoding="utf-8")).get("sha256")
        if old != hashes:
            sys.exit(f"\nthis build does not match {LOCK.relative_to(ROOT)} "
                     f"({old} locked, {hashes} built); --relock if that is meant")
    LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    print(f"\nwrote {LOCK.relative_to(ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--top", type=int, default=40)
    sub.add_parser("fetch")
    b = sub.add_parser("build")
    b.add_argument("--corpus", default="corpus-b")
    b.add_argument("--seed", type=int, default=20260917)
    b.add_argument("--relock", action="store_true")
    args = ap.parse_args()
    return {"scan": cmd_scan, "fetch": cmd_fetch, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

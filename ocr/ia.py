#!/usr/bin/env python3
"""archive.org OCR against the same proofread pages as Tier 2.

    python ocr/ia.py find       # which Tier 2 scans came from archive.org -> ia-candidates.json
    python ocr/ia.py fetch      # each item's current OCR, page by page -> data/raw/
    python ocr/ia.py compare    # old (Commons) vs current (archive.org) OCR, same passages
    python ocr/ia.py build      # -> bench-ia.{current,then,now}.jsonl, locked
    python ocr/ia.py overlap    # how much of each book the corpus already has
    python ocr/bench.py run identity unigram fim --tier 3 --split current

WHY
    The goal is archive.org text in the corpus. Before repairing any of it,
    two things need knowing: whether Tier 2's figures carry over to what
    archive.org serves today, and whether archive.org's own re-OCR (it has
    re-read many older scans with Tesseract) already beats what repair can
    do. Both can be read off books Wikisource has proofread, because many
    Commons scans were uploaded FROM archive.org, and the proofread page is
    the truth for both readings.

FIND
    A Commons file page names its source; an archive.org identifier in it
    makes the book a candidate. Then a check that is not an assumption: are
    the Commons text layer and archive.org's current text the SAME reading?
    Fifteen random pages, one run of eight words from each, looked for
    verbatim. Most match -- the Commons file is archive.org's own DjVu, OCR
    and all, and Tier 2 was already measuring archive.org -- and the few that
    do not are scans archive.org has read again since.

COMPARE
    For each proofread page, the archive.org page is found by content, not by
    number (archive.org counts covers and colour cards the Commons file may
    not), within a window around where it ought to be. Passages are cut and
    located exactly as tier2.py does, in BOTH readings, and only passages
    found in both are scored, so the two error rates are over the same text.
    Books whose readings are identical are the control: their two rates must
    agree, or the alignment is wrong.

    No model runs here. Nothing is tuned on these books; several are in
    Tier 2's test split.

BUILD: A BENCHMARK THE MODEL MAY BE RUN ON
    Three splits, cut and located as tier2.py cuts and locates:

      current  archive.org's OCR as it is served today, every usable book
      then     the older reading (the Commons layer), books archive.org has
               since read again, passages found in both readings
      now      the same passages as `then`, in archive.org's current reading

    `then` against `now` is re-reading; each against itself repaired is
    repair; `now` repaired is both. `current` is the wider question: how
    much does repair add to archive.org text as it can be downloaded.

    What keeps it honest:
      - Tier 2's TEST books are left out whole. Seeing the model's score on
        an exam book before the exam is sat is sitting it early.
      - Passages Tier 2's dev split holds are left out; the rules were tuned
        on them.
      - Contamination is checked as tier2.py checks it.
      - Nothing is tuned here: bench.py runs tier 3 under Tier 2's rules and
        Tier 2's fitted channel. That channel was learned from older OCR, so
        on Tesseract's errors it is, if anything, a handicap.
"""

import argparse
import hashlib
import html
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import bench  # noqa: E402
import metrics  # noqa: E402
import tier2  # noqa: E402
from normalize import normalize  # noqa: E402

CANDIDATES = HERE / "ia-candidates.json"
RAW = ROOT / "data" / "raw" / "archive-org-ocr"
IA = "https://archive.org"
AGENT = tier2.AGENT

IA_ID = re.compile(r"archive\.org/(?:details|stream|download)/([A-Za-z0-9._-]+)")
IA_TEMPLATE = re.compile(r"\{\{\s*IA\s*\|\s*([A-Za-z0-9._-]+)", re.I)
# Identifiers Commons pages cite that are not the scan: a collection, and the
# copyright catalogues a 20th-century book's file page quotes as evidence.
NOT_A_SCAN = re.compile(r"^(rogerwagner|catalogofcopyrig.*|MN\d+ucmf.*)$")

SAMPLE = 15            # pages checked for "same reading"
RUN = 8                # words in each checked run
WINDOW = 12            # archive.org pages searched either side of the expected one
MIN_OVERLAP = 0.3      # word overlap below which a page was not found
PER_BOOK = 40          # passages any one book may give the benchmark
LOCK = HERE / "ia.lock.json"


def get(url: str, timeout: int = 120) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def words(text: str) -> List[str]:
    return re.findall(r"[a-zæœ]{3,}", text.lower())


def commons_layer(book: str) -> Dict[str, str]:
    path = tier2.raw_path(book)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("pages") or {}


# ---------------------------------------------------------------------------
# find
# ---------------------------------------------------------------------------

def source_ids(books: List[str]) -> Dict[str, List[str]]:
    """archive.org identifiers each Commons file page cites, fifty a request."""
    found: Dict[str, List[str]] = {}
    for k in range(0, len(books), 50):
        chunk = books[k:k + 50]
        query = urllib.parse.urlencode({
            "action": "query", "format": "json", "prop": "revisions",
            "rvprop": "content", "rvslots": "main",
            "titles": "|".join(f"File:{b}" for b in chunk)})
        pages = json.loads(get(f"{tier2.API}?{query}"))["query"]["pages"].values()
        for page in pages:
            text = (page.get("revisions") or [{}])[0].get("slots", {}) \
                .get("main", {}).get("*", "")
            ids = set(IA_ID.findall(text)) | set(IA_TEMPLATE.findall(text))
            found[page["title"][5:]] = sorted(i for i in ids if not NOT_A_SCAN.match(i))
        time.sleep(1)
    return found


def item(identifier: str) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(get(f"{IA}/metadata/{identifier}", timeout=60))
    except Exception:
        return None
    return data if data.get("metadata") else None


def first(value: Any) -> Any:
    return value[0] if isinstance(value, list) else value


def ocr_file(data: Dict[str, Any], suffix: str, book: str) -> Optional[str]:
    """The item's file with this suffix; with several, the one named like the book."""
    names = [f["name"] for f in data.get("files", []) if f["name"].endswith(suffix)]
    if len(names) > 1:
        stem = re.sub(r"\.(djvu|pdf)$", "", book).replace(" ", "_")
        names.sort(key=lambda n: (not n.startswith(stem), n))
    return names[0] if names else None


def djvu_txt(identifier: str, name: str) -> str:
    return get(f"{IA}/download/{identifier}/{urllib.parse.quote(name)}").decode("utf-8", "replace")


def same_reading(book: str, identifier: str, name: str, seed: int) -> Optional[str]:
    """"k/n": how many of n sampled Commons runs archive.org has verbatim."""
    layer = [t for t in commons_layer(book).values() if len(words(t)) > 3 * RUN]
    if not layer:
        return None
    theirs = " ".join(words(djvu_txt(identifier, name)))
    rng = random.Random(seed)
    sample = rng.sample(layer, min(SAMPLE, len(layer)))
    hits = 0
    for page in sample:
        w = words(page)
        mid = len(w) // 2
        hits += " ".join(w[mid:mid + RUN]) in theirs
    return f"{hits}/{len(sample)}"


def cmd_find(args) -> int:
    pages: Counter = Counter()
    levels: Counter = Counter()
    with tier2.PAGES.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            pages[row["book"]] += 1
            levels[row["book"]] += row["level"] >= 4
    split: Dict[str, str] = {}
    for s in ("dev", "test"):
        with (bench.OUT / f"bench-tier2.{s}.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                split[json.loads(line)["doc"]] = s
    books = sorted(b for b in pages if tier2.raw_path(b).exists())
    print(f"{len(books)} Tier 2 books with an OCR layer; reading their Commons pages ...")
    ids = source_ids(books)

    rows = []
    for book, cited in sorted(ids.items()):
        for identifier in cited:
            data = item(identifier)
            row: Dict[str, Any] = {"book": book, "ia": identifier,
                                   "pages": pages[book], "validated": levels[book],
                                   "tier2": split.get(book)}
            if data is None:
                row["status"] = "missing"
            else:
                md = data["metadata"]
                row.update({"title": first(md.get("title")),
                            "date": first(md.get("date") or md.get("year")),
                            "engine": first(md.get("ocr")) or
                            (f"archive.org module {md['ocr_module_version']}"
                             if md.get("ocr_module_version") else None),
                            "djvu_txt": ocr_file(data, "_djvu.txt", book),
                            "djvu_xml": ocr_file(data, "_djvu.xml", book)})
                if not row["djvu_xml"] or not row["djvu_txt"]:
                    row["status"] = "no OCR"
                else:
                    try:
                        same = same_reading(book, identifier, row["djvu_txt"], args.seed)
                    except Exception as error:      # archive.org 5xx: say so, go on
                        row["status"] = f"fetch failed ({error})"
                        rows.append(row)
                        print(f"  {row['status']:<15}        {identifier}", flush=True)
                        continue
                    row["same"] = same
                    hits, n = map(int, same.split("/")) if same else (0, 0)
                    row["status"] = ("no Commons text" if not n else
                                     "same reading" if hits >= 0.8 * n else
                                     "read again")
            rows.append(row)
            print(f"  {row['status']:<15} {row.get('same') or '':>5}  {identifier:<45} {book[:50]}",
                  flush=True)
            time.sleep(0.5)
    CANDIDATES.write_text(json.dumps({"seed": args.seed, "candidates": rows},
                                     indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n{Counter(r['status'] for r in rows)}\nwrote {CANDIDATES.relative_to(ROOT)}")
    return 0


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

OBJECT = re.compile(r"<OBJECT\b.*?</OBJECT>", re.S)
LINE = re.compile(r"<LINE>(.*?)</LINE>", re.S)
WORD = re.compile(r"<WORD[^>]*>(.*?)</WORD>", re.S)


def pages_of_djvu_xml(xml: str) -> List[str]:
    """The text of each page, lines kept, in the order the item gives them."""
    out = []
    for obj in OBJECT.findall(xml):
        lines = [" ".join(html.unescape(w) for w in WORD.findall(line))
                 for line in LINE.findall(obj)]
        out.append("\n".join(l for l in lines if l))
    return out


def raw_path(identifier: str) -> Path:
    return RAW / (re.sub(r"[^A-Za-z0-9._-]+", "_", identifier) + ".json")


def candidates() -> List[Dict[str, Any]]:
    if not CANDIDATES.exists():
        sys.exit(f"missing {CANDIDATES.relative_to(ROOT)} -- run `python ocr/ia.py find`")
    return json.loads(CANDIDATES.read_text(encoding="utf-8"))["candidates"]


def cmd_fetch(args) -> int:
    """Once per item, into data/raw/, never edited there."""
    RAW.mkdir(parents=True, exist_ok=True)
    for row in candidates():
        if not row.get("djvu_xml"):
            continue
        path = raw_path(row["ia"])
        if path.exists():
            continue
        try:
            xml = get(f"{IA}/download/{row['ia']}/{urllib.parse.quote(row['djvu_xml'])}",
                      timeout=300).decode("utf-8", "replace")
        except Exception as error:
            print(f"  {row['ia']}: {error}")
            continue
        pages = pages_of_djvu_xml(xml)
        path.write_text(json.dumps({"ia": row["ia"], "file": row["djvu_xml"],
                                    "engine": row.get("engine"), "pages": pages},
                                   ensure_ascii=False), encoding="utf-8")
        print(f"  {len(pages):>5} pages  {row['ia']}", flush=True)
        time.sleep(1.5)
    return 0


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def best_page(truth: str, theirs: List[set], expected: int) -> Optional[int]:
    """The archive.org page sharing most words with the truth, near `expected`."""
    want = set(words(truth))
    if not want:
        return None
    lo, hi = max(0, expected - WINDOW), min(len(theirs), expected + WINDOW + 1)
    # On a tie (a page printed twice, a repeated title) the nearer page wins.
    score, _, where = max(((len(want & theirs[k]) / len(want), -abs(k - expected), k)
                           for k in range(lo, hi)), default=(0.0, 0, None))
    return where if score >= MIN_OVERLAP else None


def cmd_compare(args) -> int:
    lex = set(bench.lexicon(args.corpus))
    truth_pages: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with tier2.PAGES.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            truth_pages[row["book"]].append(row)

    report = []
    print(f"  {'status':<13} {'passages':>8} {'old CER ↓':>10} {'new CER ↓':>10} "
          f"{'change':>7}  engine now / book")
    for row in candidates():
        path = raw_path(row["ia"])
        if not path.exists():
            continue
        ours = commons_layer(row["book"])
        theirs_raw = json.loads(path.read_text(encoding="utf-8"))["pages"]
        theirs = [" ".join(tier2.tidy(normalize(p)).split()) for p in theirs_raw]
        bags = [set(words(p)) for p in theirs]
        scale = len(theirs) / max(1, len(ours))
        rng = random.Random(f"{args.seed}/{row['book']}")
        pages = truth_pages[row["book"]][:]
        rng.shuffle(pages)

        pairs = []
        for page in pages[:args.pages]:
            old = ours.get(str(page["page"] - 1))
            if not old or not old.strip():
                continue
            truth = " ".join(tier2.tidy(normalize(tier2.clean_wikitext(page["wikitext"]))).split())
            at = best_page(truth, bags, int((page["page"] - 1) * scale))
            if at is None:
                continue
            a, _ = tier2.passages_of(truth, " ".join(tier2.tidy(normalize(old)).split()), lex)
            b, _ = tier2.passages_of(truth, theirs[at], lex)
            new = {p["truth"]: p for p in b}
            pairs += [(p, new[p["truth"]]) for p in a if p["truth"] in new]
        if not pairs:
            print(f"  {row['status']:<13} {0:>8}  {row['book'][:60]}")
            continue
        chars = sum(len(p["truth"]) for p, _ in pairs)
        old_cer = sum(p["cer"] * len(p["truth"]) for p, _ in pairs) / chars
        new_cer = sum(q["cer"] * len(q["truth"]) for _, q in pairs) / chars
        change = (new_cer - old_cer) / old_cer if old_cer else 0.0
        report.append({"book": row["book"], "ia": row["ia"], "status": row["status"],
                       "tier2": row.get("tier2"), "engine": row.get("engine"),
                       "passages": len(pairs), "cer_old": old_cer, "cer_new": new_cer})
        print(f"  {row['status']:<13} {len(pairs):>8} {old_cer:>10.2%} {new_cer:>10.2%} "
              f"{change:>+7.0%}  {str(row.get('engine') or '?')[:26]} / {row['book'][:40]}",
              flush=True)

    for status in ("same reading", "read again"):
        rows = [r for r in report if r["status"] == status]
        if not rows:
            continue
        n = sum(r["passages"] for r in rows)
        old = sum(r["cer_old"] * r["passages"] for r in rows) / n
        new = sum(r["cer_new"] * r["passages"] for r in rows) / n
        print(f"\n  {status}: {len(rows)} books, {n} passages, "
              f"CER old {old:.2%} -> new {new:.2%}  (↓ lower is better)")
    out = ROOT / "reports" / "ocr-ia-compare.json"
    out.write_text(json.dumps({"seed": args.seed, "pages_per_book": args.pages,
                               "corpus": args.corpus, "books": report},
                              indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def tier2_ids() -> Dict[str, str]:
    """Passage id -> Tier 2 split, and book -> split, for everything Tier 2 holds."""
    held: Dict[str, str] = {}
    for s in ("dev", "test"):
        with (bench.OUT / f"bench-tier2.{s}.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                p = json.loads(line)
                held[p["id"]] = s
                held.setdefault(p["doc"], s)
    return held


def reading(pages: List[str]) -> List[str]:
    return [" ".join(tier2.tidy(normalize(p or "")).split()) for p in pages]


def cmd_build(args) -> int:
    lex = set(bench.lexicon(args.corpus))
    held = tier2_ids()
    truth_pages: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with tier2.PAGES.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            truth_pages[row["book"]].append(row)

    found: List[Dict[str, Any]] = []
    why: Counter = Counter()
    for row in candidates():
        path = raw_path(row["ia"])
        if not path.exists() or row["status"] not in (
                "same reading", "read again", "no Commons text"):
            continue
        if held.get(row["book"]) == "test":
            why["book is in Tier 2's test split"] += 1
            continue
        now = reading(json.loads(path.read_text(encoding="utf-8"))["pages"])
        now_bags = [set(words(p)) for p in now]
        layer = commons_layer(row["book"])
        then = reading([layer.get(str(k)) for k in range(
            max((int(k) for k in layer), default=-1) + 1)])
        then_bags = [set(words(p)) for p in then]
        scale = len(now) / max(1, len(then) or len(now))
        pages = sorted(truth_pages[row["book"]], key=lambda p: (
            -p["level"], bench._order(args.seed, f"{row['book']}/{p['page']}")))
        took = 0
        for page in pages:
            if took >= 4 * PER_BOOK:
                break
            truth = " ".join(tier2.tidy(normalize(tier2.clean_wikitext(page["wikitext"]))).split())
            at = best_page(truth, now_bags, int((page["page"] - 1) * scale))
            if at is None:
                why["page not found in archive.org's reading"] += 1
                continue
            got, reasons = tier2.passages_of(truth, now[at], lex)
            why.update(reasons)
            old: Dict[str, Dict[str, Any]] = {}
            if row["status"] == "read again" and then:
                back = best_page(truth, then_bags, page["page"] - 1)
                if back is not None:
                    old = {p["truth"]: p for p in tier2.passages_of(truth, then[back], lex)[0]}
            for g in got:
                pid = f"{row['book']}/{page['page']}#{g.pop('slot')}"
                if pid in held:
                    why["passage is in Tier 2"] += 1
                    continue
                g.update({"id": pid, "doc": row["book"], "author": row["book"],
                          "ia": row["ia"], "status": row["status"],
                          "engine": row.get("engine"), "quality": page["level"],
                          "source": "archive-org-ocr", "license": "CC-BY-SA-4.0",
                          "period": None, "classes": None})
                if g["truth"] in old:
                    g["then"] = old[g["truth"]]["noisy"]
                    g["then_cer"] = old[g["truth"]]["cer"]
                found.append(g)
                took += 1
        print(f"  {took:>5} passages  {row['status']:<15} {row['book'][:60]}", flush=True)

    print(f"\n{len(found):,} passages located; checking them against "
          f"{args.corpus}.train ...", flush=True)
    dirty = tier2.contaminated(found, args.corpus)
    found = [p for p in found if p["id"] not in dirty]
    why[f"shares {tier2.SHINGLE} words running with the training split"] = len(dirty)

    found.sort(key=lambda p: bench._order(args.seed, p["id"]))
    seen: Counter = Counter()
    kept = []
    for p in found:
        if seen[p["doc"]] < PER_BOOK:
            seen[p["doc"]] += 1
            kept.append(p)
    why["over the per-book cap"] = len(found) - len(kept)

    def as_split(p: Dict[str, Any], name: str, noisy: str, cer: float) -> Dict[str, Any]:
        q = {k: v for k, v in p.items() if k not in ("then", "then_cer")}
        q.update({"noisy": noisy, "cer": cer, "split": name, "level": tier2.band(cer)})
        return q

    splits = {
        "current": [as_split(p, "current", p["noisy"], p["cer"]) for p in kept],
        "then": [as_split(p, "then", p["then"], p["then_cer"]) for p in kept if "then" in p],
        "now": [as_split(p, "now", p["noisy"], p["cer"]) for p in kept if "then" in p],
    }
    hashes = {}
    for name, rows in splits.items():
        body = "".join(json.dumps(p, ensure_ascii=False, sort_keys=True) + "\n" for p in rows)
        hashes[name] = hashlib.sha256(body.encode()).hexdigest()[:16]
        (bench.OUT / f"bench-ia.{name}.jsonl").write_text(body, encoding="utf-8")

    print("\n  dropped, and why:")
    for reason, n in why.most_common():
        print(f"    {n:>6,}  {reason}")
    for name, rows in splits.items():
        chars = sum(len(p["truth"]) for p in rows) or 1
        cer = sum(p["cer"] * len(p["truth"]) for p in rows) / chars
        print(f"\n  {name}: {len(rows)} passages from {len({p['doc'] for p in rows})} books, "
              f"CER {cer:.2%}, sha256 {hashes[name]}")

    lock = {"corpus": args.corpus, "seed": args.seed, "per_book": PER_BOOK,
            "sha256": hashes, "passages": {s: len(r) for s, r in splits.items()},
            "dropped": dict(why), "candidates": hashlib.sha256(
                CANDIDATES.read_bytes()).hexdigest()[:16]}
    if LOCK.exists() and not args.relock:
        old = json.loads(LOCK.read_text(encoding="utf-8")).get("sha256")
        if old != hashes:
            sys.exit(f"\nthis build does not match {LOCK.relative_to(ROOT)} "
                     f"({old} locked, {hashes} built); --relock if that is meant")
    LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    print(f"\nwrote {LOCK.relative_to(ROOT)}")
    return 0


# ---------------------------------------------------------------------------
# overlap
# ---------------------------------------------------------------------------

def cmd_overlap(args) -> int:
    """Per book, the share of proofread passages already in the training split.

    Every proofread page, cut into WIDTH-word passages -- no OCR involved --
    and checked as tier2.contaminated checks: a run of SHINGLE words in
    common means the model has probably read it. What is left is what
    archive.org could add.
    """
    rows = {r["book"]: r for r in candidates()}
    passages: List[Dict[str, Any]] = []
    with tier2.PAGES.open(encoding="utf-8") as fh:
        for line in fh:
            page = json.loads(line)
            if page["book"] not in rows:
                continue
            w = tier2.tidy(normalize(tier2.clean_wikitext(page["wikitext"]))).split()
            for k in range(len(w) // tier2.WIDTH):
                passages.append({"id": f"{page['book']}/{page['page']}#{k}",
                                 "doc": page["book"],
                                 "truth": " ".join(w[k * tier2.WIDTH:(k + 1) * tier2.WIDTH])})
    dirty = tier2.contaminated(passages, args.corpus)
    total = Counter(p["doc"] for p in passages)
    seen = Counter(p["doc"] for p in passages if p["id"] in dirty)
    out = []
    for book in sorted(total, key=lambda b: -seen[b] / total[b]):
        out.append({"book": book, "status": rows[book]["status"],
                    "tier2": rows[book].get("tier2"), "passages": total[book],
                    "in_corpus": seen[book]})
        print(f"  {seen[book] / total[book]:6.0%} {total[book]:6}  {book[:60]}")
    n, d = sum(total.values()), sum(seen.values())
    print(f"\n  {d:,} of {n:,} passages ({d / n:.1%}) share {tier2.SHINGLE} words "
          f"running with {args.corpus}.train")
    path = ROOT / "reports" / "ocr-ia-overlap.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=20260923)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("find")
    sub.add_parser("fetch")
    c = sub.add_parser("compare")
    c.add_argument("--corpus", default="corpus-b")
    c.add_argument("--pages", type=int, default=40, help="proofread pages sampled per book")
    b = sub.add_parser("build")
    b.add_argument("--corpus", default="corpus-b")
    b.add_argument("--relock", action="store_true")
    o = sub.add_parser("overlap")
    o.add_argument("--corpus", default="corpus-b")
    args = ap.parse_args()
    return {"find": cmd_find, "fetch": cmd_fetch, "compare": cmd_compare,
            "build": cmd_build, "overlap": cmd_overlap}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

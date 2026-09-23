#!/usr/bin/env python3
"""Build the OCR-repair benchmark once, then judge repairers against it.

    python ocr/bench.py build                         # held-out text -> dev + test
    python ocr/bench.py run identity unigram lii      # on dev, which is for tuning
    python ocr/bench.py run unigram lii --split test  # the number that is quoted
    python ocr/bench.py compare reports/ocr-A.json reports/ocr-B.json

WHERE THE TRUTH COMES FROM
    The held-out split of the corpus the model trained on, and only records
    labelled `text_provenance: transcribed`. Held out by author, so the model
    has not read these passages or anything else by who wrote them. And
    transcribed only, because the truth has to BE true: CSEL is machine-
    corrected OCR, and scoring a repairer against it would mark the repairer
    wrong every time it fixed an error the truth still carries.

DEV AND TEST ARE DIFFERENT AUTHORS
    Everything a repairer learns -- its channel, its margin, how far it looks
    right -- it learns from dev. Test is for the run that gets quoted. They
    are split by author rather than by passage so that a margin tuned to keep
    one writer's proper names intact is not then tested on the same names.

ONE PASSAGE IN FOUR IS LEFT CLEAN
    Level 0 has no damage in it, so everything a repairer changes there is
    harm and nothing else. It is the control the other levels are read
    against, and it is the most realistic one: most of any scanned page is
    right, and a tool that cannot leave right text alone is not usable on it
    whatever its recall.

THE LOCK
    ocr/bench.lock.json records the hash of each split. Two reports compare
    only if they carry the same hash, and `build` refuses to replace a locked
    benchmark without --relock, because a benchmark that drifts makes every
    earlier number unfalsifiable. The passages themselves stay under data/,
    which is never committed; they are reproducible from the seed.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from collections import Counter
from hashlib import blake2b
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import channel  # noqa: E402
import metrics  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
OUT = PROCESSED / "ocr"
LOCK = HERE / "bench.lock.json"

LEVELS = (0.0, 0.01, 0.03, 0.08)
ALPHA = re.compile(r"[A-Za-z]+")

# A form has to turn up this often to count as a word. Below it the lexicon
# fills with the corpus's own typos, and a typo in the lexicon is a wrong
# answer the repairers are licensed to give.
LEXICON_FLOOR = 3

STATS = {
    "cer_reduction": metrics.cer_reduction,
    "recall": metrics.recall,
    "recall_real_word": lambda c: metrics.recall(c, "real:"),
    "recall_nonword": lambda c: metrics.recall(c, "nonword:"),
    "precision": metrics.precision,
    "harm_per_1000": metrics.harm,
}


# ---------------------------------------------------------------------------
# Lexicon
# ---------------------------------------------------------------------------

def capitalised(corpus: str) -> Dict[str, int]:
    """How often each lexicon form carries an initial capital in training.

    Keyed lowercase like the lexicon, so a proposer can ask what share of a
    word's appearances were `Roma` rather than `roma`, and offer the other
    case where the corpus says the page probably had it.
    """
    cache = OUT / f"capitalised-{corpus}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    source = PROCESSED / f"{corpus}.train.jsonl"
    lex = lexicon(corpus)
    counts: Counter = Counter()
    with source.open(encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            if record.get("text_provenance") == "ocr":
                continue
            for w in ALPHA.findall(record["text"]):
                if w[0].isupper():
                    counts[w.lower()] += 1
    kept = {w: n for w, n in counts.items() if w in lex}
    cache.write_text(json.dumps(kept), encoding="utf-8")
    return kept


def lexicon(corpus: str) -> Dict[str, int]:
    """Word counts from the TRAINING split, minus what came from OCR.

    Training, because it is what the model also saw: the unigram baseline and
    the model then differ in what they do with the text, not in how much of
    it they were given. Minus `ocr` provenance, because a scanno that CSEL
    repeats three times would otherwise be an attested word -- the benchmark
    would count it as a silent error's disguise and the repairers would offer
    it as a correction.
    """
    cache = OUT / f"lexicon-{corpus}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    source = PROCESSED / f"{corpus}.train.jsonl"
    if not source.exists():
        sys.exit(f"missing {source.relative_to(ROOT)} -- build the corpus first")
    print(f"counting words in {source.name} (once; cached after) ...", file=sys.stderr)
    counts: Counter = Counter()
    with source.open(encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            if record.get("text_provenance") == "ocr":
                continue
            counts.update(ALPHA.findall(record["text"].lower()))
    kept = {w: n for w, n in counts.items() if n >= LEXICON_FLOOR}
    OUT.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(kept), encoding="utf-8")
    return kept


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _order(seed: int, key: str) -> int:
    return int.from_bytes(blake2b(f"{seed}:{key}".encode(), digest_size=8).digest(), "big")


def latinate(words: List[str]) -> bool:
    """Running prose, not a table, a list of numerals or an index."""
    plain = sum(1 for w in words if metrics.core(w).isalpha())
    return plain >= 0.85 * len(words)


def draw(corpus: str, n: int, width: int, per_doc: int, seed: int) -> List[Dict[str, Any]]:
    import random
    path = PROCESSED / f"{corpus}.validation.jsonl"
    if not path.exists():
        sys.exit(f"missing {path.relative_to(ROOT)}")
    found: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            if record.get("text_provenance") != "transcribed":
                continue
            words = record["text"].split()
            slots = len(words) // width
            if slots < 2:
                continue
            rng = random.Random(f"{seed}:{record['id']}")
            for slot in rng.sample(range(slots), min(per_doc, slots)):
                passage = words[slot * width:(slot + 1) * width]
                if not latinate(passage):
                    continue
                found.append({
                    "id": f"{record['id']}#{slot}",
                    "doc": record["id"],
                    "author": record.get("author"),
                    "period": record.get("period"),
                    "source": record.get("source"),
                    "license": record.get("license"),
                    "truth": " ".join(passage),
                })
    found.sort(key=lambda p: _order(seed, p["id"]))
    return found[:n]


def draw_clean(corpus: str, width: int, seed: int, groups: set, taken: set,
               per_doc: int, n: int) -> List[Dict[str, Any]]:
    """More undamaged passages, from DEV authors only, for measuring harm.

    Harm is a rare event: two words in a thousand is seven words in dev's
    sixty clean passages, and a rule with eight margins can be tuned to spare
    any seven words. Clean text costs nothing to get more of -- it needs no
    noise, only truth -- so the rule is tuned against ten times as much, and
    the limit then means something. Dev authors only, so test stays unseen.
    """
    import random
    path = PROCESSED / f"{corpus}.validation.jsonl"
    found: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            if record.get("text_provenance") != "transcribed":
                continue
            if (record.get("author") or record["id"]) not in groups:
                continue
            words = record["text"].split()
            slots = [k for k in range(len(words) // width)
                     if f"{record['id']}#{k}" not in taken]
            rng = random.Random(f"{seed}:clean:{record['id']}")
            for slot in rng.sample(slots, min(per_doc, len(slots))):
                passage = words[slot * width:(slot + 1) * width]
                if latinate(passage):
                    text = " ".join(passage)
                    found.append({"id": f"{record['id']}#{slot}", "doc": record["id"],
                                  "author": record.get("author"),
                                  "period": record.get("period"),
                                  "source": record.get("source"),
                                  "license": record.get("license"),
                                  "split": "clean", "level": 0.0, "truth": text,
                                  "noisy": text, "classes": [[] for _ in passage],
                                  "cer": 0.0})
    found.sort(key=lambda p: _order(seed, "clean:" + p["id"]))
    return found[:n]


def cmd_build(args) -> int:
    import random
    lex = lexicon(args.corpus)
    passages = draw(args.corpus, args.n, args.words, args.per_doc, args.seed)
    if len(passages) < args.n:
        print(f"only {len(passages)} usable passages (asked for {args.n})",
              file=sys.stderr)

    splits: Dict[str, List[Dict[str, Any]]] = {"dev": [], "test": []}
    for p in passages:
        # By author where there is one, so dev and test share no writer.
        group = p["author"] or p["doc"]
        splits["dev" if _order(args.seed, group) % 4 == 0 else "test"].append(p)

    hashes: Dict[str, str] = {}
    OUT.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        for k, p in enumerate(rows):
            p["split"] = split
            p["level"] = LEVELS[k % len(LEVELS)]
            noisy, edits = channel.corrupt(
                p["truth"], p["level"], random.Random(f"{args.seed}:{p['id']}"))
            p["noisy"] = noisy
            p["classes"] = channel.classes_by_word(p["truth"], edits)
            p["cer"] = metrics.align(p["truth"], noisy)[0] / len(p["truth"])
        body = "".join(json.dumps(p, ensure_ascii=False, sort_keys=True) + "\n"
                       for p in rows)
        hashes[split] = hashlib.sha256(body.encode()).hexdigest()[:16]
        (OUT / f"bench-{args.corpus}.{split}.jsonl").write_text(body, encoding="utf-8")

    dev_groups = {p["author"] or p["doc"] for p in splits["dev"]}
    clean = draw_clean(args.corpus, args.words, args.seed, dev_groups,
                       {p["id"] for p in passages}, args.clean_per_doc, args.clean)
    splits["clean"] = clean
    body = "".join(json.dumps(p, ensure_ascii=False, sort_keys=True) + "\n" for p in clean)
    hashes["clean"] = hashlib.sha256(body.encode()).hexdigest()[:16]
    (OUT / f"bench-{args.corpus}.clean.jsonl").write_text(body, encoding="utf-8")

    lock = {
        "corpus": args.corpus, "seed": args.seed, "n": args.n,
        "words": args.words, "per_doc": args.per_doc, "levels": list(LEVELS),
        "channel": channel.table_fingerprint(),
        "lexicon_words": len(lex),
        "passages": {s: len(r) for s, r in splits.items()},
        "sha256": hashes,
    }
    if LOCK.exists() and not args.relock:
        old = json.loads(LOCK.read_text(encoding="utf-8"))
        was = old.get("sha256", {})
        # A split the old lock never had is an addition, not a drift.
        if old.get("corpus") == args.corpus and any(
                was[k] != hashes.get(k) for k in was):
            sys.exit(
                f"this build does not match {LOCK.relative_to(ROOT)}:\n"
                f"  locked {old.get('sha256')}\n  built  {hashes}\n"
                "Either the corpus, the channel or the arguments changed. Every "
                "report measured on the\nlocked benchmark stops being comparable "
                "the moment it is replaced; pass --relock if that\nis what you mean.")
    LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for split, rows in splits.items():
        print(f"{split:<5} {len(rows):>4} passages  sha256 {hashes[split]}")
        if split == "clean":
            print(f"      undamaged, dev authors only, "
                  f"{sum(len(p['truth'].split()) for p in rows):,} words")
            continue
        for level in LEVELS:
            at = [p for p in rows if p["level"] == level]
            chars = sum(len(p["truth"]) for p in at)
            cer = sum(p["cer"] * len(p["truth"]) for p in at) / max(1, chars)
            print(f"      start-rate {level:<5}  {len(at):>4} passages  "
                  f"measured CER {cer:6.2%}")
    print(f"\nlexicon {len(lex):,} forms seen >= {LEXICON_FLOOR} times\n"
          f"wrote {LOCK.relative_to(ROOT)}")
    return 0


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def load_split(corpus: str, split: str, tier: int = 1) -> Tuple[List[Dict[str, Any]], str]:
    """A split and its hash, refused if it is not the one the lock records."""
    if tier == 2:
        path, lock, how = OUT / f"bench-tier2.{split}.jsonl", HERE / "tier2.lock.json", \
            "python ocr/tier2.py build"
    else:
        path, lock, how = OUT / f"bench-{corpus}.{split}.jsonl", LOCK, \
            "python ocr/bench.py build"
    if not path.exists():
        sys.exit(f"missing {path.relative_to(ROOT)} -- run `{how}`")
    body = path.read_text(encoding="utf-8")
    digest = hashlib.sha256(body.encode()).hexdigest()[:16]
    if lock.exists():
        locked = json.loads(lock.read_text(encoding="utf-8")).get("sha256", {})
        if locked.get(split) not in (None, digest):
            sys.exit(f"{path.name} is {digest} but the lock says {locked[split]} -- "
                     "rebuild, do not report against a benchmark nobody else has")
    return [json.loads(line) for line in body.splitlines()], digest


def commit() -> Dict[str, Any]:
    def git(*a: str) -> str:
        return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True,
                              text=True).stdout.strip()
    return {"commit": git("rev-parse", "--short", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--", "ocr", "src", "demos"))}


def judge(system, rows: List[Dict[str, Any]], lex: set,
          progress: bool) -> Dict[str, Any]:
    tallies: List[Counter] = []
    scores: List[float] = []
    labels: List[Optional[str]] = []
    examples: Dict[str, List[Dict[str, str]]] = {"broken": [], "miscorrected": []}
    unscored = 0
    began = time.time()

    for k, p in enumerate(rows, 1):
        repaired = system.repair(p["noisy"])
        tallies.append(metrics.tally(p["truth"], p["noisy"], repaired, lex,
                                     p.get("classes")))
        before = metrics.project(p["truth"], p["noisy"])
        after = metrics.project(p["truth"], repaired)
        for n, r in zip(before, after):
            o = metrics.outcome(n, r)
            if o in examples and len(examples[o]) < 40:
                examples[o].append({"truth": n.word, "noisy": n.text,
                                    "repaired": r.text, "passage": p["id"]})
        s = system.suspicion(p["noisy"])
        if s is None:
            unscored += 1
        else:
            scores += s
            labels += metrics.word_labels(p["truth"], p["noisy"], lex)
        if progress and k % 25 == 0:
            print(f"    {k}/{len(rows)}  {time.time() - began:5.0f}s", file=sys.stderr)

    total: Counter = Counter()
    for t in tallies:
        total.update(t)

    # Intervals per level as well as overall, because the question a level
    # answers is a yes or no -- is repair worth running on text this clean? --
    # and a point estimate of +4% on 60 passages cannot answer it.
    by_level = {}
    for level in sorted({p["level"] for p in rows}, key=str):
        mine = [t for p, t in zip(rows, tallies) if p["level"] == level]
        part: Counter = Counter()
        for t in mine:
            part.update(t)
        by_level[str(level)] = {
            **metrics.summarise(part),
            "passages": len(mine),
            "interval95": metrics.bootstrap(mine, {
                "cer_reduction": metrics.cer_reduction,
                "harm_per_1000": metrics.harm}),
        }

    # Cleanest first. Tier 1's levels are numbers and sort themselves; Tier
    # 2's are labels, and "2-6%" sorts before "<2%".
    by_level = dict(sorted(by_level.items(),
                           key=lambda kv: kv[1]["cer_before"] or 0.0))

    classes = sorted({key.split(":")[1] for key in total if key.startswith("class:")})
    by_class = {c: {"damaged": metrics.damaged(total, f"class:{c}:"),
                    "recall": metrics.recall(total, f"class:{c}:")} for c in classes}

    detection = None
    if scores:
        real_only = [(s, l == "real") for s, l in zip(scores, labels) if l != "nonword"]
        detection = {
            "words": len(scores),
            "prevalence": sum(l is not None for l in labels) / len(labels),
            "average_precision": metrics.average_precision(
                scores, [l is not None for l in labels]),
            # Non-word damage taken out of the pool altogether, so that what
            # is left is the question a dictionary cannot answer at all: of
            # words that are all attested, which ones are wrong HERE?
            "real_word_prevalence": (sum(x for _, x in real_only) / len(real_only)
                                     if real_only else None),
            "real_word_average_precision": metrics.average_precision(
                [s for s, _ in real_only], [x for _, x in real_only]),
            "passages_unscored": unscored,
        }

    return {
        "system": system.name,
        "seconds": round(time.time() - began, 1),
        "counts": {k: v for k, v in sorted(total.items()) if ":" not in k},
        "overall": metrics.summarise(total),
        "interval95": metrics.bootstrap(tallies, STATS),
        "by_level": by_level,
        "by_class": by_class,
        "detection": detection,
        "examples": examples,
        "_tallies": tallies,
        "_system": system,
    }


def pack(tallies: List[Counter]) -> Dict[str, Any]:
    """Per-passage counts, one line a passage, so reports can be compared later.

    A paired interval needs both systems' counts passage by passage, and the
    model run that produced them takes minutes where the comparison takes
    seconds. Kept as a line of integers each because a report is committed,
    and thirty keys times 560 passages pretty-printed is 17,000 lines of diff.
    """
    keys = sorted({k for t in tallies for k in t if not k.startswith("class:")})
    return {"keys": keys,
            "rows": [" ".join(str(t[k]) for k in keys) for t in tallies]}


def unpack(packed: Dict[str, Any]) -> List[Counter]:
    return [Counter(dict(zip(packed["keys"], map(int, row.split()))))
            for row in packed["rows"]]


def cmd_compare(args) -> int:
    """A minus B, paired, from two reports already on disk."""
    a, b = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (args.a, args.b))
    if a["benchmark"] != b["benchmark"]:
        sys.exit(f"not the same benchmark:\n  {a['benchmark']}\n  {b['benchmark']}")
    diff = metrics.bootstrap(unpack(a["per_passage"]), STATS,
                             against=unpack(b["per_passage"]))
    print(f"  {a['system']}  minus  {b['system']}\n  {a['benchmark']['split']} split, "
          f"{a['benchmark']['passages']} passages, paired 95% intervals\n")
    for name, lo_hi in diff.items():
        pct = name != "harm_per_1000"
        point = (a["overall"][name] or 0) - (b["overall"][name] or 0)
        print(f"    {name:<18} {fmt(point, pct).strip():>8}   " + (
            "--" if lo_hi is None else
            f"{fmt(lo_hi[0], pct).strip()} .. {fmt(lo_hi[1], pct).strip()}"))
    return 0


def fmt(x: Optional[float], pct: bool = True) -> str:
    if x is None:
        return "     --"
    return f"{x:7.1%}" if pct else f"{x:7.2f}"


def show(results: List[Dict[str, Any]]) -> None:
    print(f"\n  {'system':<28} {'CER cut':>7} {'recall':>7} {'real-wd':>7} "
          f"{'nonword':>7} {'precis.':>7} {'broken/k':>8} {'silent/k':>16}")
    for r in results:
        o = r["overall"]
        silent = (f"{o['silent_per_1000_before'] or 0:6.2f} ->"
                  f"{o['silent_per_1000_after'] or 0:6.2f}")
        print(f"  {r['system']:<28} {fmt(o['cer_reduction'])} {fmt(o['recall'])} "
              f"{fmt(o['recall_real_word'])} {fmt(o['recall_nonword'])} "
              f"{fmt(o['precision'])} {fmt(o['harm_per_1000'], False):>8} {silent:>16}")
    print("\n  by damage level: CER before -> after, and words broken per 1,000 standing")
    for r in results:
        cells = []
        for level, s in r["by_level"].items():
            cells.append(f"[{level}] {(s['cer_before'] or 0):5.1%}->"
                         f"{(s['cer_after'] or 0):5.1%} "
                         f"({fmt(s['harm_per_1000'], False).strip()})")
        print(f"  {r['system']:<28} " + "   ".join(cells))
    for r in results:
        d = r["detection"]
        if d:
            print(f"\n  detection, {r['system']}: average precision "
                  f"{fmt(d['average_precision']).strip()} (floor "
                  f"{fmt(d['prevalence']).strip()}); on real-word damage alone "
                  f"{fmt(d['real_word_average_precision']).strip()} (floor "
                  f"{fmt(d['real_word_prevalence']).strip()})")


def cmd_run(args) -> int:
    from evaluate import write_report
    from repairers import make

    lex = lexicon(args.corpus)
    caps = capitalised(args.corpus)
    known = set(lex)
    # A repairer learns its channel from the dev split OF THE TIER IT IS RUN
    # ON. A channel fitted to Tier 1's invented noise and carried to real
    # scans would be answering a question about the wrong scanner.
    dev, _ = load_split(args.corpus, "dev", args.tier)
    rows, digest = load_split(args.corpus, args.split, args.tier)
    if args.limit:
        rows = rows[:args.limit]
    pairs = [(p["truth"], p["noisy"]) for p in dev]

    if args.split == "test":
        print("TEST split. This is the run that gets quoted: nothing may be tuned "
              "against what it prints.\n", file=sys.stderr)

    results = []
    for spec in args.system:
        system = make(spec, tuned=not args.untuned, tier=args.tier)
        system.fit(pairs, lex, caps)
        print(f"  {system.name}: {len(rows)} passages from {args.split}", file=sys.stderr)
        results.append(judge(system, rows, known, not args.quiet))

    show(results)

    # Every system against the first-listed one that is not identity, over
    # the same resampled passages.
    base = next((r for r in results if r["system"] != "identity"), None)
    for r in results:
        r["against"] = None
        if base is not None and r is not base and r["system"] != "identity":
            r["against"] = {"system": base["system"],
                            "difference95": metrics.bootstrap(
                                r["_tallies"], STATS, against=base["_tallies"])}
            d = r["against"]["difference95"]
            print(f"\n  {r['system']} minus {base['system']}, paired 95% intervals:")
            for name in ("recall_real_word", "recall_nonword", "harm_per_1000"):
                lo_hi = d[name]
                pct = name != "harm_per_1000"
                print(f"    {name:<18} " + ("--" if lo_hi is None else
                      f"{fmt(lo_hi[0], pct).strip()} .. {fmt(lo_hi[1], pct).strip()}"))

    if not args.no_report:
        for r in results:
            tallies = r.pop("_tallies")
            r_system = r.pop("_system", None)
            r["per_passage"] = pack(tallies)
            slug = re.sub(r"[^a-z0-9]+", "-", r["system"].lower()).strip("-")
            stem = "ocr-" if args.tier == 1 else "ocr-t2-"
            path = write_report(f"{stem}{slug}-{args.split}", {
                "benchmark": {"tier": args.tier, "split": args.split, "sha256": digest,
                              "passages": len(tallies), "limit": args.limit or None,
                              "channel": channel.table_fingerprint()
                              if args.tier == 1 else "real"},
                "rule": r_system.rule._asdict() if hasattr(r_system, "rule") else None,
                "code": commit(),
                **r,
            }, corpus=args.corpus)
            print(f"\n  wrote {path.relative_to(ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="draw held-out passages and damage them")
    b.add_argument("--corpus", default="corpus-b")
    b.add_argument("--n", type=int, default=800)
    b.add_argument("--words", type=int, default=60, help="words per passage")
    b.add_argument("--per-doc", type=int, default=4,
                   help="most passages any one document may contribute")
    b.add_argument("--clean", type=int, default=600,
                   help="extra undamaged passages from dev authors, for harm")
    b.add_argument("--clean-per-doc", type=int, default=12)
    b.add_argument("--seed", type=int, default=20260917)
    b.add_argument("--relock", action="store_true",
                   help="replace a locked benchmark; earlier reports stop comparing")

    r = sub.add_parser("run", help="judge one or more repairers")
    r.add_argument("system", nargs="+",
                   help="identity | unigram | lii | lii-left | fim, with optional "
                        ":checkpoint and @right-context-words")
    r.add_argument("--corpus", default="corpus-b")
    r.add_argument("--split", choices=("dev", "test", "clean"), default="dev")
    r.add_argument("--untuned", action="store_true",
                   help="ignore ocr/rules.json; decide on the raw sum")
    r.add_argument("--tier", type=int, choices=(1, 2), default=1,
                   help="1: synthetic damage. 2: real scans (ocr/tier2.py build)")
    r.add_argument("--limit", type=int, help="first N passages only; a smoke test")
    r.add_argument("--no-report", action="store_true")
    r.add_argument("--quiet", action="store_true")

    c = sub.add_parser("compare", help="A minus B, paired, from two reports")
    c.add_argument("a")
    c.add_argument("b")

    args = ap.parse_args()
    return {"build": cmd_build, "run": cmd_run, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

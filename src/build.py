#!/usr/bin/env python3
"""Assemble the final corpus from the interim records.

Reads data/interim/*.jsonl, drops what dedupe.py marked, optionally restricts to
a period, and writes data/processed/ — the corpus everything downstream trains
on, plus a summary of exactly what went into it.

HELD-OUT SPLIT
    Validation is held out as WHOLE WORKS, never random lines. A line-level
    split leaks: the model sees the surrounding paragraphs of every held-out
    line, so validation loss measures memorisation of a context it already has
    rather than generalisation to unseen text. Whole works also let perplexity
    be read per author, which is the interesting number for this project.

    Works are chosen deterministically, stratified by period so the held-out set
    is not accidentally all late-antique prose, and capped in size so a single
    long work cannot consume the budget.

TRACKS
    Track A draws only on free-licence sources and is the corpus that gets
    published. Track B adds non-commercial material and is research-only. The
    track is recorded in the summary and in every record, and provenance.py
    --check will refuse a Track A build that contains anything else.

Usage:
    python src/build.py --report                 # what would be built
    python src/build.py                          # Track A, all periods
    python src/build.py --track B                # research build
    python src/build.py --period CLASSICAL       # the Latin-0 corpus
    python src/build.py --val-fraction 0.05
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"

# No single work may take more than this share of the validation budget, so the
# split cannot come down to one outlier (Livy alone is 515k words).
MAX_VAL_SHARE = 0.25
SEED = 20260815

# Licence tiers each track may contain. Mirrors fetch.py's TRACKS; duplicated
# deliberately so a build cannot succeed just because a fetch happened to.
TRACKS = {"A": {"green"}, "B": {"green", "amber"}}


def source_tiers() -> Dict[str, str]:
    """Licence tier per source id, read from the manifest.

    The corpus records carry a licence string but not a tier, and the tier is
    what decides track membership — so it is looked up here rather than trusted
    from whatever happened to be written into the records.
    """
    try:
        import yaml
    except ImportError:
        sys.exit("PyYAML required — pip install -r requirements.txt")
    with (ROOT / "manifest" / "sources.yaml").open(encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh)
    return {s["id"]: s.get("tier", "unknown") for s in manifest["sources"]}


def load() -> List[Dict[str, Any]]:
    records = []
    for path in sorted(INTERIM.glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            records.extend(json.loads(line) for line in fh)
    return records


def choose_validation(records: List[Dict[str, Any]], fraction: float) -> set:
    """Pick whole works for validation, stratified by period."""
    total = sum(r["n_words"] for r in records)
    budget = total * fraction
    by_period: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_period[rec.get("period") or "UNASSIGNED"].append(rec)

    rng = random.Random(SEED)
    chosen: set = set()
    for period, group in sorted(by_period.items()):
        share = budget * sum(r["n_words"] for r in group) / total
        cap = share * MAX_VAL_SHARE
        # Prefer mid-sized works: big enough to give a stable perplexity,
        # small enough not to swallow the budget.
        eligible = [r for r in group if 2000 <= r["n_words"] <= max(cap, 2000)]
        rng.shuffle(eligible)
        used = 0.0
        for rec in eligible:
            if used + rec["n_words"] > share:
                continue
            chosen.add(rec["id"])
            used += rec["n_words"]
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="analyse only, write nothing")
    ap.add_argument("--track", choices=sorted(TRACKS), default="A",
                    help="A = redistributable (default); B = research only")
    ap.add_argument("--period", action="append",
                    help="restrict to a period (repeatable); default is all")
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--min-words", type=int, default=0,
                    help="drop records shorter than this")
    args = ap.parse_args()

    records = load()
    if not records:
        sys.exit(f"no records in {INTERIM} — run the extract/enrich/normalize/dedupe stages")

    tiers = source_tiers()
    allowed = TRACKS[args.track]

    kept, reasons = [], Counter()
    for rec in records:
        if rec.get("is_duplicate"):
            reasons["duplicate"] += rec["n_words"]
            continue
        tier = tiers.get(rec.get("source"), "unknown")
        if tier not in allowed:
            reasons[f"not-in-track-{args.track} ({tier})"] += rec["n_words"]
            continue
        if args.period and (rec.get("period") not in args.period):
            reasons["period-filtered"] += rec["n_words"]
            continue
        if rec["n_words"] < args.min_words:
            reasons["too-short"] += rec["n_words"]
            continue
        if not rec.get("normalization"):
            reasons["NOT-NORMALIZED"] += rec["n_words"]
        kept.append(rec)

    if not kept:
        sys.exit("nothing left after filtering")

    val_ids = choose_validation(kept, args.val_fraction)
    train = [r for r in kept if r["id"] not in val_ids]
    val = [r for r in kept if r["id"] in val_ids]

    tw = sum(r["n_words"] for r in train)
    vw = sum(r["n_words"] for r in val)

    print(f"track            {args.track} — "
          f"{'redistributable' if args.track == 'A' else 'RESEARCH ONLY, not redistributable'}")
    print(f"corpus           {len(kept):,} records, {tw + vw:,} words")
    print(f"  train          {len(train):,} records, {tw:,} words")
    print(f"  validation     {len(val):,} records, {vw:,} words "
          f"({100 * vw / (tw + vw):.1f}%)")
    if args.period:
        print(f"  period filter  {', '.join(args.period)}")
    print("\nexcluded:")
    for reason, words in reasons.most_common():
        print(f"  {reason:<18} {words:>10,} words")

    periods = Counter()
    for rec in kept:
        periods[rec.get("period") or "UNASSIGNED"] += rec["n_words"]
    print("\nperiods:")
    for period, words in periods.most_common():
        print(f"  {period:<12} {words:>10,}  {100 * words / (tw + vw):5.1f}%")

    print("\nheld-out works:")
    for rec in sorted(val, key=lambda r: -r["n_words"])[:10]:
        print(f"  {rec['n_words']:>7,}w  {(rec['period'] or '?'):<10} "
              f"{(rec['author'] or '?')[:26]:<26} {(rec['work'] or '?')[:34]}")
    if len(val) > 10:
        print(f"  ... and {len(val) - 10} more")

    if args.report:
        print("\n--report: nothing written")
        return 0

    PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("validation", val)):
        out = PROCESSED / f"corpus.{name}.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for rec in rows:
                rec = dict(rec, split=name, track=args.track)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\nwrote {out.relative_to(ROOT)} — {len(rows):,} records, "
              f"{out.stat().st_size / 1e6:.1f} MB")

    summary = {
        "track": args.track,
        "redistributable": args.track == "A",
        "records": len(kept),
        "words": tw + vw,
        "train_records": len(train), "train_words": tw,
        "validation_records": len(val), "validation_words": vw,
        "period_filter": args.period,
        "periods": dict(periods),
        "licenses": dict(Counter(r.get("license") for r in kept)),
        "sources": dict(Counter(r.get("source") for r in kept)),
        "seed": SEED,
    }
    path = PROCESSED / "corpus.summary.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

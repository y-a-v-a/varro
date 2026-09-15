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
from hashlib import blake2b
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


# Text provenance: how a source's text came to be text. Not a licence question
# and not a quality one — it is about whether a later model would be learning
# from an earlier model's output.
#
# The corpus already contains OCR-derived text: Open Greek and Latin's CSEL is
# machine-corrected OCR, about 6M words of it, and has been since the third
# source landed. Labelling it is what makes "what does this model look like
# without OCR-derived text?" a question that can be answered rather than
# argued about.
#
# `model-corrected` is the one that needs a policy rather than a label. Training
# a model on text an earlier model corrected is a feedback loop, and repeated
# without guardrails it narrows the corpus toward whatever the model already
# believed — the symptom being that everything looks cleaner while growing less
# various. --exclude-provenance keeps it out entirely; --max-share admits it in
# a stated proportion. Neither is a default: the default is to report the mix
# and leave the decision to a person.
PROVENANCE_CLASSES = ("transcribed", "ocr", "model-corrected", "unknown")

# What a record with no `text_provenance` counts as. Conservative on purpose:
# text whose origin was never recorded is not evidence that a human typed it.
UNKNOWN_PROVENANCE = "unknown"


def drop_order(record: Dict[str, Any], seed: int) -> Tuple[int, int]:
    """Shortest first, ties broken by a seeded hash of the record id.

    blake2b rather than hash(): Python's is randomised per process, which
    would make two runs of the same build disagree about which records were
    shed — the one property a seed exists to provide.
    """
    digest = blake2b(f"{seed}:{record['id']}".encode(), digest_size=8).digest()
    return record["n_words"], int.from_bytes(digest, "big")


def apportion(records: List[Dict[str, Any]], caps: Dict[str, float],
              seed: int) -> Tuple[List[Dict[str, Any]], Counter]:
    """Hold each provenance class to at most its share of the corpus, by words.

    Drops whole records, smallest first, so the cap is met by shedding the
    least text. Returns what was dropped, so a build reports the shrinkage
    rather than quietly applying it.

    WHERE THE ORDER COMES FROM
        Only ties need deciding: the sort is by length, and among records of
        the same length something has to choose. That used to be a seeded
        shuffle applied before the sort — but Python's sort is stable, so the
        shuffle reached nothing except ties anyway, and the generator was
        re-seeded inside the loop so every class and every pass replayed the
        same permutation over a differently-ordered list. It read as though it
        randomised the choice; it randomised tie order, sometimes.

        Ties are now broken by a hash of the record id under the seed, which
        is what the docstring always claimed: independent of the order records
        were read in, identical on every pass, and reproducible from the seed
        alone.
    """
    if not caps:
        return records, Counter()
    dropped: Counter = Counter()
    keep_ids = {r["id"] for r in records}

    # Caps interact, so one pass is not enough: capping `ocr` counts
    # model-corrected text in its denominator, and if that text is then
    # dropped by its own cap, ocr is over again. Each pass only removes, so
    # repeating until nothing moves terminates, and the fixed point satisfies
    # every cap at once.
    for _ in range(len(caps) + 1):
        moved = False
        for cls, share in caps.items():
            if share >= 1.0:
                continue
            kept_now = [r for r in records if r["id"] in keep_ids]
            # Same default as everywhere else. A record written before the
            # field existed has no provenance, and reading that as None here
            # while reading it as "unknown" in the filter and the report meant
            # a cap on `unknown` silently matched nothing.
            in_class = [r for r in kept_now
                        if r.get("text_provenance", UNKNOWN_PROVENANCE) == cls]
            held = sum(r["n_words"] for r in in_class)
            other = sum(r["n_words"] for r in kept_now) - held
            # The budget is relative to the text NOT in this class, because
            # dropping shrinks the denominator too. Capping 500 words of 1000
            # at 20% by dropping to 200 leaves 200/700 = 29%. Solving
            # held/(other + held) <= share gives held <= share*other/(1-share).
            budget = share * other / (1.0 - share)
            if held <= budget:
                continue
            for rec in sorted(in_class, key=lambda r: drop_order(r, seed)):
                if held <= budget:
                    break
                keep_ids.discard(rec["id"])
                held -= rec["n_words"]
                dropped[cls] += rec["n_words"]
                moved = True
        if not moved:
            break
    return [r for r in records if r["id"] in keep_ids], dropped


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


def choose_validation(records: List[Dict[str, Any]], fraction: float,
                      group_by: str = "author") -> set:
    """Hold out validation, stratified by period, grouped so nothing leaks.

    Whole works are not a strong enough unit. Holding out one work by Augustine
    while training on thirty others measures interpolation to an unseen text by a
    thoroughly familiar author — which is worth knowing, but is not
    generalisation. Measured on the first version of this split: 18 of 31
    validation authors also appeared in training, covering 498,089 of 856,617
    validation words (58%).

    Grouping by author closes that. `--group-by work` restores the old behaviour
    for a cheaper, more optimistic loss signal; the two answer different
    questions and it is worth reporting both.
    """
    total = sum(r["n_words"] for r in records)
    budget = total * fraction

    def key(rec: Dict[str, Any]) -> str:
        if group_by == "author":
            return (rec.get("author") or f"~anon~{rec['id']}").strip().lower()
        return rec["id"]

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[key(rec)].append(rec)

    # A group belongs to whichever period holds most of its words, so an author
    # is never split across strata either.
    by_period: Dict[str, List[List[Dict[str, Any]]]] = defaultdict(list)
    for members in groups.values():
        weights: Counter = Counter()
        for rec in members:
            weights[rec.get("period") or "UNASSIGNED"] += rec["n_words"]
        by_period[weights.most_common(1)[0][0]].append(members)

    rng = random.Random(SEED)
    chosen: set = set()
    for period, group_list in sorted(by_period.items()):
        period_words = sum(r["n_words"] for g in group_list for r in g)
        share = budget * period_words / total
        cap = max(share * MAX_VAL_SHARE, 2000)
        eligible = [g for g in group_list
                    if 2000 <= sum(r["n_words"] for r in g) <= cap]
        rng.shuffle(eligible)
        used = 0.0
        for members in eligible:
            words = sum(r["n_words"] for r in members)
            if used + words > share:
                continue
            chosen.update(r["id"] for r in members)
            used += words
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="analyse only, write nothing")
    ap.add_argument("--track", choices=sorted(TRACKS), default="A",
                    help="A = redistributable (default); B = research only")
    ap.add_argument("--period", action="append",
                    help="restrict to a period (repeatable); default is all")
    ap.add_argument("--name", default="corpus",
                    help="output basename, so variant builds do not clobber each "
                         "other (e.g. --name classical --period CLASSICAL)")
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--group-by", choices=("author", "work"), default="author",
                    help="held-out unit: whole authors (default, no leakage) or "
                         "whole works (cheaper signal, same authors in train)")
    ap.add_argument("--exclude-provenance", action="append", default=[],
                    choices=PROVENANCE_CLASSES, metavar="CLASS",
                    help="drop a text-provenance class entirely (repeatable)")
    ap.add_argument("--max-share", action="append", default=[], metavar="CLASS=F",
                    help="cap a class at a share of the corpus, e.g. "
                         "model-corrected=0.15 (repeatable)")
    ap.add_argument("--min-words", type=int, default=0,
                    help="drop records shorter than this")
    args = ap.parse_args()

    records = load()
    if not records:
        sys.exit(f"no records in {INTERIM} — run the extract/enrich/normalize/dedupe stages")

    tiers = source_tiers()
    allowed = TRACKS[args.track]

    # dedupe resolves each track separately, so read the verdict for this one.
    # Track A's marks never depend on records outside Track A — adding a
    # research-only source must not change the published corpus.
    dup_field = "is_duplicate" if args.track == "A" else "is_duplicate_b"

    kept, reasons = [], Counter()
    for rec in records:
        if rec.get(dup_field, rec.get("is_duplicate")):
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
        if rec.get("text_provenance", UNKNOWN_PROVENANCE) in args.exclude_provenance:
            reasons[f"provenance-excluded ({rec.get('text_provenance')})"] += rec["n_words"]
            continue
        if not rec.get("normalization"):
            # Counted AND excluded. Previously this was counted and then kept,
            # which let un-normalized text into a build that claimed to be
            # normalized — the report said something was wrong and the corpus
            # shipped it anyway.
            reasons["NOT-NORMALIZED (excluded)"] += rec["n_words"]
            continue
        kept.append(rec)

    if not kept:
        sys.exit("nothing left after filtering")

    caps = {}
    for spec in args.max_share:
        if "=" not in spec:
            sys.exit(f"--max-share wants CLASS=FRACTION, got {spec!r}")
        cls, _, value = spec.partition("=")
        if cls not in PROVENANCE_CLASSES:
            sys.exit(f"--max-share: unknown class {cls!r}; "
                     f"choose from {', '.join(PROVENANCE_CLASSES)}")
        caps[cls] = float(value)
    kept, apportioned = apportion(kept, caps, SEED)
    if not kept:
        sys.exit("nothing left after apportioning")

    val_ids = choose_validation(kept, args.val_fraction, args.group_by)
    train = [r for r in kept if r["id"] not in val_ids]
    val = [r for r in kept if r["id"] in val_ids]

    tw = sum(r["n_words"] for r in train)
    vw = sum(r["n_words"] for r in val)

    print(f"track            {args.track} — "
          f"{'redistributable' if args.track == 'A' else 'RESEARCH ONLY, not redistributable'}")
    print(f"corpus           {len(kept):,} records, {tw + vw:,} words")

    # Always reported, with or without a policy. A mix nobody looks at is how
    # machine-corrected text would grow into the corpus unnoticed.
    provenance = Counter()
    for rec in kept:
        provenance[rec.get("text_provenance", UNKNOWN_PROVENANCE)] += rec["n_words"]
    print("  text provenance")
    for cls in PROVENANCE_CLASSES:
        if provenance[cls]:
            share = provenance[cls] / (tw + vw)
            capped = f"   (capped at {caps[cls]:.1%})" if cls in caps else ""
            print(f"    {cls:<16} {provenance[cls]:>12,}  {share:>6.2%}{capped}")
    for cls, words in sorted(apportioned.items()):
        print(f"    dropped to fit   {words:>12,} words of {cls}")
    print(f"  train          {len(train):,} records, {tw:,} words")
    print(f"  validation     {len(val):,} records, {vw:,} words "
          f"({100 * vw / (tw + vw):.1f}%), held out by {args.group_by}")
    # None is not an author. Counting it as one made an all-anonymous corpus
    # look 61% leaked when named-author leakage was actually zero — a metric
    # that cries wolf stops being read.
    named_train = {r.get("author") for r in train if r.get("author")}
    shared = {r.get("author") for r in val if r.get("author")} & named_train
    leaked = sum(r["n_words"] for r in val if r.get("author") in shared)
    anonymous = sum(r["n_words"] for r in val if not r.get("author"))
    if vw:
        print(f"  author overlap {len(shared)} named author(s), {leaked:,} "
              f"validation words ({100 * leaked / vw:.1f}%)")
        print(f"  anonymous      {anonymous:,} validation words "
              f"({100 * anonymous / vw:.1f}%) — each held out as its own group")
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
        out = PROCESSED / f"{args.name}.{name}.jsonl"
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
        "text_provenance": dict(provenance),
        "provenance_excluded": sorted(args.exclude_provenance),
        "provenance_caps": caps,
        "periods": dict(periods),
        "licenses": dict(Counter(r.get("license") for r in kept)),
        "sources": dict(Counter(r.get("source") for r in kept)),
        "seed": SEED,
        "group_by": args.group_by,
    }
    path = PROCESSED / f"{args.name}.summary.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

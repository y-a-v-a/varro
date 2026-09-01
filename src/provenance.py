#!/usr/bin/env python3
"""Generate the attribution notice and audit builds against the manifest.

Provenance is a standing obligation here, not a release-time chore. Two things
have to stay true continuously, and both are easy to break silently:

  1. Every source that reached the corpus is credited under its actual licence,
     with the exact version that was ingested. CC BY-SA obliges attribution on
     redistribution, and "we will write that up before release" is how projects
     end up unable to reconstruct what they trained on.

  2. A Track A build contains only redistributable material. The guard exists in
     fetch.py and build.py, but guards run at the moment of action and this one
     has to hold afterwards too — a manifest edit, a hand-copied directory or a
     stale interim file can each defeat them after the fact.

`--check` is the gate: it exits non-zero if the built corpus disagrees with the
manifest, if a Track A build carries non-green material, or if ATTRIBUTION.md is
missing or out of date. Run it after any manifest change, and before publishing
anything at all.

Usage:
    python src/provenance.py             # report + write ATTRIBUTION.md
    python src/provenance.py --report    # report only
    python src/provenance.py --check     # audit; non-zero exit on any problem
    python src/provenance.py --corpus corpus-b   # Track B, the NC sources too

TRACK B NEEDS ITS OWN NOTICE
    ATTRIBUTION.md describes Track A, and correctly omits the amber sources
    because they are not in it. But LATIN-LII trains on Track B, and LASLA is
    CC BY-NC-SA -- the BY is a live requirement, not softened by the NC. Track B
    went uncredited until --corpus existed, because nothing could address it.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    sys.exit("PyYAML required — pip install -r requirements.txt")

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "manifest" / "sources.yaml"
LOCKFILE = ROOT / "manifest" / "fetch.lock.json"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
ATTRIBUTION = ROOT / "ATTRIBUTION.md"

TRACKS = {"A": {"green"}, "B": {"green", "amber"}}

# ShareAlike propagates: a corpus containing any CC BY-SA material must itself
# be offered under CC BY-SA. Public-domain material can sit inside that happily;
# it simply carries no obligation of its own.
SHAREALIKE = {"CC-BY-SA-4.0", "CC-BY-SA-3.0"}
ATTRIBUTION_ONLY = {"CC-BY-4.0", "CC-BY-3.0"}
NO_OBLIGATION = {"public-domain", "CC0-1.0"}

# Mozilla-style file-level copyleft. Kept as its own class rather than folded
# into SHAREALIKE, because the obligation is genuinely a different shape: it
# attaches per file rather than to the work as a whole, and it requires the
# licence text to travel with the distribution. Both are redistributable and
# commercially usable, so the corpus's effective licence is unaffected — but
# a reader deciding what they must ship needs to be told which one applies.
FILE_COPYLEFT = {"UB-Public-License-1.0"}

# NonCommercial. Kept as its own class, and checked BEFORE every other, because
# the obligation is a different kind of thing: ShareAlike and attribution are
# duties you discharge when you redistribute, while NonCommercial is a
# restriction on what the result may ever be used for. It is absorbing — one NC
# source constrains the whole corpus, whatever the other licences permit — so
# the order of the tests in corpus_licence() is load-bearing, not stylistic.
#
# This is the boundary the Track A / Track B split exists to hold, which is why
# Track A carries none of these and Track B's weights are research-only.
NONCOMMERCIAL_SHAREALIKE = {"CC-BY-NC-SA-4.0"}
NONCOMMERCIAL = {"NonCommercial"}


def load_manifest() -> Dict[str, Dict[str, Any]]:
    with MANIFEST.open(encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    return {s["id"]: s for s in doc["sources"]}


def load_lock() -> Dict[str, Any]:
    if not LOCKFILE.exists():
        return {}
    with LOCKFILE.open(encoding="utf-8") as fh:
        return json.load(fh).get("sources", {})


def attribution_path(corpus: str) -> Path:
    """Track A keeps the plain name; every other build is suffixed.

    Track A's notice is the one that ships with anything published, so it keeps
    ATTRIBUTION.md. A Track B notice is not a lesser version of it -- it credits
    a different set of sources under stricter terms -- so it gets its own file
    rather than overwriting the one a reader expects to be redistributable.
    """
    return ROOT / ("ATTRIBUTION.md" if corpus == "corpus"
                   else f"ATTRIBUTION-{corpus}.md")


def load_corpus(corpus: str = "corpus") -> Tuple[List[Dict[str, Any]], str,
                                                 Optional[str]]:
    """Prefer the built corpus; fall back to interim so this works pre-build."""
    built = sorted(PROCESSED.glob(f"{corpus}.*.jsonl"))
    if built:
        records = []
        for path in built:
            with path.open(encoding="utf-8") as fh:
                records.extend(json.loads(line) for line in fh)
        track = records[0].get("track") if records else None
        return records, "data/processed", track
    records = []
    for path in sorted(INTERIM.glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            records.extend(json.loads(line) for line in fh)
    records = [r for r in records if not r.get("is_duplicate")]
    return records, "data/interim (no build yet)", None


def summarise(records: List[Dict[str, Any]],
              manifest: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    words: Counter = Counter()
    docs: Counter = Counter()
    for rec in records:
        words[rec.get("source")] += rec.get("n_words", 0)
        docs[rec.get("source")] += 1

    total = sum(words.values()) or 1
    rows = []
    for source_id, count in words.most_common():
        entry = manifest.get(source_id, {})
        rows.append({
            "id": source_id,
            "name": entry.get("name", source_id),
            "url": entry.get("url"),
            "license": entry.get("license", "UNKNOWN"),
            "tier": entry.get("tier", "UNKNOWN"),
            "words": count,
            "documents": docs[source_id],
            "share": count / total,
            "in_manifest": source_id in manifest,
        })
    return rows


def corpus_licence(rows: List[Dict[str, Any]]) -> str:
    licences = {r["license"] for r in rows}

    # NonCommercial first, because it is absorbing. Testing ShareAlike ahead of
    # it would report a corpus containing NC material as plain CC BY-SA and
    # silently drop the one term that makes it non-redistributable -- which is
    # exactly what this function did before it was ever pointed at Track B.
    if licences & (NONCOMMERCIAL_SHAREALIKE | NONCOMMERCIAL):
        if licences & (SHAREALIKE | NONCOMMERCIAL_SHAREALIKE):
            return "CC-BY-NC-SA-4.0"
        return "NonCommercial — research only"

    if licences & SHAREALIKE:
        return "CC-BY-SA-4.0"
    if licences & FILE_COPYLEFT:
        # No CC identifier describes this. Naming it is more use than
        # inventing a label the source does not carry.
        return "CC-BY-SA-4.0 + UBPL-1.0"
    if licences & ATTRIBUTION_ONLY:
        return "CC-BY-4.0"
    if licences and licences <= NO_OBLIGATION:
        return "public-domain"
    return "MIXED — resolve before redistributing"


def render(rows: List[Dict[str, Any]], lock: Dict[str, Any],
           track: Optional[str], effective: str,
           corpus: str = "corpus") -> str:
    total_words = sum(r["words"] for r in rows)
    lines = [
        "# Attribution" + ("" if corpus == "corpus" else f" — {corpus}"),
        "",
        "This corpus is assembled from the sources below. Each is credited under",
        "the licence it was released with, pinned to the exact version ingested.",
        "",
        f"Generated by `src/provenance.py` on {date.today().isoformat()}. Do not edit by hand —",
        "regenerate it, so it cannot drift from what was actually built.",
        "",
        f"- **Corpus**: `{corpus}`",
        f"- **Track**: {track or 'unbuilt'}"
        + ("  — redistributable" if track == "A"
           else "  — RESEARCH ONLY, must not be redistributed" if track == "B" else ""),
        f"- **Documents**: {len(set()) or sum(r['documents'] for r in rows):,}",
        f"- **Words**: {total_words:,}",
        f"- **Effective licence of the assembled corpus**: `{effective}`",
        "",
    ]

    if effective.startswith("CC-BY-NC") or effective.startswith("NonCommercial"):
        lines += [
            "**This corpus contains NonCommercial material and cannot be used",
            "commercially — nor can anything trained on it.** The restriction is",
            "absorbing: it applies to the corpus as a whole regardless of the more",
            "permissive licences alongside it, and it travels to model weights.",
            "",
            "Attribution is still owed in full. NonCommercial restricts what the",
            "result may be used FOR; it does not excuse crediting the sources, and the",
            "BY in CC BY-NC-SA is the same BY as anywhere else.",
            "",
            "The redistributable half of this project is the Track A corpus, credited",
            "in `ATTRIBUTION.md`.",
            "",
        ]

    if effective in SHAREALIKE:
        lines += [
            "ShareAlike propagates. Because the corpus contains CC BY-SA material, the",
            "corpus as a whole must be offered under CC BY-SA 4.0. Public-domain material",
            "sits inside that without difficulty; it simply carries no obligation itself.",
            "",
        ]

    lines += ["## Sources", ""]
    for row in rows:
        pin = lock.get(row["id"], {}).get("resolved_pin")
        lines += [
            f"### {row['name']}",
            "",
            f"- Licence: **{row['license']}**",
            f"- Source: {row['url']}",
        ]
        if pin:
            lines.append(f"- Version ingested: `{pin}`")
        lines += [
            f"- Contribution: {row['words']:,} words across {row['documents']:,} documents "
            f"({row['share']:.1%} of the corpus)",
            "",
        ]

    lines += [
        "## Underlying texts",
        "",
        "The Latin texts themselves are ancient and long out of copyright. What the",
        "licences above cover is the modern editorial and encoding work: the constituted",
        "text of a critical edition, conjectures, punctuation, and the digital markup.",
        "Those claims vary by jurisdiction — the US rejects 'sweat of the brow' after",
        "*Feist*, while Germany and Italy grant a neighbouring right in scientific",
        "editions — but this project honours the licences as offered rather than",
        "litigating them.",
        "",
    ]
    return "\n".join(lines) + "\n"


def check(rows: List[Dict[str, Any]], manifest: Dict[str, Dict[str, Any]],
          track: Optional[str], rendered: str,
          corpus: str = "corpus") -> List[str]:
    problems: List[str] = []

    for row in rows:
        if not row["in_manifest"]:
            problems.append(
                f"source `{row['id']}` is in the corpus but not in the manifest "
                f"({row['words']:,} words) — provenance cannot be established")
        if row["license"] == "UNKNOWN":
            problems.append(f"source `{row['id']}` has no recorded licence")

    if track:
        allowed = TRACKS.get(track, set())
        for row in rows:
            if row["tier"] not in allowed:
                problems.append(
                    f"Track {track} build contains `{row['id']}` at tier "
                    f"`{row['tier']}` ({row['words']:,} words) — not redistributable")
    else:
        problems.append("no built corpus found; run src/build.py before publishing")

    target = attribution_path(corpus)
    flag = "" if corpus == "corpus" else f" --corpus {corpus}"
    if not target.exists():
        problems.append(f"{target.name} is missing — run "
                        f"`python src/provenance.py{flag}`")
    else:
        current = target.read_text(encoding="utf-8")
        # Compare ignoring the generation date, which changes every day.
        def strip_date(text: str) -> str:
            return "\n".join(l for l in text.splitlines() if "Generated by" not in l)
        if strip_date(current) != strip_date(rendered):
            problems.append(
                f"{target.name} is out of date with the current corpus — "
                f"run `python src/provenance.py{flag}` to regenerate")

    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="print only, write nothing")
    ap.add_argument("--check", action="store_true",
                    help="audit and exit non-zero on any problem")
    ap.add_argument("--corpus", default="corpus",
                    help="which built corpus to describe (default: corpus, "
                         "which is Track A). `corpus-b` is Track B, and writes "
                         "ATTRIBUTION-corpus-b.md")
    args = ap.parse_args()

    manifest = load_manifest()
    lock = load_lock()
    records, origin, track = load_corpus(args.corpus)
    if not records:
        sys.exit(f"no `{args.corpus}` corpus found — run the pipeline first")

    rows = summarise(records, manifest)
    effective = corpus_licence(rows)
    rendered = render(rows, lock, track, effective, args.corpus)

    total = sum(r["words"] for r in rows)
    print(f"corpus source     {origin}")
    print(f"track             {track or 'unbuilt'}")
    print(f"documents         {sum(r['documents'] for r in rows):,}")
    print(f"words             {total:,}")
    print(f"effective licence {effective}\n")
    print(f"{'source':<32} {'tier':<8} {'licence':<16} {'words':>11}  share")
    print("-" * 78)
    for row in rows:
        flag = "" if row["in_manifest"] else "  <-- NOT IN MANIFEST"
        print(f"{row['id'][:32]:<32} {row['tier']:<8} {row['license']:<16} "
              f"{row['words']:>11,}  {row['share']:>5.1%}{flag}")
    print("-" * 78)

    obligations: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        if row["license"] in NONCOMMERCIAL_SHAREALIKE:
            obligations["attribution + ShareAlike + NonCommercial"].append(row["id"])
        elif row["license"] in NONCOMMERCIAL:
            obligations["NonCommercial; attribution expected"].append(row["id"])
        elif row["license"] in SHAREALIKE:
            obligations["attribution + ShareAlike"].append(row["id"])
        elif row["license"] in FILE_COPYLEFT:
            obligations["attribution + per-file copyleft, ship licence"].append(row["id"])
        elif row["license"] in ATTRIBUTION_ONLY:
            obligations["attribution"].append(row["id"])
        elif row["license"] in NO_OBLIGATION:
            obligations["none"].append(row["id"])
        else:
            obligations["UNRESOLVED"].append(row["id"])
    print("\nobligations:")
    for kind, ids in sorted(obligations.items()):
        print(f"  {kind:<26} {', '.join(ids)}")

    if args.check:
        problems = check(rows, manifest, track, rendered, args.corpus)
        if problems:
            print("\nPROVENANCE CHECK FAILED:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1
        print("\nprovenance check passed")
        return 0

    if not args.report:
        target = attribution_path(args.corpus)
        target.write_text(rendered, encoding="utf-8")
        print(f"\nwrote {target.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

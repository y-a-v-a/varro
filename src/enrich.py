#!/usr/bin/env python3
"""Apply manifest/authorities.yaml to extracted records: date and period them.

Rewrites data/interim/*.jsonl in place, setting `approximate_date`, `period`,
`date_source` and `date_confidence` per record. Interim files are build
artifacts regenerable from data/raw/ via extract.py, so in-place is safe and
keeps the pipeline to one stage per concern.

Kept separate from extract.py deliberately: the authority list is scholarly
metadata that will be revised many times, and re-running this takes a second
where re-extracting takes a minute.

Resolution order, most specific first:
    1. works[]   — exact match on the `work` field (anonymous but datable texts)
    2. authors[] — exact match on the `author` field
    3. source_priors — a period, never a date, and marked `inferred`
    4. nothing — both fields left null, and counted

Usage:
    python src/enrich.py --report        # resolve and print coverage, write nothing
    python src/enrich.py                 # rewrite data/interim/*.jsonl
    python src/enrich.py --audit         # list unmatched authors and dead aliases
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    sys.exit("PyYAML required — pip install -r requirements.txt")

ROOT = Path(__file__).resolve().parent.parent
AUTHORITIES = ROOT / "manifest" / "authorities.yaml"
INTERIM = ROOT / "data" / "interim"


# ---------------------------------------------------------------------------
# Authority
# ---------------------------------------------------------------------------

class Authority:
    def __init__(self, path: Path = AUTHORITIES):
        with path.open(encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
        self.periods: Dict[str, Dict[str, int]] = doc["periods"]
        self.source_priors: Dict[str, Optional[str]] = doc.get("source_priors") or {}
        self.authors = self._index(doc.get("authors") or [], "authors")
        self.works = self._index(doc.get("works") or [], "works")
        self.alias_hits: Counter = Counter()

    @staticmethod
    def _index(entries: List[Dict[str, Any]], label: str) -> Dict[str, Dict[str, Any]]:
        index: Dict[str, Dict[str, Any]] = {}
        for entry in entries:
            for alias in entry.get("aliases", []):
                key = alias.strip().lower()
                if key in index:
                    raise SystemExit(
                        f"authorities.yaml: alias {alias!r} appears twice in {label}"
                    )
                index[key] = entry
        return index

    def period_for(self, floruit: Optional[int]) -> Optional[str]:
        if floruit is None:
            return None
        for name, bounds in self.periods.items():
            low = bounds.get("min")
            high = bounds.get("max")
            if (low is None or floruit >= low) and (high is None or floruit <= high):
                return name
        return None

    def resolve(self, record: Dict[str, Any]) -> Tuple[Optional[int], Optional[str], str, str]:
        """Return (approximate_date, period, date_source, confidence)."""
        work = (record.get("work") or "").strip().lower()
        entry = self.works.get(work)
        origin = "work"

        if entry is None:
            author = (record.get("author") or "").strip().lower()
            entry = self.authors.get(author)
            origin = "author"

        if entry is not None:
            for alias in entry.get("aliases", []):
                if alias.strip().lower() in (work, (record.get("author") or "").strip().lower()):
                    self.alias_hits[alias] += 1
            floruit = entry.get("floruit")
            confidence = entry.get("confidence", "unknown")
            if floruit is not None:
                return floruit, self.period_for(floruit), origin, confidence
            # Entry exists but asserts a period without a date (corporate authors),
            # or explicitly declines to date (period: null) and should fall through.
            period = entry.get("period")
            if period:
                return None, period, origin, confidence

        prior = self.source_priors.get(record.get("source"))
        if prior:
            return None, prior, "source-prior", "inferred"
        return None, None, "none", "none"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def load(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def write_atomic(path: Path, records: List[Dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="print coverage, write nothing")
    ap.add_argument("--audit", action="store_true",
                    help="list unresolved authors and unused aliases, write nothing")
    args = ap.parse_args()

    authority = Authority()
    files = sorted(INTERIM.glob("*.jsonl"))
    if not files:
        sys.exit(f"no extracted records in {INTERIM} — run src/extract.py first")

    by_period: Counter = Counter()
    by_origin: Counter = Counter()
    by_confidence: Counter = Counter()
    unresolved: Counter = Counter()
    total_words = 0

    for path in files:
        records = load(path)
        for rec in records:
            date, period, origin, confidence = authority.resolve(rec)
            rec["approximate_date"] = date
            rec["period"] = period
            rec["date_source"] = origin
            rec["date_confidence"] = confidence

            words = rec.get("n_words", 0)
            total_words += words
            by_period[period or "UNASSIGNED"] += words
            by_origin[origin] += words
            by_confidence[confidence] += words
            if origin in ("source-prior", "none"):
                unresolved[(rec.get("author") or "?", rec.get("source"))] += words

        if not (args.report or args.audit):
            write_atomic(path, records)

    def table(title: str, counter: Counter) -> None:
        print(f"\n{title}")
        for key, words in counter.most_common():
            bar = "#" * int(40 * words / total_words)
            print(f"  {str(key):<16} {words:>10,}  {100*words/total_words:5.1f}%  {bar}")

    print(f"records dated across {len(files)} source file(s); {total_words:,} words")
    table("period", by_period)
    table("resolved via", by_origin)
    table("confidence", by_confidence)

    if args.audit:
        print(f"\nunresolved authors (fell through to prior or nothing): {len(unresolved)}")
        for (author, source), words in unresolved.most_common(30):
            print(f"  {words:>8,}  {source[:18]:<18} {author}")
        # Aliases that matched nothing are usually a typo or a name the upstream
        # source has since changed — either way the entry is silently inert.
        matched = {alias.lower() for alias in authority.alias_hits}
        unused = sorted((set(authority.authors) | set(authority.works)) - matched)
        print(f"\naliases matching nothing in the corpus: {len(unused)}")
        for alias in unused:
            print(f"  {alias}")

    if not (args.report or args.audit):
        print(f"\nrewrote {len(files)} file(s) in {INTERIM.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

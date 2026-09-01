#!/usr/bin/env python3
"""Apply the curated authorities to extracted records: date, period, form, genre.

Rewrites data/interim/*.jsonl in place, setting `approximate_date`, `period`,
`date_source` and `date_confidence` per record. Interim files are build
artifacts regenerable from data/raw/ via extract.py, so in-place is safe and
keeps the pipeline to one stage per concern.

Kept separate from extract.py deliberately: the authority list is scholarly
metadata that will be revised many times, and re-running this takes a second
where re-extracting takes a minute.

Dating (manifest/authorities.yaml), most specific first:
    1. works[]   — exact match on the `work` field (anonymous but datable texts)
    2. authors[] — exact match on the `author` field
    3. source_priors — a period, never a date, and marked `inferred`
    4. nothing — both fields left null, and counted

Form and genre (manifest/genres.yaml), most specific first:
    1. works[]          — exact work title
    2. title_patterns[] — regex on the title; carries most of the load, because
                          the largest authors are polygeneric (Cicero writes
                          letters, speeches, rhetoric and philosophy) and the
                          title is the reliable discriminator
    3. authors[]        — author default, for the monogeneric majority
    4. UNKNOWN          — recorded as such rather than guessed

Form and genre are separate axes on purpose: Terentianus Maurus writes grammar
in verse, and Boethius alternates verse with prose inside one work.

Usage:
    python src/enrich.py --report        # resolve and print coverage, write nothing
    python src/enrich.py                 # rewrite data/interim/*.jsonl
    python src/enrich.py --audit         # list unmatched authors and dead aliases
"""

import argparse
import json
import re
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
GENRES = ROOT / "manifest" / "genres.yaml"
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
        anchors = doc.get("pl_volume_anchors") or []
        self.pl_anchors = sorted((a["volume"], a["year"]) for a in anchors)

    def year_for_pl_volume(self, volume: Optional[int]) -> Optional[int]:
        """Interpolate a year from Migne's chronological volume ordering.

        Piecewise between anchors rather than linear across the whole range:
        Migne gives Augustine sixteen volumes and the tenth century about six,
        so a straight line from volume 1 to 217 dates Jerome a century early.
        """
        if volume is None or not self.pl_anchors:
            return None
        if volume <= self.pl_anchors[0][0]:
            return self.pl_anchors[0][1]
        if volume >= self.pl_anchors[-1][0]:
            return self.pl_anchors[-1][1]
        for (v0, y0), (v1, y1) in zip(self.pl_anchors, self.pl_anchors[1:]):
            if v0 <= volume <= v1:
                span = max(1, v1 - v0)
                return int(round(y0 + (volume - v0) * (y1 - y0) / span))
        return None

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

        # Below the curated authorities, above everything inferred: a date the
        # source itself states for this work. Tesserae dates all 748 of its
        # files individually, which is both more precise than an author-level
        # floruit and available for authors no authority lists — Corippus,
        # Dracontius, or an 1835 Latin life of George Washington.
        #
        # It ranks under the authorities because those are human-checked and
        # this is taken on trust, and it is recorded under its own origin so
        # its share of the corpus stays visible in the report, exactly as
        # `pl-volume` and `source-prior` are.
        stated = record.get("approximate_date")
        if isinstance(stated, int):
            return stated, self.period_for(stated), "source-metadata", "stated"

        # Below that, above the source prior: a Patrologia Latina volume
        # number dates a text without naming its author, which is the only
        # signal most Wikisource pages carry.
        year = self.year_for_pl_volume(record.get("pl_volume"))
        if year is not None:
            return year, self.period_for(year), "pl-volume", "inferred"

        prior = self.source_priors.get(record.get("source"))
        if prior:
            return None, prior, "source-prior", "inferred"
        return None, None, "none", "none"


class Genres:
    """Form and genre resolution from manifest/genres.yaml."""

    def __init__(self, path: Path = GENRES):
        with path.open(encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
        self.valid_forms = set(doc.get("forms") or [])
        self.valid_genres = set(doc.get("genres") or [])
        self.patterns = [(re.compile(p["match"], re.IGNORECASE), p)
                         for p in (doc.get("title_patterns") or [])]
        self.authors = self._index(doc.get("authors") or [])
        self.works = self._index(doc.get("works") or [])
        for entry in list(self.authors.values()) + list(self.works.values()):
            if entry.get("form") not in self.valid_forms:
                raise SystemExit(f"genres.yaml: bad form {entry.get('form')!r}")
            if entry.get("genre") not in self.valid_genres:
                raise SystemExit(f"genres.yaml: bad genre {entry.get('genre')!r}")

    @staticmethod
    def _index(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        index: Dict[str, Dict[str, Any]] = {}
        for entry in entries:
            for alias in entry.get("aliases", []):
                index[alias.strip().lower()] = entry
        return index

    def resolve(self, record: Dict[str, Any]) -> Tuple[str, str, str]:
        """Return (form, genre, origin)."""
        work = (record.get("work") or "").strip()
        author = (record.get("author") or "").strip().lower()

        entry = self.works.get(work.lower())
        if entry:
            return entry["form"], entry["genre"], "work"

        for pattern, spec in self.patterns:
            if work and pattern.search(work):
                return spec.get("form", "UNKNOWN"), spec["genre"], "pattern"

        entry = self.authors.get(author)
        if entry:
            return entry["form"], entry["genre"], "author"

        return "UNKNOWN", "UNKNOWN", "none"


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
    genres = Genres()
    files = sorted(INTERIM.glob("*.jsonl"))
    if not files:
        sys.exit(f"no extracted records in {INTERIM} — run src/extract.py first")

    by_period: Counter = Counter()
    by_origin: Counter = Counter()
    by_confidence: Counter = Counter()
    unresolved: Counter = Counter()
    by_form: Counter = Counter()
    by_genre: Counter = Counter()
    by_genre_origin: Counter = Counter()
    total_words = 0

    for path in files:
        records = load(path)
        for rec in records:
            date, period, origin, confidence = authority.resolve(rec)
            rec["approximate_date"] = date
            rec["period"] = period
            rec["date_source"] = origin
            rec["date_confidence"] = confidence

            form, genre, gorigin = genres.resolve(rec)
            rec["form"] = form
            rec["genre"] = genre
            rec["genre_source"] = gorigin

            words = rec.get("n_words", 0)
            by_form[form] += words
            by_genre[genre] += words
            by_genre_origin[gorigin] += words
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
    table("form", by_form)
    table("genre", by_genre)
    table("genre resolved via", by_genre_origin)

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

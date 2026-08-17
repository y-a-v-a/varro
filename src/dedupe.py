#!/usr/bin/env python3
"""Find and mark duplicate works across and within sources.

The sources overlap heavily: Tertullian's Adversus Marcionem is in both Perseus
and CSEL, Boethius' Consolatio in three places. Concatenating them would train
on the same text repeatedly under a *different* orthography each time, which is
worse than plain repetition — it teaches edition-house style as if it were
Latin. It also corrupts the one dial the ladder in feasibility.md depends on:
deliberate epoch count.

Detection is content-based, not title-based. Titles vary across projects
("Adversus Marcionem" / "Adv. Marcionem"), so metadata equality is used only as
corroboration.

Method
    Text is folded for comparison only (never rewritten): lowercased,
    u/v and i/j merged, non-letters stripped. Without that fold, Perseus's
    `uiri` and CSEL's `viri` look like different words and similarity collapses.

    Each document is reduced to a bottom-k sketch of its 5-gram hashes
    (k-minimum values). Jaccard is estimated from sketch overlap, and
    containment derived from it — containment matters because one witness is
    often a subset of another rather than its twin.

    Hashing is blake2b, not Python's hash(), which is per-process randomised
    and would make runs irreproducible.

Duplicates are MARKED, never deleted: `is_duplicate`, `duplicate_of`,
`duplicate_similarity`. build.py filters them. Keeping the losing record makes
the decision auditable and lets the policy change without re-extracting.

Usage:
    python src/dedupe.py --report          # distribution + clusters, no writes
    python src/dedupe.py                   # mark duplicates in data/interim/
    python src/dedupe.py --jaccard 0.4 --containment 0.7
"""

import argparse
import heapq
import json
import random
import re
import sys
from collections import defaultdict
from hashlib import blake2b
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
INTERIM = ROOT / "data" / "interim"

SKETCH_K = 1024
SHINGLE = 5

# Containment alone may not condemn a very short record. At 5-grams a text of a
# few dozen words can be 80% "contained" in another through shared formulae
# (liturgical tags, salutations, citation boilerplate) without being the same
# text. Jaccard is still allowed to mark such records at any length.
MIN_CONTAINED = 100

# Which witness wins when the same work appears in several sources.
#
# Perseus first: hand-encoded TEI from critical editions, and the cleanest text
# measured so far. DigilibLT next: TEI upstream, but reaching us through a
# CoNLL-U round-trip that re-joins tokenised text. CSEL last: machine-corrected
# OCR with known hyphenation breaks and garbled Greek (see README limitations).
#
# This ranks *text fidelity*, not scholarly authority — CSEL is the standard
# critical series for much of the patristic material it carries.
SOURCE_PRIORITY = [
    "perseus-canonical-latinLit",
    "digiliblt",
    "opengreekandlatin-csel",
]

FOLD = str.maketrans({"v": "u", "j": "i", "V": "u", "J": "i"})
NON_LETTER = re.compile(r"[^a-z\s]+")


def fold(text: str) -> List[str]:
    """Orthography-insensitive token stream, for comparison only."""
    text = text.lower().translate(FOLD)
    return NON_LETTER.sub(" ", text).split()


def sketch(tokens: Sequence[str], k: int = SKETCH_K,
           n: int = SHINGLE) -> Tuple[Set[int], int]:
    """Bottom-k sketch of the document's n-gram hashes, and its true cardinality.

    The cardinality returned is the number of DISTINCT shingles, not the number
    of shingle positions. Jaccard is estimated over sets, so containment — which
    divides an estimated intersection by a document size — has to divide by a set
    size too. Using positions mixes the two: repetition inflates the denominator
    and containment is understated (measured at ~8% repetition on a typical
    record, so the error is real but conservative).
    """
    if len(tokens) < n:
        return set(), 0
    heap: List[int] = []          # max-heap via negation
    seen: Set[int] = set()
    for i in range(len(tokens) - n + 1):
        shingle = " ".join(tokens[i:i + n]).encode()
        value = int.from_bytes(blake2b(shingle, digest_size=8).digest(), "big")
        if value in seen:
            continue
        seen.add(value)
        if len(heap) < k:
            heapq.heappush(heap, -value)
        elif -heap[0] > value:
            heapq.heapreplace(heap, -value)
    return {-x for x in heap}, len(seen)


def jaccard(a: Sequence[int], b: Sequence[int], a_set: Set[int], b_set: Set[int],
            k: int = SKETCH_K) -> float:
    """KMV estimator over two pre-sorted sketches.

    Walks the bottom-k of the union by linear merge rather than materialising
    `a | b` and re-sorting it — the same estimate, but this is the inner loop of
    an all-pairs comparison and the set construction dominated the runtime.
    """
    if not a or not b:
        return 0.0
    i = j = taken = both = 0
    while taken < k and (i < len(a) or j < len(b)):
        if j >= len(b) or (i < len(a) and a[i] < b[j]):
            if a[i] in b_set:
                both += 1
            i += 1
        elif i >= len(a) or b[j] < a[i]:
            if b[j] in a_set:
                both += 1
            j += 1
        else:                      # present in both
            both += 1
            i += 1
            j += 1
        taken += 1
    return both / taken if taken else 0.0


def containment(j: float, size_a: int, size_b: int) -> float:
    """Fraction of the smaller document present in the larger.

    Derived from Jaccard: |A∩B| = J(|A|+|B|)/(1+J), then divided by min(|A|,|B|).
    Catches the case where one witness is a subset of a fuller one, which
    Jaccard alone scores low.
    """
    if j <= 0 or not size_a or not size_b:
        return 0.0
    intersection = j * (size_a + size_b) / (1 + j)
    return min(1.0, intersection / min(size_a, size_b))


# ---------------------------------------------------------------------------

def load_all() -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    by_file: Dict[str, List[Dict[str, Any]]] = {}
    records: List[Dict[str, Any]] = []
    for path in sorted(INTERIM.glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh]
        by_file[str(path)] = rows
        records.extend(rows)
    return records, by_file


def priority(record: Dict[str, Any]) -> int:
    try:
        return SOURCE_PRIORITY.index(record.get("source", ""))
    except ValueError:
        return len(SOURCE_PRIORITY)


SUBSUMES_BELOW = 0.8


def choose_keeper(members: List[int], records: List[Dict[str, Any]]) -> List[int]:
    """Order a cluster so the surviving witness comes first.

    Two different relationships hide in one cluster and they need opposite rules:

      equivalence — the same work in several sources, near-identical length.
        Source priority decides, because what differs is text quality.

      containment — one record is part of a fuller one: Perseus's selection of
        Jerome's letters inside CSEL's complete corpus, a single Historia
        Augusta life inside the whole, one panegyric inside Panegyrici Latini.
        Here the larger record must win regardless of source, or the pipeline
        discards 290k words to keep 43k.

    Resolved by filtering to records within SUBSUMES_BELOW of the longest — so a
    materially shorter record can never win — and only then applying source
    priority among the comparable ones.
    """
    longest = max(records[i].get("n_words", 0) for i in members)
    cutoff = SUBSUMES_BELOW * longest
    comparable = [i for i in members if records[i].get("n_words", 0) >= cutoff]
    comparable.sort(key=lambda i: (priority(records[i]), -records[i].get("n_words", 0)))
    rest = sorted((i for i in members if i not in set(comparable)),
                  key=lambda i: -records[i].get("n_words", 0))
    return comparable + rest


def verify(records: List[Dict[str, Any]], probes: int = 40,
           span: int = 8, floor: float = 0.15) -> int:
    """Check every marked duplicate really does occur inside its keeper.

    Sketch-based similarity is an estimate, and containment derived from it gets
    noisy when one record dwarfs the other. This is the ground truth: sample
    `probes` random word-spans from the duplicate and test literal membership in
    the keeper. Cross-edition duplicates land around 50-90%; part-of-collection
    duplicates at 100%; a genuine false positive would score near zero.

    A single probe is not enough — the Excerpta Valesiana looked like a false
    positive on its opening alone and verifies at 70% across 40 probes.
    """
    by_id = {rec["id"]: rec for rec in records}
    cache: Dict[str, str] = {}

    def folded(rec: Dict[str, Any]) -> str:
        if rec["id"] not in cache:
            cache[rec["id"]] = " ".join(fold(rec["text"]))
        return cache[rec["id"]]

    rng = random.Random(7)
    results: List[Tuple[Optional[float], Dict[str, Any], Dict[str, Any]]] = []
    for rec in records:
        if not rec.get("is_duplicate"):
            continue
        keeper = by_id.get(rec["duplicate_of"])
        if keeper is None:
            print(f"  DANGLING duplicate_of: {rec['id']}", file=sys.stderr)
            continue
        words = folded(rec).split()
        if len(words) < span * 5:
            results.append((None, rec, keeper))
            continue
        haystack = folded(keeper)
        hits = 0
        for _ in range(probes):
            start = rng.randrange(0, len(words) - span)
            if " ".join(words[start:start + span]) in haystack:
                hits += 1
        results.append((hits / probes, rec, keeper))

    results.sort(key=lambda r: (r[0] is not None, r[0] if r[0] is not None else 0.0))
    print(f"{'verbatim':>9}  {'words':>9}  duplicate -> keeper")
    suspect = []
    for score, rec, keeper in results:
        label = "    n/a" if score is None else f"{score:>8.0%}"
        flag = ""
        if score is not None and score < floor:
            flag = "   <-- SUSPECT"
            suspect.append(rec)
        print(f"{label}  {rec['n_words']:>9,}  {(rec['work'] or '?')[:36]:<36}"
              f" -> {(keeper['work'] or '?')[:28]}{flag}")
    print(f"\nchecked {len(results)} marked duplicates; "
          f"{len(suspect)} below {floor:.0%} verbatim overlap "
          f"({sum(r['n_words'] for r in suspect):,} words)")
    return 1 if suspect else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="analyse only, write nothing")
    ap.add_argument("--verify", action="store_true",
                    help="verbatim-check existing duplicate marks and exit")
    ap.add_argument("--jaccard", type=float, default=0.5,
                    help="Jaccard at or above which two works are the same (default 0.5)")
    ap.add_argument("--containment", type=float, default=0.8,
                    help="containment at or above which the smaller is redundant (default 0.8)")
    ap.add_argument("--show", type=int, default=25, help="clusters to print")
    args = ap.parse_args()

    records, by_file = load_all()
    if not records:
        sys.exit(f"no records in {INTERIM} — run extract.py and enrich.py first")

    if args.verify:
        return verify(records)

    print(f"sketching {len(records):,} records ...", flush=True)
    sizes: List[int] = []
    sorted_sketches: List[Tuple[int, ...]] = []
    set_sketches: List[Set[int]] = []
    sketch_heartbeat = max(1, len(records) // 10)
    for done, rec in enumerate(records, 1):
        if done % sketch_heartbeat == 0:
            print(f"  sketched {done:,}/{len(records):,}", flush=True)
        tokens = fold(rec["text"])
        s, cardinality = sketch(tokens)
        sizes.append(cardinality)
        set_sketches.append(s)
        sorted_sketches.append(tuple(sorted(s)))

    # Candidate prefilter: documents sharing no sketch value at all cannot be
    # duplicates, and that is the overwhelming majority of pairs. An inverted
    # index over sketch values finds the few that do, turning an O(n^2) sweep
    # into work proportional to actual overlap.
    postings: Dict[int, List[int]] = defaultdict(list)
    for idx, s in enumerate(set_sketches):
        for value in s:
            postings[value].append(idx)
    candidates: Set[Tuple[int, int]] = set()
    for docs in postings.values():
        if len(docs) < 2:
            continue
        for x in range(len(docs)):
            for y in range(x + 1, len(docs)):
                candidates.add((docs[x], docs[y]))
    print(f"candidate pairs after prefilter: {len(candidates):,} "
          f"(of {len(records)*(len(records)-1)//2:,} possible)", flush=True)

    # A heartbeat, because this loop is the long pole and used to run silent:
    # at five sources it scores over a million pairs and says nothing for
    # minutes, which is indistinguishable from a hang to anything watching.
    pairs: List[Tuple[float, float, int, int]] = []
    heartbeat = max(1, len(candidates) // 20)
    for scored, (i, j) in enumerate(candidates, 1):
        if scored % heartbeat == 0:
            print(f"  scoring pairs: {scored:,}/{len(candidates):,} "
                  f"({scored/len(candidates):.0%}), {len(pairs):,} above threshold",
                  flush=True)
        est = jaccard(sorted_sketches[i], sorted_sketches[j],
                      set_sketches[i], set_sketches[j])
        cont = containment(est, sizes[i], sizes[j])
        # No Jaccard floor before the containment test. A short text inside a
        # long one has a legitimately tiny Jaccard — Salvian's 188-word appendix
        # sits at J=0.03 inside his Epistulae while being 84% contained — so any
        # cheap "skip if J is low" gate silently defeats containment detection
        # exactly where sizes differ most, which is where it is needed.
        if est >= args.jaccard:
            pairs.append((est, cont, i, j))
        elif cont >= args.containment and min(sizes[i], sizes[j]) >= MIN_CONTAINED:
            pairs.append((est, cont, i, j))

    pairs.sort(reverse=True)
    print(f"candidate duplicate pairs: {len(pairs)}\n")

    # Union-find over the surviving pairs.
    parent = list(range(len(records)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, _, i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    clusters: Dict[int, List[int]] = defaultdict(list)
    for idx in range(len(records)):
        clusters[find(idx)].append(idx)
    clusters = {root: members for root, members in clusters.items() if len(members) > 1}

    best_pair = {(min(i, j), max(i, j)): (e, c) for e, c, i, j in pairs}

    dropped_words = 0
    shown = 0
    print(f"clusters: {len(clusters)}")
    for members in sorted(clusters.values(), key=lambda m: -sum(records[i]["n_words"] for i in m)):
        members[:] = choose_keeper(members, records)
        keeper, losers = members[0], members[1:]
        dropped_words += sum(records[i]["n_words"] for i in losers)
        if shown < args.show:
            shown += 1
            k = records[keeper]
            print(f"\n  KEEP  {k['n_words']:>7,}w  {k['source'][:22]:<22} "
                  f"{(k['author'] or '?')[:22]:<22} {(k['work'] or '?')[:34]}")
            for i in losers:
                r = records[i]
                pair = best_pair.get((min(keeper, i), max(keeper, i)))
                sim = f"J={pair[0]:.2f} C={pair[1]:.2f}" if pair else "(indirect)"
                print(f"  drop  {r['n_words']:>7,}w  {r['source'][:22]:<22} "
                      f"{(r['author'] or '?')[:22]:<22} {(r['work'] or '?')[:34]}  {sim}")

    total = sum(r["n_words"] for r in records)
    print(f"\nwould drop {dropped_words:,} words of {total:,} ({100*dropped_words/total:.1f}%)")

    if args.report:
        print("\n--report: nothing written")
        return 0

    keep_of: Dict[int, int] = {}
    for members in clusters.values():
        ordered = choose_keeper(members, records)
        for loser in ordered[1:]:
            keep_of[loser] = ordered[0]

    index = {id(rec): i for i, rec in enumerate(records)}
    for path, rows in by_file.items():
        for rec in rows:
            i = index[id(rec)]
            if i in keep_of:
                winner = records[keep_of[i]]
                pair = best_pair.get((min(i, keep_of[i]), max(i, keep_of[i])))
                rec["is_duplicate"] = True
                rec["duplicate_of"] = winner["id"]
                rec["duplicate_similarity"] = round(pair[0], 4) if pair else None
            else:
                rec["is_duplicate"] = False
                rec["duplicate_of"] = None
                rec["duplicate_similarity"] = None
        tmp = Path(path).with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for rec in rows:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp.replace(Path(path))

    print(f"\nmarked {len(keep_of):,} records as duplicates across {len(by_file)} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

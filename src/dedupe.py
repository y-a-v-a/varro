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
    python src/dedupe.py --verify          # verbatim-check both tracks' marks
    python src/dedupe.py --verify --track A
"""

import argparse
import heapq
import json
import os
import random
import re
import sys
from bisect import bisect_right
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from hashlib import blake2b
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from encoding import chunk_texts  # noqa: E402

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


# How much text one pass hands to the worker pool.
#
# ProcessPoolExecutor.map submits every task up front, so mapping over the
# whole corpus would pickle all 500MB of it into the task queues at once. The
# records are already resident — load_all reads them all — so a window costs
# nothing to assemble; it exists purely to bound what is in flight. 64MB is
# roughly nine seconds of serial sketching, long enough that the synchronise at
# each window boundary disappears against it.
SKETCH_WINDOW_CHARS = 64_000_000


def sketch_record(text: str) -> Tuple[Tuple[int, ...], int]:
    """Fold and sketch one record: the unit of work a worker process gets.

    Module-level and taking a plain string, because a process pool has to
    pickle both the function and its argument. Returns the sketch already
    sorted, which the scoring loop needs anyway and which is cheaper to send
    back than a set.
    """
    values, cardinality = sketch(fold(text))
    return tuple(sorted(values)), cardinality


def sketch_all(records: List[Dict[str, Any]], jobs: int,
               window: int = SKETCH_WINDOW_CHARS
               ) -> Iterator[Tuple[Tuple[int, ...], int]]:
    """Sketch every record, in order, across `jobs` processes.

    Sketching is the long pole here and it is pure Python over every 5-gram of
    every document — 7.1 Mchar/s on one core, and the corpus only grows. It is
    also embarrassingly parallel: blake2b is deterministic, so the answer does
    not depend on which worker computed it or in what order. Measured on 48.6M
    characters of real records: 6.80s on one core, 1.99s across eight, results
    identical.

    `jobs=1` stays in this process. A pool that never had a second worker is
    just a slower way to call a function, and it keeps a profiler, a debugger
    and the tests in one process.
    """
    if jobs <= 1:
        for record in records:
            yield sketch_record(record["text"])
        return
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for batch in chunk_texts((r["text"] for r in records), window):
            yield from pool.map(sketch_record, batch,
                                chunksize=max(1, len(batch) // (jobs * 4)))


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


def build_postings(set_sketches: Sequence[Set[int]]) -> Dict[int, List[int]]:
    """Sketch value -> the documents holding it, each list in ascending order.

    The inverted index behind the candidate prefilter: documents sharing no
    sketch value at all cannot be duplicates, and that is the overwhelming
    majority of pairs.
    """
    postings: Dict[int, List[int]] = defaultdict(list)
    for idx, values in enumerate(set_sketches):
        for value in values:
            postings[value].append(idx)
    return postings


def neighbours(i: int, set_sketches: Sequence[Set[int]],
               postings: Dict[int, List[int]]) -> Set[int]:
    """Documents after `i` sharing at least one sketch value with it.

    Only those after it: every pair is examined once, from its lower index.
    Postings are ascending by construction, so bisect skips the prefix rather
    than filtering it.
    """
    found: Set[int] = set()
    for value in set_sketches[i]:
        docs = postings[value]
        found.update(docs[bisect_right(docs, i):])
    return found


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


# Licence tiers, worst-first, as a keeper must be chosen for the MOST
# restrictive track a cluster could be built into.
#
# A cluster is resolved once and every track is then built from the same marks.
# So if a green record is marked as a duplicate of an amber keeper, a Track A
# build drops the duplicate for being a duplicate AND the keeper for being
# non-redistributable, and the text disappears from the published corpus with
# nothing reporting it. Ranking green above amber before anything else makes
# that impossible: whenever a cluster contains a redistributable witness, the
# survivor is one.
TIER_RANK = {"green": 0, "amber": 1, "unknown": 2, "eval": 3}


def source_tiers() -> Dict[str, str]:
    """source id -> licence tier, from the manifest."""
    try:
        import yaml
    except ImportError:
        return {}
    path = ROOT / "manifest" / "sources.yaml"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh)
    return {s["id"]: s.get("tier", "unknown") for s in manifest.get("sources", [])}


_TIERS: Dict[str, str] = {}


def priority(record: Dict[str, Any]) -> Tuple[int, int]:
    """Keeper ranking: licence tier first, then curation quality."""
    global _TIERS
    if not _TIERS:
        _TIERS = source_tiers() or {"": "green"}
    tier = _TIERS.get(record.get("source", ""), "unknown")
    try:
        rank = SOURCE_PRIORITY.index(record.get("source", ""))
    except ValueError:
        rank = len(SOURCE_PRIORITY)
    return (TIER_RANK.get(tier, len(TIER_RANK)), rank)


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


# Which record fields carry a track's verdict. dedupe writes two — Track A's
# marks must not move when a research-only source is added — and anything
# reading them has to say which it means.
TRACK_SUFFIX = {"A": "", "B": "_b"}


def verify_track(records: List[Dict[str, Any]], track: str, probes: int,
                 span: int, floor: float) -> List[Dict[str, Any]]:
    """Check every duplicate this track marked really occurs inside its keeper.

    Sketch-based similarity is an estimate, and containment derived from it gets
    noisy when one record dwarfs the other. This is the ground truth: sample
    `probes` random word-spans from the duplicate and test literal membership in
    the keeper. Cross-edition duplicates land around 50-90%; part-of-collection
    duplicates at 100%; a genuine false positive would score near zero.

    A single probe is not enough — the Excerpta Valesiana looked like a false
    positive on its opening alone and verifies at 70% across 40 probes.
    """
    suffix = TRACK_SUFFIX[track]
    by_id = {rec["id"]: rec for rec in records}
    cache: Dict[str, str] = {}

    def folded(rec: Dict[str, Any]) -> str:
        if rec["id"] not in cache:
            cache[rec["id"]] = " ".join(fold(rec["text"]))
        return cache[rec["id"]]

    rng = random.Random(7)
    results: List[Tuple[Optional[float], Dict[str, Any], Dict[str, Any]]] = []
    for rec in records:
        if not rec.get("is_duplicate" + suffix):
            continue
        keeper = by_id.get(rec.get("duplicate_of" + suffix))
        if keeper is None:
            print(f"  DANGLING duplicate_of{suffix}: {rec['id']}", file=sys.stderr)
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
    print(f"\nTrack {track} — {len(results)} marked duplicate(s)")
    if not results:
        print("  (none marked)")
        return []
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
    print(f"  checked {len(results)}; {len(suspect)} below {floor:.0%} verbatim "
          f"overlap ({sum(r['n_words'] for r in suspect):,} words)")
    return suspect


def verify(records: List[Dict[str, Any]], tracks: Sequence[str] = ("A", "B"),
           probes: int = 40, span: int = 8, floor: float = 0.15) -> int:
    """Verify each track's marks in turn.

    Both by default. dedupe resolves the tracks separately and writes two
    independent verdicts, but this checked only `is_duplicate` — Track A's —
    so every drop LATIN-LII's corpus was built on went unverified. A check
    that silently covers half of what it is pointed at is worse than one that
    admits it covers half.
    """
    suspect = []
    for track in tracks:
        suspect.extend(verify_track(records, track, probes, span, floor))
    return 1 if suspect else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="analyse only, write nothing")
    ap.add_argument("--verify", action="store_true",
                    help="verbatim-check existing duplicate marks and exit")
    ap.add_argument("--track", action="append", choices=sorted(TRACK_SUFFIX),
                    metavar="TRACK",
                    help="which track's marks --verify checks (repeatable); "
                         "default is both, because both are built")
    ap.add_argument("--jaccard", type=float, default=0.5,
                    help="Jaccard at or above which two works are the same (default 0.5)")
    ap.add_argument("--containment", type=float, default=0.8,
                    help="containment at or above which the smaller is redundant (default 0.8)")
    ap.add_argument("--show", type=int, default=25, help="clusters to print")
    ap.add_argument("--jobs", type=int, default=None, metavar="N",
                    help="processes used for sketching (default: every core; "
                         "1 keeps everything in this process)")
    args = ap.parse_args()

    records, by_file = load_all()
    if not records:
        sys.exit(f"no records in {INTERIM} — run extract.py and enrich.py first")

    if args.verify:
        return verify(records, tuple(args.track or sorted(TRACK_SUFFIX)))

    jobs = args.jobs if args.jobs is not None else (os.cpu_count() or 1)
    print(f"sketching {len(records):,} records across {jobs} process(es) ...",
          flush=True)
    sizes: List[int] = []
    sorted_sketches: List[Tuple[int, ...]] = []
    set_sketches: List[Set[int]] = []
    sketch_heartbeat = max(1, len(records) // 10)
    for done, (ordered, cardinality) in enumerate(sketch_all(records, jobs), 1):
        if done % sketch_heartbeat == 0:
            print(f"  sketched {done:,}/{len(records):,}", flush=True)
        sizes.append(cardinality)
        set_sketches.append(set(ordered))
        sorted_sketches.append(ordered)

    # Candidate prefilter: documents sharing no sketch value at all cannot be
    # duplicates, and that is the overwhelming majority of pairs. An inverted
    # index over sketch values finds the few that do, turning an O(n^2) sweep
    # into work proportional to actual overlap.
    postings = build_postings(set_sketches)

    # WHY THE PAIRS ARE STREAMED RATHER THAN COLLECTED
    #   The candidate set used to be materialised whole: every posting list of
    #   length m contributed m(m-1)/2 tuples to one Python set, and nothing
    #   bounded m. A shingle common to many witnesses — liturgy, scripture,
    #   citation boilerplate, the formulae that made MIN_CONTAINED necessary —
    #   is exactly the case where m grows, and the cost is quadratic in it. At
    #   12,988 records that set held 1,105,761 pairs; it is the corpus growing
    #   that makes it a problem, which is the direction this project only ever
    #   moves in.
    #
    #   Capping the posting lists would fix the memory and silently cost
    #   recall in the worst place: n witnesses of one work share nearly all
    #   their sketch values, so a cap low enough to help is a cap that hides
    #   exactly the duplicates worth finding. Scoring each document against
    #   its neighbours as they are found costs the same time, examines the
    #   identical set of pairs, and never holds more than one document's
    #   neighbourhood. Results are unchanged: `pairs` is sorted before anything
    #   reads it, so the order they were produced in was never observable.
    #
    #   Postings are ascending, so bisect skips the half of each list that
    #   would only re-derive pairs already scored from the other side.
    crowded = max((len(d) for d in postings.values()), default=0)
    upper = sum(len(d) * (len(d) - 1) // 2 for d in postings.values())
    print(f"candidate pairs after prefilter: at most {upper:,} "
          f"(of {len(records)*(len(records)-1)//2:,} possible); "
          f"most-shared shingle is in {crowded:,} document(s)", flush=True)

    # A heartbeat, because this loop is the long pole and used to run silent:
    # at five sources it scores over a million pairs and says nothing for
    # minutes, which is indistinguishable from a hang to anything watching.
    pairs: List[Tuple[float, float, int, int]] = []
    heartbeat = max(1, len(records) // 20)
    scored = 0
    for i in range(len(records)):
        if (i + 1) % heartbeat == 0:
            print(f"  scoring pairs: document {i+1:,}/{len(records):,}, "
                  f"{scored:,} pairs scored, {len(pairs):,} above threshold",
                  flush=True)
        for j in neighbours(i, set_sketches, postings):
            scored += 1
            est = jaccard(sorted_sketches[i], sorted_sketches[j],
                          set_sketches[i], set_sketches[j])
            cont = containment(est, sizes[i], sizes[j])
            # No Jaccard floor before the containment test. A short text inside
            # a long one has a legitimately tiny Jaccard — Salvian's 188-word
            # appendix sits at J=0.03 inside his Epistulae while being 84%
            # contained — so any cheap "skip if J is low" gate silently defeats
            # containment detection exactly where sizes differ most, which is
            # where it is needed.
            if est >= args.jaccard:
                pairs.append((est, cont, i, j))
            elif cont >= args.containment and min(sizes[i], sizes[j]) >= MIN_CONTAINED:
                pairs.append((est, cont, i, j))

    pairs.sort(reverse=True)
    print(f"scored {scored:,} candidate pairs; "
          f"candidate duplicate pairs: {len(pairs)}\n")

    # Direct pairwise evidence, consulted by resolve() below.
    best_pair = {(min(i, j), max(i, j)): (e, c) for e, c, i, j in pairs}

    def resolve(eligible: Set[int]) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
        """Cluster and choose keepers among `eligible` records only.

        Restricting the pairs, not just the keeper choice, is what makes a
        track's result independent of records outside it. A record from another
        track can otherwise bridge two clusters that would have stayed separate
        — union-find is transitive — and that changes which witness survives
        among records the other track never contained.

        Measured 2026-08-25: ingesting LASLA, whose 19 authors are Perseus's
        core, moved 256,605 words out of Track A while adding 85,352 to Track B.
        Almost none of that was a green record losing to an amber keeper; it was
        green-on-green churn caused by amber records joining their clusters.
        Adding research-only material must not touch the published corpus.
        """
        parent = {i: i for i in eligible}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for _, _, i, j in pairs:
            if i not in parent or j not in parent:
                continue
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)

        grouped: Dict[int, List[int]] = defaultdict(list)
        for idx in eligible:
            grouped[find(idx)].append(idx)
        grouped = {root: m for root, m in grouped.items() if len(m) > 1}

        # A record is only ever discarded in favour of one it was DIRECTLY
        # measured against. Union-find is transitive and textual similarity is
        # not: patristic and medieval prose shares scripture, liturgy and
        # citation, so A~B and B~C chains into one blob where a single keeper
        # swallows everything.
        #
        # Measured before this guard: Jerome's Epistulae, 290,455 words, was
        # absorbing 401 records and 12,831,872 words — Remigius on the Psalms,
        # Haymo's homilies, Augustine on John, none of them Jerome's letters and
        # none of them ever compared with them. That is 16% of the corpus
        # discarded on evidence that did not exist.
        #
        # Clustering still finds the candidates; it just no longer licenses a
        # drop on its own. Members with no direct pair to the keeper are
        # re-resolved among themselves, so genuine sub-groups inside a bridged
        # cluster still collapse.
        keepers: Dict[int, int] = {}
        for members in grouped.values():
            remaining = list(members)
            while len(remaining) > 1:
                ordered = choose_keeper(remaining, records)
                keeper, rest = ordered[0], ordered[1:]
                attached = [i for i in rest
                            if (min(i, keeper), max(i, keeper)) in best_pair]
                for loser in attached:
                    keepers[loser] = keeper
                remaining = [i for i in rest if i not in set(attached)]
        return grouped, keepers

    # One pass per track. Track A sees only redistributable records, so its
    # marks are identical whether or not any amber source has been fetched.
    tiers_by_source = source_tiers()
    def tier_of(i: int) -> str:
        return tiers_by_source.get(records[i].get("source", ""), "unknown")

    eligible_a = {i for i in range(len(records)) if tier_of(i) == "green"}
    eligible_b = {i for i in range(len(records)) if tier_of(i) in ("green", "amber")}

    clusters, keep_a = resolve(eligible_a)
    _, keep_b = resolve(eligible_b) if eligible_b != eligible_a else (None, keep_a)


    # WHAT THIS REPORTS, AND WHY IT IS NOT THE CLUSTER
    #   A cluster is a set of candidates, not a verdict. resolve() drops a
    #   record only in favour of one it was DIRECTLY measured against, because
    #   union-find is transitive and textual similarity is not — the guard
    #   above this exists because Jerome's Epistulae was otherwise absorbing
    #   401 records and 12.8M words it had never been compared with.
    #
    #   This summary used to sum whole clusters instead: keeper = members[0],
    #   everything else counted as dropped. So the headline figure was the
    #   number the guard was written to prevent, printed as though it were
    #   what the pipeline does, and `--report` overstated the loss by exactly
    #   the amount the guard saves. It is now read off the marks that will
    #   actually be written.
    dropped_words = sum(records[i]["n_words"] for i in keep_a)
    shown = 0
    print(f"clusters: {len(clusters)}")
    for members in sorted(clusters.values(), key=lambda m: -sum(records[i]["n_words"] for i in m)):
        members[:] = choose_keeper(members, records)
        keeper = members[0]
        if shown >= args.show:
            continue
        shown += 1
        k = records[keeper]
        print(f"\n  KEEP  {k['n_words']:>7,}w  {k['source'][:22]:<22} "
              f"{(k['author'] or '?')[:22]:<22} {(k['work'] or '?')[:34]}")
        for i in members[1:]:
            r = records[i]
            against = keep_a.get(i)
            if against is None:
                # In the cluster, but attached to nothing it was measured
                # against. It survives, and saying so is the point.
                verdict, sim = "stays", "(no direct pair)"
            else:
                pair = best_pair.get((min(against, i), max(against, i)))
                sim = f"J={pair[0]:.2f} C={pair[1]:.2f}" if pair else "(indirect)"
                if against != keeper:
                    sim += f" -> {(records[against]['work'] or '?')[:20]}"
                verdict = "drop "
            print(f"  {verdict} {r['n_words']:>7,}w  {r['source'][:22]:<22} "
                  f"{(r['author'] or '?')[:22]:<22} {(r['work'] or '?')[:34]}  {sim}")

    total = sum(r["n_words"] for r in records)
    clustered = sum(records[i]["n_words"] for m in clusters.values() for i in m[1:])
    print(f"\nwould drop {dropped_words:,} words of {total:,} "
          f"({100*dropped_words/total:.1f}%) — Track A")
    print(f"  {clustered - dropped_words:,} further words are in a cluster but "
          f"attached to no record they were directly measured against, and stay")

    if args.report:
        print("\n--report: nothing written")
        return 0

    keep_of = keep_a
    index = {id(rec): i for i, rec in enumerate(records)}
    for path, rows in by_file.items():
        for rec in rows:
            i = index[id(rec)]
            # Two independent verdicts, one per track. build.py reads the one
            # matching the track it is assembling.
            for suffix, marks in (("", keep_a), ("_b", keep_b)):
                if i in marks:
                    winner = records[marks[i]]
                    pair = best_pair.get((min(i, marks[i]), max(i, marks[i])))
                    rec["is_duplicate" + suffix] = True
                    rec["duplicate_of" + suffix] = winner["id"]
                    rec["duplicate_similarity" + suffix] = (
                        round(pair[0], 4) if pair else None)
                else:
                    rec["is_duplicate" + suffix] = False
                    rec["duplicate_of" + suffix] = None
                    rec["duplicate_similarity" + suffix] = None
        tmp = Path(path).with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for rec in rows:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp.replace(Path(path))

    print(f"\nmarked {len(keep_a):,} records as duplicates in Track A, "
          f"{len(keep_b):,} in Track B, across {len(by_file)} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

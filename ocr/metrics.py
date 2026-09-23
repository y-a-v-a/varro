#!/usr/bin/env python3
"""What "repaired" means, as arithmetic. No model, no corpus, no randomness
except the bootstrap's own seed.

THE UNIT IS THE WORD OF THE TRUE TEXT
    A repairer is handed noisy text and returns text. Both are compared with
    the truth character by character, and the comparison is then read off one
    TRUE word at a time: is this word standing in the output, whole, with a
    word boundary either side of it? That is the only question asked, and it
    is asked of the noisy text and of the repaired text alike.

    Counting by true word is what lets segmentation errors be scored at all.
    `de the sauro` has three words and `dethesauro` has one; neither lines up
    word-for-word with `de thesauro`, and any metric built on pairing noisy
    words with true words has to invent a rule for that. Projected onto the
    truth there is nothing to invent: `thesauro` is not standing in either
    text, so it is damaged in both.

FIVE OUTCOMES, BECAUSE "ACCURACY" HIDES THE ONE THAT MATTERS
                        standing after repair     not standing after repair
    standing before     kept                      BROKEN
    damaged before      fixed                     missed / MISCORRECTED

    `missed` is a damaged word the repairer left exactly as it found it.
    `miscorrected` is a damaged word it changed into something still wrong.
    `broken` is a word that was right until the repairer touched it.

    Character error rate nets all of this into one number, and a repairer that
    fixes thirty scannos while rewriting ten proper names into commoner words
    shows a healthy CER reduction. For a corpus that is the wrong trade: a
    scanno is visible to the next audit and a fluent wrong word is not. So
    every report carries `broken` and `miscorrected` separately, and SILENT
    errors -- wrong words that are themselves attested Latin -- before and
    after. A repairer can lower CER and raise the silent count, and one that
    does has made the text worse in the way that cannot be found again.
"""

import random
import re
from collections import Counter
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# (kind, i, j): kind is '=' match, 's' substitution, 'd' a true character with
# nothing opposite it, 'i' a hypothesis character with nothing opposite it.
# i indexes the truth and j the hypothesis; for 'd' and 'i' the index on the
# empty side is where the missing character would have stood.
Op = Tuple[str, int, int]

WORD = re.compile(r"\S+")

# Trimmed before a word is looked up, never before it is compared: `tres,` and
# `tres` are different strings to the outcome table and the same entry to the
# lexicon.
LEAD_PUNCT = "(\"'"
TRAIL_PUNCT = ".,;:!?)\"'"

OUTCOMES = ("kept", "broken", "fixed", "missed", "miscorrected")


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def _banded(a: str, b: str, k: int) -> Optional[Tuple[int, List[Op]]]:
    """Levenshtein alignment restricted to |i - j| <= k, or None if the true
    distance exceeds k.

    A path of cost d never strays more than d cells from the diagonal, so a
    result <= k found inside the band is the exact distance and not a bound.
    """
    n, m = len(a), len(b)
    if abs(n - m) > k:
        return None
    inf = n + m + 1
    prev_lo, prev = 0, list(range(min(m, k) + 1))
    back: List[Tuple[int, List[str]]] = [(0, ["i"] * len(prev))]

    for i in range(1, n + 1):
        lo, hi = max(0, i - k), min(m, i + k)
        row: List[int] = []
        ops: List[str] = []
        ca = a[i - 1]
        for j in range(lo, hi + 1):
            best, op = inf, "d"
            p = j - 1 - prev_lo
            # Diagonal first and strict < after it, so that among equal-cost
            # alignments a substitution is preferred to an insert-delete pair.
            # The outcome table does not depend on which is chosen; the
            # fitted channel does, and `e -> c` is a more useful thing to have
            # learned than `e -> nothing` next to `nothing -> c`.
            if 0 <= p < len(prev):
                same = ca == b[j - 1]
                best, op = prev[p] + (0 if same else 1), "=" if same else "s"
            p += 1
            if 0 <= p < len(prev) and prev[p] + 1 < best:
                best, op = prev[p] + 1, "d"
            if j - 1 >= lo and row[j - 1 - lo] + 1 < best:
                best, op = row[j - 1 - lo] + 1, "i"
            row.append(best)
            ops.append(op)
        back.append((lo, ops))
        prev_lo, prev = lo, row

    distance = prev[m - prev_lo]
    if distance > k:
        return None

    path: List[Op] = []
    i, j = n, m
    while i > 0 or j > 0:
        lo, ops = back[i]
        op = ops[j - lo]
        if op in "=s":
            i, j = i - 1, j - 1
            path.append((op, i, j))
        elif op == "d":
            i -= 1
            path.append((op, i, j))
        else:
            j -= 1
            path.append((op, i, j))
    path.reverse()
    return distance, path


def align(truth: str, hyp: str) -> Tuple[int, List[Op]]:
    """Exact edit distance and one minimal alignment.

    The band is doubled until the answer fits inside it. OCR text is mostly
    right, so the distance is small next to the length and a narrow band is
    nearly always enough: a 400-character passage at 3% error needs 33 columns
    a row rather than 400, which is the difference between a benchmark that
    takes a minute and one that takes ten.
    """
    k = 8
    limit = max(len(truth), len(hyp), 1)
    while True:
        found = _banded(truth, hyp, min(k, limit))
        if found is not None:
            return found
        k *= 2


def edit_regions(truth: str, hyp: str,
                 ops: Optional[List[Op]] = None) -> List[Tuple[str, str]]:
    """Each maximal run of non-matching columns, as (true text, what stood there).

    Runs rather than single columns because the errors worth learning are not
    one character wide. `m` read as `rn` aligns as a substitution and an
    insertion side by side, and only taken together do they say `m -> rn`.
    """
    if ops is None:
        _, ops = align(truth, hyp)
    regions: List[Tuple[str, str]] = []
    t: List[str] = []
    h: List[str] = []
    for kind, i, j in ops:
        if kind == "=":
            if t or h:
                regions.append(("".join(t), "".join(h)))
                t, h = [], []
            continue
        if kind in "sd":
            t.append(truth[i])
        if kind in "si":
            h.append(hyp[j])
    if t or h:
        regions.append(("".join(t), "".join(h)))
    return regions


# ---------------------------------------------------------------------------
# Projection onto the words of the truth
# ---------------------------------------------------------------------------

class Standing:
    """One true word, and what occupies its place in a hypothesis."""
    __slots__ = ("word", "intact", "text", "span")

    def __init__(self, word: str, intact: bool, text: str,
                 span: Tuple[int, int]):
        self.word, self.intact, self.text, self.span = word, intact, text, span


def project(truth: str, hyp: str,
            ops: Optional[List[Op]] = None) -> List[Standing]:
    """For every word of the truth: is it standing in `hyp`, and if not, what is?

    `text` is the whole whitespace-delimited stretch of the hypothesis that
    the word's characters landed in. For a merge that is the run-together
    token, reported for both of the true words inside it; for a split it is
    every fragment. A word is intact only if that stretch IS the word -- same
    letters, and a boundary either side -- so `urbem` inside `inurbem` is not
    intact however perfectly its five letters aligned.
    """
    if ops is None:
        _, ops = align(truth, hyp)
    # before[i]: hypothesis characters consumed before true character i.
    before = [0] * (len(truth) + 1)
    present = [False] * len(truth)
    consumed = 0
    for kind, i, j in ops:
        if kind == "i":
            consumed += 1
            continue
        before[i] = consumed
        if kind != "d":
            present[i] = True
            consumed += 1
    before[len(truth)] = consumed

    out: List[Standing] = []
    for match in WORD.finditer(truth):
        s, e = match.span()
        lo = before[s]
        hi = before[e - 1] + (1 if present[e - 1] else 0)
        wide_lo, wide_hi = lo, hi
        while wide_lo > 0 and not hyp[wide_lo - 1].isspace():
            wide_lo -= 1
        while wide_hi < len(hyp) and not hyp[wide_hi].isspace():
            wide_hi += 1
        text = hyp[wide_lo:wide_hi]
        intact = (wide_lo, wide_hi) == (lo, hi) and text == match.group()
        out.append(Standing(match.group(), intact, text, (wide_lo, wide_hi)))
    return out


def outcome(noisy: Standing, repaired: Standing) -> str:
    if noisy.intact:
        return "kept" if repaired.intact else "broken"
    if repaired.intact:
        return "fixed"
    return "missed" if repaired.text == noisy.text else "miscorrected"


def core(word: str) -> str:
    """The word as the lexicon knows it: edge punctuation off, lower case."""
    return word.lstrip(LEAD_PUNCT).rstrip(TRAIL_PUNCT).lower()


def attested(text: str, lexicon: Set[str]) -> bool:
    """Every fragment is a known word -- so nothing here would catch the eye."""
    parts = [core(p) for p in text.split()]
    return bool(parts) and all(p in lexicon for p in parts)


# ---------------------------------------------------------------------------
# One passage -> counts
# ---------------------------------------------------------------------------

def tally(truth: str, noisy: str, repaired: str, lexicon: Set[str],
          word_classes: Optional[Sequence[Iterable[str]]] = None) -> Counter:
    """Everything a report needs from one passage, as additive counts.

    Additive on purpose: a corpus-level rate is a ratio of sums, never a mean
    of per-passage ratios, and the bootstrap resamples these counters whole.

    Keys, where <o> is one of OUTCOMES:
        chars, dist_before, dist_after          character error, micro-averaged
        words, <o>                              the outcome table
        real:<o> / nonword:<o>                  damaged words, by whether the
                                                damage is itself attested Latin
        oov:<o> / inlex:<o>                     by whether the TRUE word is in
                                                the lexicon (names and hapax
                                                are where `broken` lives)
        class:<c>:<o>                           by injected error class, when
                                                the benchmark knows it
        silent_before, silent_after             wrong AND attested
    """
    d_before, ops_n = align(truth, noisy)
    d_after, ops_r = align(truth, repaired)
    before = project(truth, noisy, ops_n)
    after = project(truth, repaired, ops_r)

    c: Counter = Counter()
    c["chars"] = len(truth)
    c["dist_before"] = d_before
    c["dist_after"] = d_after
    c["words"] = len(before)

    for k, (n, r) in enumerate(zip(before, after)):
        o = outcome(n, r)
        c[o] += 1
        c[("inlex:" if core(n.word) in lexicon else "oov:") + o] += 1
        if not n.intact:
            real = attested(n.text, lexicon)
            c[("real:" if real else "nonword:") + o] += 1
            c["silent_before"] += real
            if word_classes is not None:
                for cls in set(word_classes[k]):
                    c[f"class:{cls}:{o}"] += 1
        if not r.intact:
            c["silent_after"] += attested(r.text, lexicon)
    return c


def word_labels(truth: str, noisy: str, lexicon: Set[str]) -> List[Optional[str]]:
    """One label per NOISY word, for judging detection.

    None for a word that is standing as it should; otherwise 'real' or
    'nonword'. This is the one place the noisy text's own words are the unit,
    because a detector is handed them and nothing else.
    """
    good = {s.span for s in project(truth, noisy) if s.intact}
    labels: List[Optional[str]] = []
    for match in WORD.finditer(noisy):
        if match.span() in good:
            labels.append(None)
        else:
            labels.append("real" if core(match.group()) in lexicon else "nonword")
    return labels


# ---------------------------------------------------------------------------
# Counts -> the numbers a report quotes
# ---------------------------------------------------------------------------

def _ratio(a: float, b: float) -> Optional[float]:
    return a / b if b else None


def damaged(c: Counter, prefix: str = "") -> int:
    return sum(c[prefix + o] for o in ("fixed", "missed", "miscorrected"))


def recall(c: Counter, prefix: str = "") -> Optional[float]:
    """Of the damaged words, how many are now standing."""
    return _ratio(c[prefix + "fixed"], damaged(c, prefix))


def precision(c: Counter) -> Optional[float]:
    """Of the true words whose place the repairer changed, how many ended right."""
    return _ratio(c["fixed"], c["fixed"] + c["miscorrected"] + c["broken"])


def harm(c: Counter, prefix: str = "") -> Optional[float]:
    """Words broken per 1,000 that were standing. The number to watch."""
    return _ratio(1000 * c[prefix + "broken"],
                  c[prefix + "kept"] + c[prefix + "broken"])


def cer_reduction(c: Counter) -> Optional[float]:
    """Relative: 1.0 removed every error, 0 did nothing, negative made it worse."""
    return _ratio(c["dist_before"] - c["dist_after"], c["dist_before"])


def summarise(c: Counter) -> Dict[str, Optional[float]]:
    return {
        "cer_before": _ratio(c["dist_before"], c["chars"]),
        "cer_after": _ratio(c["dist_after"], c["chars"]),
        "cer_reduction": cer_reduction(c),
        "word_error_before": _ratio(damaged(c), c["words"]),
        "word_error_after": _ratio(
            c["missed"] + c["miscorrected"] + c["broken"], c["words"]),
        "recall": recall(c),
        "recall_real_word": recall(c, "real:"),
        "recall_nonword": recall(c, "nonword:"),
        "precision": precision(c),
        "harm_per_1000": harm(c),
        "harm_per_1000_oov": harm(c, "oov:"),
        "harm_per_1000_inlex": harm(c, "inlex:"),
        "silent_per_1000_before": _ratio(1000 * c["silent_before"], c["words"]),
        "silent_per_1000_after": _ratio(1000 * c["silent_after"], c["words"]),
    }


def average_precision(scores: Sequence[float],
                      positive: Sequence[bool]) -> Optional[float]:
    """Area under precision-recall, for a detector that ranks words.

    Average precision rather than ROC because the classes are lopsided: at 1%
    character error roughly one word in twenty is damaged, and ROC flatters
    any ranking under that imbalance. The floor is the share of positives.
    Ties are broken pessimistically -- positives last -- so that a detector
    giving every word the same score earns the floor and not a lucky order.
    """
    total = sum(positive)
    if not total:
        return None
    order = sorted(range(len(scores)), key=lambda k: (-scores[k], positive[k]))
    hits, area = 0, 0.0
    for rank, k in enumerate(order, 1):
        if positive[k]:
            hits += 1
            area += hits / rank
    return area / total


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------

def _total(tallies: Sequence[Counter], picks: Iterable[int]) -> Counter:
    out: Counter = Counter()
    for k in picks:
        out.update(tallies[k])
    return out


def bootstrap(tallies: Sequence[Counter],
              stats: Dict[str, Callable[[Counter], Optional[float]]],
              reps: int = 1000, seed: int = 0,
              against: Optional[Sequence[Counter]] = None
              ) -> Dict[str, Optional[Tuple[float, float]]]:
    """95% intervals for each of `stats`, resampling PASSAGES, not words.

    Words inside a passage are not independent -- one hard proper name recurs,
    one merge damages two neighbours -- so resampling words would report an
    interval several times too narrow. The passage is the unit that was drawn
    at random, so the passage is the unit resampled.

    With `against`, each interval is for stat(tallies) - stat(against) over
    the SAME resampled passages. Two systems run on one benchmark share most
    of their luck; differencing inside each resample cancels it, which is why
    a paired interval can exclude zero when the two separate ones overlap.
    """
    n = len(tallies)
    if n == 0 or (against is not None and len(against) != n):
        return {name: None for name in stats}
    rng = random.Random(seed)
    values: Dict[str, List[float]] = {name: [] for name in stats}
    for _ in range(reps):
        picks = [rng.randrange(n) for _ in range(n)]
        mine = _total(tallies, picks)
        theirs = _total(against, picks) if against is not None else None
        for name, stat in stats.items():
            v = stat(mine)
            if theirs is not None:
                w = stat(theirs)
                v = None if v is None or w is None else v - w
            if v is not None:
                values[name].append(v)
    out: Dict[str, Optional[Tuple[float, float]]] = {}
    for name, vs in values.items():
        if len(vs) < reps // 2:
            out[name] = None
            continue
        vs.sort()
        out[name] = (vs[int(0.025 * len(vs))],
                     vs[min(len(vs) - 1, int(0.975 * len(vs)))])
    return out


# ---------------------------------------------------------------------------
# Finding a passage inside a page
# ---------------------------------------------------------------------------

def locate(pattern: str, text: str) -> Tuple[int, int, int]:
    """Where in `text` does `pattern` occur, approximately? (start, end, distance)

    Sellers' variant of the edit-distance table: the top row is all zeros, so
    a match may begin anywhere in the text for free, and the answer is the
    smallest value in the bottom row, so it may end anywhere too. Real scans
    need this where `align` will not do: a page of OCR carries a running head,
    footnotes and an apparatus that the transcription leaves out, and a global
    alignment would have to pay for every character of them.

    The start is recovered by carrying it through the table rather than by a
    backtrace, which keeps memory at two rows however long the page is.
    """
    m, n = len(pattern), len(text)
    if not m:
        return 0, 0, 0
    prev = [0] * (n + 1)
    prev_start = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        cur_start = [0] * (n + 1)
        pc = pattern[i - 1]
        for j in range(1, n + 1):
            best, start = prev[j - 1] + (pc != text[j - 1]), prev_start[j - 1]
            if prev[j] + 1 < best:
                best, start = prev[j] + 1, prev_start[j]
            if cur[j - 1] + 1 < best:
                best, start = cur[j - 1] + 1, cur_start[j - 1]
            cur[j], cur_start[j] = best, start
        prev, prev_start = cur, cur_start
    end = min(range(n + 1), key=lambda j: prev[j])
    return prev_start[end], end, prev[end]

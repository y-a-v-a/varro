#!/usr/bin/env python3
"""The noise: a channel that damages clean Latin, and a way to learn a channel
back from damaged/clean pairs.

WHY THERE ARE TWO THINGS IN THIS FILE AND THEY MUST NOT MEET
    `corrupt` is the benchmark's side. It knows CONFUSIONS, the table of what
    a scanner does to a page.

    `fit` is the repairer's side. It is handed pairs of (true, noisy) text
    from the DEV split and counts what changed into what. That is all a
    repairer is allowed to know about the noise, because it is all anyone
    repairing real scans would know: nobody is given the scanner's confusion
    matrix, they hand-correct a few pages and estimate it.

    A repairer that imported CONFUSIONS would be inverting a function it had
    been shown, and would post a number no real page could reproduce.
    tests/test_ocr.py fails if ocr/repairers.py ever names it.

WHERE THE TABLE CAME FROM, AND WHAT THAT MAKES TIER 1 WORTH
    Written by hand from the confusions the OCR literature reports again and
    again for roman type -- the minim cluster (m/rn/in/ni, u/n, ii/u), closed
    against open bowls (c/e/o), ascender pairs (l/i/t/1, h/b/li, d/cl), the
    long s read as f, digits for letters -- plus the segmentation errors that
    character tables leave out. The weights are judgement, not measurement.

    So Tier 1 measures repair of THIS noise. It is the right tool for
    comparing two repairers and for the clean-text control, and the wrong
    thing to quote as "fixes N% of OCR errors". That sentence needs Tier 2,
    real scanner output, and the honest way to improve this table is to run
    `fit` over Tier 2 pairs and replace these weights with those counts.

Usage:
    python ocr/channel.py "Gallia est omnis diuisa in partes tres" --rate 0.08
"""

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import edit_regions  # noqa: E402

# (what the page says, what the scanner read, weight, class)
CONFUSIONS: List[Tuple[str, str, float, str]] = [
    # shape: one glyph read as two, or two as one. The reason a character-
    # level spell checker is not enough -- these change the letter count.
    ("m", "rn", 6, "shape"), ("m", "in", 2, "shape"), ("m", "ni", 1, "shape"),
    ("rn", "m", 3, "shape"), ("in", "m", 1, "shape"), ("ni", "m", 1, "shape"),
    ("h", "li", 2, "shape"), ("li", "h", 1, "shape"),
    ("d", "cl", 2, "shape"), ("cl", "d", 2, "shape"),
    ("u", "ii", 2, "shape"), ("ii", "u", 2, "shape"),
    ("n", "ri", 1, "shape"), ("ri", "n", 1, "shape"),
    # letter: one for one. These are the ones that make real words --
    # cum/eum, uiri/uiti, nos/uos -- which is where a dictionary goes blind.
    ("e", "c", 5, "letter"), ("c", "e", 4, "letter"),
    ("u", "n", 5, "letter"), ("n", "u", 5, "letter"),
    ("i", "l", 3, "letter"), ("l", "i", 3, "letter"),
    ("s", "f", 4, "letter"), ("f", "s", 2, "letter"),
    ("h", "b", 2, "letter"), ("b", "h", 1, "letter"),
    ("a", "o", 2, "letter"), ("o", "a", 1, "letter"),
    ("o", "c", 2, "letter"), ("e", "o", 2, "letter"),
    ("t", "c", 2, "letter"), ("c", "t", 1, "letter"),
    ("t", "l", 1, "letter"), ("l", "t", 1, "letter"),
    ("r", "t", 1, "letter"), ("t", "r", 1, "letter"),
    ("q", "g", 1, "letter"), ("g", "q", 1, "letter"),
    # digit: never a real word, always a non-word. The easy class, kept so
    # that "easy" has a number next to it.
    ("l", "1", 3, "digit"), ("i", "1", 2, "digit"), ("o", "0", 2, "digit"),
    ("s", "5", 1, "digit"), ("b", "6", 1, "digit"), ("g", "9", 1, "digit"),
]

# Generic damage, applicable at any letter. Weighted against the table above.
DROP_WEIGHT = 1.5      # a letter lost outright
SPLIT_WEIGHT = 1.5     # a space that was never there: `the sauro`
SPECK_WEIGHT = 0.7     # dirt read as punctuation
RANDOM_WEIGHT = 1.0    # any letter for any other; the part no table predicts
SPECKS = ".,'-"
LETTERS = "abcdefghilmnopqrstux"

# The total weight of an average letter of running Latin, measured on the
# held-out split. Damage starts at a letter with probability
# rate * (that letter's total weight) / WEIGHT_NORM, so `rate` stays close to
# the share of letters damaged while an `m`, which has six ways to go wrong,
# still goes wrong more often than a `p`, which has none of its own. Dividing
# each letter's options by their own total instead would damage every letter
# equally often and make the generic classes -- the only ones a `p` has --
# half of all errors, which is not what a scanned page looks like.
WEIGHT_NORM = 9.0

# A space is lost less readily than a letter is misread; without this a
# passage at 8% would run a fifth of its words together.
MERGE_FACTOR = 0.6


class Edit(NamedTuple):
    start: int          # in the TRUE text
    end: int
    truth: str
    noisy: str
    cls: str


def table_fingerprint() -> str:
    """Changes whenever the noise does, so a lock file can notice."""
    blob = json.dumps([CONFUSIONS, DROP_WEIGHT, SPLIT_WEIGHT, SPECK_WEIGHT,
                       RANDOM_WEIGHT, SPECKS, LETTERS, WEIGHT_NORM, MERGE_FACTOR])
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _options(text: str, i: int) -> List[Tuple[str, str, float, str]]:
    ch = text[i]
    if not ("a" <= ch <= "z"):
        # Capitals and punctuation are left alone. Scanners do misread them;
        # leaving them out keeps every injected error one whose true form the
        # lexicon could in principle hold, and a capital is a proper name
        # often enough that damaging it would mostly measure the lexicon.
        return []
    found = [c for c in CONFUSIONS if text.startswith(c[0], i)]
    found.append((ch, "", DROP_WEIGHT, "drop"))
    found.append((ch, "?", RANDOM_WEIGHT, "random"))
    found.append((ch, ch + "?", SPECK_WEIGHT, "speck"))
    if i + 1 < len(text) and text[i + 1].isalpha() and i > 0 and text[i - 1].isalpha():
        found.append((ch, ch + " ", SPLIT_WEIGHT, "split"))
    return found


def corrupt(text: str, rate: float, rng: random.Random) -> Tuple[str, List[Edit]]:
    """Damage `text`, visiting each position once.

    `rate` is roughly the chance that damage STARTS at a character, not the
    character error rate that results -- `m -> rn` costs two edits and a split
    costs one. The benchmark therefore records the error rate it measured
    rather than the one it asked for.
    """
    out: List[str] = []
    edits: List[Edit] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == " ":
            inside = 0 < i < len(text) - 1 and text[i - 1] != " " and text[i + 1] != " "
            if inside and rng.random() < rate * MERGE_FACTOR:
                edits.append(Edit(i, i + 1, " ", "", "merge"))
            else:
                out.append(ch)
            i += 1
            continue
        options = _options(text, i)
        heat = sum(o[2] for o in options) / WEIGHT_NORM
        if not options or rng.random() >= rate * heat:
            out.append(ch)
            i += 1
            continue
        truth, noisy, _, cls = rng.choices(
            options, weights=[o[2] for o in options])[0]
        if cls == "random":
            noisy = rng.choice([c for c in LETTERS if c != ch])
        elif cls == "speck":
            noisy = ch + rng.choice(SPECKS)
        out.append(noisy)
        edits.append(Edit(i, i + len(truth), truth, noisy, cls))
        i += len(truth)
    return "".join(out), edits


def classes_by_word(text: str, edits: Sequence[Edit]) -> List[List[str]]:
    """Which error classes touched each word of the true text.

    Touch, not overlap: a lost space belongs to the word on each side of it,
    and both are damaged by it.
    """
    from metrics import WORD
    out: List[List[str]] = []
    for match in WORD.finditer(text):
        s, e = match.span()
        out.append([x.cls for x in edits if x.start <= e and x.end >= s])
    return out


# ---------------------------------------------------------------------------
# The repairer's side
# ---------------------------------------------------------------------------

class Channel(NamedTuple):
    """log P(scanner wrote `noisy` | page said `truth`), estimated from pairs."""
    logp: Dict[Tuple[str, str], float]
    pairs: int

    def ops(self) -> List[Tuple[str, str, float]]:
        return [(t, n, lp) for (t, n), lp in self.logp.items()]


def fit(pairs: Sequence[Tuple[str, str]], min_count: int = 2,
        max_len: int = 3) -> Channel:
    """Count what turned into what across aligned (true, noisy) pairs.

    min_count keeps a single accident from becoming a rule. max_len drops the
    regions where two errors landed side by side and the alignment cannot say
    which letters answer to which; they are real damage and useless evidence.

    The denominator for an operation is how often its true side occurred at
    all, so the result is a rate -- `e` was read as `c` in 0.4% of its
    appearances -- and a repairer can weigh it against a language model's
    opinion in the same units.
    """
    seen: Counter = Counter()
    chars = 0
    text: List[str] = []
    for truth, noisy in pairs:
        chars += len(truth)
        text.append(truth)
        for t, n in edit_regions(truth, noisy):
            if len(t) <= max_len and len(n) <= max_len:
                seen[(t, n)] += 1
    whole = "\n".join(text)
    logp: Dict[Tuple[str, str], float] = {}
    for (t, n), count in seen.items():
        if count < min_count:
            continue
        chances = whole.count(t) if t else chars
        if chances:
            logp[(t, n)] = math.log(min(1.0, count / chances))
    return Channel(logp, len(pairs))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text")
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    noisy, edits = corrupt(args.text, args.rate, random.Random(args.seed))
    print(noisy)
    for e in edits:
        print(f"  {e.cls:<7} {e.truth!r} -> {e.noisy!r} at {e.start}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

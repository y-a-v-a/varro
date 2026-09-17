#!/usr/bin/env python3
"""Which of two readings is better Latin, and where they part company.

THE SMALLEST EXPERIMENT THERE IS
    A perplexity says how surprised the model is by a text. Put two texts
    side by side that differ in ONE thing and the same number becomes a
    judgement about that thing: does the model know that `diuisa` agrees with
    `Gallia` and `diuisus` does not? That `in` wants the accusative for
    motion? That `ut` wants a subjunctive? A minimal pair is the smallest
    test that can be run on a language model, and the suite here runs a
    couple of dozen of them so the shape of what a 52M-parameter model of
    Latin has learned can be seen at a glance -- and, with several --name
    flags, what arrived with model size.

WHAT THE MARGIN MEANS
    Bits. +4 bits says the model finds the first reading sixteen times as
    likely as the second. The token table underneath says where that cost
    was paid: usually on the changed token itself, but often also AFTER it,
    when the wrong form makes the rest of the sentence harder to predict.
    That spill-over is the model using context, and it is the same mechanism
    lacuna.py relies on when it judges a supplement by the text after it.

WHAT IT CANNOT SEPARATE
    Grammar from frequency. `bonorum` is a commoner word than `bonarum`, and
    a model of Latin has learned both facts at once, so a pair that pits a
    rare right form against a common wrong one measures the sum. The suite
    marks the pairs where that is the story, and --benchmark sidesteps it:
    every corrupted form it scores is a word attested in the held-out split,
    so the model cannot win by noticing a non-word.

Usage:
    python3 demos/paria.py "puella rosam amat" "puella rosam amant"
    python3 demos/paria.py --suite
    python3 demos/paria.py --suite --name latin-vii --name latin-lii-latinitas
    python3 demos/paria.py --benchmark 200
"""

import argparse
import collections
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    DEFAULT_MODEL, PERIODS, ROOT, Loaded, announce_folding, bits, load,
    pieces, score_readings, to_corpus_orthography,
)
from lacuna import draw_trials  # noqa: E402


class Pair(NamedTuple):
    phenomenon: str
    a: str            # the reading a grammar would print
    b: str            # the same sentence with one thing wrong
    note: str = ""    # what the pair tests, or why it is unfair
    expected: Optional[str] = "a"   # None: no right answer, a preference


# THE SUITE
#   Grouped by what is being tested. Every pair changes as little as it can:
#   one ending, one word, one position. The `expected` column is "a" where a
#   grammar has an opinion and None where Latin word order genuinely allows
#   both, in which case the demo reports a preference and claims nothing.
#
#   `frequency trap` marks pairs whose wrong reading is the COMMONER string.
#   They are kept because a demo that hid its failures would be lying about
#   what the number measures.
SUITE: Tuple[Pair, ...] = (
    Pair("agreement: subject and verb",
         "puella rosam amat", "puella rosam amant",
         "singular subject, plural verb"),
    Pair("agreement: subject and verb",
         "milites castra muniunt", "milites castra munit",
         "plural subject, singular verb"),
    Pair("agreement: subject and verb, at a distance",
         "milites qui in castris manserant fortiter pugnauerunt",
         "milites qui in castris manserant fortiter pugnauit",
         "the verb is seven words from its subject"),
    Pair("agreement: gender of a participle",
         "Gallia est omnis diuisa in partes tres",
         "Gallia est omnis diuisus in partes tres",
         "feminine subject, masculine participle"),
    Pair("agreement: gender of an adjective",
         "magnus exercitus", "magna exercitus",
         "masculine noun, feminine adjective"),
    Pair("agreement: gender of an adjective, at a distance",
         "puella quam heri in foro uidi pulchra est",
         "puella quam heri in foro uidi pulcher est",
         "the adjective is seven words from its noun"),
    Pair("agreement: gerundive",
         "Carthago delenda est", "Carthago delendus est",
         "feminine city, masculine gerundive"),
    Pair("agreement: relative pronoun",
         "uir qui uenit", "uir quae uenit",
         "masculine antecedent, feminine relative"),
    Pair("agreement: relative pronoun",
         "mulier quae uenit", "mulier qui uenit",
         "feminine antecedent, masculine relative"),
    Pair("agreement: adjective and noun, genitive plural",
         "bonarum puellarum", "bonorum puellarum",
         "frequency trap: `bonorum` is the commoner string",
         ),
    Pair("case: after a preposition",
         "cum amico ambulabat", "cum amicum ambulabat",
         "`cum` takes the ablative"),
    Pair("case: after a preposition",
         "in urbem uenit", "in urbis uenit",
         "motion into: accusative, not genitive"),
    Pair("case: after a preposition",
         "de bello Gallico", "de bellum Gallicum",
         "`de` takes the ablative"),
    Pair("case: direct object",
         "Caesar Galliam uicit", "Caesar Gallia uicit",
         "accusative object, ablative given"),
    Pair("mood: subjunctive after ut",
         "rogo ut uenias", "rogo ut uenis",
         "indicative given"),
    Pair("mood: subjunctive in a cum-clause",
         "cum Caesar in Galliam uenisset, Heluetii",
         "cum Caesar in Galliam uenerat, Heluetii",
         "pluperfect indicative given"),
    Pair("indirect statement",
         "dixit se uenturum esse", "dixit se ueniet",
         "a finite verb where an infinitive belongs"),
    Pair("voice",
         "hostes urbem oppugnabant", "hostes urbem oppugnabantur",
         "passive with a direct object"),
    Pair("form of a preposition before a vowel",
         "ab urbe condita", "a urbe condita",
         "`a` is not written before a vowel"),
    Pair("form of a preposition before a consonant",
         "a Caesare uictus est", "ab Caesare uictus est",
         "both forms are attested; `a` is the norm"),
    Pair("number: after a numeral",
         "tres partes", "tres pars",
         "plural numeral, singular noun"),
    Pair("word order",
         "Caesar Galliam uicit", "Galliam Caesar uicit",
         "both are Latin; this is a preference, not a verdict", None),
    Pair("word order",
         "Caesar Galliam uicit", "uicit Galliam Caesar",
         "verb first is rarer in prose; still not wrong", None),
)


# ---------------------------------------------------------------------------
# Pure parts
# ---------------------------------------------------------------------------

def align(a: Sequence[int], b: Sequence[int]) -> Tuple[int, int]:
    """(prefix, suffix): how many tokens two readings share at each end.

    The middle is where they differ. The suffix is not allowed to overlap the
    prefix, so `a b c` against `a c` aligns as prefix 1, suffix 1, and the
    middle is `b` against nothing.
    """
    prefix = 0
    while prefix < min(len(a), len(b)) and a[prefix] == b[prefix]:
        prefix += 1
    suffix = 0
    while (suffix < min(len(a), len(b)) - prefix
           and a[-1 - suffix] == b[-1 - suffix]):
        suffix += 1
    return prefix, suffix


# Endings that swap for one another within a declension or conjugation. The
# table is deliberately small and deliberately shallow: it does not know
# which declension a word belongs to, and it does not need to, because the
# result is only used when it is ATTESTED in the held-out split. A swap that
# produces a non-word is discarded before the model ever sees it.
ENDINGS: Dict[str, Tuple[str, ...]] = {
    "us": ("um", "i", "o"), "um": ("us", "i", "o"),
    "i": ("us", "um", "o"), "o": ("us", "um", "i"),
    "a": ("am", "ae", "as"), "am": ("a", "ae", "as"),
    "ae": ("a", "am", "as"), "as": ("a", "am", "ae"),
    "orum": ("arum", "is"), "arum": ("orum", "is"),
    "is": ("es", "ibus", "e"), "ibus": ("is", "es"),
    "es": ("is", "ibus", "em"), "em": ("es", "e", "is"),
    "t": ("nt",), "nt": ("t",),
    "ur": ("unt", "it"), "unt": ("ur", "it"), "it": ("unt", "ur"),
}
WORD = re.compile(r"^[a-z]+$")


def corrupt_ending(word: str, attested: Set[str],
                   rng: random.Random) -> Optional[str]:
    """A different, attested inflection of `word`, or None."""
    if not WORD.match(word):
        return None
    for ending in sorted(ENDINGS, key=len, reverse=True):
        if word.endswith(ending) and len(word) > len(ending) + 2:
            stem = word[:-len(ending)]
            options = [stem + alt for alt in ENDINGS[ending]
                       if stem + alt in attested and stem + alt != word]
            return rng.choice(options) if options else None
    return None


def perturb(left: List[str], word: str, right: List[str], attested: Set[str],
            rng: random.Random) -> Dict[str, Optional[str]]:
    """The original passage and three corruptions of it, as strings.

    swap     the hidden word changes places with the one after it
    ending   the hidden word takes another attested inflection
    shuffle  every word of the passage, in random order -- the reference
             case, which any model with a notion of syntax must win
    """
    original = " ".join(left + [word] + right)
    swapped = None
    if right and WORD.match(right[0]) and right[0] != word:
        swapped = " ".join(left + [right[0], word] + right[1:])
    bent = corrupt_ending(word, attested, rng)
    ended = " ".join(left + [bent] + right) if bent else None
    words = left + [word] + right
    shuffled = words[:]
    rng.shuffle(shuffled)
    if shuffled == words:
        shuffled = None
    return {"original": original, "swap": swapped, "ending": ended,
            "shuffle": " ".join(shuffled) if shuffled else None}


# ---------------------------------------------------------------------------
# One pair, in detail
# ---------------------------------------------------------------------------

def compare(m: Loaded, a: str, b: str, period: Optional[str]) -> Dict:
    ra, rb = score_readings(m, "", [a, b], "", period)
    return {
        "a": a, "b": b,
        "bits_a": bits(ra["logprob"]), "bits_b": bits(rb["logprob"]),
        "margin": (ra["logprob"] - rb["logprob"]) / math.log(2),
        "tokens_a": list(zip(pieces(m, ra["ids"]), map(bits, ra["logprobs"]))),
        "tokens_b": list(zip(pieces(m, rb["ids"]), map(bits, rb["logprobs"]))),
    }


def show_pair(result: Dict) -> None:
    ta, tb = result["tokens_a"], result["tokens_b"]
    margin = result["margin"]
    print(f"  A  {result['a']!r:<52} {result['bits_a']:>7.1f} bits")
    print(f"  B  {result['b']!r:<52} {result['bits_b']:>7.1f} bits")
    who = "A" if margin >= 0 else "B"
    print(f"\n  {who} is preferred by {abs(margin):.2f} bits "
          f"({2 ** abs(margin):,.0f}x as likely)\n")

    prefix, suffix = align([p for p, _ in ta], [p for p, _ in tb])
    mid_a, mid_b = ta[prefix:len(ta) - suffix], tb[prefix:len(tb) - suffix]
    print(f"  {'A token':<14} {'bits':>6}   {'B token':<14} {'bits':>6}   "
          f"{'B - A':>6}")

    def row(pa, pb, mark=""):
        da = f"{pa[0]!r:<14} {pa[1]:>6.1f}" if pa else " " * 21
        db = f"{pb[0]!r:<14} {pb[1]:>6.1f}" if pb else " " * 21
        diff = f"{pb[1] - pa[1]:>+6.1f}" if pa and pb else " " * 6
        print(f"  {da}   {db}   {diff}{mark}")

    for i in range(prefix):
        row(ta[i], tb[i])
    if mid_a or mid_b:
        for i in range(max(len(mid_a), len(mid_b))):
            row(mid_a[i] if i < len(mid_a) else None,
                mid_b[i] if i < len(mid_b) else None,
                "   <-- differs" if i == 0 else "")
        cost = sum(x for _, x in mid_b) - sum(x for _, x in mid_a)
        print(f"  {'':<21}   {'':<21}   {cost:>+6.1f}   on the changed span")
    for i in range(suffix):
        row(ta[len(ta) - suffix + i], tb[len(tb) - suffix + i])
    if suffix:
        after = (sum(x for _, x in tb[len(tb) - suffix:])
                 - sum(x for _, x in ta[len(ta) - suffix:]))
        print(f"  {'':<21}   {'':<21}   {after:>+6.1f}   on the text after it")
    print()


def cmd_pair(args, models: List[Loaded]) -> int:
    a, b = args.a, args.b
    if not args.raw:
        a, ca = to_corpus_orthography(a)
        b, cb = to_corpus_orthography(b)
        merged = {k: ca.get(k, 0) + cb.get(k, 0) for k in set(ca) | set(cb)}
        announce_folding(merged, False)
    else:
        announce_folding({}, True)
    for m in models:
        print(f"{m.describe()}\n")
        show_pair(compare(m, a, b, args.period))
    return 0


# ---------------------------------------------------------------------------
# The suite
# ---------------------------------------------------------------------------

def run_suite(m: Loaded, pairs: Sequence[Pair],
              period: Optional[str]) -> List[float]:
    """Margin in bits, A over B, for every pair -- batched in one call."""
    texts: List[str] = []
    for pair in pairs:
        texts += [pair.a, pair.b]
    rows = score_readings(m, "", texts, "", period)
    return [(rows[2 * i]["logprob"] - rows[2 * i + 1]["logprob"]) / math.log(2)
            for i in range(len(pairs))]


def cmd_suite(args, models: List[Loaded]) -> int:
    pairs = SUITE
    margins = {m.name: run_suite(m, pairs, args.period) for m in models}
    names = [m.name for m in models]
    short = [n.replace("latin-", "") for n in names]

    for m in models:
        print(m.describe())
    print(f"\n{len(pairs)} pairs. Positive bits: the model prefers the "
          f"reading a grammar would print.\n")

    width = max(len(p.phenomenon) for p in pairs)
    col = max(10, max(len(s) for s in short) + 1)
    head = "  ".join(f"{s:>{col}}" for s in short)
    print(f"  {'phenomenon':<{width}}  {head}   reading  |  alternative")
    for i, pair in enumerate(pairs):
        cells = []
        for name in names:
            x = margins[name][i]
            flag = " " if pair.expected is None or x > 0 else "!"
            cells.append(f"{x:>+{col - 1}.2f}{flag}")
        print(f"  {pair.phenomenon:<{width}}  {'  '.join(cells)}   "
              f"{pair.a}  |  {pair.b}")
        if args.notes and pair.note:
            print(f"  {'':<{width}}  {'':<{len(head)}}   ({pair.note})")

    judged = [i for i, p in enumerate(pairs) if p.expected == "a"]
    print()
    for name, s in zip(names, short):
        right = sum(margins[name][i] > 0 for i in judged)
        med = statistics.median(abs(margins[name][i]) for i in judged)
        print(f"  {s:<{col}} {right:>2}/{len(judged)} pairs the right way, "
              f"median margin {med:.1f} bits")
    unjudged = len(pairs) - len(judged)
    if unjudged:
        print(f"\n  {unjudged} word-order pairs have no right answer and are "
              f"not counted.")
    print("  ! marks a pair the model got backwards. The frequency-trap pair "
          "is expected there.")
    if args.out:
        payload = {"models": names, "period": args.period,
                   "pairs": [p._asdict() for p in pairs],
                   "margins": margins}
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n",
                                  encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


# ---------------------------------------------------------------------------
# Benchmark -- attested corruptions of held-out text
# ---------------------------------------------------------------------------

def attested_words(corpus: str, min_count: int = 3) -> Set[str]:
    """Every word that occurs at least min_count times in the held-out split.

    The floor keeps one-off OCR errors out of the set: a corruption that
    turns a word into a scanno would be a test of spelling, not of grammar.
    """
    path = ROOT / "data" / "processed" / f"{corpus}.validation.jsonl"
    if not path.exists():
        sys.exit(f"missing {path.relative_to(ROOT)}")
    counts: collections.Counter = collections.Counter()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            counts.update(json.loads(line)["text"].split())
    return {w for w, n in counts.items() if n >= min_count}


def cmd_benchmark(args, models: List[Loaded]) -> int:
    trials = draw_trials(args.corpus, args.benchmark, args.seed, args.span,
                         min_len=4)
    if not trials:
        sys.exit("no usable passages in the held-out split")
    attested = attested_words(args.corpus)
    rng = random.Random(args.seed)
    variants = [perturb(t["left"].split(), t["answer"], t["right"].split(),
                        attested, rng) for t in trials]
    kinds = ("swap", "ending", "shuffle")

    for m in models:
        print(f"{m.describe()}\n")
        print(f"{len(trials)} passages of {2 * args.span + 1} words from "
              f"{args.corpus}.validation, each scored against three\n"
              f"corruptions of itself. A corruption 'wins' when the model "
              f"finds it likelier than the original.\n")
        wins: Dict[str, List[bool]] = {k: [] for k in kinds}
        margins: Dict[str, List[float]] = {k: [] for k in kinds}
        upsets: Dict[str, List[Tuple[float, str, str]]] = {k: [] for k in kinds}

        for trial, v in zip(trials, variants):
            texts = [v["original"]] + [v[k] for k in kinds if v[k]]
            rows = score_readings(m, "", texts, "", args.period)
            base = rows[0]["logprob"]
            j = 1
            for k in kinds:
                if not v[k]:
                    continue
                margin = (base - rows[j]["logprob"]) / math.log(2)
                wins[k].append(margin > 0)
                margins[k].append(margin)
                if margin <= 0:
                    upsets[k].append((margin, v["original"], v[k]))
                j += 1

        print(f"  {'corruption':<10} {'n':>5}  {'original preferred':>19}  "
              f"{'median margin':>14}")
        for k in kinds:
            n = len(wins[k])
            if not n:
                continue
            print(f"  {k:<10} {n:>5}  {sum(wins[k]):>7} {sum(wins[k])/n:>10.1%}  "
                  f"{statistics.median(margins[k]):>10.1f} bits")

        if args.show and upsets["ending"]:
            print(f"\n  inflections the model preferred to the original "
                  f"(first {args.show}):")
            for margin, orig, bad in upsets["ending"][:args.show]:
                o, b = orig.split(), bad.split()
                k = next(i for i in range(len(o)) if o[i] != b[i])
                lo, hi = max(0, k - 4), k + 5
                print(f"    {-margin:>5.1f} bits   ...{' '.join(o[lo:k])} "
                      f"[{o[k]} -> {b[k]}] {' '.join(o[k + 1:hi])}...")
        print()

        if args.out:
            payload = {
                "model": m.name, "corpus": args.corpus, "n": len(trials),
                "span": args.span,
                "results": {k: {"n": len(wins[k]), "original_preferred":
                                sum(wins[k]), "margins": margins[k]}
                            for k in kinds},
            }
            out = args.out if len(models) == 1 else \
                args.out.replace(".json", f"-{m.name}.json")
            Path(out).write_text(json.dumps(payload, indent=2) + "\n",
                                 encoding="utf-8")
            print(f"wrote {out}\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", nargs="?", help="the first reading")
    ap.add_argument("b", nargs="?", help="the second reading")
    ap.add_argument("--name", action="append",
                    help=f"checkpoint to load; repeatable, so several can be "
                         f"compared (default {DEFAULT_MODEL})")
    ap.add_argument("--period", choices=PERIODS,
                    help="condition on a period tag")
    ap.add_argument("--raw", action="store_true",
                    help="score the text exactly as typed, without folding it "
                         "to the corpus u/v spelling")
    ap.add_argument("--suite", action="store_true",
                    help="run the built-in minimal pairs")
    ap.add_argument("--notes", action="store_true",
                    help="suite: print what each pair tests")
    ap.add_argument("--benchmark", type=int, metavar="N",
                    help="score N held-out passages against corruptions of "
                         "themselves")
    ap.add_argument("--corpus", default="corpus-b",
                    help="which held-out split to benchmark on")
    ap.add_argument("--span", type=int, default=12,
                    help="benchmark: words of context either side")
    ap.add_argument("--show", type=int, default=8,
                    help="benchmark: how many upsets to print")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", help="write a JSON report here")
    args = ap.parse_args()

    if not (args.suite or args.benchmark) and not (args.a and args.b):
        ap.error("give two readings, or --suite, or --benchmark N")

    models = [load(n) for n in (args.name or [DEFAULT_MODEL])]
    for m in models:
        if args.period and not m.tagged:
            print(f"WARNING: {m.name} was trained without period tags; "
                  f"<{args.period}> carries no learned meaning here\n",
                  file=sys.stderr)
    if args.benchmark:
        return cmd_benchmark(args, models)
    if args.suite:
        return cmd_suite(args, models)
    return cmd_pair(args, models)


if __name__ == "__main__":
    sys.exit(main())

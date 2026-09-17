#!/usr/bin/env python3
"""What a perplexity actually measures: the same words, rearranged.

THE EXPERIMENT
    Take a passage. Shuffle its sentences; shuffle the words inside each
    sentence; shuffle every word; reverse it. Every one of those texts has
    exactly the vocabulary and exactly the word frequencies of the original,
    so a frequency table -- a bag of words -- would give them all the same
    score. The model does not, and the ladder of numbers it gives is a
    direct reading of how much it knows about Latin beyond which words exist:
    the ORDER they come in, which is to say syntax.

HOW TO READ THE LADDER
    Shuffling the sentences barely moves the number, because a sentence is
    mostly predicted from itself. Shuffling within sentences moves it a lot,
    because agreement, case government and idiom all live inside the
    sentence. A text in a language the model has never seen -- there is an
    English translation of the Caesar sample in demos/samples/ -- sits an
    order of magnitude above, which is the separation the corpus audit runs
    on. Everything here uses one forward pass per text; no candidates, no
    sampling, no held-out split, so it runs on any passage in a second.

WHY SEVERAL TRIALS
    A single shuffle is one draw from a distribution. Three are averaged by
    default, and the spread is printed, so a lucky permutation is not quoted
    as a fact about the model.

Usage:
    python3 demos/ordo.py demos/samples/clean.txt
    python3 demos/ordo.py FILE --against demos/samples/english.txt
    echo "Gallia est omnis divisa in partes tres" | python3 demos/ordo.py -
"""

import argparse
import math
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    DEFAULT_MODEL, PERIODS, Loaded, announce_folding, load,
    to_corpus_orthography,
)
from score import score_text  # noqa: E402

SENTENCE = re.compile(r"(?<=[.!?;:])\s+")


def sentences(text: str) -> List[str]:
    return [s for s in SENTENCE.split(" ".join(text.split())) if s]


def shuffle_sentences(text: str, rng: random.Random) -> str:
    parts = sentences(text)
    rng.shuffle(parts)
    return " ".join(parts)


def shuffle_within_sentences(text: str, rng: random.Random) -> str:
    out = []
    for s in sentences(text):
        words = s.split()
        rng.shuffle(words)
        out.append(" ".join(words))
    return " ".join(out)


def shuffle_words(text: str, rng: random.Random) -> str:
    words = text.split()
    rng.shuffle(words)
    return " ".join(words)


def reverse_words(text: str, rng: random.Random) -> str:
    return " ".join(reversed(text.split()))


# Each rung: a label, the transformation, and whether it is random (and so
# worth repeating). Ordered from least to most disruptive, which is also the
# order the perplexities come out in on every text tried.
LADDER = (
    ("as written", None, False),
    ("sentences shuffled", shuffle_sentences, True),
    ("words shuffled within each sentence", shuffle_within_sentences, True),
    ("all words shuffled", shuffle_words, True),
    ("word order reversed", reverse_words, False),
)


def measure(m: Loaded, text: str, period: Optional[str]) -> Dict:
    """Perplexity per token, and the same information per WORD.

    Perplexity is per token, and the tokenizer was fitted on Latin: a Latin
    word is about 1.7 tokens here, an English one over 3. Each of those
    small English pieces is individually easy to guess, so per-token
    perplexity understates how foreign the text is. Bits per word is the
    cost of the same text in a unit that does not depend on how it was cut.
    """
    result = score_text(m.model, m.tok, text, m.context, period)
    n_words = max(1, len(text.split()))
    total_bits = -result["mean_logprob"] * result["n_tokens"] / math.log(2)
    return {"perplexity": result["perplexity"],
            "tokens_per_word": result["n_tokens"] / n_words,
            "bits_per_word": total_bits / n_words}


def ladder(m: Loaded, text: str, trials: int, seed: int,
           period: Optional[str]) -> List[Dict]:
    """Perplexity per rung, averaged over trials where the rung is random."""
    rows: List[Dict] = []
    n_sentences = len(sentences(text))
    for label, fn, random_ in LADDER:
        if fn is shuffle_sentences and n_sentences < 2:
            continue   # one sentence has no sentence order to shuffle
        values = []
        for t in range(trials if random_ else 1):
            variant = text if fn is None else fn(text, random.Random(seed + t))
            values.append(measure(m, variant, period))
        ppl = [v["perplexity"] for v in values]
        rows.append({"label": label, "perplexity": statistics.mean(ppl),
                     "spread": (max(ppl) - min(ppl)) if len(ppl) > 1 else 0.0,
                     "trials": len(values),
                     "tokens_per_word": values[0]["tokens_per_word"],
                     "bits_per_word": statistics.mean(
                         v["bits_per_word"] for v in values)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", default="-",
                    help="a text file, or - for stdin")
    ap.add_argument("--against", action="append", default=[],
                    help="another file to score on the same scale, e.g. a "
                         "translation; repeatable")
    ap.add_argument("--name", default=DEFAULT_MODEL)
    ap.add_argument("--period", choices=PERIODS,
                    help="score as if the text were of this era")
    ap.add_argument("--trials", type=int, default=3,
                    help="how many shuffles to average (default 3)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--raw", action="store_true",
                    help="score the text exactly as typed, without folding it "
                         "to the corpus u/v spelling")
    args = ap.parse_args()

    text = sys.stdin.read() if args.file == "-" \
        else Path(args.file).read_text(encoding="utf-8")
    if not text.strip():
        sys.exit("empty input")
    changes: Dict[str, int] = {}
    if not args.raw:
        text, changes = to_corpus_orthography(text)

    m = load(args.name)
    if args.period and not m.tagged:
        print(f"WARNING: {m.name} was trained without period tags; "
              f"<{args.period}> carries no learned meaning here\n",
              file=sys.stderr)
    print(f"{m.describe()}\n")
    announce_folding(changes, args.raw)

    rows = ladder(m, text, args.trials, args.seed, args.period)
    base = rows[0]["perplexity"]
    n_words = len(text.split())
    print(f"{n_words} words, {len(sentences(text))} sentences. Every rung "
          f"below the first has the same words and\nthe same word "
          f"frequencies as the original; only the order changes.\n")
    print(f"  {'text':<38} {'perplexity':>10}  {'x as written':>12}  "
          f"{'tok/word':>8}  {'bits/word':>9}")
    for row in rows:
        spread = (f"  (±{row['spread'] / 2:.1f} over {row['trials']} shuffles)"
                  if row["trials"] > 1 else "")
        print(f"  {row['label']:<38} {row['perplexity']:>10.1f}  "
              f"{row['perplexity'] / base:>11.2f}x  "
              f"{row['tokens_per_word']:>8.2f}  {row['bits_per_word']:>9.1f}"
              f"{spread}")

    for path in args.against:
        other = Path(path).read_text(encoding="utf-8")
        if not args.raw:
            other, _ = to_corpus_orthography(other)
        r = measure(m, other, args.period)
        print(f"  {Path(path).name:<38} {r['perplexity']:>10.1f}  "
              f"{r['perplexity'] / base:>11.2f}x  {r['tokens_per_word']:>8.2f}"
              f"  {r['bits_per_word']:>9.1f}")
    if args.against:
        print(f"\nPerplexity is per token, and a text the tokenizer was not "
              f"fitted on is cut into more,\nsmaller pieces, each easy to "
              f"guess from the last. Compare the bits per word instead:\n"
              f"that is what the whole text cost, in a unit that does not "
              f"depend on the cutting.")

    worst = max(rows, key=lambda r: r["perplexity"])
    print(f"\nA bag of words scores every rung the same. This model is "
          f"{worst['perplexity'] / base:.1f}x more surprised\nby "
          f"'{worst['label']}' than by the text as written: that ratio is "
          f"what it knows about\nLatin word order, over and above which words "
          f"Latin has.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

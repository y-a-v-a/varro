#!/usr/bin/env python3
"""Show where a Latin text stops looking like Latin.

WHAT IT IS FOR
    A single perplexity for a document says something is wrong with it. It does
    not say where. This prints the text back with every word shaded by how
    surprised the model was to see it, so the eye lands on the damage: an OCR
    scanno, a mojibake run, a stretch of Greek, an apparatus criticus that
    survived extraction, a modern interpolation in an ancient text.

    It is the same measurement as `src/score.py --audit`, which is what found
    the bad documents in this corpus. The difference is only that the audit
    ranks documents for a pipeline and this renders one document for a person.

WHY WORDS AND NOT TOKENS
    A byte-level BPE splits `praetereundum` into pieces no reader thinks in.
    Surprisal is summed back up to the whole word, so a long rare word is not
    penalised for being several tokens and the display lines up with the text.

WHAT SURPRISE IS NOT
    Not error. The model reports improbability and cannot tell a genuine
    Plautine archaism from a typo -- both are rare. Proper names, hapax
    legomena and technical vocabulary light up exactly the way corruption does.
    A hot word is a place to look, never a verdict.

Usage:
    python3 demos/surprisal.py FILE
    echo "Gallia est omnis divisa in partes tres" | python3 demos/surprisal.py -
    python3 demos/surprisal.py FILE --period CLASSICAL --worst 15
    python3 demos/surprisal.py FILE --plain > report.txt
"""

import argparse
import math
import statistics
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    DEFAULT_MODEL, PERIODS, Loaded, announce_folding, load,
    to_corpus_orthography,
)
from score import score_text  # noqa: E402

# Five bands, dim to loud. The thresholds are in bits of surprisal per word,
# chosen against this model's own median rather than in the abstract: a typical
# word here costs about 9 bits, so the bands start where a word is already
# twice as surprising as typical and end where it is a 1-in-a-million event.
BANDS = (
    (12.0, "\033[2;37m", "."),   # unremarkable
    (18.0, "\033[0m", " "),      # ordinary
    (24.0, "\033[33m", "-"),     # notable
    (32.0, "\033[35m", "+"),     # suspicious
    (1e9, "\033[1;31m", "#"),    # look here
)
RESET = "\033[0m"


class Word(NamedTuple):
    text: str
    bits: float
    tokens: int
    # (piece, bits) per token, so a hot word can be opened up: an OCR scanno
    # like `rnercatores` is not uniformly strange, it is one impossible
    # piece followed by ordinary ones, and that is visible only here.
    pieces: Tuple[Tuple[str, float], ...] = ()


def to_words(tok, ids: List[int], logprobs: List[Optional[float]],
             prefix_len: int) -> List[Word]:
    """Regroup per-token surprisal onto whole words.

    A byte-level BPE marks a word boundary by putting the space INSIDE the
    following token, so a piece that starts with a space starts a new word.
    Pieces are accumulated as text rather than re-decoded per word, which keeps
    multi-byte characters intact across a split.
    """
    words: List[Word] = []
    buf, bits, count = "", 0.0, 0
    parts: List[Tuple[str, float]] = []

    for i in range(prefix_len, len(ids)):
        lp = logprobs[i]
        if lp is None:
            continue
        piece = tok.decode([ids[i]])
        if (piece.startswith(" ") or piece.startswith("\n")) and buf:
            words.append(Word(buf, bits, count, tuple(parts)))
            buf, bits, count, parts = "", 0.0, 0, []
        buf += piece
        bits += -lp / math.log(2)
        count += 1
        parts.append((piece, -lp / math.log(2)))

    if buf:
        words.append(Word(buf, bits, count, tuple(parts)))
    return [w._replace(text=w.text.strip()) for w in words if w.text.strip()]


def band(bits: float):
    for threshold, colour, mark in BANDS:
        if bits < threshold:
            return colour, mark
    return BANDS[-1][1], BANDS[-1][2]


def render(words: List[Word], plain: bool, width: int) -> None:
    line = 0
    for word in words:
        colour, mark = band(word.bits)
        piece = f"{word.text}{mark}" if plain else f"{colour}{word.text}{RESET}"
        if line + len(word.text) + 1 > width:
            print()
            line = 0
        print(piece, end=" ")
        line += len(word.text) + 1
    print("\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="a text file, or - for stdin")
    ap.add_argument("--name", default=DEFAULT_MODEL)
    ap.add_argument("--period", choices=PERIODS,
                    help="score as if the text were of this era")
    ap.add_argument("--worst", type=int, default=12,
                    help="how many hot words to list underneath")
    ap.add_argument("--raw", action="store_true",
                    help="score the text exactly as typed, without folding it "
                         "to the corpus u/v spelling")
    ap.add_argument("--plain", action="store_true",
                    help="mark bands with punctuation instead of colour")
    ap.add_argument("--pieces", action="store_true",
                    help="open up each hot word into the tokens it was "
                         "scored as, with the bits each one cost")
    ap.add_argument("--width", type=int, default=78)
    args = ap.parse_args()

    text = sys.stdin.read() if args.file == "-" \
        else Path(args.file).read_text(encoding="utf-8")
    if not text.strip():
        sys.exit("empty input")

    changes: dict = {}
    if not args.raw:
        text, changes = to_corpus_orthography(text)

    m: Loaded = load(args.name)
    if args.period and not m.tagged:
        print(f"WARNING: {m.name} was trained without period tags; "
              f"<{args.period}> carries no learned meaning here\n",
              file=sys.stderr)

    result = score_text(m.model, m.tok, text, m.context, args.period)
    words = to_words(m.tok, result["ids"], result["logprobs"],
                     result["prefix_len"])
    if not words:
        sys.exit("nothing scorable in that input")

    print(f"{m.describe()}\n")
    announce_folding(changes, args.raw)
    render(words, args.plain, args.width)

    per_word = [w.bits for w in words]
    median = statistics.median(per_word)
    print(f"{len(words):,} words, {result['n_tokens']:,} tokens, "
          f"perplexity {result['perplexity']:.1f}")
    print(f"median {median:.1f} bits/word, "
          f"90th percentile {sorted(per_word)[int(len(per_word) * .9)]:.1f}, "
          f"max {max(per_word):.1f}")
    if args.plain:
        print("bands:  . unremarkable   (blank) ordinary   - notable   "
              "+ suspicious   # look here")

    hot = sorted(words, key=lambda w: -w.bits)[:args.worst]
    print(f"\n{'bits':>6}  {'toks':>4}  word")
    for word in hot:
        print(f"{word.bits:>6.1f}  {word.tokens:>4}  {word.text}")
        if args.pieces:
            print(f"{'':>14}{describe_pieces(word)}")
    if args.pieces:
        print("\nA word's bits are the sum of its pieces, and a damaged word "
              "pays twice: the tokenizer\nhas no piece for a scanno, so the "
              "word is cut into more pieces than its correct form\n(mercatores "
              "is 2 tokens; rnercatores is 4), and every one of them is then "
              "improbable.")
    return 0


def describe_pieces(word: Word) -> str:
    """`' rn' 21.3 + 'erc' 9.8 + 'at' 3.0 + 'ores' 8.8` -- the surprise, itemised."""
    return " + ".join(f"{p!r} {b:.1f}" for p, b in word.pieces)


if __name__ == "__main__":
    sys.exit(main())

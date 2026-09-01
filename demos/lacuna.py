#!/usr/bin/env python3
"""Fill a hole in a damaged Latin text, and say how confident that is.

THE PROBLEM THIS IS SHAPED AROUND
    A papyrus tears, a stone chips, a scribe's eye skips a line. What an editor
    prints in the gap is a conjecture, and the apparatus argues for it in prose.
    This puts a number next to the argument: for each proposed supplement, the
    probability the model assigns to the rest of the line once that supplement
    is in place.

WHY A LEFT-TO-RIGHT MODEL CAN DO THIS AT ALL
    It cannot read the gap from both sides -- nothing in a causal transformer
    looks right. But it can score a whole sequence, and the score of
    `fill + everything after it` does depend on the text after the gap. So the
    model PROPOSES forward and is JUDGED backwards over the right context. The
    --blind flag shows the difference that makes: it ranks the same candidates
    on the left context alone, which is what plain generation would give you.

WHAT THE NUMBER IS NOT
    Not a probability that the conjecture is historically correct. It is
    P(text | this reading) under a 52M-parameter model of Latin -- a fluency
    judgement, blind to metre, to the letter-count the stone allows, and to
    everything an editor knows about the author. Use --pattern for the letters
    that survive; bring the rest of the judgement yourself.

Usage:
    python3 demos/lacuna.py --text "Gallia est omnis divisa in [...] tres"
    python3 demos/lacuna.py FILE --words 2 --blind
    python3 demos/lacuna.py --text "arma [...] cano" -c virumque -c ducemque
    python3 demos/lacuna.py --benchmark 40      # measured, not demonstrated
"""

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    DEFAULT_MODEL, PERIODS, ROOT, Loaded, announce_folding, encode, load,
    prefix_ids, score_readings, to_corpus_orthography,
)

# The conventions an editor already writes: [...] for a gap of unknown length,
# [---] for the epigraphic dash, [?] for a doubtful one.
GAP = re.compile(r"\[(?:\.{3}|-{2,}|\?+)\]")

# A candidate containing one of these is not a word the model proposed; it is a
# special token, or a fragment of one, that leaked through decoding.
SPECIAL = re.compile(r"[<>|]")

WORD = re.compile(r"^[A-Za-z]+$")


def split_gaps(text: str) -> Tuple[List[str], int]:
    """Segments either side of each gap. n segments means n-1 gaps."""
    parts = GAP.split(text)
    return parts, len(parts) - 1


def propose(m: Loaded, left: str, n_words: int, k: int, temperature: float,
            top_k: int, seed: int, period: Optional[str]) -> List[str]:
    """Sample k candidate fills from the left context alone.

    Batched: every candidate shares a prompt and grows by one token per step,
    so k samples cost one forward pass per step rather than k. At k=32 that is
    the difference between a demo that pauses and one that does not.

    The mx.eval per step is not an optimisation, it is the difference between
    running and being killed. MLX is lazy, so without it the graph for every
    step of every row stays live until the final read, and a dozen steps of a
    52M model at batch 32 exhausts 16GB.
    """
    import mlx.core as mx

    base = (prefix_ids(m.tok, period) + encode(m, left))[-(m.context - 1):]
    if not base:
        base = prefix_ids(m.tok, period)
    ids = mx.array([base] * k)
    key = mx.random.key(seed)

    for _ in range(n_words * 6 + 8):
        logits = m.model(ids[:, -m.context:])[:, -1].astype(mx.float32)
        logits = logits / max(temperature, 1e-6)
        if top_k and top_k < logits.shape[-1]:
            kth = mx.sort(logits, axis=-1)[:, -top_k][:, None]
            logits = mx.where(logits < kth,
                              mx.full(logits.shape, -1e9, logits.dtype), logits)
        key, use = mx.random.split(key)
        ids = mx.concatenate(
            [ids, mx.random.categorical(logits, key=use)[:, None]], axis=1)
        mx.eval(ids)

    out: List[str] = []
    for row in ids[:, len(base):].tolist():
        words = m.tok.decode(row).split()
        if len(words) >= n_words:
            candidate = " ".join(words[:n_words])
            if not SPECIAL.search(candidate):
                out.append(candidate)
    return out


def rank(m: Loaded, left: str, candidates: List[str], right: str,
         period: Optional[str]) -> List[Dict]:
    scored = score_readings(m, left, candidates, right, period)
    scored.sort(key=lambda r: -r["logprob"])
    return scored


def show(rows: List[Dict], limit: int, answer: Optional[str],
         heading: str) -> None:
    """Ranked candidates, in bits below the best reading.

    Bits rather than raw log-probability because the gap between two readings
    is then a statement anyone can check: 3 bits worse means the model puts
    that reading at one eighth the likelihood of the winner.
    """
    if not rows:
        print("  (no candidates)")
        return
    best = rows[0]["logprob"]
    print(f"  {heading}")
    print(f"  {'bits worse':>10}  {'toks':>4}  supplement")
    shown = rows[:limit]
    for row in shown:
        delta = (best - row["logprob"]) / math.log(2)
        mark = "   <-- the reading in the manuscripts" \
            if answer is not None and row["fill"] == answer else ""
        print(f"  {delta:>10.2f}  {row['fill_tokens']:>4}  {row['fill']!r}{mark}")
    if answer is not None and not any(r["fill"] == answer for r in shown):
        for i, row in enumerate(rows):
            if row["fill"] == answer:
                delta = (best - row["logprob"]) / math.log(2)
                print(f"  {delta:>10.2f}  {row['fill_tokens']:>4}  "
                      f"{row['fill']!r}   <-- the reading in the manuscripts, "
                      f"rank {i + 1} of {len(rows)}")
                break
    print()


def cmd_fill(args, m: Loaded) -> int:
    text = Path(args.file).read_text(encoding="utf-8") if args.file else args.text
    if not text:
        sys.exit("give --text or a FILE")
    text = " ".join(text.split())

    # Split BEFORE folding. src/normalize.py strips square brackets as
    # editorial marks, so folding the whole string first would delete the very
    # gap this command exists to fill.
    parts, n_gaps = split_gaps(text)
    if n_gaps == 0:
        sys.exit("no gap in the text -- mark one with [...] "
                 "(also accepted: [---], [?])")

    if not args.raw:
        folded, changes = zip(*(to_corpus_orthography(p) for p in parts))
        parts = list(folded)
        merged: Dict[str, int] = {}
        for change in changes:
            for what, n in change.items():
                merged[what] = merged.get(what, 0) + n
        args.candidate = [to_corpus_orthography(c)[0] for c in args.candidate]
        if args.answer:
            args.answer = to_corpus_orthography(args.answer)[0]
        announce_folding(merged, False)
    else:
        announce_folding({}, True)

    if args.answer and n_gaps > 1:
        # It labelled the same word as "the manuscript reading" at every gap,
        # which is false at all but one of them.
        print(f"  [--answer ignored: it names one reading and this text has "
              f"{n_gaps} gaps]\n")
        args.answer = None

    pattern = re.compile(args.pattern, re.I) if args.pattern else None
    # Pieces, not one string: normalize() strips each segment, so rejoining
    # with "" would run the words either side of a gap together.
    filled: List[str] = [parts[0].strip()]

    for gap in range(n_gaps):
        left = " ".join(p for p in filled if p)
        right = parts[gap + 1].strip()

        print(f"gap {gap + 1} of {n_gaps}:  ...{left[-44:]}  "
              f"[{args.words} word(s)]  {right[:44]}...\n")

        proposals = propose(m, left, args.words, args.candidates,
                            args.temperature, args.top_k, args.seed + gap,
                            args.period)
        pool = list(dict.fromkeys(args.candidate + proposals))
        if pattern:
            kept = [c for c in pool if pattern.search(c)]
            if kept:
                print(f"  [--pattern {args.pattern!r} dropped "
                      f"{len(pool) - len(kept)} of {len(pool)} candidates]")
                pool = kept
            else:
                print(f"  [--pattern {args.pattern!r} matched nothing; "
                      f"showing the unfiltered pool]")
        if args.answer and args.answer not in pool:
            pool.append(args.answer)
        if not pool:
            sys.exit("no candidates survived -- raise --candidates")

        rows = rank(m, left, pool, right, args.period)
        show(rows, args.top, args.answer,
             f"{len(pool)} candidates, judged on the fill AND the text after it")

        if args.blind:
            show(rank(m, left, pool, "", args.period), args.top, args.answer,
                 "the same candidates judged on the left context alone "
                 "-- what plain generation sees")

        filled.append(rows[0]["fill"])
        filled.append(parts[gap + 1].strip())

    print("reconstruction:")
    print(f"  {' '.join(' '.join(p for p in filled if p).split())}")
    return 0


# ---------------------------------------------------------------------------
# Benchmark -- the same task, scored against text the model has never read
# ---------------------------------------------------------------------------

def draw_trials(corpus: str, n: int, seed: int, span: int,
                min_len: int) -> List[Dict]:
    """Hide one word in each of n passages from the HELD-OUT split.

    Held-out matters more here than anywhere else in the project. Asking a
    model to restore a word it was trained on measures recall, not
    reconstruction, and would report a number several times too high.
    """
    path = ROOT / "data" / "processed" / f"{corpus}.validation.jsonl"
    if not path.exists():
        sys.exit(f"missing {path.relative_to(ROOT)}")

    rng = random.Random(seed)
    reservoir: List[Dict] = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if len(reservoir) < n * 4:
                reservoir.append(json.loads(line))
            else:
                j = rng.randrange(i + 1)
                if j < n * 4:
                    reservoir[j] = json.loads(line)

    trials: List[Dict] = []
    for record in reservoir:
        words = record["text"].split()
        if len(words) < 2 * span + 4:
            continue
        centre = rng.randrange(span, len(words) - span - 1)
        choices = [k for k in range(max(span, centre - 12),
                                    min(len(words) - span, centre + 12))
                   if len(words[k]) >= min_len and WORD.match(words[k])]
        if not choices:
            continue
        k = rng.choice(choices)
        trials.append({
            "id": record.get("id", "?"),
            "left": " ".join(words[max(0, k - span):k]),
            "answer": words[k],
            "right": " ".join(words[k + 1:k + 1 + span]),
        })
        if len(trials) == n:
            break
    return trials


def cmd_benchmark(args, m: Loaded) -> int:
    """How often does the right word win? Two numbers, and they differ.

    PROPOSED  the model sampled the true word at all, from the left context.
              This is a recall ceiling: a word never proposed cannot be chosen.
    CHOSEN    the true word outranked every proposal once the right context was
              scored. This is the number the demo actually claims, and it is
              measured with the truth forced into the pool, so a failure to
              propose is not silently counted as a failure to judge.
    """
    trials = draw_trials(args.corpus, args.benchmark, args.seed, args.span,
                         args.min_len)
    if not trials:
        sys.exit("no usable passages in the held-out split")

    proposed = chosen = chosen_blind = 0
    ranks: List[int] = []
    print(f"{m.describe()}\n")
    print(f"{args.benchmark} passages from {args.corpus}.validation, one word "
          f"hidden in each,\n{args.candidates} candidates sampled per gap, "
          f"{args.span} words of context either side.\n")
    print(f"{'#':>4}  {'rank':>5}  {'answer':<18} {'model chose':<18} proposed?")

    for i, trial in enumerate(trials, 1):
        pool = propose(m, trial["left"], 1, args.candidates, args.temperature,
                       args.top_k, args.seed + i, args.period)
        pool = list(dict.fromkeys(pool))
        was_proposed = trial["answer"] in pool
        proposed += was_proposed
        if not was_proposed:
            pool.append(trial["answer"])

        rows = rank(m, trial["left"], pool, trial["right"], args.period)
        order = [r["fill"] for r in rows]
        place = order.index(trial["answer"]) + 1
        ranks.append(place)
        chosen += place == 1
        blind = rank(m, trial["left"], pool, "", args.period)
        chosen_blind += blind[0]["fill"] == trial["answer"]

        print(f"{i:>4}  {place:>5}  {trial['answer']:<18} {order[0]:<18} "
              f"{'yes' if was_proposed else 'no'}")

    n = len(trials)
    median = sorted(ranks)[n // 2]
    print(f"\nproposed by the model      {proposed:>4}/{n}  {proposed/n:>6.1%}")
    print(f"chosen, with right context {chosen:>4}/{n}  {chosen/n:>6.1%}")
    print(f"chosen, left context only  {chosen_blind:>4}/{n}  "
          f"{chosen_blind/n:>6.1%}   <-- what generation alone gets")
    print(f"median rank of the true word  {median}   "
          f"top-5 {sum(r <= 5 for r in ranks)/n:.1%}")

    if args.out:
        payload = {
            "model": m.name, "corpus": args.corpus, "n": n,
            "candidates": args.candidates, "span": args.span,
            "proposed": proposed, "chosen": chosen,
            "chosen_blind": chosen_blind, "ranks": ranks,
        }
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n",
                                  encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", help="a text file with [...] in it")
    ap.add_argument("--text", help="the damaged text, inline")
    ap.add_argument("--name", default=DEFAULT_MODEL)
    ap.add_argument("--words", type=int, default=1,
                    help="how many words the gap holds (default 1)")
    ap.add_argument("--candidates", type=int, default=32,
                    help="how many supplements to sample (default 32)")
    ap.add_argument("-c", "--candidate", action="append", default=[],
                    help="a conjecture of your own; repeatable, always scored")
    ap.add_argument("--pattern",
                    help="regex the fill must match -- the letters the stone "
                         "still shows, e.g. '^p.*s$'")
    ap.add_argument("--answer", help="the true reading, marked in the table")
    ap.add_argument("--period", choices=PERIODS,
                    help="condition on a period tag")
    ap.add_argument("--raw", action="store_true",
                    help="score the text exactly as typed, without folding it "
                         "to the corpus u/v spelling")
    ap.add_argument("--blind", action="store_true",
                    help="also rank without the right context, for comparison")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--benchmark", type=int, metavar="N",
                    help="measure recovery on N held-out passages instead")
    ap.add_argument("--corpus", default="corpus-b",
                    help="which held-out split to benchmark on")
    ap.add_argument("--span", type=int, default=25,
                    help="benchmark: words of context either side")
    ap.add_argument("--min-len", type=int, default=4,
                    help="benchmark: shortest word worth hiding")
    ap.add_argument("--out", help="benchmark: write the report here")
    args = ap.parse_args()

    m = load(args.name)
    if args.period and not m.tagged:
        print(f"WARNING: {m.name} was trained without period tags; "
              f"<{args.period}> carries no learned meaning here\n",
              file=sys.stderr)
    if args.benchmark:
        return cmd_benchmark(args, m)
    print(f"{m.describe()}\n")
    return cmd_fill(args, m)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Date a piece of Latin by asking which era it reads as.

HOW IT WORKS, AND WHY IT IS A FAIR COMPARISON
    Every document in training was streamed as `<PERIOD> ... text ...`, so the
    era tag is a thing the model was conditioned on rather than a thing it has
    to infer. Turn that around and you have a classifier for free: score the
    SAME tokens four times, changing only the tag in front of them, and see
    which conditioning fits best.

    Only the prefix differs between the four runs, so the perplexities are
    measured over an identical token sequence. That is not true of most
    perplexity comparisons and it is what makes the ranking meaningful.

WHY THE MARGINS ARE SMALL, AND WHY THAT IS FINE
    A period tag is two tokens in front of a few hundred, so its influence is
    diluted -- the best and worst era typically differ by about 2.5% of
    perplexity. Small is not the same as absent: on held-out text this picks
    the right era of four about two thirds of the time, against 25% for a coin.
    What matters is that a 2.5% margin is reported honestly rather than dressed
    up as a confident answer.

WHY THIS WOULD NOT HAVE WORKED ON THE SMALLER MODEL
    Measured on LATIN-XXV-2 the same procedure gets 53.5% over three eras with
    a median margin of 0.3% -- barely distinguishable from no conditioning at
    all. Era estimation is a capability that arrived with model size and corpus
    breadth, not a property of period tagging as such. Run --calibrate before
    trusting this on any checkpoint it has not been measured on.

Usage:
    python3 demos/aetas.py FILE
    echo "Gallia est omnis divisa in partes tres" | python3 demos/aetas.py -
    python3 demos/aetas.py --calibrate 12      # confusion matrix, held-out
"""

import argparse
import collections
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    DEFAULT_MODEL, PERIODS, ROOT, Loaded, announce_folding, load,
    to_corpus_orthography,
)
from score import score_text  # noqa: E402

REPORTS = ROOT / "reports"

# Margin bands in bits per token, for the empirical confidence table. The
# boundaries are round numbers rather than fitted ones: fitting bucket edges to
# the same data that fills the buckets would report its own noise back as
# confidence.
BANDS = (0.002, 0.006, 0.015)
BAND_NAMES = ("very weak", "weak", "moderate", "clear")


def band_of(margin: float) -> str:
    for edge, name in zip(BANDS, BAND_NAMES):
        if margin < edge:
            return name
    return BAND_NAMES[-1]


def era_scores(m: Loaded, text: str) -> Dict[str, Dict[str, float]]:
    """Perplexity and mean log-probability per token, under each era tag."""
    out: Dict[str, Dict[str, float]] = {}
    for period in PERIODS:
        result = score_text(m.model, m.tok, text, m.context, period)
        out[period] = {
            "perplexity": result["perplexity"],
            "mean_logprob": result["mean_logprob"],
            "n_tokens": result["n_tokens"],
        }
    return out


def verdict(scores: Dict[str, Dict[str, float]]) -> Tuple[str, str, float]:
    """Best era, runner-up, and the margin between them in bits per token."""
    order = sorted(scores, key=lambda p: -scores[p]["mean_logprob"])
    best, second = order[0], order[1]
    margin = ((scores[best]["mean_logprob"] - scores[second]["mean_logprob"])
              / math.log(2))
    return best, second, margin


def load_calibration(name: str) -> Optional[Dict]:
    path = REPORTS / f"era-{name}.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Estimating one text
# ---------------------------------------------------------------------------

def cmd_estimate(args, m: Loaded) -> int:
    text = sys.stdin.read() if args.file == "-" \
        else Path(args.file).read_text(encoding="utf-8")
    if not text.strip():
        sys.exit("empty input")

    changes: Dict[str, int] = {}
    if not args.raw:
        text, changes = to_corpus_orthography(text)
    announce_folding(changes, args.raw)

    scores = era_scores(m, text)
    best, second, margin = verdict(scores)
    n_tokens = scores[best]["n_tokens"]

    print(f"  {'era':<12} {'perplexity':>11}   {'vs best':>8}")
    for period in sorted(scores, key=lambda p: scores[p]["perplexity"]):
        ppl = scores[period]["perplexity"]
        ratio = ppl / scores[best]["perplexity"]
        mark = "  <-- best fit" if period == best else ""
        print(f"  {period:<12} {ppl:>11.2f}   {ratio:>7.4f}x{mark}")

    print(f"\n{n_tokens:,} tokens.  {best} over {second} by "
          f"{margin:.4f} bits/token ({band_of(margin)}).")

    calibration = load_calibration(m.name)
    if calibration is None:
        print(f"\nNo calibration for {m.name}. The ranking above is unmeasured "
              f"-- run:\n  python3 demos/aetas.py --calibrate 12 --name {m.name}")
    else:
        report_confidence(calibration, band_of(margin), n_tokens)
    return 0


def report_confidence(calibration: Dict, band: str, n_tokens: int) -> None:
    """What a margin this size was worth on text of known date."""
    bucket = calibration["bands"].get(band)
    print(f"\nOn {calibration['n']} held-out documents of known date, this "
          f"model was right\n{calibration['accuracy']:.1%} of the time overall "
          f"(chance is {1/len(PERIODS):.0%}).")
    if bucket and bucket["n"]:
        print(f"Among the {bucket['n']} of them whose margin was also "
              f"'{band}', it was right {bucket['accuracy']:.1%}.")
    else:
        print(f"No calibration document landed in the '{band}' band, so there "
              f"is no measured accuracy for a margin this size.")
    if n_tokens < calibration["tokens_per_doc"] // 2:
        print(f"NOTE: this text is {n_tokens:,} tokens against "
              f"{calibration['tokens_per_doc']:,} in calibration. Short texts "
              f"give noisier margins than the table above reflects.")


# ---------------------------------------------------------------------------
# Calibration -- the confusion matrix that says whether to believe any of this
# ---------------------------------------------------------------------------

def draw_dated(corpus: str, per_period: int, seed: int,
               words: int) -> List[Tuple[str, str]]:
    """Held-out documents that carry a period, balanced across periods.

    Balanced on purpose. corpus-b.validation is 134 NEOLATIN against 19
    CLASSICAL, and an unbalanced sample would let a model that always guessed
    the majority era look competent.
    """
    path = ROOT / "data" / "processed" / f"{corpus}.validation.jsonl"
    if not path.exists():
        sys.exit(f"missing {path.relative_to(ROOT)}")

    by_period: Dict[str, List[str]] = collections.defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            period = record.get("period")
            if period in PERIODS and len(record["text"].split()) > words:
                by_period[period].append(
                    " ".join(record["text"].split()[:words]))

    rng = random.Random(seed)
    sample: List[Tuple[str, str]] = []
    for period, texts in sorted(by_period.items()):
        take = min(per_period, len(texts))
        if take < per_period:
            print(f"  [only {take} held-out {period} documents available]")
        sample += [(period, t) for t in rng.sample(texts, take)]
    return sample


def cmd_calibrate(args, m: Loaded) -> int:
    sample = draw_dated(args.corpus, args.calibrate, args.seed, args.words)
    if not sample:
        sys.exit("no dated documents in that held-out split")

    print(f"{len(sample)} held-out documents, {args.words} words each, "
          f"{len(PERIODS)} eras.\n")
    matrix: collections.Counter = collections.Counter()
    per_band: Dict[str, List[bool]] = collections.defaultdict(list)
    rows: List[Dict] = []
    token_counts: List[int] = []

    for i, (truth, text) in enumerate(sample, 1):
        scores = era_scores(m, text)
        best, second, margin = verdict(scores)
        matrix[(truth, best)] += 1
        per_band[band_of(margin)].append(best == truth)
        token_counts.append(scores[best]["n_tokens"])
        rows.append({"true": truth, "predicted": best, "runner_up": second,
                     "margin_bits": margin,
                     "n_tokens": scores[best]["n_tokens"],
                     "perplexity": {p: s["perplexity"]
                                    for p, s in scores.items()}})
        # Only when a person is watching: \r into a redirected file runs the
        # whole run onto one unreadable line.
        if sys.stdout.isatty():
            print(f"  {i:>3}/{len(sample)}  {truth:<10} -> {best:<10} "
                  f"{margin:+.4f} bits/token", end="\r", flush=True)

    correct = sum(matrix[(p, p)] for p in PERIODS)
    n = len(sample)
    if sys.stdout.isatty():
        print(" " * 70, end="\r")
    print(f"accuracy {correct}/{n} = {correct/n:.1%}   "
          f"(chance {1/len(PERIODS):.0%})\n")

    width = max(len(p) for p in PERIODS)
    print(f"{'':<{width}}  " + "  ".join(f"{p[:4]:>5}" for p in PERIODS)
          + "   <- predicted")
    for truth in PERIODS:
        cells = "  ".join(f"{matrix[(truth, pred)]:>5}" for pred in PERIODS)
        total = sum(matrix[(truth, pred)] for pred in PERIODS)
        hit = matrix[(truth, truth)]
        rate = f"{hit/total:.0%}" if total else "-"
        print(f"{truth:<{width}}  {cells}   {rate:>4} recall")

    print(f"\n{'margin band':<12} {'n':>4} {'accuracy':>9}")
    bands: Dict[str, Dict] = {}
    for name in BAND_NAMES:
        hits = per_band.get(name, [])
        bands[name] = {"n": len(hits),
                       "accuracy": (sum(hits) / len(hits)) if hits else None}
        shown = f"{bands[name]['accuracy']:.1%}" if hits else "-"
        print(f"{name:<12} {len(hits):>4} {shown:>9}")

    payload = {
        "model": m.name, "corpus": args.corpus, "n": n,
        "tokens_per_doc": int(sum(token_counts) / len(token_counts)),
        "words_per_doc": args.words,
        "accuracy": correct / n,
        "periods": list(PERIODS),
        "matrix": {f"{t}->{p}": c for (t, p), c in sorted(matrix.items())},
        "bands": bands,
        "results": rows,
    }
    path = REPORTS / f"era-{m.name}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {path.relative_to(ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", default="-",
                    help="a text file, or - for stdin")
    ap.add_argument("--name", default=DEFAULT_MODEL)
    ap.add_argument("--raw", action="store_true",
                    help="score the text exactly as typed, without folding it "
                         "to the corpus u/v spelling")
    ap.add_argument("--calibrate", type=int, metavar="N",
                    help="measure instead: N held-out documents per era")
    ap.add_argument("--corpus", default="corpus-b")
    ap.add_argument("--words", type=int, default=250,
                    help="calibration: words taken from each document")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    m = load(args.name)
    print(f"{m.describe()}\n")
    if not m.tagged:
        sys.exit(f"{m.name} was trained without period tags -- every era "
                 f"prefix would carry the same (absent) meaning, and the "
                 f"ranking would be noise. Use a model trained with "
                 f"period_tags: true.")
    if args.calibrate:
        return cmd_calibrate(args, m)
    return cmd_estimate(args, m)


if __name__ == "__main__":
    sys.exit(main())

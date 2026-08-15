#!/usr/bin/env python3
"""Evaluation harness — baselines, memorisation, and a versioned report.

Built BEFORE any model exists, deliberately. An evaluation written after the
fact tends to measure what the model happens to be good at; written first, it
states in advance what would count as success.

WHAT IT DOES NOW, WITH NO MODEL

  Baselines. An n-gram language model over the training split, scored on the
  held-out split. This is the floor Latin-0 has to beat: a 2-5M parameter
  transformer that cannot outperform an interpolated bigram has learned nothing.
  Reported per period, genre and form, because a single averaged perplexity over
  Plautus and the Digesta says very little.

  Memorisation. The instrument that matters most for this project. At 2-5M
  parameters trained many epochs over 9.5M tokens, memorisation is the failure
  mode, and the plan's own success criterion ("novel Latin that does not occur in
  the corpus") is unfalsifiable without it. Generated text is cut into windows
  and each is checked verbatim against the training text.

  Self-test. The instrument is validated on inputs whose answer is known:
  training text must score ~100% memorised, held-out text ~0%. An unvalidated
  metric is worse than none, because it will be believed.

EVERY RUN EMITS VERSIONED JSON to data/processed/eval/. Documentation should read
those numbers rather than transcribe them — hand-copied figures are how this
project's docs drifted from its builds before.

Usage:
    python src/evaluate.py --self-test          # validate the instrument
    python src/evaluate.py --baselines          # n-gram floor, sliced
    python src/evaluate.py --memorization FILE  # score generated text
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
TOKENIZERS = PROCESSED / "tokenizers"
REPORTS = PROCESSED / "eval"

SCHEMA_VERSION = 1
DEFAULT_VOCAB = 4000

# Window length for verbatim memorisation, in words. Long enough that ordinary
# Latin collocations do not trip it, short enough to catch real regurgitation.
MEMO_WINDOW = 12


def load_split(split: str) -> List[Dict[str, Any]]:
    path = PROCESSED / f"corpus.{split}.jsonl"
    if not path.exists():
        sys.exit(f"missing {path.relative_to(ROOT)} — run src/build.py first")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def load_tokenizer(vocab: int):
    path = TOKENIZERS / f"latin-bpe-{vocab}.json"
    if not path.exists():
        sys.exit(f"missing {path.relative_to(ROOT)} — run src/tokenizer.py first")
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(path))


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class NgramBaseline:
    """Interpolated bigram over BPE tokens — the floor a real model must clear.

    Deliberately simple. A trigram would score better and cost far more memory;
    the point is not to build a good language model but to establish what
    "learned nothing" looks like on this corpus, per slice.
    """

    def __init__(self, vocab_size: int, lambda_bigram: float = 0.7):
        self.unigram: Counter = Counter()
        self.bigram: Dict[int, Counter] = defaultdict(Counter)
        self.total = 0
        self.vocab_size = vocab_size
        self.lam = lambda_bigram

    def train(self, sequences: Iterable[List[int]]) -> None:
        for ids in sequences:
            previous = None
            for token in ids:
                self.unigram[token] += 1
                self.total += 1
                if previous is not None:
                    self.bigram[previous][token] += 1
                previous = token

    def logprob(self, ids: List[int]) -> Tuple[float, int]:
        """Total log-probability and token count, add-1 smoothed on unigrams."""
        total_lp = 0.0
        previous = None
        denominator = self.total + self.vocab_size
        for token in ids:
            p_uni = (self.unigram[token] + 1) / denominator
            if previous is not None and self.bigram.get(previous):
                context = self.bigram[previous]
                p_bi = context.get(token, 0) / sum(context.values())
                p = self.lam * p_bi + (1 - self.lam) * p_uni
            else:
                p = p_uni
            total_lp += math.log(max(p, 1e-12))
            previous = token
        return total_lp, len(ids)


def perplexity(logprob: float, count: int) -> Optional[float]:
    if not count:
        return None
    return math.exp(-logprob / count)


def slice_baselines(model: NgramBaseline, tok, records: List[Dict[str, Any]],
                    axis: str) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0, 0])
    for rec in records:
        ids = tok.encode(rec["text"]).ids
        lp, n = model.logprob(ids)
        key = str(rec.get(axis) or "UNKNOWN")
        buckets[key][0] += lp
        buckets[key][1] += n
        buckets[key][2] += 1
    return {
        key: {"perplexity": perplexity(lp, n), "tokens": n, "documents": docs}
        for key, (lp, n, docs) in sorted(buckets.items(),
                                         key=lambda kv: -kv[1][1])
    }


# ---------------------------------------------------------------------------
# Memorisation
# ---------------------------------------------------------------------------

def build_haystack(records: List[Dict[str, Any]]) -> str:
    """One normalised string of all training text, for exact substring search.

    Exact search rather than a hashed n-gram index: it cannot report a false
    positive, and at this corpus size (~113MB) Python's substring search is fast
    enough for the few hundred probes a generation check needs.
    """
    return "\n".join(" ".join(rec["text"].split()) for rec in records)


def memorisation(text: str, haystack: str,
                 window: int = MEMO_WINDOW) -> Dict[str, Any]:
    words = " ".join(text.split()).split()
    if len(words) < window:
        return {"windows": 0, "memorised": 0, "fraction": None,
                "longest_verbatim_words": 0}
    hits = 0
    windows = 0
    for start in range(len(words) - window + 1):
        windows += 1
        if " ".join(words[start:start + window]) in haystack:
            hits += 1

    # Longest verbatim run, found by extending from the first matching window.
    longest = 0
    start = 0
    while start <= len(words) - window:
        if " ".join(words[start:start + window]) in haystack:
            end = start + window
            while end < len(words) and " ".join(words[start:end + 1]) in haystack:
                end += 1
            longest = max(longest, end - start)
            start = end - window + 1
        start += 1

    return {"windows": windows, "memorised": hits,
            "fraction": hits / windows if windows else None,
            "longest_verbatim_words": longest}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def corpus_fingerprint() -> Dict[str, Any]:
    path = PROCESSED / "corpus.summary.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        summary = json.load(fh)
    return {k: summary.get(k) for k in
            ("track", "records", "words", "train_words", "validation_words",
             "group_by", "seed", "period_filter")}


def write_report(name: str, payload: Dict[str, Any]) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus": corpus_fingerprint(),
        **payload,
    }
    path = REPORTS / f"{name}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


# ---------------------------------------------------------------------------

def cmd_baselines(args) -> Dict[str, Any]:
    tok = load_tokenizer(args.vocab)
    train = load_split("train")
    val = load_split("validation")

    print(f"training {args.vocab:,}-token bigram baseline on {len(train):,} documents ...",
          flush=True)
    model = NgramBaseline(args.vocab)
    model.train(tok.encode(rec["text"]).ids for rec in train)
    print(f"  {model.total:,} tokens, {len(model.bigram):,} bigram contexts")

    overall_lp = overall_n = 0
    for rec in val:
        lp, n = model.logprob(tok.encode(rec["text"]).ids)
        overall_lp += lp
        overall_n += n
    overall = perplexity(overall_lp, overall_n)
    print(f"\nheld-out perplexity (interpolated bigram): {overall:.1f}")
    print("This is the floor. A model that cannot beat it has learned nothing.\n")

    slices = {axis: slice_baselines(model, tok, val, axis)
              for axis in ("period", "genre", "form")}
    for axis, buckets in slices.items():
        print(f"{axis}:")
        for key, stats in list(buckets.items())[:8]:
            print(f"  {key:<16} ppl {stats['perplexity']:>8.1f}   "
                  f"{stats['tokens']:>9,} tokens  {stats['documents']:>3} docs")
        print()

    return {"baselines": {"model": "interpolated-bigram",
                          "lambda_bigram": model.lam,
                          "vocab": args.vocab,
                          "train_tokens": model.total,
                          "overall_perplexity": overall,
                          "by": slices}}


def cmd_self_test(args) -> Dict[str, Any]:
    """Validate the memorisation instrument on inputs whose answer is known."""
    train = load_split("train")
    val = load_split("validation")
    haystack = build_haystack(train)
    print(f"haystack: {len(haystack):,} characters of training text\n")

    results = {}
    sample = " ".join(train[0]["text"].split()[:300])
    results["training_text"] = memorisation(sample, haystack)
    held = " ".join(val[0]["text"].split()[:300])
    results["held_out_text"] = memorisation(held, haystack)
    # Latin word salad: real words, impossible order.
    import random
    rng = random.Random(11)
    words = " ".join(train[1]["text"].split()).split()[:2000]
    rng.shuffle(words)
    results["shuffled_words"] = memorisation(" ".join(words[:300]), haystack)

    print(f"{'input':<18} {'windows':>8} {'memorised':>10} {'fraction':>9} {'longest':>8}")
    for name, r in results.items():
        frac = f"{r['fraction']:.1%}" if r["fraction"] is not None else "n/a"
        print(f"{name:<18} {r['windows']:>8,} {r['memorised']:>10,} {frac:>9} "
              f"{r['longest_verbatim_words']:>8}")

    # Explicit None checks, not `x or default`. A fraction of 0.0 is falsy, and
    # 0.0 is precisely the correct answer for held-out text — the `or` idiom
    # substituted the default and failed the instrument for behaving correctly.
    def within(value: Optional[float], low: float, high: float) -> bool:
        return value is not None and low <= value <= high

    ok = (within(results["training_text"]["fraction"], 0.95, 1.0)
          and within(results["held_out_text"]["fraction"], 0.0, 0.05)
          and within(results["shuffled_words"]["fraction"], 0.0, 0.05))
    print(f"\ninstrument {'VALID' if ok else 'FAILED'} — "
          f"training text must score ~100%, held-out ~0%")
    return {"self_test": {"results": results, "valid": ok,
                          "window_words": MEMO_WINDOW}}


def overlap_fraction(text: str, haystack: str, window: int,
                     cap: int = 250) -> Optional[float]:
    """Share of sampled windows occurring verbatim in the haystack."""
    words = " ".join(text.split()).split()
    available = max(0, len(words) - window + 1)
    if not available:
        return None
    step = max(1, available // min(cap, available))
    tried = hits = 0
    for start in range(0, available, step):
        tried += 1
        if " ".join(words[start:start + window]) in haystack:
            hits += 1
    return hits / tried if tried else None


def cmd_memorisation(args) -> Dict[str, Any]:
    """Score text for verbatim training overlap, against a real-Latin control.

    A bare 0% is uninterpretable on its own: it could mean the model invents
    everything, or merely that the window is too long for any text to match.
    Genuine unseen Latin — a held-out work by a real author — sets the scale.
    Overlap at or below that line is not regurgitation; well above it is.
    """
    train = load_split("train")
    haystack = build_haystack(train)
    text = sys.stdin.read() if args.memorization == "-" else \
        Path(args.memorization).read_text(encoding="utf-8")

    result = memorisation(text, haystack, window=args.window)
    frac = f"{result['fraction']:.1%}" if result["fraction"] is not None else "n/a"
    print(f"windows {result['windows']:,}   memorised {result['memorised']:,}   "
          f"fraction {frac}   longest verbatim run "
          f"{result['longest_verbatim_words']} words\n")

    control_text = load_split("validation")[0]["text"]
    curve = {}
    print(f"{'window':>7} {'this text':>11} {'real unseen Latin':>19}")
    for w in (3, 4, 5, 6, 8, 12):
        here = overlap_fraction(text, haystack, w)
        there = overlap_fraction(control_text, haystack, w)
        curve[w] = {"text": here, "control": there}
        print(f"{w:>7} {here:>10.1%} {there:>18.1%}")

    return {"memorization": result, "window": args.window, "curve": curve}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baselines", action="store_true",
                    help="n-gram floor on the held-out split, sliced")
    ap.add_argument("--self-test", action="store_true",
                    help="validate the memorisation instrument on known inputs")
    ap.add_argument("--memorization", metavar="FILE",
                    help="score a file (or - for stdin) for verbatim training overlap")
    ap.add_argument("--vocab", type=int, default=DEFAULT_VOCAB)
    ap.add_argument("--window", type=int, default=MEMO_WINDOW,
                    help="verbatim window in words (default 12)")
    args = ap.parse_args()

    if args.baselines:
        payload, name = cmd_baselines(args), "baselines"
    elif args.self_test:
        payload, name = cmd_self_test(args), "self-test"
    elif args.memorization:
        payload, name = cmd_memorisation(args), "memorization"
    else:
        ap.print_help()
        return 0

    path = write_report(name, payload)
    print(f"\nwrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

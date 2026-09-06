#!/usr/bin/env python3
"""Ask a trained model how surprising a piece of Latin is.

train.py measures perplexity on a held-out split to judge the model. This
turns the same machinery around: the model becomes the instrument and the text
becomes the thing measured. What it produces is a log-probability per token,
which is the input to four separate jobs:

  AUDIT the corpus.       Score every document; the outliers are OCR damage,
                          mojibake, surviving apparatus and wrong-language
                          text. The vowel-ratio heuristic in the known-
                          limitations list is a crude version of this.
  DATE anonymous text.    Score the same tokens under each period tag and see
                          which conditioning fits best. 43% of the corpus has
                          no period, and this is a real signal about it.
  RANK OCR hypotheses.    P(text) in a noisy-channel model. Needs no knowledge
                          of the world, only a sense of what Latin sequences
                          are plausible, which is exactly what this model has.
  LOCALISE oddity.        Per-token surprisal points at where a passage stops
                          looking like Latin, which a single perplexity cannot.

WHAT THIS IS NOT
    A grammar checker. The model reports improbability, not error, and cannot
    tell the two apart: genuine Plautine archaism and a scanning mistake are
    both surprising. For agreement and case government use a morphological
    analyser or a treebank parser. This is a screen, not an authority.

Usage:
    python src/score.py --name latin-xxv FILE...      # perplexity per file
    echo "Gallia est omnis" | python src/score.py --name latin-xxv -
    python src/score.py --name latin-xxv --tokens FILE   # per-token surprisal
    python src/score.py --name latin-xxv --audit classical --limit 200
    python src/score.py --name latin-xxv --period-fit FILE
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
CHECKPOINTS = ROOT / "checkpoints"
REPORTS = ROOT / "reports"

PERIODS = ("CLASSICAL", "LATE", "MEDIEVAL", "NEOLATIN")


# ---------------------------------------------------------------------------
# Windowing — pure, so it can be tested without a model
# ---------------------------------------------------------------------------

def window_plan(n_tokens: int, context: int,
                stride: int) -> List[Tuple[int, int, int]]:
    """(start, end, first_target) windows covering every target exactly once.

    A token's log-probability is only defined given what precedes it, so a
    document longer than the context window has to be scored in overlapping
    slices. The overlap is not optional: scoring back-to-back blocks would
    give the first token of every block no left context at all and charge the
    model for a discontinuity it never saw. Each window therefore re-reads
    `context - stride` tokens purely as context and scores only what is new,
    so no token is counted twice and none is scored cold.
    """
    if n_tokens < 2:
        return []
    plans: List[Tuple[int, int, int]] = []
    start = 0
    while True:
        end = min(start + context, n_tokens)
        first_target = 1 if start == 0 else start + context - stride
        if first_target < end:
            plans.append((start, end, first_target))
        if end >= n_tokens:
            break
        start += stride
    return plans


# Thresholds, calibrated on the 2026-08-19 audit of the 79M-word corpus
# rather than chosen. See classify() for what each one separates.
DEFAULT_CEILING = 70.0
DEFAULT_FLOOR = 3.0
DEFAULT_FLOOR_VARIETY = 0.05


def classify(perplexity_value: Optional[float], distinct_ratio: Optional[float],
             ceiling: float = DEFAULT_CEILING, floor: float = DEFAULT_FLOOR,
             floor_variety: float = DEFAULT_FLOOR_VARIETY) -> str:
    """Whether a document looks like Latin, foreign/corrupt, or degenerate.

    High perplexity catches text that is not Latin — the audit found a Lorem
    ipsum stub filed as De Oratore at 212, a macaronic German carol, a
    treatise on French pronunciation. Low perplexity catches text carrying
    almost no information: Wikisource navigation indexes, placeholder pages
    whose entire content is bare dashes.

    The low tail needs a human, because it does not separate defect from
    fact. EDCS chunks of Roman epigraphy land there too, and they are not a
    bug: brick stamps and lead water-pipe stamps were mass-produced with
    identical text, so `Legio XXII Primigenia` really does occur hundreds of
    times over. Checked against the raw dump — 34 distinct records carry
    `Officina Marciana`, and the extracted document has 23 occurrences, so
    nothing is being duplicated. Genuinely repetitive, genuinely Latin, and
    still close to worthless as training data. This flags it; it does not
    presume to say which of those it is.

    Low perplexity ALONE is not evidence of that, which is why the floor test
    also requires low lexical variety. Genuinely formulaic Latin — epigraphy,
    liturgy, legal boilerplate — is predictable and legitimate. Measured on
    the corpus, degenerate documents sit at 0.006-0.022 distinct tokens per
    token while the least varied legitimate document reaches 0.129 and the
    median is 0.395, so 0.05 separates them with room on both sides.

    The ceiling sits above the corpus's own p99 of 61.4 and far below the
    495.8 that English scores, so it flags what does not belong without
    condemning difficult Latin.
    """
    if perplexity_value is None:
        return "ok"
    if perplexity_value > ceiling:
        return "surprising"
    if (perplexity_value < floor and distinct_ratio is not None
            and distinct_ratio < floor_variety):
        return "degenerate"
    return "ok"


def perplexity(logprobs: List[float]) -> Optional[float]:
    if not logprobs:
        return None
    return math.exp(-sum(logprobs) / len(logprobs))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(name: str):
    """Weights plus the architecture recorded when they were trained."""
    import mlx.core as mx  # noqa: F401  (imported for the caller's benefit)
    sys.path.insert(0, str(ROOT / "src"))
    from train import GPT

    report_path = REPORTS / f"train-{name}.json"
    if not report_path.exists():
        sys.exit(f"missing {report_path.relative_to(ROOT)} — train {name} first")
    with report_path.open(encoding="utf-8") as fh:
        run = json.load(fh)
    cfg = run["config"]

    weights = CHECKPOINTS / f"{name}.safetensors"
    if not weights.exists():
        sys.exit(f"missing {weights.relative_to(ROOT)}")

    from tokenizers import Tokenizer
    sys.path.insert(0, str(ROOT / "src"))
    from encoding import resolve_tokenizer
    tok = Tokenizer.from_file(resolve_tokenizer(cfg["vocab"], run))

    model = GPT(cfg["vocab"], cfg["dim"], cfg["layers"], cfg["heads"], cfg["context"])
    model.load_weights(str(weights))
    model.eval()
    mx.eval(model.parameters())
    return model, tok, cfg, run


def prefix_ids(tok, period: Optional[str]) -> List[int]:
    """What the model saw before each document during training.

    train.py streams documents as `<PERIOD> ... text ... <|endoftext|>`, so a
    document's first real token was predicted from the tag, and the tag from
    the previous document's end-of-text. Reproducing that here is what makes a
    score comparable with the perplexity the training run reported.
    """
    ids = [tok.token_to_id("<|endoftext|>")]
    if period:
        tag = tok.token_to_id(f"<{period}>")
        if tag is None:
            sys.exit(f"tokenizer has no <{period}> token")
        ids.append(tag)
    return ids


def token_logprobs(model, ids: List[int], context: int,
                   stride: Optional[int] = None) -> List[Optional[float]]:
    """log P(ids[i] | ids[:i]) for every i >= 1; index 0 is always None."""
    import mlx.core as mx

    stride = stride or max(1, context // 2)
    out: List[Optional[float]] = [None] * len(ids)
    for start, end, first_target in window_plan(len(ids), context, stride):
        window = ids[start:end]
        logits = model(mx.array([window]))[0].astype(mx.float32)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        # Row t holds the distribution over window[t + 1].
        targets = mx.array(window[1:])
        picked = mx.take_along_axis(logp[:-1], targets[:, None], axis=-1)[:, 0]
        values = picked.tolist()
        for t, value in enumerate(values):
            absolute = start + t + 1
            if absolute >= first_target:
                out[absolute] = value
    return out


def score_text(model, tok, text: str, context: int,
               period: Optional[str] = None) -> Dict[str, Any]:
    """Perplexity of `text`, plus the pieces needed to look inside it."""
    prefix = prefix_ids(tok, period)
    body = tok.encode(text).ids
    ids = prefix + body
    logprobs = token_logprobs(model, ids, context)
    scored = [lp for lp in logprobs[len(prefix):] if lp is not None]
    return {
        "n_tokens": len(scored),
        "perplexity": perplexity(scored),
        "mean_logprob": (sum(scored) / len(scored)) if scored else None,
        "ids": ids,
        "logprobs": logprobs,
        "prefix_len": len(prefix),
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def read_input(path: str) -> str:
    return sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")


def cmd_files(args, model, tok, cfg) -> None:
    print(f"{'perplexity':>11}  {'tokens':>8}  text")
    print("-" * 60)
    for path in args.files:
        text = read_input(path)
        result = score_text(model, tok, text, cfg["context"], args.period)
        ppl = result["perplexity"]
        label = "-" if ppl is None else f"{ppl:.1f}"
        print(f"{label:>11}  {result['n_tokens']:>8,}  {path}")
        if args.tokens:
            show_tokens(tok, result, args.worst)


def show_tokens(tok, result: Dict[str, Any], worst: int) -> None:
    """Per-token surprisal, in bits, with the least likely tokens called out.

    Bits rather than nats because the unit is interpretable: a token costing
    10 bits is one the model gave a 1-in-1024 chance. That is the number that
    says *where* a passage stops looking like Latin.
    """
    ids, logprobs = result["ids"], result["logprobs"]
    rows = [(i, ids[i], -lp / math.log(2))
            for i, lp in enumerate(logprobs) if lp is not None
            and i >= result["prefix_len"]]
    if not rows:
        return
    surprisals = [bits for _, _, bits in rows]
    print(f"    median {statistics.median(surprisals):.1f} bits/token, "
          f"max {max(surprisals):.1f}")
    print(f"    {'bits':>6}  token")
    for i, tid, bits in sorted(rows, key=lambda r: -r[2])[:worst]:
        piece = tok.decode([tid]).replace("\n", "\\n")
        context = tok.decode(ids[max(0, i - 8):i]).replace("\n", " ")
        print(f"    {bits:>6.1f}  {piece!r:<14} after …{context[-44:]}")
    print()


def cmd_period_fit(args, model, tok, cfg) -> None:
    """Which register does this text look like? Same tokens, different prefix.

    Only the conditioning changes between rows, so the perplexities are
    measured over an identical token sequence and compare directly — which is
    not true of most perplexity comparisons in this project.
    """
    if not cfg.get("period_tags"):
        print(f"WARNING: {args.name} was trained without period tags; "
              f"these numbers carry no learned meaning\n", file=sys.stderr)
    for path in args.files:
        text = read_input(path)
        scores = []
        for period in PERIODS:
            result = score_text(model, tok, text, cfg["context"], period)
            scores.append((result["perplexity"], period, result["n_tokens"]))
        baseline = score_text(model, tok, text, cfg["context"], None)
        scores.sort()
        best = scores[0]
        print(f"{path}  ({best[2]:,} tokens)")
        print(f"  {'period':<12} {'perplexity':>11}   vs best")
        for ppl, period, _ in scores:
            marker = "  <-- best fit" if period == best[1] else ""
            print(f"  {period:<12} {ppl:>11.1f}   {ppl / best[0]:>5.3f}x{marker}")
        print(f"  {'(untagged)':<12} {baseline['perplexity']:>11.1f}")
        print()


def cmd_audit(args, model, tok, cfg) -> Dict[str, Any]:
    """Score every document in a split and flag both tails.

    Not a ranking to browse — a gate. Documents outside either threshold are
    findings, and --check turns the word share they account for into an exit
    code, the way provenance.py --check does for licensing.
    """
    path = PROCESSED / f"{args.audit}.{args.split}.jsonl"
    if not path.exists():
        sys.exit(f"missing {path.relative_to(ROOT)} — run src/build.py")

    with path.open(encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]
    if args.limit:
        records = records[:args.limit]

    scored: List[Dict[str, Any]] = []
    for n, record in enumerate(records, 1):
        text = record["text"]
        # A long document is sampled rather than read whole: the point is a
        # comparable estimate per document, not an exact one. The slice comes
        # from the middle because front matter — titles, salutations,
        # editorial headers — is unrepresentative of the body. It is also
        # where the EDCS repetition artifact hides, which a prefix would miss.
        if args.max_chars and len(text) > args.max_chars:
            middle = len(text) // 2
            half = args.max_chars // 2
            text = text[middle - half:middle + half]

        # 43% of the full corpus has period UNASSIGNED, and there is no
        # <UNASSIGNED> token — train.py silently prefixes nothing for those,
        # so scoring has to do the same or it would charge the model for a
        # conditioning it never received.
        period = args.period or record.get("period")
        if not cfg.get("period_tags") or period not in PERIODS:
            period = None

        result = score_text(model, tok, text, cfg["context"], period)
        if result["perplexity"] is None:
            continue
        body = result["ids"][result["prefix_len"]:]
        distinct = len(set(body)) / len(body) if body else None
        scored.append({
            "id": record.get("id"), "author": record.get("author"),
            "work": record.get("work"), "source": record.get("source"),
            "period": record.get("period"), "n_words": record.get("n_words"),
            "tokens_scored": result["n_tokens"],
            "perplexity": result["perplexity"],
            "distinct_ratio": distinct,
            "verdict": classify(result["perplexity"], distinct,
                                args.ceiling_ppl, args.floor_ppl,
                                args.floor_variety),
        })
        if n % 250 == 0 or n == len(records):
            print(f"  scored {n:,}/{len(records):,}", flush=True)

    values = sorted(s["perplexity"] for s in scored)
    quantiles = {q: values[min(len(values) - 1, int(len(values) * q))]
                 for q in (0.01, 0.25, 0.5, 0.75, 0.95, 0.99)}
    total_words = sum(s["n_words"] or 0 for s in scored)
    print(f"\n{len(scored):,} documents, {total_words:,} words")
    print("  " + "  ".join(f"p{int(q*100)} {v:.1f}" for q, v in quantiles.items()))

    tails = {
        "surprising": ("NOT LATIN, OR DAMAGED", f"perplexity > {args.ceiling_ppl:g}",
                       lambda s: -s["perplexity"]),
        "degenerate": ("NO INFORMATION", f"perplexity < {args.floor_ppl:g} and "
                       f"distinct tokens < {args.floor_variety:g}",
                       lambda s: s["perplexity"]),
    }
    summary: Dict[str, Any] = {}
    for verdict, (title, rule, order) in tails.items():
        hits = sorted([s for s in scored if s["verdict"] == verdict], key=order)
        words = sum(s["n_words"] or 0 for s in hits)
        share = words / total_words if total_words else 0.0
        summary[verdict] = {"documents": len(hits), "words": words, "share": share}
        print(f"\n{title} — {rule}")
        print(f"  {len(hits):,} documents, {words:,} words ({share:.2%})")
        if hits:
            print(f"  {'ppl':>8} {'distinct':>9} {'words':>9}  {'source':<22} work")
        for s in hits[:args.worst]:
            print(f"  {s['perplexity']:>8.2f} {(s['distinct_ratio'] or 0):>9.3f} "
                  f"{(s['n_words'] or 0):>9,}  {(s['source'] or '?')[:22]:<22} "
                  f"{(s['work'] or s['id'] or '?')[:38]}")
        if len(hits) > args.worst:
            print(f"  ... and {len(hits) - args.worst:,} more (full list in --out)")

    flagged_words = sum(v["words"] for v in summary.values())
    flagged_share = flagged_words / total_words if total_words else 0.0
    print(f"\ntotal flagged: {flagged_words:,} words ({flagged_share:.2%}) "
          f"of {total_words:,}")

    ok = flagged_share <= args.budget
    if args.check:
        verdict = "PASS" if ok else "FAIL"
        print(f"{verdict}: flagged {flagged_share:.2%}, budget {args.budget:.2%}")

    return {"corpus": args.audit, "split": args.split, "model": args.name,
            "max_chars": args.max_chars,
            "thresholds": {"ceiling_ppl": args.ceiling_ppl,
                           "floor_ppl": args.floor_ppl,
                           "floor_variety": args.floor_variety,
                           "budget": args.budget},
            "quantiles": quantiles, "total_words": total_words,
            "flagged": summary, "flagged_share": flagged_share,
            "passed": ok, "documents": scored}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="files to score, or - for stdin")
    ap.add_argument("--name", default="latin-xxv")
    ap.add_argument("--period", choices=PERIODS,
                    help="condition on a period tag")
    ap.add_argument("--tokens", action="store_true",
                    help="per-token surprisal, to localise what is odd")
    ap.add_argument("--period-fit", action="store_true",
                    help="score under every period tag and rank the fit")
    ap.add_argument("--audit", metavar="CORPUS",
                    help="score every document of a built corpus and rank it")
    ap.add_argument("--split", default="train", choices=("train", "validation"))
    ap.add_argument("--limit", type=int, help="audit only the first N documents")
    ap.add_argument("--max-chars", type=int, default=8000,
                    help="characters sampled per document when auditing "
                         "(0 for the whole text)")
    ap.add_argument("--worst", type=int, default=20,
                    help="how many of each tail to print")
    ap.add_argument("--check", action="store_true",
                    help="gate mode: exit non-zero if the flagged word share "
                         "exceeds --budget")
    ap.add_argument("--ceiling-ppl", type=float, default=DEFAULT_CEILING,
                    help=f"above this a document is not Latin, or is damaged "
                         f"(default {DEFAULT_CEILING:g}; the corpus p99 is 61.4 "
                         f"and English scores ~496)")
    ap.add_argument("--floor-ppl", type=float, default=DEFAULT_FLOOR,
                    help=f"below this, with low variety, a document carries no "
                         f"information (default {DEFAULT_FLOOR:g})")
    ap.add_argument("--floor-variety", type=float, default=DEFAULT_FLOOR_VARIETY,
                    help=f"distinct tokens per token; degenerate documents "
                         f"measure 0.006-0.022, legitimate ones 0.129 and up "
                         f"(default {DEFAULT_FLOOR_VARIETY:g})")
    ap.add_argument("--budget", type=float, default=0.005,
                    help="share of words that may be flagged before --check "
                         "fails (default 0.005)")
    ap.add_argument("--out", help="write a JSON report here")
    args = ap.parse_args()

    if not args.files and not args.audit:
        ap.print_help()
        return 0

    model, tok, cfg, run = load_model(args.name)
    print(f"{args.name}: {run['parameters']/1e6:.2f}M parameters, trained on "
          f"{cfg.get('corpus')}, held-out perplexity "
          f"{run['final']['val_perplexity']:.1f}\n")

    payload: Optional[Dict[str, Any]] = None
    if args.audit:
        payload = cmd_audit(args, model, tok, cfg)
    elif args.period_fit:
        cmd_period_fit(args, model, tok, cfg)
    else:
        cmd_files(args, model, tok, cfg)

    if args.out and payload is not None:
        Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                                  encoding="utf-8")
        print(f"\nwrote {args.out}")

    # Non-zero on a failed gate, so this can sit in a build the way
    # provenance.py --check does.
    if args.check and payload is not None and not payload.get("passed", True):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

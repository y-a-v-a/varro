#!/usr/bin/env python3
"""Compare training runs side by side.

Latin-0 and Latin-2 are meant to be read against each other, not in isolation:
same pipeline, same tokenizer, different corpus size and model size. This reads
the versioned reports in reports/ rather than any transcribed figure, so the
comparison cannot drift from what was actually measured.

Usage:
    python src/compare.py                    # every run found
    python src/compare.py latin-0 latin-2
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

FLOOR_NOTE = "bigram floor from reports/baselines-*.json"

MARKS = {"ok": "", "stale": "!", "unverified": "?", "no-floor": "-"}

EXPLANATIONS = {
    "stale": "measured on a different build of this corpus than the current floor",
    "unverified": "report predates corpus fingerprinting — build unknown, "
                  "assume it differs",
    "no-floor": "no baseline for this corpus — run evaluate.py --baselines",
}


def load_runs(names: List[str]) -> List[Dict[str, Any]]:
    paths = ([REPORTS / f"train-{n}.json" for n in names] if names
             else sorted(REPORTS.glob("train-*.json")))
    runs = []
    for path in paths:
        if not path.exists():
            print(f"  (no report for {path.stem})", file=sys.stderr)
            continue
        with path.open(encoding="utf-8") as fh:
            runs.append(json.load(fh))
    return runs


def floor_for(corpus: str) -> tuple[float | None, Dict[str, Any] | None]:
    """The floor for `corpus`, and the whole report it came from."""
    path = REPORTS / f"baselines-{corpus}.json"
    if not path.exists():
        return None, None
    with path.open(encoding="utf-8") as fh:
        report = json.load(fh)
    return report["baselines"]["overall_perplexity"], report


def comparability(run: Dict[str, Any], floor: float | None,
                  floor_report: Dict[str, Any] | None) -> str:
    """Whether this run's perplexity may be quoted against this floor.

    A corpus keeps its name across rebuilds, so `baselines-corpus.json` and
    `train-latin-2.json` can describe entirely different text. Every model
    trained before the corpus grew to 79M words is in exactly that position:
    same corpus name, different corpus, incomparable perplexity. Printing
    those side by side under one `x floor` column is the mistake this guard
    exists to prevent.
    """
    if floor is None:
        return "no-floor"
    run_fingerprint = run.get("corpus") or None
    floor_fingerprint = (floor_report or {}).get("corpus") or None
    if run_fingerprint is None or floor_fingerprint is None:
        return "unverified"
    if run_fingerprint != floor_fingerprint:
        return "stale"
    # The tokenizer is the other half of "same held-out text". Retraining the
    # 4k vocabulary on a larger corpus moved the classical floor 158.8 -> 149.8
    # while the classical corpus itself changed by three words.
    run_tokenizer = run.get("tokenizer")
    floor_tokenizer = (floor_report or {}).get("tokenizer")
    if run_tokenizer is None or floor_tokenizer is None:
        return "unverified"
    return "ok" if run_tokenizer == floor_tokenizer else "stale"


def memorisation_for(name: str) -> Dict[str, Any] | None:
    path = REPORTS / f"memorization-{name}.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    runs = load_runs(sys.argv[1:])
    if not runs:
        sys.exit("no run reports in reports/ — train something first")

    rows = []
    for run in runs:
        cfg = run["config"]
        final = run.get("final") or {}
        corpus = cfg.get("corpus", "corpus")
        floor, floor_report = floor_for(corpus)
        ppl = final.get("val_perplexity")
        memo = memorisation_for(run["run"])
        status = comparability(run, floor, floor_report)
        rows.append({
            "name": run["run"],
            "params": run["parameters"],
            "corpus": corpus,
            "tokens": run["train_tokens"],
            "epochs": run["epochs"],
            "ppl": ppl,
            "floor": floor if status == "ok" else None,
            "gain": (floor / ppl) if (status == "ok" and ppl) else None,
            "memo": (memo or {}).get("memorization", {}).get("fraction"),
            "hours": (final.get("elapsed_s") or 0) / 3600,
            "status": status,
        })

    head = (f"{'run':<22} {'params':>9} {'corpus':>10} {'tokens':>12} {'ep':>5} "
            f"{'ppl':>8} {'floor':>8} {'x floor':>8} {'memo':>6} {'hours':>6}  ")
    print(head)
    print("-" * len(head))
    for r in rows:
        ppl = f"{r['ppl']:.1f}" if r["ppl"] else "-"
        floor = f"{r['floor']:.1f}" if r["floor"] else "-"
        gain = f"{r['gain']:.2f}x" if r["gain"] else "-"
        memo = f"{r['memo']:.1%}" if r["memo"] is not None else "-"
        mark = MARKS[r["status"]]
        print(f"{r['name'][:22]:<22} {r['params']/1e6:>8.2f}M {r['corpus'][:10]:>10} "
              f"{r['tokens']:>12,} {r['epochs']:>5.1f} {ppl:>8} {floor:>8} "
              f"{gain:>8} {memo:>6} {r['hours']:>6.1f}  {mark}")
    print(f"\n  ppl      held-out perplexity, lower is better")
    print(f"  x floor  how many times better than the {FLOOR_NOTE}")
    print(f"  memo     share of 12-word windows found verbatim in training text")

    flagged = {r["status"] for r in rows} - {"ok"}
    if flagged:
        print()
        for status in ("stale", "unverified", "no-floor"):
            if status in flagged:
                print(f"  {MARKS[status]:<3} {EXPLANATIONS[status]}")
        print("\n  Perplexities in flagged rows are held-out scores over different\n"
              "  text and do not compare with each other or with the current floor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

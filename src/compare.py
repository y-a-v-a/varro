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


def floor_for(corpus: str) -> float | None:
    path = REPORTS / f"baselines-{corpus}.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)["baselines"]["overall_perplexity"]


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
        floor = floor_for(corpus)
        ppl = final.get("val_perplexity")
        memo = memorisation_for(run["run"])
        rows.append({
            "name": run["run"],
            "params": run["parameters"],
            "corpus": corpus,
            "tokens": run["train_tokens"],
            "epochs": run["epochs"],
            "ppl": ppl,
            "floor": floor,
            "gain": (floor / ppl) if (floor and ppl) else None,
            "memo": (memo or {}).get("memorization", {}).get("fraction"),
            "hours": (final.get("elapsed_s") or 0) / 3600,
        })

    head = (f"{'run':<22} {'params':>9} {'corpus':>10} {'tokens':>12} {'ep':>5} "
            f"{'ppl':>8} {'floor':>8} {'x floor':>8} {'memo':>6} {'hours':>6}")
    print(head)
    print("-" * len(head))
    for r in rows:
        ppl = f"{r['ppl']:.1f}" if r["ppl"] else "-"
        floor = f"{r['floor']:.1f}" if r["floor"] else "-"
        gain = f"{r['gain']:.2f}x" if r["gain"] else "-"
        memo = f"{r['memo']:.1%}" if r["memo"] is not None else "-"
        print(f"{r['name'][:22]:<22} {r['params']/1e6:>8.2f}M {r['corpus'][:10]:>10} "
              f"{r['tokens']:>12,} {r['epochs']:>5.1f} {ppl:>8} {floor:>8} "
              f"{gain:>8} {memo:>6} {r['hours']:>6.1f}")
    print(f"\n  ppl      held-out perplexity, lower is better")
    print(f"  x floor  how many times better than the {FLOOR_NOTE}")
    print(f"  memo     share of 12-word windows found verbatim in training text")
    return 0


if __name__ == "__main__":
    sys.exit(main())

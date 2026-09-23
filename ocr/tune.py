#!/usr/bin/env python3
"""Find the rule under which a repairer stops breaking what was right.

    python ocr/tune.py trace lii          # ask the model once, write it all down
    python ocr/tune.py sweep lii          # replay that under many rules
    python ocr/tune.py sweep lii --save   # ... and keep the winner in rules.json
    python ocr/tune.py ceiling unigram    # how far could a perfect scorer get?

WHY THE MODEL IS ASKED ONCE
    A rule decides which proposed changes stand. It does not change what is
    proposed or what the scorer thinks of it. So one pass over dev with every
    decision recorded -- the candidates, the scorer's nats for each, what kind
    of word it was -- can be replayed under any rule in milliseconds, where
    asking LATIN-LII again takes half an hour. The replay is approximate in
    one respect: later words were scored with the left context the RECORDED
    run chose, not the one the replayed rule would have produced. A winner is
    therefore always confirmed with a real `bench.py run`.

WHAT IS OPTIMISED
    Recall on dev, subject to harm. The limit in ocr/README.md (H2) is two
    words broken per thousand on undamaged text; the sweep aims below it, at
    TARGET, because a limit met exactly on the text it was tuned on is a limit
    missed on the text it was not. Harm is measured on dev's clean quarter
    plus the clean split -- 39,600 words, where dev alone has 3,600 and the
    limit would be seven words.

NOTHING HERE READS THE TEST SPLIT.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import bench  # noqa: E402
import metrics  # noqa: E402
from repairers import Rule, decide, make, page_flags  # noqa: E402

TARGET = 1.5          # words broken per 1,000 standing, on clean text
CEILING = 2.0         # ... and on the damaged levels, where H2 sets no limit
                      # but a repairer that breaks more than this is not one

GRID = {
    "weight": (0.6, 0.8, 1.0, 1.25, 1.5),
    "margin": (0, 1, 2, 3, 4, 6, 8),
    "known": (0, 2, 4, 6, 10),
    "unknown": (0, 1, 2, 4),
    "capital": (0, 2, 4, 8, 100),
    "join": (0, 2, 4, 8, 100),
    "split": (0, 2, 4, 8),
    "short": (0, 2, 4, 8),
    "repeated": (0, 2, 4, 8, 100),
    "seen": (0, 1, 2, 4),
}


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def trace_path(corpus: str, name: str, tier: int = 1) -> Path:
    stem = "trace-" if tier == 1 else "trace-t2-"
    return bench.OUT / f"{stem}{corpus}-{slug(name)}.json"


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------

def cmd_trace(args) -> int:
    lex = bench.lexicon(args.corpus)
    dev, dev_hash = bench.load_split(args.corpus, "dev", args.tier)
    # Tier 2 brings its own control: the passages its scanner read perfectly.
    clean, clean_hash = ([], None) if args.tier == 2 else \
        bench.load_split(args.corpus, "clean")
    system = make(args.system, tuned=False)
    system.fit([(p["truth"], p["noisy"]) for p in dev], lex, bench.capitalised(args.corpus))

    rows = dev + clean
    if args.limit:
        rows = rows[:args.limit]
    out = []
    for k, p in enumerate(rows, 1):
        system.trace = []
        system.repair(p["noisy"])
        out.append({"id": p["id"], "split": p["split"], "level": p["level"],
                    "truth": p["truth"], "noisy": p["noisy"],
                    "entries": system.trace})
        if k % 50 == 0:
            print(f"  {k}/{len(rows)}", file=sys.stderr)
    path = trace_path(args.corpus, system.name, args.tier)
    path.write_text(json.dumps({"system": system.name, "tier": args.tier,
                                "dev": dev_hash,
                                "clean": clean_hash, "passages": out}),
                    encoding="utf-8")
    print(f"wrote {path.relative_to(bench.ROOT)}")
    return 0


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def replay(passage: Dict[str, Any], rule: Rule) -> str:
    words = passage["noisy"].split()
    entries = {e["k"]: e for e in passage["entries"]}
    out: List[str] = []
    k = 0
    while k < len(words):
        e = entries.get(k)
        if e is None:
            out.append(words[k])
            k += 1
            continue
        flags = e["flags"]
        if "repeated" not in flags:
            # A trace from before the page was consulted; consult it now.
            flags = {**flags, **page_flags(words, k, e["pool"])}
        text, _, used = e["pool"][decide(rule, e["pool"], e["lang"], flags)]
        out.append(text)
        k += used
    return " ".join(out)


class Replayer:
    def __init__(self, trace: Dict[str, Any], lex: set):
        self.passages = trace["passages"]
        self.lex = lex
        self.tier = trace.get("tier", 1)
        self.memo: Dict[Tuple[int, str], Counter] = {}

    def score(self, rule: Rule) -> Dict[str, float]:
        damaged: Counter = Counter()
        clean: Counter = Counter()
        for n, p in enumerate(self.passages):
            text = replay(p, rule)
            key = (n, text)
            if key not in self.memo:
                self.memo[key] = metrics.tally(p["truth"], p["noisy"], text, self.lex)
            (clean if p["level"] in (0, "0") else damaged).update(self.memo[key])
        pooled = clean + damaged
        return {
            "recall": metrics.recall(damaged) or 0.0,
            "recall_real": metrics.recall(damaged, "real:") or 0.0,
            "precision": metrics.precision(damaged) or 0.0,
            "harm_clean": metrics.harm(clean) or 0.0,
            "harm_damaged": metrics.harm(damaged) or 0.0,
            "harm_pooled": metrics.harm(pooled) or 0.0,
            "broken_clean": clean["broken"],
            "clean_words": clean["words"],
            "standing_words": pooled["kept"] + pooled["broken"],
        }

    def rank(self, s: Dict[str, float]) -> Tuple[bool, float]:
        """Inside the limit beats outside it; inside, recall; outside, less harm."""
        if self.ok(s):
            return (True, s["recall"])
        return (False, -(s["harm_clean"] + s["harm_damaged"]))

    def ok(self, s: Dict[str, float]) -> bool:
        """Under the limit, measured where there are enough words to measure it.

        Tier 1 has a clean control of 36,000 words, and the limit is applied
        there and, more loosely, on the damaged levels. Tier 2's control is
        the fifty-odd passages a scanner happened to read perfectly -- three
        thousand words, where the limit is four broken words and the fifth
        decides everything. The first sweep tuned to it, and what it found
        was a rule shaped by four words. So on Tier 2 the limit is applied to
        harm over EVERY standing word, control and damaged passages pooled:
        thirty-two thousand of them, and the same rate the report prints.
        """
        if self.tier == 2:
            return s["harm_pooled"] <= TARGET
        return s["harm_clean"] <= TARGET and s["harm_damaged"] <= CEILING


def cmd_sweep(args) -> int:
    lex = set(bench.lexicon(args.corpus))
    path = trace_path(args.corpus, args.system, args.tier)
    if not path.exists():
        have = sorted(f.name for f in bench.OUT.glob(f"trace-{args.corpus}-*.json"))
        sys.exit(f"no {path.name}; traces on disk: {have}")
    trace = json.loads(path.read_text(encoding="utf-8"))
    r = Replayer(trace, lex)

    base = r.score(Rule())
    print(f"{trace['system']}\n  raw sum      recall {base['recall']:6.1%}  "
          f"harm clean {base['harm_clean']:5.2f}  damaged {base['harm_damaged']:5.2f}")

    # Coordinate ascent from a cautious start. The surface is lumpy -- harm
    # moves a word at a time -- so several passes, and every rule that met the
    # limit is kept, not only the last.
    rule = Rule(margin=6, known=6, capital=8, join=8, split=4, short=4)
    best = r.score(rule)
    seen = 1
    for sweep in range(4):
        moved = False
        for field, values in GRID.items():
            for v in values:
                trial = rule._replace(**{field: float(v)})
                if trial == rule:
                    continue
                s = r.score(trial)
                seen += 1
                better = r.rank(s) > r.rank(best)
                if better:
                    rule, best, moved = trial, s, True
        print(f"  pass {sweep + 1}     recall {best['recall']:6.1%}  "
              f"harm clean {best['harm_clean']:5.2f}  damaged "
              f"{best['harm_damaged']:5.2f}   {seen} rules tried")
        if not moved:
            break

    print(f"\n  {rule}\n  recall {best['recall']:.1%} (real-word "
          f"{best['recall_real']:.1%}), precision {best['precision']:.1%}\n"
          f"  broke {best['broken_clean']} of {best['clean_words']:,} clean words "
          f"= {best['harm_clean']:.2f} per 1,000; on damaged text "
          f"{best['harm_damaged']:.2f}; over all {best['standing_words']:,} "
          f"standing words {best['harm_pooled']:.2f}")
    if not r.ok(best):
        print("  NO RULE IN THE GRID MEETS THE LIMIT")

    if args.frontier:
        print("\n  what each step of caution costs (margin swept, rest as above):")
        for m in (0, 1, 2, 3, 4, 6, 8, 12, 16):
            s = r.score(rule._replace(margin=float(m)))
            print(f"    margin {m:>2}   recall {s['recall']:6.1%}   harm clean "
                  f"{s['harm_clean']:5.2f}   damaged {s['harm_damaged']:5.2f}"
                  f"   pooled {s['harm_pooled']:5.2f}")

    if args.save and r.ok(best):
        path = HERE / "rules.json"
        saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() \
            else {"rules": {}, "tuned_on": {}}
        key = trace["system"] + ("#tier2" if args.tier == 2 else "")
        saved["rules"][key] = rule._asdict()
        saved["tuned_on"][key] = {
            "dev": trace["dev"], "clean": trace["clean"], "target": TARGET,
            "replayed": best}
        path.write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
        print(f"\n  saved to {path.relative_to(bench.ROOT)}")
    return 0


def oracle(passage: Dict[str, Any]) -> str:
    """The best text any rule could produce from what was proposed.

    Each decision is made by someone who can see the truth: take the
    candidate that IS the true text at that place, if one was offered, and
    otherwise leave the page alone. What this cannot fix, no scorer can --
    the right word was never on the table -- so its recall is the proposer's
    ceiling, and the distance from a real system's recall up to it is what a
    better scorer could still win.
    """
    truth, noisy = passage["truth"], passage["noisy"]
    words = list(metrics.WORD.finditer(noisy))
    want: Dict[Tuple[int, int], List[str]] = {}
    for st in metrics.project(truth, noisy):
        want.setdefault(st.span, []).append(st.word)
    entries = {e["k"]: e for e in passage["entries"]}
    out: List[str] = []
    k = 0
    while k < len(words):
        e = entries.get(k)
        pick, used = words[k].group(), 1
        if e is not None:
            for text, _, n in e["pool"][1:]:
                if k + n > len(words):
                    continue
                span = (words[k].start(), words[k + n - 1].end())
                if " ".join(want.get(span, [])) == text:
                    pick, used = text, n
                    break
        out.append(pick)
        k += used
    return " ".join(out)


def cmd_ceiling(args) -> int:
    lex = set(bench.lexicon(args.corpus))
    path = trace_path(args.corpus, args.system, args.tier)
    trace = json.loads(path.read_text(encoding="utf-8"))
    total: Counter = Counter()
    for p in trace["passages"]:
        if p["level"] not in (0, "0"):
            total.update(metrics.tally(p["truth"], p["noisy"], oracle(p), lex))
    print(f"{trace['system']}, tier {trace.get('tier', 1)}: with every decision made "
          f"by someone who can see the truth,\n  recall {metrics.recall(total):.1%}"
          f"  (real-word {metrics.recall(total, 'real:'):.1%}, "
          f"non-word {metrics.recall(total, 'nonword:'):.1%}) "
          f"of {metrics.damaged(total):,} damaged words\n"
          f"  The rest were never offered the right word.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trace")
    t.add_argument("system")
    t.add_argument("--corpus", default="corpus-b")
    t.add_argument("--limit", type=int)
    t.add_argument("--tier", type=int, choices=(1, 2), default=1)
    s = sub.add_parser("sweep")
    s.add_argument("system", help="a system's name as its trace records it, "
                                  "e.g. unigram or lii:latin-lii-latinitas")
    s.add_argument("--corpus", default="corpus-b")
    s.add_argument("--tier", type=int, choices=(1, 2), default=1)
    s.add_argument("--frontier", action="store_true")
    s.add_argument("--save", action="store_true")
    c = sub.add_parser("ceiling", help="what the proposer could reach with a perfect scorer")
    c.add_argument("system")
    c.add_argument("--corpus", default="corpus-b")
    c.add_argument("--tier", type=int, choices=(1, 2), default=1)
    args = ap.parse_args()
    return {"trace": cmd_trace, "sweep": cmd_sweep, "ceiling": cmd_ceiling}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

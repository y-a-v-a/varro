#!/usr/bin/env python3
"""Mine conversational turn-pairs from the corpus, for instruction tuning.

There is no Latin instruction dataset and translating an English one would break
the purity thesis in spirit — the model would see no English, but its
conversational behaviour would be English behaviour in Latin dress.

The corpus supplies its own. Latin has an unusually rich dialogue literature and
roughly a fifth of this corpus is conversational in form. Donatus' Ars minor is
a catechism from beginning to end: a second-century Latin teacher's instruction
dataset, written for this exact purpose fourteen centuries early.

TWO EXTRACTION PATHS

  speaker turns — drama only, and the highest quality signal available, because
    <speaker> markup survived extraction. A turn is a named speaker followed by
    an utterance, and the pair is (previous turn, this turn):

        Staphyla   Nam cur me miseram uerberas?
        Euclio     Ut misera sis atque ut te dignam mala malam aetatem exigas.

  question -> answer — everywhere else. A sentence ending in `?` followed by the
    sentence that answers it. This is what makes catechism usable.

WHAT THIS IS NOT
    SFT changes behaviour, not knowledge. Pairs mined here teach a model to take
    a turn in Latin, in register. They cannot teach it anything true, and a
    fine-tuned model that sounds confident is still an interlocutor rather than
    an assistant.

Usage:
    python src/dialogue.py --report        # what would be mined
    python src/dialogue.py                 # write data/processed/dialogue.jsonl
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
INTERIM = ROOT / "data" / "interim"

MIN_WORDS = 3
MAX_WORDS = 60

# A sentence must START at a boundary, or the match is a fragment lifted from
# the middle of a longer clause. The first measurement of this corpus without
# the anchor returned passages beginning mid-word.
SENTENCE = re.compile(r'(?:^|(?<=[.!?;:]))\s*([^.!?]{6,400}[.!?])')

# A speaker name stands ALONE on its line — one to three capitalised words with
# no terminal punctuation. Matching "capitalised word followed by text" instead
# returns every verse line that happens to start with a capital, which produced
# a first pass full of speakers named "Satiust", "Sum" and "Neque".
SPEAKER_LINE = re.compile(r'^([A-Z][a-zA-Z]{1,18}(?:\s+[A-Z][a-zA-Z]{1,18}){0,2})$')

# Real characters recur; a stray capitalised line does not. Requiring several
# appearances is what separates the two without needing a cast list.
MIN_SPEAKER_APPEARANCES = 3


def words(text: str) -> int:
    return len(text.split())


def usable(text: str) -> bool:
    n = words(text)
    if not (MIN_WORDS <= n <= MAX_WORDS):
        return False
    letters = sum(c.isalpha() for c in text)
    return letters / max(1, len(text)) > 0.6


def speaker_turns(record: Dict[str, Any]) -> Iterator[Tuple[str, str, str, str]]:
    """(speaker_a, utterance_a, speaker_b, utterance_b) from dramatic dialogue.

    Two passes. The first finds which standalone capitalised lines recur often
    enough to be characters rather than accidents of verse capitalisation; the
    second segments the text on those, accumulating everything between one
    speaker line and the next as that speaker's turn.
    """
    lines = [l.strip() for l in record["text"].split("\n") if l.strip()]

    counts: Counter = Counter()
    for line in lines:
        match = SPEAKER_LINE.match(line)
        if match:
            counts[match.group(1)] += 1
    cast = {name for name, n in counts.items() if n >= MIN_SPEAKER_APPEARANCES}
    if len(cast) < 2:
        return

    turns: List[Tuple[str, List[str]]] = []
    for line in lines:
        if line in cast:
            turns.append((line, []))
        elif turns:
            turns[-1][1].append(line)

    joined = [(name, " ".join(body).strip()) for name, body in turns if body]
    for (sa, ua), (sb, ub) in zip(joined, joined[1:]):
        if sa == sb:
            continue
        if usable(ua) and usable(ub):
            yield sa, ua, sb, ub


def question_answer(record: Dict[str, Any]) -> Iterator[Tuple[str, str]]:
    """(question, answer) where a sentence ends in `?` and the next replies."""
    text = " ".join(record["text"].split())
    sentences = [m.group(1).strip() for m in SENTENCE.finditer(text)]
    for question, answer in zip(sentences, sentences[1:]):
        if not question.endswith("?") or answer.endswith("?"):
            continue
        if usable(question) and usable(answer):
            yield question, answer


def mine(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Counter]:
    pairs: List[Dict[str, Any]] = []
    stats: Counter = Counter()
    seen: set = set()

    for rec in records:
        base = {k: rec.get(k) for k in
                ("author", "work", "genre", "period", "source", "license")}

        if rec.get("genre") == "DRAMA":
            for sa, ua, sb, ub in speaker_turns(rec):
                key = (ua[:60], ub[:60])
                if key in seen:
                    continue
                seen.add(key)
                pairs.append({**base, "kind": "turn", "speaker_prompt": sa,
                              "prompt": ua, "speaker_response": sb, "response": ub})
                stats["turn"] += 1

        for question, answer in question_answer(rec):
            key = (question[:60], answer[:60])
            if key in seen:
                continue
            seen.add(key)
            pairs.append({**base, "kind": "qa", "speaker_prompt": None,
                          "prompt": question, "speaker_response": None,
                          "response": answer})
            stats["qa"] += 1
            stats[f"qa:{rec.get('genre')}"] += 1

    return pairs, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="analyse only, write nothing")
    ap.add_argument("--split", default="train",
                    help="which corpus split to mine (default train — never "
                         "mine validation, it would leak into SFT)")
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--samples", type=int, default=6)
    args = ap.parse_args()

    path = PROCESSED / f"{args.corpus}.{args.split}.jsonl"
    if not path.exists():
        sys.exit(f"missing {path.relative_to(ROOT)} — run src/build.py first")
    with path.open(encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]

    pairs, stats = mine(records)
    print(f"mined {len(pairs):,} pairs from {len(records):,} documents "
          f"({args.corpus}.{args.split})\n")
    print(f"  speaker turns (drama)   {stats['turn']:>7,}")
    print(f"  question -> answer      {stats['qa']:>7,}")
    print("\n  Q->A by genre:")
    for key, count in stats.most_common():
        if key.startswith("qa:"):
            print(f"    {key[3:]:<16} {count:>7,}")

    print("\nsamples:")
    shown = 0
    for kind in ("turn", "qa"):
        for pair in pairs:
            if pair["kind"] != kind or shown >= args.samples:
                continue
            tag = (f"{pair['speaker_prompt']} -> {pair['speaker_response']}"
                   if kind == "turn" else f"{pair['genre']}")
            print(f"  [{kind} · {pair['author']} · {tag}]")
            print(f"     > {pair['prompt'][:88]}")
            print(f"     < {pair['response'][:88]}")
            shown += 1
            if shown % 3 == 0:
                break

    if args.report:
        print("\n--report: nothing written")
        return 0

    out = PROCESSED / "dialogue.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)} — {len(pairs):,} pairs, "
          f"{out.stat().st_size/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

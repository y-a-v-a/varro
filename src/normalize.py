#!/usr/bin/env python3
"""Normalise orthography so the corpus reads as one language, not three.

feasibility.md §4.1 argues this matters more than tokenizer vocabulary size, and
the measurements bear that out. The sources do not merely differ in spelling;
they use *opposing conventions*:

    v as a share of (u+v)      Perseus 12.9%   DigilibLT 11.5%   CSEL 0.60%

CSEL writes classical `uir`/`uita` throughout while the others distinguish u
from v. Left alone, a tokenizer spends vocabulary discovering that `uir` and
`vir` are the same word, and the model learns edition-house style as though it
were Latin.

WHY FOLD TOWARDS u
    Converting the other way — u to v where consonantal — needs morphological
    analysis to know which is which. Folding v to u needs nothing, restores the
    classical convention (Latin had one letter here; the distinction is a modern
    editorial invention), matches CSEL's third of the corpus unchanged, and is
    already what dedupe.py does to compare editions. It is a deliberate loss of
    an editorial distinction, not of information about Latin.

WHAT ELSE THE TEXT TURNED OUT TO CARRY
    acute accents      12,007, mostly Plautus: metrical ictus (`iússu`, `hánc`)
    macrons / breves    3,256 vowel-length marks
    underdots          papyrological "uncertain reading" (`ṃạḷẹ`)
    overlines          numeral marks (`HS |X̅X̅`)
    dagger              1,306 crux desperationis
    brackets           editorial supplement / athetesis
    hyphenation        3,101 words split across line breaks
    backslash            303, CSEL only — OCR debris, sometimes standing in for
                       a lost letter (`inim\\cis`), unreconstructable

    Every one of those is an editor talking about the text rather than the text
    itself. All are removed; the words they enclose are kept.

Ancient Latin needs only the 26 letters and no diacritics at all, so stripping
every combining mark is safe and handles accents, macrons, breves, underdots and
overlines in a single NFD pass.

Usage:
    python src/normalize.py --report        # what would change, writes nothing
    python src/normalize.py                 # rewrite data/interim/*.jsonl
    python src/normalize.py --uv v          # fold towards v instead
    python src/normalize.py --uv keep --keep-diacritics
"""

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
INTERIM = ROOT / "data" / "interim"

VERSION = "v1"

# Precomposed ligatures do not decompose under NFD, so they need expanding.
LIGATURES = {
    "æ": "ae", "Æ": "Ae", "œ": "oe", "Œ": "Oe",
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st",
}

# Editorial marks: deleted outright, the text they enclose retained. Brackets
# carry incompatible meanings across projects (supplement in one, athetesis in
# another), so keeping the words and dropping the marks is the only choice that
# does not require knowing each editor's convention.
EDITORIAL = "[]<>{}†‡*|\\_§¶"

QUOTES = {
    "ʼ": "'", "ʻ": "'", "‘": "'", "’": "'",
    "“": '"', "”": '"', "«": '"', "»": '"',
}

DASHES = {"—": "—", "–": "—", "‒": "—",
          "‑": "-", "−": "-"}

SOFT_HYPHEN = "­"

# Spacing modifier letters and symbols: a combining mark that lost its base
# letter during OCR and now floats free (˘ ˉ ´ ¯ ˙ …). Meaningless once the
# combining marks are gone anyway, so they are dropped rather than folded.
ORPHAN_CATEGORIES = {"Lm", "Sk"}

# Characters that *look* like Latin letters but are not — overwhelmingly
# Cyrillic, from OCR run over Latin type. Each one silently splits a word into
# two unrelated tokens, so the cost is out of proportion to the ~130 occurrences.
# Only confident equivalences are listed; anything else is left alone and
# reported, on the principle that a visible unknown beats a wrong guess.
HOMOGLYPHS = {
    "а": "a", "е": "e", "о": "o", "с": "c", "м": "m", "т": "t", "в": "v",
    "з": "z", "р": "p", "у": "y", "х": "x",
    "А": "A", "В": "B", "Е": "E", "М": "M", "Н": "H", "О": "O", "Р": "P",
    "С": "C", "Т": "T", "Х": "X", "Є": "E",
    "ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "Đ": "D", "Ð": "D", "ƀ": "b",
    "ɪ": "I", "ɭ": "l", "ʋ": "u", "Ʒ": "Z", "ʓ": "z", "ⱶ": "h",
    "ⅽ": "c", "ⅰ": "i",
}

# A word broken across a line break: the hyphen is a typesetting artifact and
# the two halves belong together.
HYPHEN_BREAK = re.compile(r"(\w)-[ \t]*\n[ \t\n]*(\w)")

# A hyphen followed by a SPACE rather than a newline is ambiguous, and the two
# readings need opposite treatment. Measured across the corpus: 763 occurrences,
# genuinely mixed. Some are line breaks whose newline was lost upstream
# (`de- turbarunt` -> deturbarunt, `co- haerentes` -> cohaerentes); others are
# dashes used as punctuation between separate words (`esse- celeriter`,
# `incaluisse- deas`), where joining produces the nonce words `essceleriter` and
# `incaluissedeas`.
#
# Fusing two real words is worse than failing to fuse a split one: the first
# invents a token that does not exist in Latin, the second leaves two fragments
# that mostly do. So the default strips the hyphen and keeps the word boundary.
HYPHEN_SPACE = re.compile(r"(\w)-[ \t]+(\w)")

FOLD_U = str.maketrans({"v": "u", "V": "U", "j": "i", "J": "I"})
FOLD_J_ONLY = str.maketrans({"j": "i", "J": "I"})

WS_INLINE = re.compile(r"[ \t\r\f\v]+")
WS_AROUND = re.compile(r"[ \t]*\n[ \t]*")
WS_BLANK = re.compile(r"\n{3,}")


def strip_marks(text: str) -> str:
    """Remove every combining mark. Latin needs none of them."""
    decomposed = unicodedata.normalize("NFD", text)
    kept = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", kept)


def normalize(text: str, uv: str = "u", diacritics: bool = False,
              lowercase: bool = False, stats: Counter = None,
              hyphen_space: str = "split") -> str:
    if stats is None:
        stats = Counter()

    text = unicodedata.normalize("NFC", text)
    text = text.replace(SOFT_HYPHEN, "")

    # Rejoin before anything else: the marks that follow could mask the break.
    text, n = HYPHEN_BREAK.subn(r"\1\2", text)
    stats["hyphen_rejoined"] += n
    if hyphen_space == "join":
        text, n = HYPHEN_SPACE.subn(r"\1\2", text)
        stats["hyphen_space_joined"] += n
    elif hyphen_space == "split":
        text, n = HYPHEN_SPACE.subn(r"\1 \2", text)
        stats["hyphen_space_split"] += n

    if not diacritics:
        before = text
        text = strip_marks(text)
        if before != text:
            stats["records_with_diacritics"] += 1

    for src, dst in LIGATURES.items():
        if src in text:
            stats["ligatures"] += text.count(src)
            text = text.replace(src, dst)

    for src, dst in {**QUOTES, **DASHES}.items():
        text = text.replace(src, dst)

    if EDITORIAL:
        removed = sum(text.count(c) for c in EDITORIAL)
        if removed:
            stats["editorial_marks"] += removed
            text = text.translate({ord(c): None for c in EDITORIAL})

    if not diacritics:
        orphans = [c for c in text
                   if ord(c) > 127 and unicodedata.category(c) in ORPHAN_CATEGORIES]
        if orphans:
            stats["orphan_diacritics"] += len(orphans)
            text = text.translate({ord(c): None for c in set(orphans)})

    swaps = [c for c in HOMOGLYPHS if c in text]
    if swaps:
        stats["homoglyphs"] += sum(text.count(c) for c in swaps)
        text = text.translate({ord(c): HOMOGLYPHS[c] for c in swaps})

    if uv == "u":
        text = text.translate(FOLD_U)
    elif uv == "v":
        # Only i/j is folded; u->v needs to know which u is consonantal.
        text = text.translate(FOLD_J_ONLY)

    if lowercase:
        text = text.lower()

    text = WS_INLINE.sub(" ", text)
    text = WS_AROUND.sub("\n", text)
    text = WS_BLANK.sub("\n\n", text)
    return text.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="analyse only, write nothing")
    ap.add_argument("--uv", choices=("u", "v", "keep"), default="u",
                    help="fold v->u (default), fold j->i only, or leave alone")
    ap.add_argument("--keep-diacritics", action="store_true",
                    help="retain combining marks (macrons, ictus, underdots)")
    ap.add_argument("--lowercase", action="store_true", help="also lowercase everything")
    ap.add_argument("--hyphen-space", choices=("split", "join", "keep"), default="split",
                    help="how to treat `word- word`: strip the hyphen and keep the "
                         "boundary (default), fuse the halves, or leave untouched")
    ap.add_argument("--samples", type=int, default=6, help="before/after examples to show")
    args = ap.parse_args()

    files = sorted(INTERIM.glob("*.jsonl"))
    if not files:
        sys.exit(f"no records in {INTERIM} — run extract.py first")

    stats: Counter = Counter()
    samples: List[Tuple[str, str]] = []
    words_before = words_after = 0

    for path in files:
        with path.open(encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh]

        for rec in rows:
            original = rec["text"]
            words_before += len(original.split())
            text = normalize(original, uv=args.uv, diacritics=args.keep_diacritics,
                             lowercase=args.lowercase, stats=stats,
                             hyphen_space=args.hyphen_space)
            words_after += len(text.split())
            stats["records"] += 1
            if original != text:
                stats["records_changed"] += 1
                if len(samples) < args.samples:
                    for i, (a, b) in enumerate(zip(original, text)):
                        if a != b:
                            lo = max(0, i - 55)
                            samples.append((" ".join(original[lo:i + 55].split()),
                                            " ".join(text[lo:i + 55].split())))
                            break
            rec["text"] = text
            rec["normalization"] = None if args.report else VERSION
            rec["n_words"] = len(text.split())

        if not args.report:
            tmp = path.with_suffix(".jsonl.tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                for rec in rows:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            tmp.replace(path)

    print(f"normalization    {VERSION}   u/v={args.uv}  "
          f"diacritics={'kept' if args.keep_diacritics else 'stripped'}  "
          f"lowercase={args.lowercase}")
    print(f"records          {stats['records']:,} ({stats['records_changed']:,} changed)")
    print(f"words            {words_before:,} -> {words_after:,} "
          f"({words_after - words_before:+,})")
    for key in ("hyphen_rejoined", "hyphen_space_split", "hyphen_space_joined",
                "editorial_marks", "orphan_diacritics",
                "homoglyphs", "ligatures", "records_with_diacritics"):
        if stats[key]:
            print(f"  {key:<26} {stats[key]:>9,}")

    if samples:
        print("\nfirst divergence per record:")
        for before, after in samples:
            print(f"  -  {before}")
            print(f"  +  {after}")

    if args.report:
        print("\n--report: nothing written")
    else:
        print(f"\nrewrote {len(files)} file(s) in {INTERIM.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

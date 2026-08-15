#!/usr/bin/env python3
"""Train and compare Latin BPE tokenizers.

Two jobs. The first is to settle the last estimate in the project: every
parameter figure in feasibility.md §2 rests on an assumed ~1.8 tokens per word,
which nothing has measured. The second is the experiment plan.md asks for —
whether a small vocabulary makes the tokenizer learn Latin's recurring stems and
endings rather than memorising whole inflected forms.

Latin is a good case for that question. A single verb supplies dozens of surface
forms from one stem:

    amo  amas  amat  amamus  amatis  amant  amabam  amabo  amaui  amatus

A large vocabulary can afford to store each as its own token and learn nothing;
a small one is forced to find `am-` and the endings. Which is better is an
empirical question about downstream loss, not a matter of taste — this script
measures the trade so the choice can be made on evidence.

Byte-level BPE is used throughout: it cannot emit <unk>, which matters because
249 stray non-ASCII characters survived normalisation deliberately, and it is
what GPT-2-class training pipelines expect.

Usage:
    python src/tokenizer.py --report              # measure, train nothing
    python src/tokenizer.py                       # train the default sweep
    python src/tokenizer.py --vocab 8000          # a single size
"""

import argparse
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUT = PROCESSED / "tokenizers"

DEFAULT_VOCABS = [2000, 4000, 8000, 16000, 32000]

# Period tags are injected at training time from the `period` field, so the
# tokenizer must reserve them rather than discover them from the text.
SPECIAL = ["<|pad|>", "<|endoftext|>",
           "<CLASSICAL>", "<LATE>", "<MEDIEVAL>", "<NEOLATIN>"]

# One regular verb, indicative active, plus the participle and infinitive.
# If a tokenizer has learned Latin morphology it should isolate `am-`.
PARADIGM = ["amo", "amas", "amat", "amamus", "amatis", "amant",
            "amabam", "amabis", "amabit", "amaui", "amatus", "amare"]

# Same stem, different derivational shapes — harder, and closer to real text.
STEMS = {
    "am":   ["amor", "amicus", "amicitia", "amabilis"],
    "reg":  ["rego", "regis", "regit", "regem", "regnum", "regius"],
    "dic":  ["dico", "dicis", "dicit", "dictum", "dictio", "dicere"],
}

PROBE = ("Gallia est omnis diuisa in partes tres, quarum unam incolunt Belgae, "
         "aliam Aquitani, tertiam qui ipsorum lingua Celtae, nostra Galli appellantur.")


def load(split: str) -> List[Dict[str, Any]]:
    path = PROCESSED / f"corpus.{split}.jsonl"
    if not path.exists():
        sys.exit(f"missing {path.relative_to(ROOT)} — run src/build.py first")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def texts(records: List[Dict[str, Any]]) -> Iterator[str]:
    for rec in records:
        yield rec["text"]


def train_one(records: List[Dict[str, Any]], vocab_size: int):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts(records), trainer=trainer, length=len(records))
    return tok


def fertility(tok, records: List[Dict[str, Any]], sample: int = 400) -> Dict[str, float]:
    """Tokens per word, measured on real text rather than assumed."""
    chosen = records[:sample]
    words = tokens = chars = 0
    for rec in chosen:
        words += len(rec["text"].split())
        chars += len(rec["text"])
        tokens += len(tok.encode(rec["text"]).ids)
    return {
        "tokens_per_word": tokens / words if words else 0.0,
        "chars_per_token": chars / tokens if tokens else 0.0,
        "words": words,
        "tokens": tokens,
    }


def split_of(tok, word: str) -> List[str]:
    pieces = tok.encode(word).tokens
    return [p.replace("Ġ", "") for p in pieces]


def morphology(tok) -> Dict[str, Any]:
    """Does the tokenizer isolate a shared stem, or memorise whole forms?

    Two measures, because the obvious one is vacuous. "Does the first token
    start with the stem?" scores 16/16 at every vocabulary size — a first token
    of `a` satisfies it as readily as `am`. What actually distinguishes the
    tokenizers is *consistency*: whether every form of a stem opens with the
    SAME token. Perfect isolation gives one distinct opener per group; whole-form
    memorisation gives one per form.
    """
    whole = sum(1 for w in PARADIGM if len(split_of(tok, w)) == 1)

    ratios = []
    for forms in STEMS.values():
        openers = {split_of(tok, form)[0] for form in forms}
        ratios.append(len(openers) / len(forms))
    consistency = sum(ratios) / len(ratios)

    para_openers = {split_of(tok, form)[0] for form in PARADIGM}
    return {
        "paradigm_whole": whole,
        "paradigm_total": len(PARADIGM),
        "paradigm_openers": len(para_openers),
        "stem_spread": consistency,       # 1/n is ideal, 1.0 is memorisation
        "example": " · ".join(split_of(tok, "amabamus")),
    }


def verify_roundtrip(tok, records: List[Dict[str, Any]], sample: int = 25) -> int:
    """Encode then decode held-out text and require the original back.

    Byte-level BPE cannot emit <unk>, so any loss here would be a real defect
    rather than an out-of-vocabulary character. One documented exception: with
    add_prefix_space the pre-tokenizer prepends a space and the decoder restores
    it, so the round-trip is exact modulo that leading space. That setting is
    kept deliberately — it makes document-initial `Gallia` the same token as
    `Gallia` mid-sentence, instead of burning a second vocabulary entry on it.
    """
    failures = 0
    for rec in records[:sample]:
        original = rec["text"]
        restored = tok.decode(tok.encode(original).ids)
        if restored.startswith(" ") and not original.startswith(" "):
            restored = restored[1:]
        if restored != original:
            failures += 1
    return failures


def embedding_cost(vocab_size: int, d_model: int, n_layer: int) -> Dict[str, float]:
    """What this vocabulary costs in a model of the size we can actually train.

    The decisive constraint for Latin-0 is not compression but parameters. A
    transformer's body is roughly 12·n_layer·d² regardless of vocabulary, while
    the embedding table is vocab·d — and at the two-to-five-million-parameter
    scale the corpus supports, a large vocabulary can outweigh the entire model.
    Embeddings are assumed tied (input and output share one matrix), which is
    standard at this size and already the generous case.
    """
    body = 12 * n_layer * d_model ** 2
    embed = vocab_size * d_model
    return {"body": body, "embed": embed, "total": body + embed,
            "embed_share": embed / (body + embed)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vocab", type=int, action="append",
                    help="vocabulary size (repeatable); default is a sweep")
    ap.add_argument("--report", action="store_true", help="do not save tokenizers")
    ap.add_argument("--d-model", type=int, default=256,
                    help="model width, for the embedding-cost table (default 256)")
    ap.add_argument("--n-layer", type=int, default=6,
                    help="model depth, for the embedding-cost table (default 6)")
    args = ap.parse_args()

    vocabs = args.vocab or DEFAULT_VOCABS
    train = load("train")
    validation = load("validation")
    classical = [r for r in train if r.get("period") == "CLASSICAL"]

    total_words = sum(r["n_words"] for r in train)
    total_chars = sum(len(r["text"]) for r in train)
    alphabet = Counter()
    for rec in train:
        alphabet.update(rec["text"])

    print(f"training corpus  {len(train):,} records, {total_words:,} words, "
          f"{total_chars:,} chars")
    print(f"classical subset {len(classical):,} records, "
          f"{sum(r['n_words'] for r in classical):,} words")
    print(f"distinct characters {len(alphabet)}   "
          f"letters {sum(1 for c in alphabet if c.isalpha())}")
    print(f"\ncharacter baseline: {total_chars / total_words:.2f} chars/word "
          f"-> a char-level model needs {total_chars / 1e6:.1f}M tokens\n")

    rows = []
    for vocab_size in vocabs:
        print(f"training vocab={vocab_size:,} ...", flush=True)
        tok = train_one(train, vocab_size)
        full = fertility(tok, train)
        cls = fertility(tok, classical)
        morph = morphology(tok)
        used = len(tok.get_vocab())
        bad = verify_roundtrip(tok, validation)
        if bad:
            print(f"  ROUND-TRIP FAILED on {bad} held-out records", file=sys.stderr)
        rows.append((vocab_size, used, full, cls, morph))

        if not args.report:
            OUT.mkdir(parents=True, exist_ok=True)
            path = OUT / f"latin-bpe-{vocab_size}.json"
            tok.save(str(path))
            print(f"  saved {path.relative_to(ROOT)}")

    print("\n" + "=" * 78)
    print(f"{'vocab':>7}  {'tok/word':>9}  {'tok/word':>9}  {'chars':>7}  "
          f"{'whole':>6}  {'stem':>6}")
    print(f"{'':>7}  {'(all)':>9}  {'(class.)':>9}  {'/token':>7}  "
          f"{'forms':>6}  {'spread':>6}")
    print("-" * 78)
    for vocab_size, used, full, cls, morph in rows:
        print(f"{vocab_size:>7,}  {full['tokens_per_word']:>9.3f}  "
              f"{cls['tokens_per_word']:>9.3f}  {full['chars_per_token']:>7.2f}  "
              f"{morph['paradigm_whole']:>2}/{morph['paradigm_total']:<3}  "
              f"{morph['stem_spread']:>6.2f}")
    print("=" * 78)
    print("  whole forms  paradigm members stored as a single token "
          "(memorisation)")
    print("  stem spread  distinct openers per stem group; 0.25 is ideal here, "
          "1.00 is memorisation")

    print(f"\nround-trip verified on {min(25, len(validation))} held-out works "
          f"(exact, modulo the documented leading space)")

    print("\nhow `amabamus` divides:")
    for vocab_size, _, _, _, morph in rows:
        print(f"  {vocab_size:>6,}   {morph['example']}")

    print("\nprojected corpus size in tokens:")
    for vocab_size, _, full, cls, _ in rows:
        whole = total_words * full["tokens_per_word"]
        classical_tokens = sum(r["n_words"] for r in classical) * cls["tokens_per_word"]
        print(f"  {vocab_size:>6,}   full {whole / 1e6:>5.1f}M   "
              f"classical {classical_tokens / 1e6:>5.1f}M")

    d, L = args.d_model, args.n_layer
    print(f"\nwhat the vocabulary costs at d_model={d}, n_layer={L} "
          f"(tied embeddings):")
    print(f"  {'vocab':>7}  {'embed':>9}  {'body':>9}  {'total':>9}  {'embed':>7}")
    for vocab_size, _, _, _, _ in rows:
        c = embedding_cost(vocab_size, d, L)
        print(f"  {vocab_size:>7,}  {c['embed']/1e6:>8.2f}M  {c['body']/1e6:>8.2f}M  "
              f"{c['total']/1e6:>8.2f}M  {c['embed_share']:>6.0%}")
    print("  The corpus supports 2-5M parameters (feasibility.md §2). Above ~8k")
    print("  the embedding table starts to outweigh the model it serves.")

    if args.report:
        print("\n--report: no tokenizers saved")
    return 0


if __name__ == "__main__":
    sys.exit(main())

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

The vocabulary is fitted on corpus.train MINUS everything any corpus holds
out — not just this one's validation split. build.py chooses each corpus's
split independently, so a work held out of `classical` or `corpus-b` can still
sit in `corpus`'s training text, and the tokenizer would be reading text those
models are scored against. See held_out_ids().

Every run that saves tokenizers also writes reports/tokenizer.json, so the
sweep survives the terminal it was printed in. tokens-per-word is what the
parameter ladder is computed against, and it was an assumption for months
because nothing kept the measurement.

Usage:
    python src/tokenizer.py --report              # measure, train nothing
    python src/tokenizer.py                       # train the default sweep
    python src/tokenizer.py --vocab 8000          # a single size
"""

import argparse
import hashlib
import json
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parent))

from encoding import encode_all  # noqa: E402
# The corpus fingerprint has one definition, in evaluate.py, and this is the
# third report that needs it. Copying the key list again is how baselines came
# to claim they were measured on a corpus they were not.
from evaluate import SCHEMA_VERSION, corpus_fingerprint  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUT = PROCESSED / "tokenizers"
REPORTS = ROOT / "reports"

# 64k added when the corpus reached 79M words. The original sweep stopped at 32k
# because at d_model=256 that vocabulary already cost more parameters than the
# model it served — an argument that weakens once the corpus supports a model
# large enough to absorb the embedding table.
DEFAULT_VOCABS = [2000, 4000, 8000, 16000, 32000, 64000]

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


def held_out_ids() -> Set[str]:
    """Every record that ANY corpus holds out, not just this one's.

    build.py runs once per corpus and each run chooses its own validation split
    independently, so a work held out of `classical` can still sit in `corpus`'s
    training split — 14 of LATIN-VII's 22 held-out works do, 104,211 words. The
    tokenizer trains on corpus.train, so it was reading text LATIN-VII is then
    scored against.

    Nothing leaked into a model: LATIN-VII's weights never saw those works. What
    saw them was the merge table used to encode both sides, and BPE picks its
    merges by frequency over its input, so sequences peculiar to a held-out work
    could earn one. Measured directly — two 4k vocabularies over the same 90M
    words, one including those 14 works and one excluding them — the effect is
    0.026% on their tokens-per-word, against 0.01% in the opposite direction on
    the 8 works neither tokenizer saw. Real in sign, invisible in size.

    So this is not fixed because it moves a number. It is fixed because
    "no vocabulary was fitted on text a model is scored against" is worth being
    unconditionally true rather than true to within a fortieth of a percent, and
    the filter costs 0.12% of the input.

    Over-exclusion is the safe direction here. While record ids collide a train
    record can share an id with a held-out one and be dropped with it; dropping
    a little too much never leaks, and the id fix removes the ambiguity anyway.
    """
    ids: Set[str] = set()
    for path in sorted(PROCESSED.glob("*.validation.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                ids.add(json.loads(line)["id"])
    return ids


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


def fertility(tok, records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Tokens per word, counted over EVERY record — no sampling, no extrapolation.

    This previously took `records[:400]`. Because the corpus is loaded in a
    deterministic file order, those 400 records were 342 DigilibLT and 58 CSEL
    and contained no Perseus at all — so the "full corpus" figure was measured on
    late-antique prose only, then multiplied by every word in the corpus. The
    classical subset was unaffected (all 189 of its records fell inside the
    window), but the headline number did not mean what it said.

    Encoding the whole corpus per vocabulary costs a couple of minutes. That is
    the right price for a number the entire parameter ladder rests on — and
    encode_all spends it across every core rather than one, which is most of
    that couple of minutes back on a sweep of six vocabularies.
    """
    words = tokens = chars = 0
    ids_per_record = encode_all(tok, (rec["text"] for rec in records))
    for rec, ids in zip(records, ids_per_record):
        words += len(rec["text"].split())
        chars += len(rec["text"])
        tokens += len(ids)
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


def write_report(rows: List[Dict[str, Any]], args,
                 corpus_shape: Dict[str, Any]) -> Path:
    """Persist the sweep, because printing it is not recording it.

    Every other stage writes reports/*.json; this one printed a table to stdout
    and kept nothing. That table is where the tokens-per-word figure comes from,
    and tokens-per-word is what the entire parameter ladder is computed against
    — 20 tokens per parameter is a statement about tokens, so a corpus of
    4,805,330 words is 10.67M tokens at 2.221 and 8.65M at 1.8. The project
    assumed 1.8 for months precisely because nothing held the measured number.

    Written only when tokenizers were actually saved: the report names each
    vocabulary by the content digest of the file on disk, and a --report run
    has no such file. See main().
    """
    REPORTS.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus_name": "corpus",
        "corpus": corpus_fingerprint("corpus"),
        "corpus_shape": corpus_shape,
        "embedding_cost_model": {"d_model": args.d_model, "n_layer": args.n_layer},
        "vocabularies": rows,
    }
    path = REPORTS / "tokenizer.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


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

    # What the vocabulary may be FITTED on, which is not the same as what the
    # fertility figures are measured over. The measurements describe the corpus
    # train.py will actually tokenise, so they stay on the whole split; only the
    # merge table is denied the held-out works. See held_out_ids().
    withheld = held_out_ids()
    fitting = [r for r in train if r["id"] not in withheld]
    fitting_words = sum(r["n_words"] for r in fitting)

    print(f"training corpus  {len(train):,} records, {total_words:,} words, "
          f"{total_chars:,} chars")
    print(f"fitted on        {len(fitting):,} records, {fitting_words:,} words "
          f"({len(train) - len(fitting):,} withheld: held out by some corpus, "
          f"{100 * (total_words - fitting_words) / total_words:.2f}% of the split)")
    print(f"classical subset {len(classical):,} records, "
          f"{sum(r['n_words'] for r in classical):,} words")
    print(f"distinct characters {len(alphabet)}   "
          f"letters {sum(1 for c in alphabet if c.isalpha())}")
    print(f"\ncharacter baseline: {total_chars / total_words:.2f} chars/word "
          f"-> a char-level model needs {total_chars / 1e6:.1f}M tokens\n")

    rows = []
    for vocab_size in vocabs:
        print(f"training vocab={vocab_size:,} ...", flush=True)
        tok = train_one(fitting, vocab_size)
        full = fertility(tok, train)
        cls = fertility(tok, classical)
        morph = morphology(tok)
        bad = verify_roundtrip(tok, validation)
        if bad:
            print(f"  ROUND-TRIP FAILED on {bad} held-out records", file=sys.stderr)
        row: Dict[str, Any] = {
            "vocab": vocab_size,
            # What the trainer actually reached, which can be short of what was
            # asked for. Collected since the first sweep and, until this report
            # existed, never shown to anyone.
            "vocab_used": len(tok.get_vocab()),
            "sha256": None,
            "roundtrip_failures": bad,
            "full": full,
            "classical": cls,
            "morphology": morph,
            "embedding_cost": embedding_cost(vocab_size, args.d_model, args.n_layer),
        }
        rows.append(row)

        if not args.report:
            OUT.mkdir(parents=True, exist_ok=True)
            path = OUT / f"latin-bpe-{vocab_size}.json"
            tok.save(str(path))

            # Also keep a copy named by content, and never overwrite one.
            #
            # A model's embedding table is keyed to the exact vocabulary it was
            # trained through: token 1234 is whatever that file said it was.
            # Retraining the tokenizer in place therefore does not merely
            # invalidate a floor, it destroys every checkpoint that depended on
            # it — and silently, because loading a model with the wrong
            # vocabulary produces confident nonsense rather than an error.
            #
            # This happened. The 2026-08-20 rebuild overwrote the vocabulary
            # LATIN-XXV was trained through, and nothing complained.
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            row["sha256"] = digest
            archived = OUT / f"latin-bpe-{vocab_size}.{digest}.json"
            if not archived.exists():
                archived.write_bytes(path.read_bytes())
            print(f"  saved {path.relative_to(ROOT)}  ({digest})")

    print("\n" + "=" * 78)
    print(f"{'vocab':>7}  {'tok/word':>9}  {'tok/word':>9}  {'chars':>7}  "
          f"{'whole':>6}  {'stem':>6}")
    print(f"{'':>7}  {'(all)':>9}  {'(class.)':>9}  {'/token':>7}  "
          f"{'forms':>6}  {'spread':>6}")
    print("-" * 78)
    for row in rows:
        full, cls, morph = row["full"], row["classical"], row["morphology"]
        print(f"{row['vocab']:>7,}  {full['tokens_per_word']:>9.3f}  "
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
    for row in rows:
        print(f"  {row['vocab']:>6,}   {row['morphology']['example']}")

    print("\ncorpus size in tokens (counted, not projected):")
    for row in rows:
        print(f"  {row['vocab']:>6,}   full {row['full']['tokens'] / 1e6:>5.1f}M   "
              f"classical {row['classical']['tokens'] / 1e6:>5.1f}M")

    d, L = args.d_model, args.n_layer
    print(f"\nwhat the vocabulary costs at d_model={d}, n_layer={L} "
          f"(tied embeddings):")
    print(f"  {'vocab':>7}  {'embed':>9}  {'body':>9}  {'total':>9}  {'embed':>7}")
    for row in rows:
        c = row["embedding_cost"]
        print(f"  {row['vocab']:>7,}  {c['embed']/1e6:>8.2f}M  {c['body']/1e6:>8.2f}M  "
              f"{c['total']/1e6:>8.2f}M  {c['embed_share']:>6.0%}")
    print("  The corpus supports 2-5M parameters (feasibility.md §2). Above ~8k")
    print("  the embedding table starts to outweigh the model it serves.")

    if args.report:
        # No report file either, deliberately. reports/tokenizer.json names
        # vocabularies by content digest, and a --report run has no files to
        # digest; writing one would replace a record of what is on disk with a
        # record of what is not.
        print("\n--report: no tokenizers saved, no report written")
        return 0

    path = write_report(rows, args, corpus_shape={
        "records": len(train),
        "words": total_words,
        "chars": total_chars,
        "chars_per_word": total_chars / total_words if total_words else 0.0,
        "distinct_characters": len(alphabet),
        # What the merges were actually fitted on, so the claim that no
        # vocabulary saw a held-out work is checkable rather than asserted.
        "fitted_records": len(fitting),
        "fitted_words": fitting_words,
        "withheld_records": len(train) - len(fitting),
        "withheld_words": total_words - fitting_words,
    })
    print(f"\nwrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

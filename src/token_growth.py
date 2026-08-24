"""Training tokens in each corpus, measured through one shared vocabulary.

Counted, not projected, and all four corpora go through the SAME 4,000-token
vocabulary — the Track A one the models are trained with. A separate vocabulary
per corpus would make the token counts incomparable, which is the whole point
of measuring them side by side.
"""
import json
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path("/Users/vincentb/Projects/claudius")
P = ROOT / "data" / "processed"
tok = Tokenizer.from_file(str(P / "tokenizers" / "latin-bpe-4000.json"))

CORPORA = [("corpus", "Track A · full"), ("classical", "Track A · classical"),
           ("corpus-b", "Track B · full"), ("classical-b", "Track B · classical")]

rows = []
for name, label in CORPORA:
    summary = json.loads((P / f"{name}.summary.json").read_text())
    n = 0
    with (P / f"{name}.train.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            n += len(tok.encode(json.loads(line)["text"]).ids)
    rows.append({"name": name, "label": label, "train_tokens": n,
                 "words": summary["words"], "records": summary["records"],
                 "train_words": summary["train_words"],
                 "sources": summary["sources"]})
    print(f"  {label:<24} {n:>13,} train tokens", flush=True)

a, cls_a, b, cls_b = rows
print("\n" + "=" * 68)
print(f"{'corpus':<24}{'records':>9}{'words':>14}{'train tokens':>15}")
for r in rows:
    print(f"{r['label']:<24}{r['records']:>9,}{r['words']:>14,}{r['train_tokens']:>15,}")

print("\nTrack B minus Track A:")
print(f"  full        {b['train_tokens'] - a['train_tokens']:>+14,} tokens "
      f"({(b['train_tokens']/a['train_tokens']-1)*100:+.2f}%)")
print(f"  classical   {cls_b['train_tokens'] - cls_a['train_tokens']:>+14,} tokens "
      f"({(cls_b['train_tokens']/cls_a['train_tokens']-1)*100:+.2f}%)")

print("\nrecords by source, Track B:")
for s, n in sorted(b["sources"].items(), key=lambda kv: -kv[1]):
    print(f"  {s:<30} {n:>7,}")

# What 4 epochs of each supports at Chinchilla's ~20 tokens per parameter.
print(f"\n{'corpus':<24}{'4-epoch budget':>17}{'supports':>12}")
for r in rows:
    eff = r["train_tokens"] * 4
    print(f"{r['label']:<24}{eff/1e9:>16.3f}B{eff/20/1e6:>11.1f}M")

(ROOT / "data" / "processed" / "token-growth.json").write_text(
    json.dumps(rows, indent=2), encoding="utf-8")

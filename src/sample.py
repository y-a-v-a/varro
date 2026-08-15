#!/usr/bin/env python3
"""Generate Latin from a trained checkpoint.

Exists mainly to feed `evaluate.py --memorization`. A perplexity number cannot
distinguish a model that learned Latin from one that memorised its corpus; only
looking at what it writes, and checking whether that text already exists, can.

Architecture is read from the run's report rather than re-specified, so a sample
cannot silently be drawn from a differently-shaped model than the one trained.

Usage:
    python src/sample.py                              # unconditional samples
    python src/sample.py --prompt "Gallia est omnis"
    python src/sample.py --n 20 --out samples.txt     # for memorisation scoring
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
TOKENIZERS = PROCESSED / "tokenizers"
CHECKPOINTS = ROOT / "checkpoints"
REPORTS = ROOT / "reports"

sys.path.insert(0, str(ROOT / "src"))

try:
    import mlx.core as mx
except ImportError:
    sys.exit("MLX required — pip install mlx")

from train import GPT  # noqa: E402  (same architecture, read from the report)


def load_run(name: str) -> dict:
    path = REPORTS / f"train-{name}.json"
    if not path.exists():
        sys.exit(f"missing {path.relative_to(ROOT)} — train the model first")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def generate(model, tok, prompt: str, length: int, temperature: float,
             top_k: int, context: int, key) -> str:
    if prompt:
        ids = tok.encode(prompt).ids
    else:
        ids = [tok.token_to_id("<|endoftext|>")]

    for _ in range(length):
        window = ids[-context:]
        logits = model(mx.array([window]))[0, -1]
        logits = logits / max(temperature, 1e-6)
        if top_k and top_k < logits.size:
            kth = mx.sort(logits)[-top_k]
            logits = mx.where(logits < kth, mx.full(logits.shape, -1e9), logits)
        key, use = mx.random.split(key)
        nxt = mx.random.categorical(logits, key=use).item()
        ids.append(int(nxt))

    text = tok.decode(ids)
    return text.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="latin-0")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--length", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", help="write samples here, one per blank-line block")
    args = ap.parse_args()

    run = load_run(args.name)
    cfg = run["config"]

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZERS / f"latin-bpe-{cfg['vocab']}.json"))

    model = GPT(cfg["vocab"], cfg["dim"], cfg["layers"], cfg["heads"], cfg["context"])
    weights = CHECKPOINTS / f"{args.name}.safetensors"
    if not weights.exists():
        sys.exit(f"missing {weights.relative_to(ROOT)}")
    model.load_weights(str(weights))
    model.eval()
    mx.eval(model.parameters())

    print(f"{args.name}: {run['parameters']/1e6:.2f}M parameters, "
          f"final held-out perplexity {run['final']['val_perplexity']:.1f}\n")

    key = mx.random.key(args.seed)
    samples: List[str] = []
    for i in range(args.n):
        key, use = mx.random.split(key)
        text = generate(model, tok, args.prompt, args.length,
                        args.temperature, args.top_k, cfg["context"], use)
        samples.append(text)
        print(f"--- {i+1} " + "-" * 62)
        print(" ".join(text.split()))
        print()

    if args.out:
        Path(args.out).write_text("\n\n".join(samples) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

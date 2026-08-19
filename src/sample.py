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



def resolve_tokenizer(vocab: int, run: dict) -> str:
    """The exact vocabulary this run was trained through, or a clear refusal.

    A model's embeddings are keyed to specific token ids: with the wrong
    vocabulary, token 1234 means a different string and the model emits
    confident nonsense rather than failing. That is the worst failure mode
    available here, so it is checked rather than assumed.

    tokenizer.py keeps a content-named copy of every vocabulary it trains, so
    a model outlives a retrained tokenizer as long as its copy is on disk.
    """
    import hashlib

    want = (run.get("tokenizer") or {}).get("sha256")
    current = TOKENIZERS / f"latin-bpe-{vocab}.json"
    if want is None:
        # Reports written before tokenizers were fingerprinted cannot be
        # checked; say so rather than imply the check passed.
        print(f"  [no tokenizer fingerprint in this run's report — using "
              f"{current.name} unverified]", file=sys.stderr)
        return str(current)

    archived = TOKENIZERS / f"latin-bpe-{vocab}.{want}.json"
    if archived.exists():
        return str(archived)
    if current.exists():
        have = hashlib.sha256(current.read_bytes()).hexdigest()[:16]
        if have == want:
            return str(current)
        sys.exit(
            f"tokenizer mismatch: {run.get('run')} was trained through "
            f"vocabulary {want}, and {current.name} is {have}.\n"
            f"Loading it anyway would map every token to the wrong string and "
            f"produce fluent nonsense.\n"
            f"Recover it by rebuilding the corpus at the commit that trained "
            f"this model, or retrain the model on the current vocabulary.")
    sys.exit(f"no tokenizer for vocab {vocab} in {TOKENIZERS.relative_to(ROOT)}")


def load_run(name: str) -> dict:
    path = REPORTS / f"train-{name}.json"
    if not path.exists():
        sys.exit(f"missing {path.relative_to(ROOT)} — train the model first")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def generate(model, tok, prompt: str, length: int, temperature: float,
             top_k: int, context: int, key, period: Optional[str] = None) -> str:
    """Generate, optionally conditioned on a period tag.

    Models trained with period tags saw each document prefixed by its register,
    so starting a sample with <CLASSICAL> asks for Rome rather than the late
    antique prose that is three quarters of the mixed corpus. A model trained
    without tags ignores it — the token exists in the vocabulary but carries no
    learned meaning, which is why the tag is reported alongside every sample.
    """
    ids = [tok.token_to_id("<|endoftext|>")]
    if period:
        tag = tok.token_to_id(f"<{period}>")
        if tag is None:
            sys.exit(f"tokenizer has no <{period}> token")
        ids.append(tag)
    if prompt:
        ids.extend(tok.encode(prompt).ids)

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
    ap.add_argument("--period", choices=("CLASSICAL", "LATE", "MEDIEVAL", "NEOLATIN"),
                    help="condition on a period tag (models trained with tags only)")
    ap.add_argument("--out", help="write samples here, one per blank-line block")
    args = ap.parse_args()

    run = load_run(args.name)
    cfg = run["config"]

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(resolve_tokenizer(cfg["vocab"], run))

    model = GPT(cfg["vocab"], cfg["dim"], cfg["layers"], cfg["heads"], cfg["context"])
    weights = CHECKPOINTS / f"{args.name}.safetensors"
    if not weights.exists():
        sys.exit(f"missing {weights.relative_to(ROOT)}")
    model.load_weights(str(weights))
    model.eval()
    mx.eval(model.parameters())

    tagged = cfg.get("period_tags")
    note = ""
    if args.period and not tagged:
        note = f"  [WARNING: {args.name} was trained WITHOUT period tags; " \
               f"<{args.period}> carries no learned meaning here]"
    print(f"{args.name}: {run['parameters']/1e6:.2f}M parameters, corpus "
          f"{cfg.get('corpus')}, final held-out perplexity "
          f"{run['final']['val_perplexity']:.1f}"
          + (f", conditioned on <{args.period}>" if args.period else "") + note + "\n")

    key = mx.random.key(args.seed)
    samples: List[str] = []
    for i in range(args.n):
        key, use = mx.random.split(key)
        text = generate(model, tok, args.prompt, args.length,
                        args.temperature, args.top_k, cfg["context"], use,
                        period=args.period)
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

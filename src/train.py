#!/usr/bin/env python3
"""Train a GPT from random initialisation on Latin alone.

No pretrained weights, no multilingual initialisation, no English anywhere in
the pipeline that produced the data. The model's entire linguistic universe is
the corpus this repository builds.

WHY A SCRIPT RATHER THAN THE GUI
    plan.md nominates Language Model Builder as the laboratory, and says it need
    not be the final infrastructure. Everything else here is scripted, pinned and
    reproducible from a manifest; a training run that cannot be reproduced from
    the command line would be the one step in the pipeline nobody could re-derive.
    MLX directly keeps that property and lets evaluate.py plug straight in.

SIZING (feasibility.md §2)
    The classical corpus is 9.5M tokens at a 4k vocabulary. Repeated-data returns
    decay past ~4 epochs and are near zero past ~16, so the well-trained range is
    roughly 2-4M parameters. Latin-0 defaults to ~3.5M: d_model 192, 6 layers.
    Going wider mostly buys embedding table, which does no reasoning.

WHAT SUCCESS LOOKS LIKE, STATED IN ADVANCE
    evaluate.py --baselines puts an interpolated bigram at 195.2 perplexity on
    held-out text. Beating that is the minimum bar, not the goal. The real
    question is whether generations are Latin that does NOT appear in the corpus,
    which is why evaluate.py --memorization exists and why validation is held out
    by whole authors.

Usage:
    python src/train.py --dry-run                 # sizing and data, train nothing
    python src/train.py                           # Latin-0 on the classical corpus
    python src/train.py --corpus corpus --steps 8000
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
TOKENIZERS = PROCESSED / "tokenizers"
CHECKPOINTS = ROOT / "checkpoints"
REPORTS = PROCESSED / "eval"

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
except ImportError:
    sys.exit("MLX required — pip install mlx  (Apple Silicon only)")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiHeadAttention(dim, heads, bias=False)
        self.ln2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, 4 * dim, bias=False)
        self.fc2 = nn.Linear(4 * dim, dim, bias=False)

    def __call__(self, x, mask):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, mask)
        h = self.ln2(x)
        return x + self.fc2(nn.gelu(self.fc1(h)))


class GPT(nn.Module):
    """A small decoder-only transformer. Embeddings are tied to the output head.

    Tying matters more than usual at this scale: an untied 4k x 192 output matrix
    would add 768k parameters — a fifth of the model — to do a job the input
    embedding already does.
    """

    def __init__(self, vocab: int, dim: int, layers: int, heads: int, context: int):
        super().__init__()
        self.context = context
        self.embed = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(context, dim)
        self.blocks = [Block(dim, heads) for _ in range(layers)]
        self.ln_f = nn.LayerNorm(dim)

    def __call__(self, idx):
        _, T = idx.shape
        x = self.embed(idx) + self.pos(mx.arange(T))
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T).astype(x.dtype)
        for block in self.blocks:
            x = block(x, mask)
        return self.embed.as_linear(self.ln_f(x))


def parameter_count(model: nn.Module) -> int:
    return sum(v.size for _, v in tree_flatten(model.parameters()))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_tokens(corpus: str, split: str, tok, cache: bool = True) -> mx.array:
    """Tokenise a split once and keep it as a flat token stream."""
    path = PROCESSED / f"{corpus}.{split}.jsonl"
    if not path.exists():
        sys.exit(f"missing {path.relative_to(ROOT)} — run src/build.py")
    cached = PROCESSED / f"{corpus}.{split}.tokens.npy"
    if cache and cached.exists() and cached.stat().st_mtime > path.stat().st_mtime:
        return mx.array(mx.load(str(cached)))

    eos = tok.token_to_id("<|endoftext|>")
    stream: List[int] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            stream.extend(tok.encode(json.loads(line)["text"]).ids)
            stream.append(eos)
    arr = mx.array(stream, dtype=mx.uint16 if max(stream) < 65535 else mx.int32)
    if cache:
        mx.save(str(cached), arr)
    return arr


def batches(tokens: mx.array, batch: int, context: int, seed: int):
    """Random windows into the token stream, indefinitely."""
    limit = tokens.size - context - 1
    rng = mx.random.key(seed)
    while True:
        rng, use = mx.random.split(rng)
        starts = mx.random.randint(0, limit, (batch,), key=use)
        idx = starts[:, None] + mx.arange(context + 1)[None, :]
        chunk = tokens[idx].astype(mx.int32)
        yield chunk[:, :-1], chunk[:, 1:]


def loss_fn(model, x, y):
    logits = model(x).astype(mx.float32)
    return nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   y.reshape(-1), reduction="mean")


def evaluate_split(model, tokens: mx.array, batch: int, context: int,
                   iterations: int = 40, seed: int = 7) -> float:
    """Mean loss over fixed, evenly spaced windows — deterministic across runs."""
    limit = max(1, tokens.size - context - 1)
    stride = max(1, limit // max(1, iterations * batch))
    total, count = 0.0, 0
    position = 0
    for _ in range(iterations):
        starts = mx.array([(position + i * stride) % limit for i in range(batch)])
        idx = starts[:, None] + mx.arange(context + 1)[None, :]
        chunk = tokens[idx].astype(mx.int32)
        total += loss_fn(model, chunk[:, :-1], chunk[:, 1:]).item()
        count += 1
        position = (position + batch * stride) % limit
    return total / count


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="latin-0")
    ap.add_argument("--corpus", default="classical",
                    help="corpus basename built by src/build.py")
    ap.add_argument("--vocab", type=int, default=4000)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok_path = TOKENIZERS / f"latin-bpe-{args.vocab}.json"
    if not tok_path.exists():
        sys.exit(f"missing {tok_path.relative_to(ROOT)} — run src/tokenizer.py")
    tok = Tokenizer.from_file(str(tok_path))

    mx.random.seed(args.seed)
    model = GPT(args.vocab, args.dim, args.layers, args.heads, args.context)
    mx.eval(model.parameters())
    n_params = parameter_count(model)

    train_tokens = load_tokens(args.corpus, "train", tok)
    val_tokens = load_tokens(args.corpus, "validation", tok)

    tokens_per_step = args.batch * args.context
    total_tokens = args.steps * tokens_per_step
    epochs = total_tokens / train_tokens.size

    print(f"model        {args.name}  {n_params/1e6:.2f}M parameters "
          f"(dim {args.dim}, {args.layers} layers, {args.heads} heads, ctx {args.context})")
    print(f"corpus       {args.corpus}  {train_tokens.size:,} train tokens, "
          f"{val_tokens.size:,} validation")
    print(f"schedule     {args.steps:,} steps x {tokens_per_step:,} tokens "
          f"= {total_tokens/1e6:.1f}M  ({epochs:.1f} epochs)")
    print(f"tokens/param {total_tokens/n_params:.1f}   "
          f"(Chinchilla-optimal is ~20)")
    if epochs > 16:
        print("  WARNING: past ~16 epochs repeated data buys almost nothing "
              "(feasibility.md §2)")
    if args.dry_run:
        print("\n--dry-run: nothing trained")
        return 0

    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    stream = batches(train_tokens, args.batch, args.context, args.seed)

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        # Cosine decay from lr down to 0.1*lr.
        return 0.1 * args.lr + 0.9 * args.lr * 0.5 * (1 + math.cos(math.pi * progress))

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, Any]] = []
    started = time.time()
    running = 0.0

    for step in range(args.steps):
        optimizer.learning_rate = lr_at(step)
        x, y = next(stream)
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        running += loss.item()

        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            train_loss = running / args.eval_every
            running = 0.0
            val_loss = evaluate_split(model, val_tokens, args.batch, args.context)
            elapsed = time.time() - started
            rate = (step + 1) * tokens_per_step / elapsed
            print(f"  step {step+1:>6,}/{args.steps:,}  "
                  f"train {train_loss:.4f}  val {val_loss:.4f}  "
                  f"ppl {math.exp(val_loss):>7.1f}  "
                  f"{rate/1000:.0f}k tok/s  {elapsed/60:.1f}m", flush=True)
            history.append({"step": step + 1, "train_loss": train_loss,
                            "val_loss": val_loss,
                            "val_perplexity": math.exp(val_loss),
                            "lr": optimizer.learning_rate.item()
                            if hasattr(optimizer.learning_rate, "item")
                            else optimizer.learning_rate,
                            "elapsed_s": elapsed})

    weights = CHECKPOINTS / f"{args.name}.safetensors"
    model.save_weights(str(weights))
    final = history[-1] if history else {}
    print(f"\nsaved {weights.relative_to(ROOT)}")
    print(f"final held-out perplexity {final.get('val_perplexity', float('nan')):.1f} "
          f"(bigram floor 195.2)")

    REPORTS.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": args.name,
        "config": vars(args),
        "parameters": n_params,
        "train_tokens": int(train_tokens.size),
        "validation_tokens": int(val_tokens.size),
        "epochs": epochs,
        "history": history,
        "final": final,
    }
    path = REPORTS / f"train-{args.name}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Continue training a model so that it can fill a hole given BOTH sides of it.

WHAT THIS IS FOR, AND WHAT IT IS NOT
    A causal model already has an opinion about a gap given both sides: for a
    fixed left and right, P(fill, right | left) is proportional to
    P(fill | left, right), so scoring a candidate across the text after it --
    what demos/lacuna.py and ocr/repairers.py do -- ranks candidates exactly
    as the model's own infilling posterior would. Fill-in-the-middle training
    does not add that ability. What it adds is two things the inversion
    cannot give:

      a DIRECT estimate of P(fill | left, right), read off the fill's own
        tokens, instead of one recovered from the cost of a right context
        that is long, noisy, and truncated to what is affordable to score;
      GENERATION into a gap. A causal model can only propose from the left
        (11.7% in lacuna.py's benchmark). One trained this way proposes
        knowing what comes after.

    Whether the direct estimate is BETTER than the inverted one is an
    empirical question, and ocr/bench.py is how it gets answered.

THE FORMAT, AND WHY IT NEEDS ONLY ONE NEW SYMBOL
    A stretch of text  prefix | middle | suffix  is rewritten as

        <|pad|> suffix <|pad|> prefix middle <|endoftext|>

    Suffix first, so that prefix and middle stay contiguous and the middle is
    predicted as an ordinary continuation of the prefix -- the one thing the
    model is already good at -- with the suffix sitting in its context. The
    closing <|endoftext|> is part of the target: it is how the model says the
    middle is over and joins what follows, and a score that left it out would
    not penalise a fill that fails to connect.

    The vocabulary is 4,000 and every id is taken, but <|pad|> has never
    occurred in a training stream, so its embedding is free to mean "seam".
    One symbol serves for both seams because they are told apart by order.
    Nothing about the tokenizer or the model's shape changes, and every
    loader in the project opens the result like any other checkpoint.

CUTS FALL ON WORD BOUNDARIES
    A byte-level BPE puts the space inside the following token, so a token
    that begins with one begins a word. Middles start and end there, because
    that is what will be asked of the model: candidates are whole words,
    encoded on their own, and a model trained on middles that begin mid-word
    would be scored on a distribution it never saw.

HALF THE TEXT IS LEFT ALONE
    Bavarian et al. (2022, "Efficient Training of Language Models to Fill in
    the Middle") find left-to-right loss is unharmed with up to half the data
    transformed when training from scratch. This is NOT from scratch, and they
    also find that retrofitting costs more than it would have cost to include
    from the start -- so the run reports plain held-out loss next to the
    fill-in-the-middle loss, and a rise in the first is the price.

Usage:
    python src/fim.py --dry-run
    python src/fim.py --init-from latin-lii-latinitas --name latin-lii-fim --steps 12000
    python src/fim.py --name latin-lii-fim --resume
"""

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
CHECKPOINTS = ROOT / "checkpoints"

# A piece shorter than this has no room for three parts worth the name.
MIN_PIECE = 24

# Most middles are a word or a few, because that is the job; the rest are
# long, so the model does not learn that a gap is always short.
SHORT_SHARE = 0.7
SHORT_WORDS = (1, 2, 3)
LONG_TOKENS = 64

# Half the time the suffix is cut short. At inference it always is -- a
# repairer passes a line or two, not the rest of the book -- and a model that
# only ever saw suffixes running to the end of the window would be reading an
# unfamiliar shape.
TRUNCATE_SHARE = 0.5
SUFFIX_RANGE = (4, 64)


# ---------------------------------------------------------------------------
# The transformation -- pure, so it is tested without MLX
# ---------------------------------------------------------------------------

def word_starts(piece: Sequence[int], starts_word: Sequence[bool]) -> List[int]:
    return [i for i, t in enumerate(piece) if starts_word[t]]


def transform(piece: Sequence[int], starts_word: Sequence[bool], seam: int,
              eos: int, rng: random.Random) -> List[int]:
    """prefix|middle|suffix  ->  seam suffix seam prefix middle eos

    Returns the piece unchanged when it has too few word boundaries to cut.
    """
    ws = word_starts(piece, starts_word)
    if len(piece) < MIN_PIECE or len(ws) < 4:
        return list(piece)
    j = rng.randrange(1, len(ws) - 1)
    a = ws[j]
    if rng.random() < SHORT_SHARE:
        k = min(len(ws) - 1, j + rng.choice(SHORT_WORDS))
        b = ws[k] if k > j else len(piece)
    else:
        limit = a + rng.randint(4, LONG_TOKENS)
        later = [w for w in ws[j + 1:] if w <= limit]
        b = later[-1] if later else (ws[j + 1] if j + 1 < len(ws) else len(piece))
    prefix, middle, suffix = piece[:a], piece[a:b], piece[b:]
    if suffix and rng.random() < TRUNCATE_SHARE:
        suffix = suffix[:rng.randint(*SUFFIX_RANGE)]
    return [seam, *suffix, seam, *prefix, *middle, eos]


def transform_window(window: Sequence[int], starts_word: Sequence[bool],
                     seam: int, eos: int, rate: float,
                     rng: random.Random) -> List[int]:
    """Apply `transform` to each document piece of a window, at `rate`.

    Per piece, never across an end-of-text: a middle whose prefix is one
    author and whose suffix is another teaches nothing about either.
    """
    out: List[int] = []
    piece: List[int] = []
    for t in list(window) + [None]:
        if t is not None and t != eos:
            piece.append(t)
            continue
        if piece:
            if rng.random() < rate:
                done = transform(piece, starts_word, seam, eos, rng)
                out.extend(done)
                # transform closed the piece itself; a second end-of-text
                # after it would teach that fills end twice.
                closed = done[-1] == eos and len(done) != len(piece)
            else:
                out.extend(piece)
                closed = False
            piece = []
            if t == eos and not closed:
                out.append(eos)
        elif t == eos:
            out.append(eos)
    return out


def frame(prefix: Sequence[int], middle: Sequence[int], suffix: Sequence[int],
          seam: int, eos: int) -> List[int]:
    """The inference-time sequence for one candidate middle. Shared with the
    repairer, so training and scoring cannot drift apart on the format."""
    return [seam, *suffix, seam, *prefix, *middle, eos]


def starts_word_table(tok) -> List[bool]:
    vocab = tok.get_vocab()
    table = [False] * tok.get_vocab_size()
    for piece, idx in vocab.items():
        table[idx] = piece.startswith("Ġ")
    return table


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="latin-lii-fim")
    ap.add_argument("--init-from", dest="init_from", default="latin-lii-latinitas")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="peak; the base run peaked at 2.5e-4 and ended at 2.5e-5")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--rate", type=float, default=0.5,
                    help="share of document pieces rewritten")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--checkpoint-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
    except ImportError:
        sys.exit("MLX required — pip install mlx")
    from tokenizers import Tokenizer
    from encoding import resolve_tokenizer
    from train import (GPT, is_report_step, load_checkpoint, load_tokens,
                       loss_fn, parameter_count, save_checkpoint)

    base_path = REPORTS / f"train-{args.init_from}.json"
    if not base_path.exists():
        sys.exit(f"missing {base_path.relative_to(ROOT)}")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    cfg = base["config"]
    # The base run's own vocabulary, not whatever latin-bpe-4000.json is today.
    tok = Tokenizer.from_file(resolve_tokenizer(cfg["vocab"], base))
    seam, eos = tok.token_to_id("<|pad|>"), tok.token_to_id("<|endoftext|>")
    table = starts_word_table(tok)
    context = cfg["context"]

    mx.random.seed(args.seed)
    model = GPT(cfg["vocab"], cfg["dim"], cfg["layers"], cfg["heads"], context)
    train_tokens = load_tokens(cfg["corpus"], "train", tok,
                               period_tags=cfg.get("period_tags", True))
    val_tokens = load_tokens(cfg["corpus"], "validation", tok,
                             period_tags=cfg.get("period_tags", True))
    if int(mx.sum(train_tokens[:5_000_000] == seam).item()):
        sys.exit("<|pad|> occurs in the training stream; it cannot serve as the seam")

    def rows(tokens, n: int, rate: float, rng: random.Random):
        """n sequences of context+1 tokens, pieces rewritten at `rate`."""
        limit = tokens.size - 2 * context - 2
        out = []
        for _ in range(n):
            start = rng.randrange(limit)
            # Three short of the window, so one rewritten piece -- two seams
            # and a closing end-of-text -- still fits whole. Cutting a row to
            # length must never be what removes a middle: a long prefix with
            # its middle sliced off is a fill-in-the-middle example with no
            # fill in it.
            body = context + 1 - 3
            window = tokens[start:start + body].tolist()
            row = transform_window(window, table, seam, eos, rate, rng)
            if len(row) < context + 1:
                # A shortened suffix leaves room; fill it with what came next.
                more = tokens[start + body:start + body + context].tolist()
                row = row + more
            out.append(row[:context + 1])
        chunk = mx.array(out)
        return chunk[:, :-1], chunk[:, 1:]

    def held_out(rate: float) -> float:
        rng = random.Random(7)          # the same windows every time
        total = 0.0
        for _ in range(40):
            x, y = rows(val_tokens, args.batch, rate, rng)
            total += loss_fn(model, x, y).item()
        return total / 40

    tokens_per_step = args.batch * context
    print(f"model        {args.name} from {args.init_from}  "
          f"{parameter_count(model)/1e6:.2f}M parameters")
    print(f"schedule     {args.steps:,} steps x {tokens_per_step:,} tokens = "
          f"{args.steps * tokens_per_step / 1e6:.1f}M "
          f"({args.steps * tokens_per_step / train_tokens.size:.2f} epochs), "
          f"{args.rate:.0%} of pieces rewritten", flush=True)
    if args.dry_run:
        rng = random.Random(args.seed)
        x, _ = rows(val_tokens, 2, 1.0, rng)
        for row in x.tolist():
            print("\n" + tok.decode(row[:120], skip_special_tokens=False))
        return 0

    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)
    fingerprint = base.get("tokenizer")
    history: List[Dict[str, Any]] = []
    start_step = 0
    if args.resume:
        start_step, history = load_checkpoint(args.name, model, optimizer, fingerprint)
        print(f"resumed at step {start_step:,}")
    else:
        model.load_weights(str(CHECKPOINTS / f"{args.init_from}.safetensors"))
    mx.eval(model.parameters())

    if not args.resume:
        history.append({"step": 0, "val_loss": held_out(0.0),
                        "val_loss_fim": held_out(1.0), "elapsed_s": 0.0})
        print(f"  before       plain {history[0]['val_loss']:.4f}  "
              f"fim {history[0]['val_loss_fim']:.4f}", flush=True)

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.1 * args.lr + 0.9 * args.lr * 0.5 * (1 + math.cos(math.pi * progress))

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    rng = random.Random(args.seed + start_step)
    started, running, since = time.time(), 0.0, 0
    for step in range(start_step, args.steps):
        optimizer.learning_rate = lr_at(step)
        x, y = rows(train_tokens, args.batch, args.rate, rng)
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        running += loss.item()
        since += 1
        if is_report_step(step, args.steps, args.eval_every):
            entry = {"step": step + 1, "train_loss": running / max(1, since),
                     "val_loss": held_out(0.0), "val_loss_fim": held_out(1.0),
                     "val_perplexity": 0.0, "lr": lr_at(step),
                     "elapsed_s": time.time() - started}
            entry["val_perplexity"] = math.exp(entry["val_loss"])
            running, since = 0.0, 0
            history.append(entry)
            print(f"  step {step+1:>6,}/{args.steps:,}  train {entry['train_loss']:.4f}  "
                  f"plain {entry['val_loss']:.4f}  fim {entry['val_loss_fim']:.4f}  "
                  f"{entry['elapsed_s']/60:.0f}m", flush=True)
        if (step + 1) % args.checkpoint_every == 0:
            save_checkpoint(args.name, model, optimizer, step + 1, history, fingerprint)

    save_checkpoint(args.name, model, optimizer, args.steps, history, fingerprint)
    report = {
        "schema_version": 1,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": args.name,
        # The base run's shape, so every loader opens this like any checkpoint.
        "config": {**cfg, "name": args.name, "init_from": args.init_from,
                   "steps": args.steps, "lr": args.lr, "warmup": args.warmup,
                   "batch": args.batch, "seed": args.seed},
        "fim": {"rate": args.rate, "seam": "<|pad|>", "short_share": SHORT_SHARE,
                "short_words": SHORT_WORDS, "long_tokens": LONG_TOKENS,
                "truncate_share": TRUNCATE_SHARE, "suffix_range": SUFFIX_RANGE},
        "corpus_name": base.get("corpus_name"),
        "corpus": base.get("corpus"),
        "tokenizer": fingerprint,
        "parameters": base.get("parameters"),
        "train_tokens": int(train_tokens.size),
        "validation_tokens": int(val_tokens.size),
        "epochs": args.steps * tokens_per_step / train_tokens.size,
        "history": history,
        "final": history[-1] if history else {},
    }
    path = REPORTS / f"train-{args.name}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

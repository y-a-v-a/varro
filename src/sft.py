#!/usr/bin/env python3
"""Supervised fine-tuning on dialogue mined from the corpus itself.

Turns a base model that continues Latin into one that takes a turn and stops.
The training pairs come from src/dialogue.py — dramatic exchanges and
question/answer sentences found in the corpus — so no English-derived
instruction data enters the pipeline at any point.

LOSS IS MASKED TO THE RESPONSE. Training on prompt tokens too would teach the
model to generate questions as readily as answers, which is the opposite of the
behaviour wanted: at inference the prompt is given, and only the continuation is
the model's job.

STOPPING MATTERS AS MUCH AS CONTENT. A base model continues forever; a chat model
must end its turn. Every example terminates in <|endoftext|>, and that token is
included in the loss, so producing it is learned behaviour rather than an
inference-time hack.

CHECKPOINTS, FOR THE SAME REASON train.py HAS THEM. A run that can only be
restarted from zero is a run you will eventually lose, and Adam's moments take
hundreds of steps to rebuild — dropping them mid-run puts a visible spike in the
loss. This shares train.py's checkpoint format rather than inventing a second
one, so `--resume` means the same thing in both.

Usage:
    python src/sft.py --report                 # data shape, train nothing
    python src/sft.py                          # fine-tune latin-2
    python src/sft.py --resume                 # continue from the last checkpoint
    python src/sft.py --chat                   # talk to the result
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
REPORTS = ROOT / "reports"

sys.path.insert(0, str(ROOT / "src"))

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
except ImportError:
    sys.exit("MLX required — pip install mlx")

from train import (GPT, is_report_step, load_checkpoint,  # noqa: E402
                   save_checkpoint)

MAX_LEN = 256


def load_run(name: str) -> Dict[str, Any]:
    path = REPORTS / f"train-{name}.json"
    if not path.exists():
        sys.exit(f"missing {path.relative_to(ROOT)} — train {name} first")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def build_examples(pairs: List[Dict[str, Any]], tok,
                   max_len: int = MAX_LEN) -> Tuple[mx.array, mx.array]:
    """Tokenise pairs into padded sequences plus a loss mask over the response."""
    eos = tok.token_to_id("<|endoftext|>")
    pad = tok.token_to_id("<|pad|>")
    rows, masks = [], []
    for pair in pairs:
        prompt = tok.encode(pair["prompt"] + "\n").ids
        response = tok.encode(pair["response"]).ids + [eos]
        if len(prompt) + len(response) > max_len or len(response) < 3:
            continue
        ids = prompt + response
        mask = [0] * len(prompt) + [1] * len(response)
        padding = max_len - len(ids)
        rows.append(ids + [pad] * padding)
        masks.append(mask + [0] * padding)
    return (mx.array(rows, dtype=mx.int32), mx.array(masks, dtype=mx.float32))


def loss_fn(model, x, y, mask):
    """Cross-entropy over response tokens only."""
    logits = model(x).astype(mx.float32)
    losses = nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none")
    losses = losses.reshape(y.shape) * mask
    return losses.sum() / mx.maximum(mask.sum(), 1)


def batches(data: mx.array, masks: mx.array, batch: int, seed: int):
    n = data.shape[0]
    key = mx.random.key(seed)
    while True:
        key, use = mx.random.split(key)
        idx = mx.random.randint(0, n, (batch,), key=use)
        rows, m = data[idx], masks[idx]
        yield rows[:, :-1], rows[:, 1:], m[:, 1:]


def chat(model, tok, prompt: str, max_new: int = 120,
         temperature: float = 0.8, top_k: int = 40, seed: int = 1) -> str:
    """Generate one turn, stopping at <|endoftext|> as trained."""
    eos = tok.token_to_id("<|endoftext|>")
    ids = tok.encode(prompt + "\n").ids
    key = mx.random.key(seed)
    out: List[int] = []
    for _ in range(max_new):
        logits = model(mx.array([ids[-256:]]))[0, -1] / max(temperature, 1e-6)
        if top_k and top_k < logits.size:
            kth = mx.sort(logits)[-top_k]
            logits = mx.where(logits < kth, mx.full(logits.shape, -1e9), logits)
        key, use = mx.random.split(key)
        nxt = int(mx.random.categorical(logits, key=use).item())
        if nxt == eos:
            break
        ids.append(nxt)
        out.append(nxt)
    return tok.decode(out).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="latin-2", help="run whose weights to start from")
    ap.add_argument("--name", default="latin-2-chat")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--checkpoint-every", type=int, default=500,
                    help="save a resumable checkpoint every N steps")
    ap.add_argument("--resume", action="store_true",
                    help="continue this fine-tune from its last checkpoint")
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--chat", action="store_true", help="load and talk to --name")
    args = ap.parse_args()

    run = load_run(args.base)
    cfg = run["config"]
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZERS / f"latin-bpe-{cfg['vocab']}.json"))

    model = GPT(cfg["vocab"], cfg["dim"], cfg["layers"], cfg["heads"], cfg["context"])

    if args.chat:
        weights = CHECKPOINTS / f"{args.name}.safetensors"
        if not weights.exists():
            sys.exit(f"missing {weights.relative_to(ROOT)} — fine-tune first")
        model.load_weights(str(weights))
        model.eval()
        mx.eval(model.parameters())
        prompts = ["Quid est ueritas?", "Quid agis?", "Vnde uenis?",
                   "Quot sunt partes orationis?", "Cur me uerberas?"]
        print(f"{args.name} — one turn per prompt, stopping at <|endoftext|>\n")
        for i, p in enumerate(prompts):
            print(f"  > {p}")
            print(f"  < {chat(model, tok, p, seed=i + 1)}\n")
        return 0

    path = PROCESSED / "dialogue.jsonl"
    if not path.exists():
        sys.exit("missing data/processed/dialogue.jsonl — run src/dialogue.py")
    with path.open(encoding="utf-8") as fh:
        pairs = [json.loads(line) for line in fh]

    data, masks = build_examples(pairs, tok)
    kept, dropped = data.shape[0], len(pairs) - data.shape[0]
    supervised = float(masks.sum())

    print(f"base            {args.base} ({run['parameters']/1e6:.2f}M params)")
    print(f"dialogue pairs  {len(pairs):,} mined, {kept:,} usable "
          f"({dropped:,} dropped: too long or too short)")
    print(f"supervised      {supervised/1e6:.2f}M response tokens "
          f"(prompt tokens are masked out)")
    print(f"schedule        {args.steps:,} steps x {args.batch} "
          f"= {args.steps*args.batch/kept:.1f} epochs")
    if args.report:
        print("\n--report: nothing trained")
        return 0

    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.0)
    history: List[Dict[str, Any]] = []
    start_step = 0
    if args.resume:
        # The checkpoint holds the fine-tuned weights and the optimizer's
        # moments. Loading the base over them would silently undo the run and
        # report the result as a resumed one.
        start_step, history = load_checkpoint(args.name, model, optimizer)
        print(f"resumed {args.name} at step {start_step:,}")
    else:
        model.load_weights(str(CHECKPOINTS / f"{args.base}.safetensors"))
        mx.eval(model.parameters())

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    stream = batches(data, masks, args.batch, args.seed)
    started = time.time()
    running = 0.0
    since_report = 0

    for step in range(start_step, args.steps):
        frac = min(1.0, (step + 1) / max(1, args.warmup))
        progress = max(0.0, (step - args.warmup) / max(1, args.steps - args.warmup))
        optimizer.learning_rate = args.lr * frac * (0.1 + 0.9 * 0.5 *
                                                    (1 + math.cos(math.pi * progress)))
        x, y, m = next(stream)
        loss, grads = loss_and_grad(model, x, y, m)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        running += loss.item()
        since_report += 1

        if is_report_step(step, args.steps, args.eval_every):
            # Divide by what was actually accumulated, not by eval_every. The
            # final block is closed by reaching the end rather than by the
            # interval, so it is short whenever steps is not a multiple of
            # eval_every — the same arithmetic that reported LATIN-XXV's real
            # 3.12 train loss as 1.25. The defaults here divide evenly, which
            # is why it never showed; a --steps on the command line is all it
            # would have taken.
            mean = running / max(1, since_report)
            running, since_report = 0.0, 0
            elapsed = time.time() - started
            print(f"  step {step+1:>5,}/{args.steps:,}  loss {mean:.4f}  "
                  f"ppl {math.exp(mean):>7.1f}  {elapsed/60:.1f}m", flush=True)
            history.append({"step": step + 1, "loss": mean,
                            "perplexity": math.exp(mean), "elapsed_s": elapsed})

        if (step + 1) % args.checkpoint_every == 0:
            save_checkpoint(args.name, model, optimizer, step + 1, history)

    save_checkpoint(args.name, model, optimizer, args.steps, history)
    print(f"\nsaved checkpoints/{args.name}.safetensors")

    REPORTS.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1,
              "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "run": args.name, "base": args.base, "config": vars(args),
              "pairs_mined": len(pairs), "pairs_used": kept,
              "supervised_tokens": supervised, "history": history,
              "final": history[-1] if history else {}}
    with (REPORTS / f"sft-{args.name}.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    print(f"wrote reports/sft-{args.name}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

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

WHERE THE TIME GOES, MEASURED RATHER THAN ASSUMED
    The training step is Metal, not Python. Measured 2026-09-05 on an 8-core
    M2 (4P+4E, 10-core GPU, 16GB): a 25-step run of the LATIN-VII shape spent
    37.6s of CPU across 257s of wall clock — 0.15 of a single core, with seven
    and a half idle. Throwing cores at a training run therefore buys nothing;
    the bound is the GPU and the 100GB/s it shares with everything else, which
    is the same bound the batch-size cliff in configs/latin-xxv.yaml records.

    The one exception is tokenisation. `load_tokens` is ordinary Python calling
    a Rust tokenizer, it runs before the first step, and it is where the cores
    are. Batching the encode made it 2.4-3.2x faster: corpus.train, 183,265,520
    tokens, went 178.2s and 4.70GB peak to 55.2s and 2.68GB. The token stream
    is bit-identical, checked against the old path on three splits.

    Also measured, and NOT adopted: wrapping the step in `mx.compile` gave
    1.04x at the LATIN-VII shape and 1.24x at LATIN-XXV. The larger figure is
    worth revisiting, but it needs a full run against a recorded perplexity
    before a recipe depends on it — a fused graph that trains subtly
    differently is exactly the kind of change this project refuses to make on
    a 20-step benchmark.

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
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
TOKENIZERS = PROCESSED / "tokenizers"
CHECKPOINTS = ROOT / "checkpoints"
REPORTS = ROOT / "reports"

def rel(path: Path) -> str:
    """A path for a message, shortened to the repo when it lies inside it.

    load_recipe already refuses `relative_to(ROOT)` on the grounds that a
    missing-file message which itself crashes is worse than the missing file.
    The same is true of every other path this file names, so the rule lives in
    one function rather than being remembered at each call site.
    """
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_unflatten
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


def save_checkpoint(name: str, model, optimizer, step: int, history: List[Dict]) -> None:
    """Weights, optimizer state and position — everything needed to resume.

    A twelve-hour run that can only be restarted from zero is a twelve-hour run
    you will eventually lose. Optimizer state matters as much as weights: Adam's
    moment estimates take hundreds of steps to rebuild, and dropping them mid-run
    produces a visible loss spike.
    """
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(CHECKPOINTS / f"{name}.safetensors"))
    state = dict(tree_flatten(optimizer.state))
    arrays = {k: v for k, v in state.items() if isinstance(v, mx.array)}
    if arrays:
        mx.save_safetensors(str(CHECKPOINTS / f"{name}.opt.safetensors"), arrays)
    with (CHECKPOINTS / f"{name}.state.json").open("w", encoding="utf-8") as fh:
        json.dump({"step": step, "history": history}, fh)


def load_checkpoint(name: str, model, optimizer) -> Tuple[int, List[Dict]]:
    weights = CHECKPOINTS / f"{name}.safetensors"
    state_path = CHECKPOINTS / f"{name}.state.json"
    if not weights.exists() or not state_path.exists():
        sys.exit(f"cannot resume {name}: missing weights or state in "
                 f"{rel(CHECKPOINTS)}/")
    model.load_weights(str(weights))
    opt_path = CHECKPOINTS / f"{name}.opt.safetensors"
    if opt_path.exists():
        loaded = mx.load(str(opt_path))
        optimizer.state = tree_unflatten(list(loaded.items()))
    with state_path.open(encoding="utf-8") as fh:
        saved = json.load(fh)
    mx.eval(model.parameters())
    return saved["step"], saved.get("history", [])


def is_report_step(step: int, steps: int, eval_every: int) -> bool:
    """Whether step (0-based) closes a reporting block.

    The last block is closed by reaching the end, not by the interval, so it is
    short whenever `steps` is not a multiple of `eval_every`. Anything
    averaging over a block must divide by what it actually accumulated.
    """
    return (step + 1) % eval_every == 0 or step == steps - 1


def parameter_count(model: nn.Module) -> int:
    return sum(v.size for _, v in tree_flatten(model.parameters()))


# The subset of a corpus summary that identifies *which build* it is. Two
# builds that agree on all of these are the same held-out split over the same
# text, which is the only condition under which their perplexities compare.
FINGERPRINT_KEYS = ("track", "records", "words", "train_words",
                    "validation_words", "group_by", "seed", "period_filter")


def corpus_fingerprint(corpus: str) -> Dict[str, Any]:
    path = PROCESSED / f"{corpus}.summary.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        summary = json.load(fh)
    return {k: summary.get(k) for k in FINGERPRINT_KEYS}


def tokenizer_fingerprint(vocab: int) -> Dict[str, Any]:
    """Which tokenizer these numbers were measured through.

    Not redundant with the corpus fingerprint. The classical corpus moved by
    three words between the 2026-08-17 and 2026-08-18 builds and its bigram
    floor still went 158.8 -> 149.8, purely because the 4k vocabulary had been
    retrained on a corpus 4.6x larger: same Latin, different merges, different
    token stream. Perplexity compares only when both match.
    """
    path = TOKENIZERS / f"latin-bpe-{vocab}.json"
    if not path.exists():
        return {"vocab": vocab, "sha256": None}
    return {"vocab": vocab,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16]}


def read_floor(corpus: str, vocab: int) -> Tuple[Any, str]:
    """The bigram floor for `corpus`, and why it may not be quotable.

    Returns (perplexity, ""), (perplexity, reason) if the baseline describes a
    different build or tokenizer, or (None, "") if no baseline exists.
    """
    path = REPORTS / f"baselines-{corpus}.json"
    if not path.exists():
        return None, ""
    with path.open(encoding="utf-8") as fh:
        report = json.load(fh)
    floor = report.get("baselines", {}).get("overall_perplexity")
    recorded = report.get("corpus") or {}
    current = corpus_fingerprint(corpus)
    if not recorded or not current:
        return floor, ""
    # An older report may name the wrong corpus entirely: `corpus_name` was
    # only added when that bug was fixed, so its absence is itself suspicious.
    if report.get("corpus_name") not in (None, corpus):
        return floor, f"stamped corpus_name={report['corpus_name']!r}"
    drift = [f"{k}: {recorded.get(k)!r} -> {current.get(k)!r}"
             for k in FINGERPRINT_KEYS if recorded.get(k) != current.get(k)]
    recorded_tok = report.get("tokenizer")
    current_tok = tokenizer_fingerprint(vocab)
    if recorded_tok and recorded_tok != current_tok:
        drift.append(f"tokenizer: {recorded_tok} -> {current_tok}")
    return floor, "; ".join(drift)


# ---------------------------------------------------------------------------
# Recipes
# ---------------------------------------------------------------------------

CONFIGS = ROOT / "configs"


def load_recipe(path: str, phase: Optional[str]) -> Tuple[Dict[str, Any], str]:
    """Defaults for one phase of a recipe, plus the phase name.

    A recipe states epochs, never steps. A step count is meaningless the moment
    the corpus changes: 74,700 steps was four passes over 153M tokens, and on a
    corpus half again as large the same number is 2.7 passes with nothing to
    say the model is now undertrained. Epochs mean the same thing at any corpus
    size, which is what makes a recipe survive corpus growth — and growing the
    corpus is the point of the pipeline.
    """
    import yaml
    candidate = Path(path)
    if not candidate.exists():
        candidate = CONFIGS / path
        if not candidate.suffix:
            candidate = candidate.with_suffix(".yaml")
    if not candidate.exists():
        # Not relative_to(ROOT): a path outside the repo raises there, and a
        # missing-file message that crashes is worse than the missing file.
        sys.exit(f"no recipe at `{path}`; looked in configs/ too. "
                 f"Available: {', '.join(sorted(c.stem for c in CONFIGS.glob('*.yaml'))) or 'none'}")
    with candidate.open(encoding="utf-8") as fh:
        recipe = yaml.safe_load(fh)

    phases = recipe.get("phases") or []
    if not phases:
        sys.exit(f"{candidate.name} declares no phases")
    if phase is None:
        chosen = phases[0]
    else:
        matches = [p for p in phases if p.get("name") == phase]
        if not matches:
            sys.exit(f"{candidate.name} has no phase `{phase}` — "
                     f"choose from {', '.join(p['name'] for p in phases)}")
        chosen = matches[0]

    settings = {**(recipe.get("defaults") or {}), **chosen}
    settings["name"] = chosen["name"]
    return settings, chosen["name"]


def steps_for_epochs(epochs: float, train_tokens: int, batch: int,
                     context: int) -> int:
    """How many optimizer steps cover `epochs` passes over this corpus."""
    per_step = batch * context
    return max(1, round(epochs * train_tokens / per_step))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

PERIOD_TOKENS = {"CLASSICAL": "<CLASSICAL>", "LATE": "<LATE>",
                 "MEDIEVAL": "<MEDIEVAL>", "NEOLATIN": "<NEOLATIN>"}

# How much text to hand the tokenizer at once.
#
# `encode_batch` runs the BPE across every core; encoding one record at a time
# uses one. That is the only part of a training run where a second core has
# anything to do — the step itself is Metal, and measured at 0.15 of a core on
# an 8-core M2 (see the note on load_tokens).
#
# Bounded by characters rather than by record count because the two corpora
# differ by two orders of magnitude in record size, and the tokenizer holds an
# Encoding per record while a batch is in flight. 8MB is ~2.5M tokens per
# batch: enough to keep every core fed, small enough that Track B's 906MB
# corpus never has more than a fraction of itself resident.
ENCODE_CHUNK_CHARS = 8_000_000


def chunk_records(lines: Iterable[str],
                  budget: int = ENCODE_CHUNK_CHARS) -> Iterator[List[Dict[str, Any]]]:
    """Group JSONL lines into batches of at most `budget` characters of text."""
    batch: List[Dict[str, Any]] = []
    size = 0
    for line in lines:
        rec = json.loads(line)
        batch.append(rec)
        size += len(rec["text"])
        if size >= budget:
            yield batch
            batch, size = [], 0
    if batch:
        yield batch


def load_tokens(corpus: str, split: str, tok, cache: bool = True,
                period_tags: bool = True,
                chunk_chars: int = ENCODE_CHUNK_CHARS) -> mx.array:
    """Tokenise a split once and keep it as a flat token stream.

    Each document is prefixed with its period tag, which is what makes the
    two-phase Latinitas -> Romanitas plan possible: the model sees register as an
    explicit, conditionable signal rather than having to infer 1,500 years of
    linguistic drift from context alone. The tags are reserved in the tokenizer
    and injected here, never stored in the corpus — so the tagging scheme can
    change without rebuilding anything upstream.

    On a classical-only corpus every document gets the same tag, which is close
    to a no-op; it matters on the mixed corpus, where LATE is three quarters of
    the text and CLASSICAL is what we ultimately want generated.
    """
    path = PROCESSED / f"{corpus}.{split}.jsonl"
    if not path.exists():
        sys.exit(f"missing {rel(path)} — run src/build.py")
    suffix = "tagged" if period_tags else "plain"
    cached = PROCESSED / f"{corpus}.{split}.{suffix}.tokens.npy"
    if cache and cached.exists() and cached.stat().st_mtime > path.stat().st_mtime:
        return mx.array(mx.load(str(cached)))

    eos = tok.token_to_id("<|endoftext|>")
    tag_ids = {k: tok.token_to_id(v) for k, v in PERIOD_TOKENS.items()}
    missing = [k for k, v in tag_ids.items() if v is None]
    if period_tags and missing:
        sys.exit(f"tokenizer lacks period tokens {missing} — retrain src/tokenizer.py")

    # Ids are bounded by the vocabulary, so the width is known before a single
    # record is read. The previous `max(stream)` needed the whole corpus in a
    # Python list to ask a question the tokenizer could already answer.
    dtype = mx.uint16 if tok.get_vocab_size() <= 65535 else mx.int32

    pieces: List[mx.array] = []
    counts: Dict[str, int] = {}
    with path.open(encoding="utf-8") as fh:
        for records in chunk_records(fh, chunk_chars):
            stream: List[int] = []
            encoded = tok.encode_batch([rec["text"] for rec in records])
            for rec, ids in zip(records, encoded):
                if period_tags:
                    tag = tag_ids.get(rec.get("period") or "")
                    if tag is not None:
                        stream.append(tag)
                        counts[rec.get("period")] = counts.get(rec.get("period"), 0) + 1
                stream.extend(ids.ids)
                stream.append(eos)
            pieces.append(mx.array(stream, dtype=dtype))
    if not pieces:
        sys.exit(f"{rel(path)} is empty — run src/build.py")
    if period_tags and counts:
        print(f"  {split}: period tags " +
              ", ".join(f"{k} x{v}" for k, v in sorted(counts.items())))
    arr = mx.concatenate(pieces) if len(pieces) > 1 else pieces[0]
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
    ap.add_argument("--period-tags", action=argparse.BooleanOptionalAction, default=True,
                    help="prefix each document with its period token (default on)")
    ap.add_argument("--init-from", metavar="RUN", dest="init_from",
                    help="start from another run's weights — the second phase of a "
                         "two-phase schedule continues from the first")
    ap.add_argument("--checkpoint-every", type=int, default=500,
                    help="save a resumable checkpoint every N steps")
    ap.add_argument("--resume", action="store_true",
                    help="continue this run from its last checkpoint")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--config", metavar="RECIPE",
                    help="a recipe in configs/ (name or path); CLI flags still win")
    ap.add_argument("--phase", help="which phase of the recipe (default: the first)")
    ap.add_argument("--epochs", type=float,
                    help="passes over the corpus; steps are derived from it. "
                         "Prefer this to --steps — a step count means something "
                         "different every time the corpus changes")
    args = ap.parse_args()

    # A recipe supplies defaults; anything given explicitly on the command line
    # overrides it, so a recipe can be tried with one value changed without
    # being edited.
    if args.config:
        settings, _ = load_recipe(args.config, args.phase)
        given = {a.lstrip("-").replace("-", "_") for a in sys.argv[1:]
                 if a.startswith("--")}
        for key, value in settings.items():
            if key in given or not hasattr(args, key):
                continue
            setattr(args, key, value)
        if "steps" not in given and settings.get("epochs") is not None:
            args.epochs = settings["epochs"]
    elif args.phase:
        sys.exit("--phase only means something with --config")

    from tokenizers import Tokenizer
    tok_path = TOKENIZERS / f"latin-bpe-{args.vocab}.json"
    if not tok_path.exists():
        sys.exit(f"missing {rel(tok_path)} — run src/tokenizer.py")
    tok = Tokenizer.from_file(str(tok_path))

    # Caught here rather than as an MLX traceback six frames down. Easy to hit
    # from a recipe: override --dim without --heads and the two stop dividing.
    if args.dim % args.heads:
        sys.exit(f"dim {args.dim} is not divisible by heads {args.heads} — "
                 f"each head needs an equal slice of the width")

    mx.random.seed(args.seed)
    model = GPT(args.vocab, args.dim, args.layers, args.heads, args.context)
    mx.eval(model.parameters())
    n_params = parameter_count(model)

    train_tokens = load_tokens(args.corpus, "train", tok, period_tags=args.period_tags)
    val_tokens = load_tokens(args.corpus, "validation", tok, period_tags=args.period_tags)

    if args.init_from:
        prior = CHECKPOINTS / f"{args.init_from}.safetensors"
        if not prior.exists():
            sys.exit(f"missing {rel(prior)} — train {args.init_from} first")
        model.load_weights(str(prior))
        mx.eval(model.parameters())
        print(f"initialised from {rel(prior)} "
              f"(optimizer state is fresh, by design: a phase change wants a new "
              f"schedule, not the tail of the old one)")

    # Epochs are the durable unit; steps are derived from the corpus in hand.
    if args.epochs is not None:
        args.steps = steps_for_epochs(args.epochs, train_tokens.size,
                                      args.batch, args.context)

    tokens_per_step = args.batch * args.context
    total_tokens = args.steps * tokens_per_step
    epochs = total_tokens / train_tokens.size

    print(f"model        {args.name}  {n_params/1e6:.2f}M parameters "
          f"(dim {args.dim}, {args.layers} layers, {args.heads} heads, ctx {args.context})")
    print(f"corpus       {args.corpus}  {train_tokens.size:,} train tokens, "
          f"{val_tokens.size:,} validation", flush=True)
    derived = "  <- derived from --epochs" if args.epochs is not None else ""
    print(f"schedule     {args.steps:,} steps x {tokens_per_step:,} tokens "
          f"= {total_tokens/1e6:.1f}M  ({epochs:.1f} epochs){derived}")
    print(f"tokens/param {total_tokens/n_params:.1f}   "
          f"(Chinchilla-optimal is ~20)", flush=True)
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
    start_step = 0
    if args.resume:
        start_step, history = load_checkpoint(args.name, model, optimizer)
        print(f"resumed {args.name} at step {start_step:,}")
    started = time.time()
    running = 0.0
    since_report = 0

    for step in range(start_step, args.steps):
        optimizer.learning_rate = lr_at(step)
        x, y = next(stream)
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        running += loss.item()
        since_report += 1

        if is_report_step(step, args.steps, args.eval_every):
            # Divide by what was actually accumulated. The final report is
            # triggered by `step == args.steps - 1` rather than by the
            # interval, so when steps is not a multiple of eval_every the
            # last block is short. Dividing by eval_every regardless scaled
            # the final train loss by (short block / eval_every) — on the
            # 74,700-step LATIN-XXV run that turned a real 3.12 into 1.25
            # and reported a 7.25x train/val gap where the truth was 1.12x.
            train_loss = running / max(1, since_report)
            running, since_report = 0.0, 0
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

        if (step + 1) % args.checkpoint_every == 0:
            save_checkpoint(args.name, model, optimizer, step + 1, history)

    save_checkpoint(args.name, model, optimizer, args.steps, history)
    weights = CHECKPOINTS / f"{args.name}.safetensors"
    final = history[-1] if history else {}
    print(f"\nsaved {rel(weights)}")

    # The floor must come from a baseline trained AND evaluated on the same
    # splits as this run. Quoting a floor computed on a different corpus
    # compares perplexities over different held-out text, which is meaningless
    # — and it flattered Latin-0 by 23% before this was caught.
    #
    # Matching by filename is not enough: the corpus changes under a fixed
    # name every time a source lands, so `baselines-corpus.json` can describe
    # a corpus three rebuilds old. The baseline carries a fingerprint of the
    # corpus it was measured on; if it disagrees with the corpus just trained
    # on, the floor is stale and quoting it would repeat the original mistake
    # in a form that is harder to see.
    floor, floor_stale = read_floor(args.corpus, args.vocab)
    ppl = final.get("val_perplexity", float("nan"))
    if floor and not floor_stale:
        print(f"final held-out perplexity {ppl:.1f} against the {args.corpus} "
              f"bigram floor of {floor:.1f}  ({floor/ppl:.2f}x better)")
    elif floor_stale:
        print(f"final held-out perplexity {ppl:.1f}")
        print(f"  NOT quoting the floor: reports/baselines-{args.corpus}.json was "
              f"measured on a different build of `{args.corpus}`\n"
              f"  ({floor_stale}) — re-run evaluate.py --baselines "
              f"--corpus {args.corpus}")
    else:
        print(f"final held-out perplexity {ppl:.1f} "
              f"(no baseline for `{args.corpus}` — run evaluate.py --baselines "
              f"--corpus {args.corpus})")

    REPORTS.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": args.name,
        "config": vars(args),
        "corpus_name": args.corpus,
        "corpus": corpus_fingerprint(args.corpus),
        "tokenizer": tokenizer_fingerprint(args.vocab),
        # Which recipe, and how many passes it asked for. `steps` alone does
        # not survive corpus growth; `epochs` is what a later run reproduces.
        "recipe": args.config,
        "epochs_requested": args.epochs,
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
    print(f"wrote {rel(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

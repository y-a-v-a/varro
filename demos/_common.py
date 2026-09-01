#!/usr/bin/env python3
"""Shared plumbing for the demos: load a model, score a candidate reading.

The demos are deliberately thin. Everything that decides what a number MEANS
lives in src/score.py, which the project already tests. This module adds only
the two things a demo needs and a library command does not: a default
checkpoint that is the right one, and a way to score a *reading* -- a text with
one span swapped out -- rather than a whole document.
"""

import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from normalize import normalize as _normalize  # noqa: E402
from score import (  # noqa: E402
    PERIODS, load_model, perplexity, prefix_ids, token_logprobs,
)

# WHICH CHECKPOINT IS "LATIN-LII"
#   `latin-lii.safetensors` is NOT the 21.3-perplexity model the project
#   reports. It is the abandoned classical-narrowing run: 30.2 on classical,
#   32.5 on corpus-b, kept because deleting a measurement is worse than
#   labelling it. configs/latin-lii.yaml records why that phase was dropped.
#
#   The model to demo is the surviving single phase, which was trained under
#   --name latin-lii-latinitas and still carries that name on disk.
DEFAULT_MODEL = "latin-lii-latinitas"

# How many candidate readings share one forward pass. Sized for a 16GB machine:
# the batch materialises chunk x width x vocab logits in float32, which at
# chunk=16, width=512, vocab=4000 is 131MB and comfortable. Raising it buys
# little, because the sequences a demo scores are far shorter than the context.
SCORE_CHUNK = 16

# WHY A CACHE LIMIT, MEASURED
#   MLX keeps freed buffers in an allocator cache and, left alone, that cache
#   is unbounded. A benchmark loop over this model held 198MB of weights and
#   peaked at 805MB of working set -- and grew the cache to 11.3GB, because
#   each sampling step asks for a slightly different buffer size and every one
#   of them is kept. On a 16GB machine that is the whole machine: the run drops
#   to 14% CPU, spends its life in the pager, and is eventually killed.
#
#   The tell is that `ps` shows nothing wrong. Unified memory is not attributed
#   to the process, so the symptom looks like a slow program rather than a
#   greedy one -- the same misreading the LATIN-LII recipe records.
#
#   1GB is far above the 805MB peak, so nothing thrashes at the other end.
CACHE_LIMIT_BYTES = 1 << 30


class Loaded(NamedTuple):
    name: str
    model: object
    tok: object
    cfg: Dict
    run: Dict

    @property
    def context(self) -> int:
        return self.cfg["context"]

    @property
    def tagged(self) -> bool:
        return bool(self.cfg.get("period_tags"))

    def describe(self) -> str:
        ppl = self.run["final"]["val_perplexity"]
        return (f"{self.name}: {self.run['parameters']/1e6:.2f}M parameters, "
                f"corpus {self.cfg.get('corpus')}, held-out perplexity {ppl:.1f}")


def load(name: str = DEFAULT_MODEL) -> Loaded:
    import mlx.core as mx

    mx.set_cache_limit(CACHE_LIMIT_BYTES)
    model, tok, cfg, run = load_model(name)
    return Loaded(name, model, tok, cfg, run)


def encode(m: Loaded, text: str) -> List[int]:
    return m.tok.encode(text).ids if text else []


def to_corpus_orthography(text: str) -> Tuple[str, Dict[str, int]]:
    """Put pasted Latin into the spelling the model was actually trained on.

    WHY THIS IS NOT OPTIONAL POLISH
        The corpus is folded u/v and i/j: `divisa` is stored `diuisa`, and a
        sample of 2,000 training documents contains 228,000 u-spellings of
        these stems and ZERO v-spellings. So a `v` in the middle of a word is,
        to this model, a character that essentially never occurs there.

        Measured on Caesar's opening sentence: the single token `v` inside
        `divisa` costs 30.4 bits, against about 9 for its neighbours. Every
        ordinary word that looked corrupt in an early run of demos/surprisal.py
        -- provinciae, Helvetii, virtute, dividit -- was ordinary, and was
        flagged solely for containing a `v`. Text from any modern edition would
        have scored as damage from end to end.

    This calls src/normalize.py rather than reimplementing the fold, so the
    demos and the corpus can never drift apart on what Latin looks like.
    """
    folded = _normalize(text)
    changes = {
        "v -> u": sum(text.count(c) for c in "vV"),
        "j -> i": sum(text.count(c) for c in "jJ"),
    }
    return folded, {k: n for k, n in changes.items() if n}


def announce_folding(changes: Dict[str, int], raw: bool) -> None:
    if raw:
        print("  [--raw: input scored as typed. A v- or j-spelling will read "
              "as damage;\n   see to_corpus_orthography in demos/_common.py]\n")
    elif changes:
        summary = ", ".join(f"{n} {what}" for what, n in changes.items())
        print(f"  [folded to the corpus spelling: {summary}]\n")


def score_readings(m: Loaded, left: str, fills: List[str], right: str,
                   period: Optional[str] = None,
                   chunk: int = SCORE_CHUNK) -> List[Dict]:
    """log P(fill + right | left) for every candidate fill.

    WHY THE PIECES ARE ENCODED SEPARATELY
        Byte-level BPE merges across a join, so encoding each whole string would
        let a candidate's token count depend on the letters either side of it.
        Encoding left, fill and right apart keeps each candidate's tokens
        exactly its own, which is what makes two candidates comparable. The
        price is that the absolute number drifts slightly from what the model
        saw in training; the ranking, which is all a demo uses, does not.

    WHY THE RIGHT CONTEXT IS INSIDE THE SCORE
        This model is left-to-right and cannot read a gap from both sides. But
        the probability of the WHOLE remaining line can be computed for any
        proposed fill, and that quantity does depend on what follows the gap.
        Generating forward ignores the right context; scoring across it is what
        makes this a reconstruction rather than a continuation.

    WHY THE CANDIDATES ARE BATCHED, AND WHY THEY ARE RIGHT-PADDED
        One forward pass per candidate is the obvious implementation and it is
        roughly twenty times slower. Batching is safe here only because the
        model is causal: a right-pad sits after every token that matters, so no
        real position shifts and no real token can attend to filler. Left-
        padding the same batch would silently move every position index and
        change the scores.
    """
    import mlx.core as mx

    prefix = prefix_ids(m.tok, period)
    tail = encode(m, right)
    bodies = [encode(m, f) for f in fills]
    widest = max((len(b) for b in bodies), default=0)

    # Trim the LEFT context to fit, never the right: the right context is the
    # evidence that distinguishes the candidates, and dropping it would quietly
    # turn this back into plain generation.
    room = m.context - len(prefix) - widest - len(tail)
    if room < 0:
        tail = tail[:max(0, m.context - len(prefix) - widest)]
        room = 0
    head = encode(m, left)[-room:] if room else []

    rows = [prefix + head + body + tail for body in bodies]
    width = max(len(r) for r in rows)
    pad = m.tok.token_to_id("<|endoftext|>") or 0
    start = len(prefix) + len(head)

    out: List[Dict] = []
    for i in range(0, len(rows), chunk):
        block = rows[i:i + chunk]
        padded = mx.array([r + [pad] * (width - len(r)) for r in block])
        logits = m.model(padded).astype(mx.float32)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        picked = mx.take_along_axis(
            logp[:, :-1], padded[:, 1:, None], axis=-1)[:, :, 0]
        # Force evaluation per chunk. Without this MLX keeps every chunk's
        # graph alive until the last one is read, and a 16GB machine dies.
        mx.eval(picked)
        values = picked.tolist()

        for j, row in enumerate(block):
            body = bodies[i + j]
            # values[j][t] is log P(row[t+1] | row[:t+1]).
            varying = values[j][start - 1:len(row) - 1]
            in_fill = values[j][start - 1:start - 1 + len(body)]
            out.append({
                "fill": fills[i + j],
                "logprob": sum(varying),
                "n_scored": len(varying),
                "fill_tokens": len(body),
                "fill_logprob": sum(in_fill),
                "perplexity": perplexity(varying),
            })
    return out


def score_reading(m: Loaded, left: str, fill: str, right: str,
                  period: Optional[str] = None) -> Dict:
    return score_readings(m, left, [fill], right, period)[0]

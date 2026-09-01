#!/usr/bin/env python3
"""Standalone loader and generator for a bundled claudius model.

This file is copied into every Hugging Face bundle as `generate.py` by
`src/bundle.py`. It must stay self-contained: someone who downloads the
bundle has model.safetensors, tokenizer.json, config.json and this file,
and nothing else from the repository.

    pip install mlx tokenizers
    python generate.py --prompt "Gallia est omnis diuisa"

Prompts are folded to the corpus orthography (u for v, i for j) before
encoding — the model has effectively never seen the letter v.
Requires Apple Silicon (MLX).
"""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from tokenizers import Tokenizer

HERE = Path(__file__).parent


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


def fold(text: str) -> str:
    """Match the corpus orthography: v->u, j->i, case preserved."""
    return (text.replace("v", "u").replace("V", "U")
                .replace("j", "i").replace("J", "I"))


def load():
    cfg = json.loads((HERE / "config.json").read_text())
    model = GPT(cfg["vocab_size"], cfg["dim"], cfg["layers"],
                cfg["heads"], cfg["context"])
    model.load_weights(str(HERE / "model.safetensors"))
    model.eval()
    tok = Tokenizer.from_file(str(HERE / "tokenizer.json"))
    return model, tok, cfg


def generate(model, tok, prompt: str = "", length: int = 120,
             temperature: float = 0.8, top_k: int = 40, seed: int = 3,
             period: str = None) -> str:
    # Every training document began <|endoftext|> plus its period tag, so a
    # sample does too. config.json records the tag when the model trained on
    # a single period; a multi-period model takes one via `period`.
    ids = [tok.token_to_id("<|endoftext|>")]
    if period:
        tag = tok.token_to_id(f"<{period}>")
        if tag is None:
            raise SystemExit(f"tokenizer has no <{period}> token")
        ids.append(tag)
    if prompt:
        ids.extend(tok.encode(fold(prompt)).ids)
    key = mx.random.key(seed)
    for _ in range(length):
        window = ids[-model.context:]
        logits = model(mx.array([window]))[0, -1]
        logits = logits / max(temperature, 1e-6)
        if top_k and top_k < logits.size:
            kth = mx.sort(logits)[-top_k]
            logits = mx.where(logits < kth, mx.full(logits.shape, -1e9), logits)
        key, use = mx.random.split(key)
        ids.append(int(mx.random.categorical(logits, key=use).item()))
    return tok.decode(ids).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--length", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--period", default=None,
                    help="period tag; defaults to the one the model trained with")
    args = ap.parse_args()

    model, tok, cfg = load()
    period = args.period or cfg.get("default_period_tag")
    for i in range(args.n):
        print(generate(model, tok, args.prompt, args.length,
                       args.temperature, args.top_k, args.seed + i, period))
        print()


if __name__ == "__main__":
    main()

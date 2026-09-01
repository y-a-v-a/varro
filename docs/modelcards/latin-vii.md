---
language:
  - la
license: ${hf_license}
library_name: mlx
pipeline_tag: text-generation
tags:
  - latin
  - gpt
  - mlx
  - safetensors
  - trained-from-scratch
---

# LATIN-VII

A ${parameters_m}-parameter GPT trained **from random initialization on classical
Latin and nothing else**. No English, no modern languages, no inherited
multilingual representations — the model has never seen a word that is not Latin.

It is the redistributable model of the [claudius](https://github.com/y-a-v-a/claudius)
project, which builds an openly licensed Latin corpus and trains Latin-only
language models on it. Every source in this model's training corpus is
CC-BY-SA, public domain, or equivalently free — no NonCommercial term
anywhere in the chain, which is the reason this model exists and is the one
published here.

| | |
|---|---|
| Parameters | ${parameters} |
| Training corpus | ${corpus_words_m} words of classical Latin (Track ${track}, redistributable) |
| Tokens seen | ${tokens_seen_m} (${epochs} epochs over ${train_tokens_m} tokens, ~${tokens_per_param} tokens/param) |
| Held-out perplexity | **${ppl}** vs. a bigram floor of ${floor} on the same split — **${ratio}×** |
| Memorisation | **${memorised}** twelve-word windows found verbatim in training text |
| Trained | ${trained_date}, ~${hours}h on an Apple M2 (MLX) |

## What it writes

```
Gallia est omnis diuisa,
ut aeridibus, quae pars maxima fuit, cum res,
aut non aeris aut area, ut ante quam in lepore
dicitur. Cuius Ianus, qui ante Idus
```

Honest reading: the morphology is largely well-formed and the function-word
skeleton is genuinely Latin, but it is locally coherent and globally not — no
sentence sustains an argument, and it invents plausible non-words. That is
what ${parameters_m_round} parameters trained on ${train_tokens_m_round} tokens buys. It is a
research artifact and a demonstration, not an oracle.

## Use

Requires Apple Silicon (the model is trained and shipped in
[MLX](https://github.com/ml-explore/mlx)). This is **not** a `transformers`
architecture; a self-contained loader ships with the model:

```bash
pip install mlx tokenizers
python generate.py --prompt "Gallia est omnis diuisa"
```

Or in code:

```python
from generate import load, generate
model, tok, cfg = load()
print(generate(model, tok, prompt="Arma uirumque cano",
               period=cfg["default_period_tag"]))
```

### Two things every prompt must respect

1. **Orthography.** The corpus is normalised to the classical convention:
   `u` for `v`, `i` for `j`. The model has effectively never seen the letter
   `v` — write `diuisa`, not `divisa`. The bundled loader folds prompts
   automatically.
2. **The period tag.** Every training document began
   `<|endoftext|><${period_tag}>`. The loader prepends both; if you tokenize by
   hand, do the same. The tokenizer also holds tags for the other periods,
   but this model trained on ${period_tag_lower} text only, so they carry no
   learned meaning here.

## Architecture

Decoder-only transformer, pre-norm, GELU MLP at 4×, no biases, input
embeddings tied to the output head, learned positional embeddings.

| | |
|---|---|
| vocab | ${vocab_fmt} (byte-level BPE, trained on the corpus itself) |
| d_model | ${dim} |
| layers | ${layers} |
| heads | ${heads} |
| context | ${context} |

The 4k vocabulary is a measured choice, not a default: at that size the
tokenizer stores none of the twelve `amo` paradigm forms whole and is forced
to learn Latin's stems and endings (`amabamus` → `am · ab · amus`), and the
embedding table stays proportionate to a model this small. `tokenizer.json`
is the exact tokenizer this model trained with (project fingerprint
`${tokenizer_sha}`) — the corpus and tokenizer are fingerprinted together
because retraining BPE on the same Latin changes the token stream.

## Training data

The classical split of the claudius corpus: ${records} documents,
${corpus_words} words, everything datable to before ~AD 200 that survives in
the project's free-licence sources — Caesar, Cicero, Vergil, Livy, Ovid,
Plautus, and their contemporaries. Sources by contribution:

| source | share | licence |
|---|---:|---|
${source_table}

Exact ingested versions (git SHAs) are pinned in
[`ATTRIBUTION.md`](ATTRIBUTION.md), which travels with these weights. The
underlying texts are ancient and out of copyright; what the licences cover
is the modern editorial and encoding work.

Why this corpus is the size it is: ~10M tokens is roughly half of the
classical Latin that survives *at all* — the rest sits in copyrighted
20th-century critical editions. No crawl produces more Cicero. At the
repeated-data limit of ~16 epochs, that supports about 8M parameters, and
this model is deliberately sized just under that ceiling.

## Evaluation

Validation is held out by **whole author** — no author in the validation set
appears anywhere in training — because a work-level split would measure
interpolation to an unseen text by a thoroughly familiar writer.

**Perplexity.** ${ppl} on the held-out split, against an interpolated-bigram
baseline of ${floor} trained and evaluated on the identical split. The floor
was stated before training.

**Memorisation.** Generated text is cut into overlapping windows and each
window searched verbatim against the full training text (exact search — no
false positives possible). A held-out real author serves as control:

| window (words) | LATIN-VII output | real unseen Latin |
|---:|---:|---:|
${memo_table}

From five words up the model reproduces its training text *less* often than a
genuine unseen Latin author does. Longest verbatim run: ${longest_run} words
at the 12-word window. It does not recite.

## Limitations

- **It is not correct about anything.** It has learned what Latin looks like,
  not what is true. Do not use it to answer questions.
- Locally coherent, globally not: grammar holds for a clause, arguments do
  not survive a sentence.
- Trained on classical Latin only; medieval or ecclesiastical register is
  out of distribution.
- Verse and prose are mixed in training; it drifts between registers.
- MLX only — no PyTorch/`transformers` port yet (the weights are plain
  safetensors; a port needs only the ~50-line architecture in `generate.py`).

## Licence

Weights are released under **${effective_license}**, matching the effective
licence of the corpus they were trained on. Attribution for the corpus
sources is in [`ATTRIBUTION.md`](ATTRIBUTION.md); please keep it with any
redistribution.

## Provenance

Everything here is reproducible from the
[claudius repository](https://github.com/y-a-v-a/claudius): the manifest pins
every source to a git SHA, `configs/${name}.yaml` is the exact training
recipe (seed included), and the training/evaluation reports this card quotes
are generated files, not hand-written numbers — the bundle itself is
assembled by `src/bundle.py` from those reports.

```
@software{claudius_latin_vii_2026,
  author = {Vincent Bruijn},
  title = {LATIN-VII: a 7M-parameter GPT trained from scratch on classical Latin},
  year = {2026},
  url = {https://github.com/y-a-v-a/claudius}
}
```

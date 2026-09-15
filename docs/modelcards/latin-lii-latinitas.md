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
  - research-only
---

# LATIN-LII — research artifact, not a download

> **This model is not redistributable.** It was trained on the Varro
> **Track ${track}** corpus, which includes NonCommercial sources
> (${nc_share} of it by words). The project's position is that NonCommercial
> propagates into weights, so these are published as *results* — perplexity,
> memorisation, the demos — and never as files. See
> [`DO-NOT-REDISTRIBUTE.md`](DO-NOT-REDISTRIBUTE.md).
>
> The publishable model is **[LATIN-VII](https://huggingface.co/vincent-b/latin-vii)**,
> whose corpus carries no NonCommercial term anywhere.

A ${parameters_m}-parameter GPT trained **from random initialization on Latin
and nothing else** — no English, no modern languages, no inherited
multilingual representations. It is the largest and best model the
[Varro](https://github.com/y-a-v-a/claudius) project has trained, and the
first whose bigram floor was computed on its own corpus rather than borrowed
from another.

| | |
|---|---|
| Parameters | ${parameters} |
| Training corpus | ${corpus_words_m} words of Latin, all periods (Track ${track}) |
| Tokens seen | ${tokens_seen_m} (${epochs} epochs over ${train_tokens_m} tokens, ~${tokens_per_param} tokens/param) |
| Held-out perplexity | **${ppl}** vs. a bigram floor of ${floor} on the same split — **${ratio}×** |
| Memorisation | **${memorised}** twelve-word windows found verbatim in training text |
| Trained | ${trained_date}, ~${hours}h on an Apple M2 (MLX) |

## What its size bought, and what it did not

Two measurements from the ladder, both of which cut against expectation:

**Data dominates sizing at this scale.** Doubling parameters from LATIN-0
(3.53M) to LATIN-VII (7.28M) on the same classical corpus bought 74.3 → 70.1,
about 6%. Training on 25× more Latin bought 70.1 → 27.3 on that same
classical held-out set, about 61%. Correct Chinchilla sizing says how to
balance parameters against the tokens you have; it says nothing about how
much better you would do with more tokens.

**The classical narrowing phase was dropped.** The two-stage recipe that
worked at 27.5M parameters (classical 34.8 → 28.3) made this model *worse* at
52M: 27.3 → 30.2 on classical, 21.3 → 32.5 on the full corpus. 10.7M
classical tokens cannot occupy 52M weights. Those weights are kept under
`latin-lii-romanitas` rather than deleted — a measurement that failed is
still a measurement — and `configs/latin-lii.yaml` records why. This card
describes the surviving single phase.

## Use

Requires Apple Silicon (trained and shipped in
[MLX](https://github.com/ml-explore/mlx)). This is **not** a `transformers`
architecture; a self-contained loader ships beside the weights:

```bash
pip install mlx tokenizers
python generate.py --prompt "Gallia est omnis diuisa" --period CLASSICAL
```

```python
from generate import load, generate
model, tok, cfg = load()
print(generate(model, tok, prompt="Arma uirumque cano", period="CLASSICAL"))
```

### Three things every prompt must respect

1. **Orthography.** The corpus is normalised to the classical convention:
   `u` for `v`, `i` for `j`. The model has effectively never seen the letter
   `v` — write `diuisa`, not `divisa`. The bundled loader folds prompts
   automatically.
2. **Every document began `<|endoftext|>`.** The loader prepends it; if you
   tokenize by hand, do the same.
3. **The period tag is live on this model.** Unlike the classical-only models,
   this one trained across four periods and each document carried its own tag,
   so `<CLASSICAL>`, `<LATE>`, `<MEDIEVAL>` and `<NEOLATIN>` genuinely steer
   the register. Untagged, it defaults toward late antique prose, which is the
   plurality of the corpus.

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

`tokenizer.json` is the exact tokenizer this model trained with (project
fingerprint `${tokenizer_sha}`). The corpus and tokenizer are fingerprinted
together because retraining BPE on the same Latin changes the token stream —
same Latin, different merges, different perplexity.

## Training data

${records} documents, ${corpus_words} words spanning classical, late,
medieval and neo-Latin. Sources by contribution:

| source | share | licence |
|---|---:|---|
${source_table}

The two **(NC)** rows are the whole reason this model cannot be
redistributed. Exact ingested versions (git SHAs) are pinned in
[`ATTRIBUTION.md`](ATTRIBUTION.md), which credits every source including the
NonCommercial ones — NonCommercial restricts what a result may be used *for*;
it does not excuse crediting anyone.

## Evaluation

Validation is held out by **whole author** — no author in the validation set
appears anywhere in training — because a work-level split would measure
interpolation to an unseen text by a thoroughly familiar writer.

**Perplexity.** ${ppl} on the held-out split, against an interpolated-bigram
baseline of ${floor} trained and evaluated on the identical split — ${ratio}×
better than the floor.

That floor is *higher* than Track A's (180.62): the corpus got genuinely
harder when the medieval and neo-Latin material arrived, so the improvement
over LATIN-XXV-2 (7.00×) is a real gain and not an easier test.

**Memorisation.** Generated text is cut into overlapping windows and each
window searched verbatim against the full training text (exact search — no
false positives possible). A held-out real author serves as control:

| window (words) | LATIN-LII output | real unseen Latin |
|---:|---:|---:|
${memo_table}

Longest verbatim run: ${longest_run} words. From four words up the model
reproduces its training text *less* often than a genuine unseen Latin author
does — and this matters more here than anywhere else in the ladder, because a
train/validation gap had been read as evidence of memorisation for two days
before this measurement settled it. Distributional overfitting and verbatim
recall are different properties. Only the second is what "does it recite?"
asks, and the answer is no.

## Limitations

- **It is not correct about anything.** It has learned what Latin looks like,
  not what is true. Do not use it to answer questions.
- **Not redistributable.** See the header and `DO-NOT-REDISTRIBUTE.md`.
- Locally coherent, globally not: grammar holds for a clause, arguments do
  not survive a sentence — though markedly better than the smaller models.
- The corpus is 43% late antique and medieval Latin, much of it Migne's
  *Patrologia Latina*; untagged output drifts toward that register.
- Dates from the PL-volume method are `inferred` confidence and place a text
  on the right side of a period boundary. They are not scholarship.
- MLX only — no PyTorch/`transformers` port (the weights are plain
  safetensors; a port needs only the ~50-line architecture in `generate.py`).

## Licence

Effective licence of the training corpus: **${effective_license}**.
Attribution for every source is in [`ATTRIBUTION.md`](ATTRIBUTION.md).
The weights are a research artifact and are not distributed.

## Provenance

Everything here is reproducible from the
[Varro repository](https://github.com/y-a-v-a/claudius): the manifest pins
every source to a git SHA, `${recipe}` is the exact training recipe (seed
included), and the figures this card quotes are substituted from generated
reports by `src/bundle.py`, not hand-written.

```
@software{varro_latin_lii_2026,
  author = {Vincent Bruijn},
  title = {LATIN-LII: a 52M-parameter GPT trained from scratch on Latin},
  year = {2026},
  url = {https://github.com/y-a-v-a/claudius}
}
```

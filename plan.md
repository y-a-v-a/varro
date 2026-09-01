# Classical Latin LLM with Language Model Builder

## Goal

Build a **small generative language model trained entirely on Latin**, from random initialization
rather than fine-tuning an existing multilingual model.

The interesting constraint is purity: the model's entire linguistic universe is Latin. No English
pretraining, no translations, no inherited knowledge from a general-purpose LLM. This makes the
project useful not only as a Latin generator, but as an experiment in what a transformer learns
from a historically bounded corpus.

A second constraint, adopted deliberately: **everything ships under a clean licence.** Corpus,
tokenizer, weights, manifest and training parameters are all publishable and redistributable. This
caps the achievable model size (see *Available Latin* below), and that trade is accepted.

## Why Language Model Builder

[Language Model Builder](https://languagemodelbuilder.com/) is a good environment for the first
experiments. It runs locally on Apple Silicon using MLX (macOS 15+) and supports the complete basic
pipeline:

**dataset → tokenizer → pretraining → sampling → SFT → DPO → chat**

Most importantly, it supports **pretraining from scratch using your own dataset**, rather than
requiring an existing foundation model. Training progress, validation loss, throughput, samples and
checkpoints can be inspected directly, and runs can be stopped and resumed. Models are exported as
standard safetensors. It is free, requires no account, and runs entirely locally.

The author estimates that a MacBook Pro M5 Max can train a GPT-2-small-class model of roughly
100–150M parameters on a few billion tokens in about a week.

**That estimate is not the binding constraint on this project.** There is nowhere near a few billion
tokens of Latin available under a clean licence, so this project is *data-bound, not compute-bound*.
Every model on the ladder below trains in hours, not weeks. Compute budgeting is not where the
effort goes.

Language Model Builder should therefore be treated as the **experimental laboratory**, not
necessarily the final infrastructure.

## Available Latin

There are two distinct corpus concepts, and a licensing ceiling that cuts across both.

**Strict Classical Latin**

Restricted approximately to Roman antiquity, the surviving body of text is small. The usual figure
is on the order of **10 million words** — Cicero, Caesar, Sallust, Livy, Virgil, Horace, Ovid,
Seneca, Pliny, Tacitus and others.

**Measured, 2026-08-17: 4,539,515 words** of classical Latin, inside a corpus of
78,975,089 words from five of nine sources. The classical figure has barely moved
as the corpus grew 4.6×, because the new material is late antique and medieval —
and dedup removed the Wikisource copies of texts Perseus already supplied. Perseus is the canonical classical source and is substantially complete for the
canonical authors, so the gap to 10M does not close with another literary repository — it closes,
partly, with the Latin Library (which overlaps Perseus heavily) plus inscriptions and papyri, and
those are a different register: formulaic, abbreviated, epigraphic.

Measured at a 4k vocabulary: **9.6M tokens** of classical Latin, against
**153.0M tokens** for the full corpus. These numbers drive the parameter ladder below;
see `feasibility.md` §2 for the epoch arithmetic.

**Historical Latin**

Latin continued as a major written language for more than 1,500 years after antiquity. Adding Late
Antique, Patristic, Medieval and Neo-Latin material changes the scale — but not as much as the
published figures suggest, once licensing is enforced.

Latin BERT was trained on approximately **643 million Latin words** spanning antiquity through the
modern period. **That corpus is not reproducible here.** It draws on Corpus Corporum
(non-commercial only), the Latin Library, Perseus and assorted internet sources, assembled for
research use rather than redistribution.

Enforcing the clean-licence requirement (see `sources.md`), the realistic ceiling is:

| Scope | Words | Status |
|---|---|---|
| Strict classical, clean licence | ~7–10M | Available now |
| Historical Latin, clean licence | ~50–150M | Achievable, requires OCR work from public-domain scans |
| Historical Latin including non-commercial sources | ~250–400M | **Excluded by design** |
| Latin BERT scale | ~643M | Not redistributable |

A useful corpus architecture preserves historical period:

```
<CLASSICAL> <LATE> <MEDIEVAL> <NEOLATIN>
```

This allows the model to learn Latin from a larger corpus without pretending that all historical
Latin represents the same linguistic distribution.

## Proposed training strategy

Rather than immediately attempting a large Latin model, begin with a sequence of controlled
experiments. Parameter counts are set by the data available, not by ambition: at ~8.3M tokens of
strict classical Latin, and with repeated-data returns decaying sharply past ~4 epochs, the
classical corpus alone supports a well-trained model of only two or three million parameters. Full
arithmetic in `feasibility.md` §2.

**Latin-0 — ~2–5M parameters**

Purpose: prove the pipeline, establish the evaluation apparatus, and answer the smallest interesting
question — can a transformer trained only on Latin generate convincing Latin?

Collect a clean Classical Latin corpus, build a Latin tokenizer, train from scratch and verify that
loss decreases and recognizable Latin emerges.

This model is **not disposable**. On 8.3M classical tokens, ~2–3M parameters is the well-trained
size; ~5M is defensible as an explicitly overtrained proof at roughly 12 epochs. Pick knowingly and
record which. Its outputs are a real result, not a smoke test.

Train entirely from random initialization. Do **pretraining only**. Don't instruction-tune it.
Memorization must be *measured*, not assumed away: report n-gram overlap between generated samples
and the training set.

**Latin-1 — folded into Latin-0**

Originally a ~25M-parameter classical-only model, then ~10–15M. Against the measured corpus it needs
24–36 epochs, past the point where repetition buys anything. There is no meaningful rung between a
small classical proof and the full-corpus model, so this stage is dropped rather than shrunk into
irrelevance.

Test the base model by giving it beginnings such as:

```
Gallia est omnis divisa in partes...
Senatus populusque Romanus...
```

and, more importantly, novel Latin sentences that don't occur in the training corpus.

The question at this stage is simply: can a small transformer trained exclusively on Latin learn to
generate convincing Latin?

**Latin-2 — ~25M parameters**

Only after the small experiment works should corpus construction become serious.

Build out the remaining clean-licence sources and train **~25M parameters**. This is the ceiling the
licensing decision imposes, and it is deliberate: a 100M+ parameter model would require
non-commercially-licensed material and could not be published.

At today's measured 31M tokens, 25M parameters sits exactly on the 16-epoch line — feasible now, and
easing as the remaining six sources land.

A two-phase training strategy:

**Phase 1 — Latinitas**

Train on the full clean-licence historical corpus across periods, using the period tags above.
Goal: acquire Latin morphology, syntax, vocabulary, discourse and general linguistic structure.

**Phase 2 — Romanitas**

Continue training primarily or exclusively on genuine Classical Roman texts.
Goal: shift the learned distribution toward Classical Latin.

This is analogous to domain adaptation, except both the broad corpus and target corpus remain
Latin — so the purity thesis survives intact. The claim to defend is **"trained only on Latin"**,
not "trained only on Classical Latin".

## Tokenizer experiments

The tokenizer deserves to be treated as part of the research. Instead of automatically adopting a
tokenizer designed for English or multilingual web text, train one specifically for Latin.

Compare, for example:

* character-level tokenization
* ~4K Latin BPE vocabulary
* ~8K Latin BPE vocabulary
* ~16K Latin BPE vocabulary

Latin's morphology makes this especially interesting:

```
amo / amas / amat / amamus / amatis / amant
```

A smaller tokenizer may learn useful recurring stems, endings and morphological structures rather
than simply memorizing complete words. At ~18M tokens the larger vocabularies are likely to be
poorly estimated in the tail, so expect 4–8K to win; 16K is included as a control.

Language Model Builder includes tokenizer-building and inspection tools, making it particularly
suitable for these early comparisons.

Note that **orthographic normalization matters more than vocabulary size** — u/v, i/j, macrons,
ae/æ and editorial punctuation vary wildly across editions, and an unnormalized corpus wastes
vocabulary learning that *uir* and *vir* are the same word. See `feasibility.md` §4.1.

## Corpus work

The difficult part of the project is **not model training but dataset construction**.

Sources are surveyed and licence-checked in `sources.md`. The backbone is Perseus
`canonical-latinLit` and Open Greek and Latin (CC BY-SA 4.0, TEI-XML), with The Latin Library and
Project Gutenberg (public domain) as fill, plus epigraphic and documentary material from EDH, EDCS
and Papyri.info.

Ancient texts themselves are public domain, but modern digital editions, transcriptions, annotations
and critical apparatus can carry licenses or usage restrictions. Every source is checked before
inclusion in the redistributable corpus. Non-commercial sources (Corpus Corporum, LASLA, IntraText)
are **excluded from the published corpus** — they may be kept in a separately tagged shard for
private comparison runs, but nothing that ships depends on them.

Preprocessing requirements — orthographic normalization, deduplication, embedded-Greek policy,
apparatus stripping, and filtering out English translations bundled inside "Latin" sources — are
detailed in `feasibility.md` §4. This is where the project will actually spend its time.

Ideally the resulting dataset becomes a project in itself:

**Open Latin Corpus**

Each document retains metadata such as:

```
author
work
approximate_date
period
genre
source
license
language_variant
```

That provenance later makes unusual experiments possible — and, since no properly-licensed
historical Latin corpus currently exists, the corpus is a contribution in its own right,
independent of the models trained on it.

## The target, revised 2026-08-17

The original plan ended at a base model and said explicitly: *don't optimise for building a useful
chatbot yet*. That "yet" has now arrived — the goal is **usable weights behind a simple chat app**,
and a rerunnable pipeline that absorbs corpus growth.

That changes the destination, and it is worth being exact about what is and is not reachable.

**Not reachable, at any corpus size:** a Latin assistant that answers questions correctly. Knowing
things requires having read about them, and all surviving Latin — even the full 1,500-year tradition —
is orders of magnitude short of what that takes. No amount of engineering fixes this.

**Reachable, and more interesting:** a **Latin interlocutor**. You write Latin; it writes Latin back,
in register, taking its turn. It will not know facts. It will hold a conversational shape. That is an
honest description of the artefact and a better one than a bad assistant.

### The instruction-data problem, and the way through it

There is no Latin instruction-tuning dataset, under any licence. Translating an English one would
compromise the purity thesis: the model would never see English, but its conversational behaviour
would be derived from it.

The corpus solves this itself. **22% of it is already dialogue-shaped**, and 54,222 question→answer
pairs are directly extractable:

| genre | words | dialogue form |
|---|---:|---|
| EPISTOLOGRAPHY | 1,847,671 | letters — real request and response |
| GRAMMAR | 866,918 | catechism; Donatus' *Ars minor* is literally Q&A |
| PHILOSOPHY | 752,682 | Ciceronian dialogue |
| DRAMA | 289,580 | stage turns, with speaker names preserved |

Instruction-tuning on Plautus and Donatus rather than translated chat logs is, as far as we can tell,
new — and it keeps the claim "trained only on Latin" true through the SFT stage, not just
pretraining.

### Revised route

| step | | status |
|---|---|---|
| 0 | Dialogue-SFT proof on the existing model — does turn-taking emerge at all? | in progress |
| 1 | Pipeline driver + the six remaining sources (~24M words, doubles the corpus) | next |
| 2 | LATIN-XXV at 25M, now properly justified rather than 16-epoch aggressive | |
| 3 | Real dialogue SFT on the better base | |
| 4 | Serve it — MLX, safetensors, a laptop | |

Step 0 comes first on the plan's own principle: establish the smallest interesting result before
investing in the corpus work that would serve it.

## What not to do

Don't begin with a 500M parameter model.
Don't mix English explanations or translations into the corpus — including the instruction data.
Don't include non-commercially-licensed sources in anything intended for publication.
Don't promise an assistant. Promise an interlocutor.

~~Don't instruction-tune immediately.~~ Superseded: instruction data now comes from the corpus itself.

First establish the smallest interesting result.

## First milestone

The initial project can therefore be remarkably simple:

1. Assemble a clean Classical Latin corpus from the Tier A sources in `sources.md`.
2. Normalize orthography, deduplicate, strip apparatus and filter non-Latin (`feasibility.md` §4).
3. Measure the actual token count — every estimate here depends on it.
4. Import the corpus into Language Model Builder.
5. Create a Latin-specific tokenizer.
6. Define the evaluation suite *before* training: held-out perplexity on whole withheld works, UD
   treebank probing, memorization / n-gram overlap.
7. Configure a ~2–5M parameter GPT (Latin-0) and train from random initialization.
8. Watch training and validation loss. Save checkpoints at different stages.
9. Sample Latin from each checkpoint; evaluate grammar, vocabulary, memorization and stylistic
   resemblance.
10. Scale to Latin-2 (~25M, full corpus, Latinitas→Romanitas) once the pipeline and eval hold up.
11. Publish the model, tokenizer, corpus manifest and training parameters.

The resulting flagship artifact:

## LATIN-XXV

**25 million parameters. One language. Zero modern languages. Trained from scratch, on an openly
licensed corpus.**

Language Model Builder is a good place to start because it removes most of the ML engineering needed
for this first experiment while still exposing the interesting parts: **the corpus, tokenizer,
architecture, training process and resulting model.**

Once LATIN-XXV works, its results determine whether the next step is a broader historical-Latin
corpus at the same scale, deeper OCR work to push the clean-licence corpus toward its ~150M word
ceiling, or a custom MLX/PyTorch training pipeline with explicit historical-period weighting.

---

## What the model is for, decided on evidence

Added 2026-08-23, after LATIN-XXV was trained twice and put to work. This section
replaces speculation about applications with a rule derived from measurement.

### The rule

Scored on six texts chosen because their answers were known:

| input | perplexity |
|---|---:|
| real held-out Latin | 41.4 |
| **the same words, shuffled** | **106.4** |
| OCR-damaged Latin | 93.4 |
| English | 495.8 |
| keyboard gibberish | 3,406.5 |

The second row is the whole argument. Identical vocabulary, identical word
frequencies, scrambled order — and the model is **2.6× more surprised**. A
frequency table would score those two texts identically. That gap is precisely
what the model adds over counting words, and nothing else it does is unique.

So: **the model earns its keep where word order carries the information, and
loses wherever a frequency table already has the answer.**

Era estimation is the proof of the second half. Scored under each period marker
against 144 author-held-out dated records, it managed 50.7% where always
guessing LATE gets 46.5% and a **unigram word-frequency classifier gets 81.9%**.
It lost by 31 points to counting words, because what separates medieval from
classical Latin is largely *which* words appear, not their order. Late antique
Latin — chronologically between the two — came out near random.

### Where it fits

1. **OCR correction.** The strongest fit and the one that compounds. Choosing
   between `de thesauro` and `de the sauro`, or `rn` against `m`, is impossible
   from frequencies (both readings are real words) and straightforward from
   context. Needs no knowledge of the world. And it feeds the corpus that trains
   the next model: fifteen centuries of Latin sit un-digitised in scans, and
   `sources.md` §3 already names OCR as the route to the ~150M-word ceiling.

2. **Filling gaps in inscriptions.** Same shape — which word fits *here*, given
   the syntax either side. Formulaic epigraphy is where a small model is
   strongest, and the corpus holds half a million inscriptions. **Not literary
   lacunae:** it would produce fluent plausible Latin that is not the lost text,
   and presenting that as reconstruction would mislead.

3. **Corpus quality control.** Already proven. `score.py --audit` found a Lorem
   ipsum stub filed as *De Oratore*, a macaronic German-Latin carol, five copies
   of a treatise on French pronunciation, and 2,206 words of one navigation link
   repeated. Word frequencies would have missed the Lorem ipsum entirely — it
   *is* Latin-ish vocabulary, just in impossible order.

4. **A conversational partner for learners.** The target this plan already
   names, and a genuine fit for a small model, because taking a turn in register
   needs no facts. Register-matching emerged unprompted in the chat proof.

### What its size buys

105 MB, runs on a laptop, no API and no per-call cost, deterministic — and
**provably trained only on openly licensed Latin**, with the corpus reproducible
from pinned sources (demonstrated 2026-08-21: a rebuild three days later
reproduced the corpus to the word and the vocabulary byte for byte). For a
scholarly instrument that is not a footnote. No general-purpose model can state
what it read.

### What it is not for

It **scores, it does not explain** — it can report that a passage is improbable,
never why. It is small even for sequence work, and the corpus supports only
~30M parameters at four epochs, so there is little headroom without more text.
A frontier model would beat it at most of these tasks today, except on the two
things it uniquely has: auditable provenance, and never having seen English.

### And the results may outlast the tool

The most interesting output so far is not an application. It is the
size-versus-schedule ablation, the memorisation curves, and the corpus-cleaning
experiment in which perplexity got *worse* while every behavioural measure got
better. Bounded-corpus learning is under-studied because almost all work in the
field happens where data is effectively infinite. Latin is closed. That is a
rare thing to be able to study.

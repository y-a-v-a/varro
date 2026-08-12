# Feasibility Assessment — Classical Latin LLM

Assessment of `plan.md` against the source survey in `sources.md`. Dated 2026-08-12.

**Verdict: feasible, and the staged structure is right. Two corrections needed.**

1. The parameter counts for Latin-1 are too large for a strict-classical corpus. Latin-0 at ~5M is
   almost exactly the right size; Latin-1 at 25M is 2–6× oversized unless you accept heavy
   repetition.
2. Latin-2 at ~100M parameters and the permissive-licence requirement are in direct conflict.
   **RESOLVED — Path A adopted:** publish everything cleanly, cap the ladder at 25–40M parameters.
   `plan.md` has been updated accordingly.

Everything else — the pipeline, the tool choice, the two-phase Latinitas→Romanitas strategy, the
tokenizer experiments, the "don't do this yet" list — is sound.

---

## 1. Tooling: verified

Language Model Builder is real and does what the plan claims. Checked 2026-08-12:

- Apple Silicon only, macOS 15+ (this machine is on Darwin 25.x, so fine), built on MLX.
- Full pipeline confirmed: tokenizer building → pretraining **from scratch** → sampling → SFT →
  DPO → chat. Pretraining from random init on your own dataset is explicitly supported, which is
  the one non-negotiable requirement of this project.
- Free, no account, no subscription, fully local. By Felix Rieseberg.
- Positioned as an educational tool with an interactive textbook attached.

The "~100–150M params on a few billion tokens in about a week" figure is quoted for an **M5 Max**
specifically. Treat it as optimistic-but-plausible for that chip and scale it to your actual
hardware.

**But the throughput number is close to irrelevant to this project, because you are data-bound, not
compute-bound.** See below. A 25M-parameter model on the entire strict-classical corpus is a matter
of hours, not a week. Compute is not your constraint and you should stop thinking about it.

---

## 2. The data ceiling — the core arithmetic

This is the part of the plan that needs revising. The plan already senses it ("This is probably
insufficient unique data") but doesn't carry the number through to the parameter counts.

**Assumptions** (stated so you can re-run this with your own measurements):

- Strict classical corpus: ~10M words (agrees with `sources.md` §0, which put the realistic
  ceiling at 7–10M).
- Latin BPE at 4–8k vocab: roughly **1.6–2.2 tokens per word**. Latin is heavily inflected and
  words are long, so it fragments more than English. Call it ~1.8.
- → **~15–25M tokens** of strict classical Latin. Use ~18M as the planning number.

**Chinchilla-optimal** is ~20 tokens per parameter:

| Model | Params | Chinchilla-optimal tokens | Available | Shortfall |
|---|---|---|---|---|
| Latin-0 | 5M | 100M | ~18M | 5.5× |
| Latin-1 | 25M | 500M | ~18M | **28×** |
| Latin-2 | 100M | 2.0B | ~18M | **111×** |

You close that gap with epochs. The relevant result is Muennighoff et al. 2023 (*Scaling
Data-Constrained Language Models*): repeating data up to **~4 epochs** is worth nearly as much as
fresh data; value decays sharply after that and is approximately **zero past ~16–40 epochs**.

So the honest budget for a ~18M-token corpus:

- **4 epochs** ≈ 72M effective tokens → supports a Chinchilla-correct model of **~3.5M params**.
- **16 epochs** ≈ pushing it, diminishing hard → supports maybe **~15M params** nominally.

**Conclusions:**

- **Latin-0 at ~5M params is correctly sized.** It is not merely a disposable pipeline test — it is
  approximately the largest *well-trained* model the strict classical corpus supports. Take its
  output seriously.
- **Latin-1 at 25M params, classical-only, is oversized by 2–6×.** It will still train and will
  still produce Latin. But it will be undertrained-per-parameter and will memorise heavily, which
  directly undermines the plan's own stated evaluation ("novel Latin sentences that don't occur in
  the training corpus"). Either drop Latin-1 to ~10–15M params, or feed it the broader historical
  corpus.
- **Latin-2 at 100M requires the historical corpus.** The plan already says this. The arithmetic
  confirms it: you need on the order of 100M+ words to make 100M params defensible.

None of this is fatal. Overtraining a small model on a fixed corpus is a legitimate and interesting
regime — it is closer to how a philologist reads than how GPT-4 was trained. But it should be a
deliberate choice, not an accident, and the memorisation has to be measured rather than hoped away.

---

## 3. The central conflict: permissive licensing vs. 100M parameters

**This is the finding that matters most.**

The plan cites Latin BERT's ~643M-word corpus as evidence that historical Latin gets you to scale.
That figure is correct (642.7M words, Bamman & Burns 2020). **But that corpus is not reproducible
under your licensing criteria.** It draws on Corpus Corporum, the Latin Library, Perseus and
assorted internet sources, and it was assembled for research use, not for redistribution.

From `sources.md`, here is what the licence tiers actually yield:

| Scope | Rough word count | Licence reality |
|---|---|---|
| Strict classical, green-licence | ~7–10M | Clean. CC BY-SA + PD. |
| Historical Latin, green-licence only | ~50–150M *(estimate — needs measuring)* | Achievable but requires real OCR work from Tier C scans |
| Historical Latin, including amber | ~250–400M | Corpus Corporum alone is 226M words — but **non-commercial only**, rights varying per constituent corpus |
| Latin BERT-scale | ~643M | Not redistributable under your criteria |

So the fork is:

- **Path A — strict permissive.** Cap ambitions around 25–40M params. Publish corpus, tokenizer and
  weights with a clean licence. The corpus itself becomes a genuine contribution ("Open Latin
  Corpus" in the plan) because *nobody has assembled a properly-licensed historical Latin corpus.*
- **Path B — maximum scale.** Include Corpus Corporum and LASLA, reach 100M+ params, but the result
  is research-only and cannot be redistributed commercially. ShareAlike/NC contamination also makes
  the weights legally murky (see `sources.md` §0 — untested for model weights).

**DECIDED: Path A.** Ladder capped at 25–40M parameters; everything publishable. Amber material is
kept in a separately-tagged shard so a Path-B variant can be trained privately for comparison
without contaminating the publishable artifact.

Path A is also the more interesting *result* — "here is a properly-licensed Latin corpus and a model
trained only on it" is a contribution; "here is another model trained on scraped text" is not.

---

## 4. Missing from the plan

These are all corpus-construction problems, which the plan correctly identifies as the hard part —
but it under-specifies them. In rough order of how much damage they do if ignored:

### 4.1 Orthographic normalisation (biggest omission)
Latin digital editions vary wildly and inconsistently on:
- **u/v** and **i/j** (*uir* vs *vir*, *iam* vs *jam*)
- **vowel-length macrons** (*āmō* vs *amo*) — present in teaching texts, absent in critical editions
- **ae/æ/e** ligatures and monophthongisation
- capitalisation and modern editorial punctuation (ancient texts had neither)
- consonantal doubling and assimilation conventions (*adfero*/*affero*)

Left unnormalised, your tokenizer will waste vocabulary learning that *uir* and *vir* are the same
word, and your model will learn edition-house style rather than Latin. **This matters more than the
tokenizer vocab-size question the plan spends most of its attention on.** Decide a target
orthography, normalise everything to it, and record the original form in metadata.

### 4.2 Deduplication
Flagged already in `sources.md` §5 but worth repeating here because it interacts with the epoch
budget above: Tesserae, LatinISE, CLTK and Corpus Corporum are all substantially *derived from*
Perseus and the Latin Library. Naive concatenation duplicates Vergil five or six times. Since you
are already planning to repeat data deliberately via epochs, *accidental* duplication corrupts the
one dial you actually control. Dedup must be edition-aware — two genuine editions of the same work
are different texts, but you probably want only one.

### 4.3 Embedded Greek
**Cicero's letters code-switch into Greek constantly.** So do parts of Suetonius, Gellius, and the
technical writers. The plan's purity claim — "the model's entire linguistic universe is Latin" —
does not survive contact with the actual corpus unless you make an explicit decision. Options:
strip Greek passages, keep them, or tag them `<GRAECUM>`. Any is defensible; silently ingesting
them is not, since it makes the headline claim false.

### 4.4 Editorial apparatus and structural noise
TEI-XML sources (Perseus, DigilibLT, OGL) carry this as structured markup you can strip cleanly.
Plain-text sources (Latin Library, Gutenberg) have it inline: line numbers, section numbers, sigla,
`[]` `<>` `†` `...` for lacunae/emendations/cruces, running heads, editor's notes. **This is a
strong argument for preferring the TEI sources as your backbone and treating plain text as fill.**
It is also why the apparatus criticus is worth stripping twice over: it is both the copyrightable
part *and* pure noise.

### 4.5 Translations inside "Latin" sources
Perseus `canonical-latinLit` and several other repos ship English translations alongside the Latin
in the same repository tree. Filter on `@xml:lang` / language codes, not on repo name. Project
Gutenberg's Latin browse tag likewise catches English translations, bilingual editions, and
grammars written in English (see `sources.md` §1). The plan's rule "don't mix English into the
pretraining corpus" requires active filtering, not just source selection.

### 4.6 Evaluation design
"Sample it and see if it looks like Latin" is not enough, and the plan's milestone 9 is vague. A
minimal but real eval suite:
- **Held-out perplexity** on complete works withheld from training (whole works, not random lines —
  random-line holdout leaks context).
- **Syntactic probing** against the UD Latin treebanks (`sources.md` §1) — these are gold-standard
  and are exactly why they're listed as never-mix-into-pretraining.
- **Memorisation check** — n-gram overlap between generated samples and the training set. Given the
  epoch counts above, this is not optional. Report it.
- **Morphological well-formedness** — sample generations, check agreement and case government.
  LASLA's manually-verified annotations are the reference standard here, licence permitting.

### 4.7 No Latin instruction data exists
The plan wisely defers SFT/DPO. Worth noting *why* it will stay hard: there is no instruction-tuning
dataset in Latin under any licence. Writing one yourself, in Latin, is the only purity-preserving
option — and that is a substantial project in its own right. Keep it out of scope.

---

## 5. Recommended adjustments

Keep the staged structure. Change the following:

1. ~~Make the Path A / Path B licensing decision~~ — **done, Path A.** Everything publishable;
   non-commercial sources excluded from anything that ships.
2. **Re-size the ladder** to match the data — applied in `plan.md`:
   - Latin-0: ~5M params, strict classical. Unchanged — and treat its results as real, not disposable.
   - Latin-1: ~10–15M params, strict classical. Accepts mild overtraining; memorization measured.
   - Latin-2: ~25–40M params, clean-licence historical corpus, two-phase Latinitas→Romanitas.
     This is the Path A ceiling. LATIN-XXV sits here.
3. **Add a corpus-preprocessing milestone between steps 1 and 2** of the First Milestone list,
   covering §4.1–4.5 above. This is where the project will actually spend its time.
4. **Specify the eval suite (§4.6) before training Latin-0**, so the pipeline test also tests the
   measurement apparatus. Building the eval after you have a model is how you end up with an eval
   that flatters the model.
5. **Measure the real token count early.** Build the tokenizer on whatever corpus you have after
   two weeks of collection, and count. Every number in §2 is an estimate; one afternoon of actual
   measurement replaces all of it and may change the parameter ladder again.

## 6. On the purity thesis

Worth stating plainly, because it's the interesting part of the project and it survives all of the
above: **the two-phase historical-Latin strategy does not compromise it.** Training Phase 1 on
medieval and neo-Latin is still a model whose entire linguistic universe is Latin. No English, no
translations, no inherited multilingual representations. The `<CLASSICAL> <LATE> <MEDIEVAL>
<NEOLATIN>` period tagging in the plan is the right mechanism — it lets the model use the larger
corpus without flattening 1,500 years into one distribution, and it makes Phase 2 domain adaptation
a matter of conditioning rather than a separate training run.

The claim to defend in the writeup is therefore not "trained only on Classical Latin" but
**"trained only on Latin"** — which is both true and, as far as I can tell, unprecedented at this
scale.

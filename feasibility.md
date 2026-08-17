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

**REVISED 2026-08-15 against measured data.** The original version of this section reasoned from
an estimated ~10M classical words. Three of nine sources are now extracted, dated per work and
deduplicated, so the classical figure is measured rather than assumed — and it is **less than half**
what the plan assumed. The conclusions move accordingly.

### Measured

| | words | notes |
|---|---:|---|
| corpus, post-dedup | 17,256,637 | 908 records, 3 of 9 green sources |
| **CLASSICAL** | **4,636,779** | to AD 200; essentially all of it Perseus |
| LATE | 12,553,096 | DigilibLT + CSEL + late Perseus |
| MEDIEVAL | 79,882 | |

**Token count, now measured too** (`src/tokenizer.py`, byte-level BPE trained on the corpus):

| vocab | tokens/word, all | tokens/word, classical | full corpus | classical |
|---:|---:|---:|---:|---:|
| 2,000 | 2.348 | 2.450 | 38.5M | 10.8M |
| **4,000** | **2.054** | **2.164** | **33.7M** | **9.5M** |
| 8,000 | 1.834 | 1.937 | 30.1M | 8.5M |
| 16,000 | 1.667 | 1.759 | 27.3M | 7.8M |
| 32,000 | 1.538 | 1.620 | 25.2M | 7.1M |

Counted over every record — an earlier version of this table sampled the first 400 documents, which
in file order contained no Perseus at all and so measured late-antique prose only. The ~1.8 planning
figure was still close: it lands between the 8k and 16k vocabularies. Classical Latin runs ~5%
*higher* fertility than the corpus average, being older, more varied and more heavily verse. At 4k:

- classical → **9.5M tokens**
- full corpus → **33.7M tokens**

Both slightly *above* the earlier estimate, which loosens the ladder a little rather than tightening
it. Character-level, for reference: 6.91 chars/word, 113.3M characters.

### What that supports

Chinchilla-optimal is ~20 tokens per parameter. The gap is closed with epochs, and the constraint on
epochs is Muennighoff et al. 2023 (*Scaling Data-Constrained Language Models*): repeating up to
**~4 epochs** is worth nearly as much as fresh data, value decays sharply after, and is
approximately zero past ~16–40.

| epochs | classical only (9.5M tok) | full corpus (33.7M tok) |
|---|---|---|
| 4 (safe) | 38M eff. → **1.9M params** | 135M eff. → **6.7M params** |
| 8 | 76M eff. → **3.8M params** | 270M eff. → **13.5M params** |
| 16 (aggressive) | 152M eff. → **7.6M params** | 539M eff. → **27.0M params** |

### Conclusions, revised

- **The classical corpus is 4.6M words, not 10M.** Perseus is the canonical classical source and is
  substantially complete for the canonical authors, so this is not a gap that another TEI repository
  closes. Reaching ~10M means the Latin Library (which overlaps Perseus heavily — dedup will eat
  most of the nominal gain) plus inscriptions and papyri, and those are a *different register*:
  formulaic, abbreviated, epigraphic. Worth having, but not interchangeable with literary prose.
- **Latin-0 at ~5M params on classical-only needs ~10 epochs.** Past the comfortable range. Either
  accept it as an explicitly overtrained pipeline proof — which is what the plan says it is — or
  size it at **~2–4M params**, the well-trained range for 9.5M tokens.
- **Latin-1 at 10–15M params classical-only is no longer defensible**: 24–36 epochs. Either drop it
  to ~5M params, or train it on the full corpus with period tags, which is what Latin-2 does anyway.
  The honest move is probably to **collapse Latin-1 into Latin-0** and go straight from a small
  classical proof to the full-corpus model.
- **LATIN-XXV at 25M params sits just inside the 16-epoch line** — 27.0M by the measured numbers,
  a little more headroom than the earlier (under-measured) figure suggested. The remaining six sources have not landed yet,
  so it stays feasible and gets more comfortable as they do.
- **Vocabulary size is a parameter-budget decision, not a compression one.** At d_model=256 the
  transformer body is ~4.7M parameters regardless of vocabulary, while the embedding table is
  vocab×256. A 32k vocabulary costs 8.2M parameters — larger than the entire model it serves, 63%
  of the total. At 4k it is 18%. For a model this small the vocabulary must be small, and the
  compression gain from a larger one is not worth spending the budget on.
- The 25–40M Path A ceiling from §3 still holds. Nothing here changes the licensing decision.

Overtraining a small model on a fixed corpus remains a legitimate and interesting regime — closer to
how a philologist reads than how GPT-4 was trained. But it should be a deliberate choice, and the
memorisation has to be measured rather than hoped away. That is what makes the evaluation suite
(§4.6) load-bearing rather than optional.

---

## 3. Two tracks: what you may train on vs. what you may redistribute

**REVISED 2026-08-15.** This section originally posed a single choice — publish cleanly at ~25–40M
parameters, or reach 100M with material that could not be redistributed. That framing was wrong in
one respect: it treated "may I train on this?" and "may I redistribute this?" as the same question.
They are not, and separating them dissolves most of the conflict.

The project is now explicitly a **non-commercial research experiment with open-source code**. Under
that framing:

### Track A — redistributable

Free-licence sources only. This is the published corpus and the published weights. It remains the
contribution: **a properly licensed Latin corpus does not currently exist**, and building one is
worth doing independently of any model trained on it.

Currently 17.25M words; ~39M when the remaining six green sources land. Supports ~25M parameters.

### Track B — research only

Adds the non-commercial sources. Never redistributed; the code, the results and the provenance are
published instead.

| | words |
|---|---:|
| Track A, all green sources landed | ~39M |
| Corpus Corporum | 226M |
| IntraText | 10M |
| LASLA | 1.6M |
| **Track B total** | **~277M** |

At the measured ~2 tokens/word that is roughly **550M tokens** — 4 epochs gives 2.2B effective
tokens, which supports a **~110M parameter model** at Chinchilla-optimal ratios, no overtraining
games. The 100M-parameter ambition of the original plan is restored, and by a comfortable margin.

Two riders on that number. Corpus Corporum is ~80% of it and is mostly medieval and neo-Latin, so
Track B's period balance is very different from Track A's. And its 226M is the platform's headline
figure, not a measurement of what is actually retrievable.

### The legal basis, and its limit

Training and redistribution sit under different rules.

- **EU.** Directive 2019/790 Art. 3 permits text and data mining for scientific research on
  lawfully accessed works by research organisations, and Art. 7(1) makes that right
  **unoverridable by contract** — a licence term forbidding TDM does not bind it. Art. 4 extends a
  general TDM exception to everyone, subject to rightsholder opt-out.
- **US.** No explicit exception, but non-commercial research is the strongest fair-use posture
  there is (*HathiTrust*, *Authors Guild v. Google*).

The limit matters as much as the permission: **Art. 3 covers making and retaining copies for
mining. It does not authorise redistributing the corpus.** That is precisely why Track B exists as a
separate track rather than as a bigger Track A. Whether a trained model is a derivative of its
training data is unsettled everywhere; publishing Track B *weights* would be taking a position on
that question, so the default is not to.

### What this changes, and what it does not

- **Changed:** the ~25M ceiling is no longer a ceiling on the *project*, only on what is published.
- **Changed:** LASLA becomes available, and it is the best Latin gold standard in existence —
  1.6M words of hand-verified morphology. As held-out evaluation data it is never redistributed, so
  its non-commercial terms cost nothing at all there.
- **Unchanged:** Track A's discipline. The guard is now track-aware rather than absolute, but
  `fetch.py`, `build.py` and `provenance.py --check` all still refuse non-green material in a
  Track A build.
- **Unchanged:** tier `unknown` enters neither track. A licence nobody has read is not a permissive
  licence; silence is not permission. Musisque Deoque — the complete corpus of ancient Latin poetry
  with apparatus — sits there, and is the most valuable outstanding permission request in the
  project.

### The comparison is itself a result

Same pipeline, same tokenizer, two corpora an order of magnitude apart, on a language whose total
surviving text is bounded. That is a cleaner data-scaling experiment than most, and it is available
essentially for free once both tracks exist.

---

## 4. Missing from the plan

These are all corpus-construction problems, which the plan correctly identifies as the hard part —
but it under-specifies them. In rough order of how much damage they do if ignored.

**Status as of 2026-08-15:** 4.2–4.5 are implemented and measured; 4.1 is in progress; 4.6–4.7
remain open. Each subsection below records what was actually found, which in several cases was
worse or stranger than anticipated.

### 4.1 Orthographic normalisation (biggest omission) — DONE
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

*Confirmed in the data.* The very first text extracted, Plautus' *Amphitruo*, opens
`Mercvrivs / Vt vos in vostris voltis mercimoniis` — three normalisation decisions in two lines.
`dedupe.py` already has to fold u/v and i/j merely to *compare* editions: without that fold,
Perseus's `uiri` and CSEL's `viri` are different words and cross-source similarity collapses. That
fold is comparison-only; normalisation proper is `src/normalize.py`.

*Implemented, and the measurement justified the priority.* The sources did not merely differ, they
used opposing conventions: v as a share of (u+v) was 12.86% in Perseus and 11.46% in DigilibLT but
**0.60% in CSEL**, which writes classical `uir`/`uita` throughout. All three are now 0%. Folding is
towards `u` because the reverse needs morphology to know which u is consonantal. Also removed:
40,672 editorial marks, 3,813 line-break hyphenations, 12,007 metrical ictus accents in Plautus,
3,256 macrons/breves, 667 orphaned spacing diacritics and 233 homoglyphs (mostly Cyrillic, each of
which silently split a word into two unrelated tokens). The result is a **48-letter alphabet** —
a-z and A-Z less v/V and j/J — which is precisely classical Latin's inventory.

### 4.2 Deduplication — DONE
Flagged already in `sources.md` §5 but worth repeating here because it interacts with the epoch
budget above: Tesserae, LatinISE, CLTK and Corpus Corporum are all substantially *derived from*
Perseus and the Latin Library. Naive concatenation duplicates Vergil five or six times. Since you
are already planning to repeat data deliberately via epochs, *accidental* duplication corrupts the
one dial you actually control. Dedup must be edition-aware — two genuine editions of the same work
are different texts, but you probably want only one.

*Implemented in `src/dedupe.py`.* Content-based (bottom-k sketches of 5-gram hashes), not
title-based, because titles diverge across projects and — more dangerously — the same title can hide
different texts. 59 records / 884,721 words (4.9%) marked; 0 of 59 fall below the 15% verbatim
floor — which establishes that each duplicate's text really does occur in its keeper, not that
discarding the whole record was right. That needs cluster-by-cluster human review, still undone.
Two findings worth carrying forward: the overlap is not only cross-source (Celsus' *De Medicina*
appears three times *inside* Perseus, in three different editions), and part/whole containment is as
common as equivalence (Historia Augusta lives inside the whole, individual panegyrics inside
*Panegyrici Latini*, Perseus's selection of Jerome's letters inside CSEL's complete corpus).

### 4.3 Embedded Greek — DONE
**Cicero's letters code-switch into Greek constantly.** So do parts of Suetonius, Gellius, and the
technical writers. The plan's purity claim — "the model's entire linguistic universe is Latin" —
does not survive contact with the actual corpus unless you make an explicit decision. Options:
strip Greek passages, keep them, or tag them `<GRAECUM>`. Any is defensible; silently ingesting
them is not, since it makes the headline claim false.

*Implemented as `extract.py --greek drop|tag|keep`, default drop.* Residual Greek is now 0
characters across all three sources, down from 66,458. The systematic case was not the one
anticipated: not Cicero's letters but the **grammarians and commentators** — Priscian, Charisius,
Servius, Macrobius — quoting Greek technical vocabulary inline inside Latin sentences
(`quid sit, περὶ τῆς οὐσίας, id est de substantia`), 12,572 tokens across 163 of DigilibLT's 373
works. Markup-level filtering missed all of it; a token-level pass was required.

### 4.4 Editorial apparatus and structural noise — DONE for TEI sources
TEI-XML sources (Perseus, DigilibLT, OGL) carry this as structured markup you can strip cleanly.
Plain-text sources (Latin Library, Gutenberg) have it inline: line numbers, section numbers, sigla,
`[]` `<>` `†` `...` for lacunae/emendations/cruces, running heads, editor's notes. **This is a
strong argument for preferring the TEI sources as your backbone and treating plain text as fill.**
It is also why the apparatus criticus is worth stripping twice over: it is both the copyrightable
part *and* pure noise.

*Implemented.* `<note>` (115,202 in Perseus, 75,125 in CSEL), `<app>`/`<rdg>`/`<lem>`, `<del>`,
`<gap>` and `<bibl>` are dropped; `<choice>` resolves to `<corr>` over `<sic>` and `<reg>` over
`<orig>`, each counted and reported. Residual apparatus survives only where the *source* has it as
bare untagged text — 9 occurrences in 5.6M words of Perseus, left alone deliberately rather than
overfitted to. The plaintext sources (Latin Library, Gutenberg) will need their own treatment.

### 4.5 Translations inside "Latin" sources — DONE
Perseus `canonical-latinLit` and several other repos ship English translations alongside the Latin
in the same repository tree. Filter on `@xml:lang` / language codes, not on repo name. Project
Gutenberg's Latin browse tag likewise catches English translations, bilingual editions, and
grammars written in English (see `sources.md` §1). The plan's rule "don't mix English into the
pretraining corpus" requires active filtering, not just source selection.

### 4.6 Evaluation design — HARNESS BUILT, model-facing parts open
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

### 4.7 No Latin instruction data exists — REOPENED 2026-08-17, and solvable
Originally filed as out of scope: there is no instruction-tuning dataset in Latin under any licence,
and writing one by hand is a project in itself. Translating an English set would preserve the letter
of the purity claim (the model still sees no English) while breaking its spirit — the conversational
behaviour would be English behaviour wearing Latin.

**The third option was missed: mine the dialogue the corpus already contains.** Latin has an
unusually rich conversational literature, and 22% of the corpus is dialogue-shaped —
**54,222 question→answer pairs are directly extractable**, concentrated in theology (12,709),
letters (6,357), drama (6,134), philosophy (3,531) and grammar (3,087).

Drama is the highest-quality signal because `<speaker>` markup survived extraction, so turns carry
their speaker:

    Staphyla   Nam cur me miseram uerberas?
    Euclio     Ut misera sis atque ut te dignam mala malam aetatem exigas.

Grammar is the most instruction-like: Donatus' *Ars minor* is a catechism, question and answer, for
its entire length — a second-century Latin teacher's instruction dataset, written for exactly this
purpose fourteen hundred years early.

**Scale honesty.** SFT changes behaviour, not knowledge. A 25M-parameter model fine-tuned on this
will take conversational turns in grammatical Latin and will not know anything. The deliverable is an
interlocutor, not an assistant, and the writeup should say so plainly rather than let a demo imply
otherwise.

---

## 5. Recommended adjustments

Keep the staged structure. Change the following:

1. ~~Make the Path A / Path B licensing decision~~ — **superseded.** The project now runs two
   tracks (§3): A redistributable, B research-only. Track membership is derived from the licence
   tier and enforced in three places, and `provenance.py --check` audits it after the fact.
2. **Re-size the ladder** to match the data. Revised again 2026-08-15 now that the classical
   corpus is measured at 4.6M words rather than the assumed 10M:
   - Latin-0: **~2–3M params** on classical-only is the well-trained size; ~5M is acceptable as an
     explicitly overtrained pipeline proof (~12 epochs). Either is defensible — choose knowingly.
   - Latin-1: **collapse into Latin-0.** At 10–15M params classical-only it needs 24–36 epochs,
     which is past the point where repetition buys anything. There is no longer a meaningful rung
     between the small classical proof and the full-corpus model.
   - Latin-2 / LATIN-XXV: ~25M params on the full clean-licence corpus, two-phase
     Latinitas→Romanitas. Sits exactly at the 16-epoch line today and eases as the remaining six
     sources land. Still the Path A ceiling.
3. **Add a corpus-preprocessing milestone between steps 1 and 2** of the First Milestone list,
   covering §4.1–4.5 above. This is where the project will actually spend its time.
4. **Specify the eval suite (§4.6) before training Latin-0**, so the pipeline test also tests the
   measurement apparatus. Building the eval after you have a model is how you end up with an eval
   that flatters the model.
5. ~~Measure the real token count early.~~ **Done.** Word counts moved the classical figure from
   ~10M to 4.6M; the tokenizer then settled words→tokens at 2.054 (4k vocab), against the assumed
   1.8. Nothing in §2 is an estimate any more. **Use a 4,000-token vocabulary** — it forces
   morphological decomposition (0 of 12 paradigm forms memorised, against 7 of 12 at 32k) and keeps
   embeddings to 18% of the parameter budget.
6. **Do not add the Latin Library before normalisation and dedup are both proven.** It is the
   largest remaining source (~15M words) and the one that overlaps Perseus most heavily; adding it
   first would have made every other problem harder to see. Adding it now, with dedup verified and
   normalisation in place, is a much cheaper operation.

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

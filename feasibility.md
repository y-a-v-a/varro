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
| corpus, post-dedup | 17,271,153 | 908 records, 3 of 9 green sources |
| **CLASSICAL** | **4,638,175** | to AD 200; essentially all of it Perseus |
| LATE | 12,553,096 | DigilibLT + CSEL + late Perseus |
| MEDIEVAL | 79,882 | |

Still **estimated**: token count. Latin is heavily inflected and fragments more than English, so a
4–8k Latin BPE runs about **1.6–2.2 tokens per word**; ~1.8 is the planning figure. This is the last
unmeasured number in the chain and `normalize.py` → tokenizer will settle it.

- classical → **~8.3M tokens**
- full corpus → **~31M tokens**

### What that supports

Chinchilla-optimal is ~20 tokens per parameter. The gap is closed with epochs, and the constraint on
epochs is Muennighoff et al. 2023 (*Scaling Data-Constrained Language Models*): repeating up to
**~4 epochs** is worth nearly as much as fresh data, value decays sharply after, and is
approximately zero past ~16–40.

| epochs | classical only (8.3M tok) | full corpus (31M tok) |
|---|---|---|
| 4 (safe) | 33M eff. → **1.7M params** | 124M eff. → **6.2M params** |
| 8 | 67M eff. → **3.3M params** | 249M eff. → **12.4M params** |
| 16 (aggressive) | 134M eff. → **6.7M params** | 498M eff. → **25M params** |

### Conclusions, revised

- **The classical corpus is 4.6M words, not 10M.** Perseus is the canonical classical source and is
  substantially complete for the canonical authors, so this is not a gap that another TEI repository
  closes. Reaching ~10M means the Latin Library (which overlaps Perseus heavily — dedup will eat
  most of the nominal gain) plus inscriptions and papyri, and those are a *different register*:
  formulaic, abbreviated, epigraphic. Worth having, but not interchangeable with literary prose.
- **Latin-0 at ~5M params on classical-only now needs ~12 epochs.** That is past the comfortable
  range. Either accept it as an explicitly overtrained pipeline proof — which is what the plan says
  it is — or size it at **~2–3M params**, which is the well-trained size for 8.3M tokens.
- **Latin-1 at 10–15M params classical-only is no longer defensible**: 24–36 epochs. Either drop it
  to ~5M params, or train it on the full corpus with period tags, which is what Latin-2 does anyway.
  The honest move is probably to **collapse Latin-1 into Latin-0** and go straight from a small
  classical proof to the full-corpus model.
- **LATIN-XXV at 25M params is exactly at the 16-epoch line for the current corpus** — and the
  remaining six sources have not landed yet. It stays feasible, and gets more comfortable as they do.
- The 25–40M Path A ceiling from §3 still holds. Nothing here changes the licensing decision.

Overtraining a small model on a fixed corpus remains a legitimate and interesting regime — closer to
how a philologist reads than how GPT-4 was trained. But it should be a deliberate choice, and the
memorisation has to be measured rather than hoped away. That is what makes the evaluation suite
(§4.6) load-bearing rather than optional.

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
but it under-specifies them. In rough order of how much damage they do if ignored.

**Status as of 2026-08-15:** 4.2–4.5 are implemented and measured; 4.1 is in progress; 4.6–4.7
remain open. Each subsection below records what was actually found, which in several cases was
worse or stranger than anticipated.

### 4.1 Orthographic normalisation (biggest omission) — IN PROGRESS
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
fold is comparison-only; normalisation proper is the next step. CSEL adds a second requirement —
unrejoined hyphenation across line breaks (`de the- sauro` for *thesauro*).

### 4.2 Deduplication — DONE
Flagged already in `sources.md` §5 but worth repeating here because it interacts with the epoch
budget above: Tesserae, LatinISE, CLTK and Corpus Corporum are all substantially *derived from*
Perseus and the Latin Library. Naive concatenation duplicates Vergil five or six times. Since you
are already planning to repeat data deliberately via epochs, *accidental* duplication corrupts the
one dial you actually control. Dedup must be edition-aware — two genuine editions of the same work
are different texts, but you probably want only one.

*Implemented in `src/dedupe.py`.* Content-based (bottom-k sketches of 5-gram hashes), not
title-based, because titles diverge across projects and — more dangerously — the same title can hide
different texts. 60 records / 887,619 words (4.9%) marked, verbatim-verified at 0 false positives.
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

### 4.6 Evaluation design — OPEN, and now the critical path
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

### 4.7 No Latin instruction data exists — OPEN, deliberately out of scope
The plan wisely defers SFT/DPO. Worth noting *why* it will stay hard: there is no instruction-tuning
dataset in Latin under any licence. Writing one yourself, in Latin, is the only purity-preserving
option — and that is a substantial project in its own right. Keep it out of scope.

---

## 5. Recommended adjustments

Keep the staged structure. Change the following:

1. ~~Make the Path A / Path B licensing decision~~ — **done, Path A.** Everything publishable;
   non-commercial sources excluded from anything that ships.
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
5. ~~Measure the real token count early.~~ **Partly done, and it did change the ladder.** Word
   counts are now measured (§2) and moved the classical figure from ~10M to 4.6M. The remaining
   estimate is words→tokens (~1.8), which the tokenizer settles. Build it as soon as
   `normalize.py` lands and re-run §2 — the ladder may move once more.
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

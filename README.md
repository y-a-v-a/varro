# Claudius — an openly licensed Latin corpus and a language model trained only on Latin

Two deliverables, in order:

1. **Open Latin Corpus** — a reproducible, properly licensed corpus of Latin. No such thing
   currently exists, which makes it a contribution independent of anything trained on it.
2. **LATIN-XXV** — a 27.49M parameter GPT trained from random initialization on that corpus.
   One language. Zero modern languages. No inherited multilingual representations.
   **Trained 2026-08-19**: held-out perplexity 29.8 against a 149.8 bigram floor
   (5.03×), and 0 of 1,339 twelve-word windows appear anywhere in its training
   text. See [`CURRENT_STATE.md`](CURRENT_STATE.md) for the full result.

Published weights: **[LATIN-VII on Hugging Face](https://huggingface.co/vincent-b/latin-vii)**
— 7.28M parameters, classical Latin only, the one model in the ladder whose
training corpus is entirely free of NonCommercial terms, released CC-BY-SA-4.0.

Planning documents:

| File | What it is |
|---|---|
| [`plan.md`](plan.md) | The project plan — goals, model ladder, training strategy |
| [`feasibility.md`](feasibility.md) | Assessment of that plan: the data arithmetic, the licensing decision, what the plan misses |
| [`sources.md`](sources.md) | Human-readable survey of every Latin text source, licence-checked |
| [`manifest/sources.yaml`](manifest/sources.yaml) | Machine-readable version of the above — drives the pipeline |
| [`manifest/authorities.yaml`](manifest/authorities.yaml) | Author/work dating authority — assigns period per work |
| [`docs/explainer.html`](docs/explainer.html) | What the trained model actually is, as files on disk |
| [`docs/capabilities.html`](docs/capabilities.html) | What the model can and cannot do, measured |
| [`site/`](site/README.md) | The public website, deployable to Vercel; its figures are generated from the manifest and reports |

## Quickstart

Python 3.14 — not incidental. Annotations are evaluated lazily there and eagerly
before it, and the scripts are written against that. MLX is Apple Silicon only, so
`requirements.txt` marks it as such: the corpus half of this repo installs and runs
anywhere, and only the training half needs a Mac.

```bash
python3.14 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/fetch.py --list          # what's declared, and under what licence
python src/fetch.py --validate      # check the manifest is sound
python src/fetch.py --dry-run       # what would be fetched
python src/fetch.py                 # Track A — redistributable sources
python src/fetch.py --track B       # research build, NOT redistributable
python src/fetch.py --only perseus-canonical-latinLit

python src/provenance.py --check    # audit provenance; non-zero exit on drift
```

Perseus alone is ~175 MB / 1,361 files and clones in about ten seconds (blobless partial clone).

## Three invariants

**1. `data/raw/` is immutable.** Nothing in this repo edits a fetched file, and neither should you.
Every correction belongs downstream, in code under `src/`. The moment you hand-patch a downloaded
text, the corpus stops being reproducible and the licence trail breaks.

**2. Two tracks, and Track A stays clean.** "May I train on this?" and "May I redistribute this?"
are different questions. **Track A** draws only on free-licence sources and is what gets published —
corpus and weights. **Track B** adds non-commercial sources for research and is never redistributed;
code, results and provenance are published instead. Track membership is *derived* from the licence
tier, never asserted by hand, and enforced in three places: `fetch.py`, `build.py`, and
`provenance.py --check`. See [`feasibility.md`](feasibility.md) §3.

**3. Provenance is a standing obligation.** Every build records which sources it drew on, under
which licence, in what proportion. `python src/provenance.py --check` regenerates
[`ATTRIBUTION.md`](ATTRIBUTION.md) and exits non-zero if the corpus disagrees with the manifest, if a
Track A build carries non-redistributable material, or if the notice has drifted. Run it after any
manifest change and before publishing anything.

Each corpus needs its own notice, so run it twice. `--corpus corpus-b` writes
[`ATTRIBUTION-corpus-b.md`](ATTRIBUTION-corpus-b.md) for Track B, which credits the two
NonCommercial sources LATIN-LII trains on. NonCommercial restricts what the result may be used
*for*; it does not excuse crediting anyone, and the BY in LASLA's CC BY-NC-SA is the same BY as
anywhere else.

## Layout

```
manifest/
  sources.yaml        # authoritative: what may enter the corpus, under what licence
  fetch.lock.json     # resolved commit SHAs + checksums — committed, makes builds reproducible
src/
  fetch.py            # manifest -> data/raw/                     [done]
  extract.py          # -> data/interim/<source>.jsonl            [done: 3 of 9 sources]
                      #    Greek + English filtering live here, not in a
                      #    separate pass — they need the markup, which only
                      #    exists at extraction time.
  enrich.py           # authorities.yaml -> date + period         [done]
  normalize.py        # orthography: u/v, i/j, marks, hyphenation [done]
  dedupe.py           # content-based near-dup detection          [done]
  build.py            # -> data/processed/corpus.{train,validation} [done]
  tokenizer.py        # Latin BPE sweep + fertility measurement    [done]
  score.py            # the model as an instrument: audit, date, rank [done]
  provenance.py       # ATTRIBUTION.md + track audit gate          [done]
  evaluate.py         # baselines, memorisation, versioned reports [done]
  train.py            # MLX GPT from random init                  [done]
  sample.py           # generation, for memorisation scoring      [done]
  bundle.py           # reports -> dist/<name>/ Hugging Face bundle [done]
configs/
data/                 # gitignored — a build artifact, not source
  raw/ interim/ processed/
```

## Corpus record format

One JSON object per document, JSONL. Metadata travels *with* the text — a side index drifts out of
sync the moment anything is re-split.

```json
{
  "id": "perseus/phi0119.phi001",
  "text": "Vt vos in vostris voltis mercimoniis...",
  "author": "Plautus",
  "work": "Amphitruo",
  "approximate_date": -195,
  "period": "CLASSICAL",
  "genre": "comedy",
  "source": "perseus-canonical-latinLit",
  "source_version": "4620cf8ab32865b127dce42a61e80c58b4c06de9",
  "license": "CC-BY-SA-4.0",
  "edition": "Teubner 1895",
  "language_variant": "lat-cls",
  "normalization": "v1",
  "n_words": 8412
}
```

The `<CLASSICAL>` / `<LATE>` / `<MEDIEVAL>` / `<NEOLATIN>` control tokens from `plan.md` are **not**
corpus fields. They are injected at tokenization time from the `period` field, so the tagging scheme
can change without rebuilding the corpus.

## Adding a source

1. Add an entry to `manifest/sources.yaml`. Licence, tier and **format** are mandatory; `amber`
   additionally requires an `exclude_reason`.
2. `python src/fetch.py --validate` — the validator enforces the Path A guard, so an `amber` source
   marked `include: true` is a hard error rather than a quiet contamination.
3. `python src/fetch.py --only <id>`.
4. `python src/pipeline.py` — **only the new source is extracted.** Extract is partitioned per
   source, so its parts are invalidated independently; everything downstream then rebuilds once,
   because the corpus did change.
5. Commit the updated `fetch.lock.json`.

**If the format already has an extractor, that is the whole job — no code.** Dispatch is keyed on
the `format` field, not the source id, because sources are added far more often than formats are.
Perseus and CSEL already share one reader; the two epigraphic sources still to land are both EpiDoc,
and Latin Library and Gutenberg are both plain text, so the four remaining sources need two
extractors between them.

Format labels are as precise as the parsing requires — `tei-capitains` is not `tei-xml`, because
Corpus Corporum ships TEI with a different structure. A new format means adding one function to
`FORMAT_EXTRACTORS` at the foot of `extract.py`.

One guard worth knowing about: `extractable_sources()` refuses the `eval` tier. The UD treebanks are
CoNLL-U exactly like DigilibLT, so dispatching on format alone would extract held-out evaluation
data straight into the training corpus.

## Corpus as measured

Eight sources extracted (seven green, one amber); every green source now has a
reader. **Track A: 10,678 records, 93,849,539 words, 180.9M tokens.** Track B adds LASLA: 10,452 records,
90,835,090 words, 175.3M tokens; see [`CURRENT_STATE.md`](CURRENT_STATE.md) for the full breakdown and
[`reports/`](reports/) for the generated figures.

| Source | records | words (pre-dedup) | vs. estimate |
|---|---:|---:|---|
| vicifons | 11,753 | 86,368,214 | 3.0M — a 29× miss |
| digiliblt | 373 | 6,509,980 | 4.0M |
| opengreekandlatin-csel | 271 | 6,036,274 | 5.0M |
| perseus-canonical-latinLit | 324 | 5,595,104 | 6.0M |
| edcs-sdam | 267 | 4,356,994 | 3.0M |

Vicifons overshot because Latin Wikisource hosts Migne's *Patrologia Latina* at
scale. Dedup removes **29.8M words (27.4%)**, far above the 4.9% seen at three
sources, because Wikisource re-hosts much of what Perseus and CSEL supply.

Editorial interventions, counted and reported by every run:

| | Perseus | DigilibLT | CSEL |
|---|---:|---:|---:|
| `<note>` dropped | 115,202 | — | 75,125 |
| `<choice>` → `<corr>` / `<reg>` | 5,587 / 110 | — | — |
| Greek passages | 4,638 | 5,366 | 3,371 |
| Greek inline runs stripped | 59w | 12,572w | 471w |
| English paragraphs dropped | 38 | 39 | 16 |

Largest works: Livy *Ab urbe condita* 515,204 · Pliny *Naturalis Historia*
401,353 · Quintilian *Institutio Oratoria* 172,362.

## Dating and period

`src/enrich.py` applies [`manifest/authorities.yaml`](manifest/authorities.yaml)
to set `approximate_date` and `period` **per work** rather than per source.

```bash
python src/enrich.py --report    # coverage, writes nothing
python src/enrich.py --audit     # unresolved authors + aliases matching nothing
python src/enrich.py             # rewrite data/interim/*.jsonl in place
```

| period | words | |
|---|---:|---|
| LATE | 13,174,924 | 72.6% |
| **CLASSICAL** | **4,903,966** | **27.0%** |
| MEDIEVAL | 79,882 | 0.4% |

Resolved 84.2% by author entry, 10.6% by work entry (anonymous but datable
texts, chiefly the Justinianic legal corpus), 5.2% by source prior. Nothing is
unassigned, and no alias in the authority file matches nothing.

**What the per-work dating changed.** 14.7% of Perseus — 822,468 words — is not
classical: Tertullian, Jerome, Ammianus, Boethius, Claudian, Prudentius,
Ausonius, Minucius Felix, the *Historia Augusta*. It cuts the other way too:
Gaius and Festus arrive via DigilibLT, whose prior is LATE, but are 2nd century
and so classical. Two records are *later* than any prior would suggest —
Ps.-Boethius (a 12th c. translation) and Remigius of Auxerre (Carolingian) —
which is why `perseus-canonical-latinLit` has no source prior at all: guessing
is the bug the file exists to fix.

**Confidence is carried, not hidden.** 73.9% of the corpus is dated high
confidence, 8.2% medium, 12.7% low, 5.2% inferred from a source prior. Every
record carries `date_source` and `date_confidence`, so a training run can filter
on them. The single most consequential judgement call is Tertullian (557k words,
floruit c.205), who falls just past the AD 200 cutoff; the entry says so.

So the corpus available for Latin-0 / Latin-1 is **4.9M classical words**, of
which 4.66M are high-confidence dated. The remaining 13.3M is Latin-2 Phase 1
(*Latinitas*) material.

## Text provenance

Every source declares **how its text came to be text**, and every record carries it:

| class | meaning |
|---|---|
| `transcribed` | keyed or proofread by people |
| `ocr` | machine-read, possibly machine-corrected by a third party |
| `model-corrected` | corrected by a model from *this* project |
| `unknown` | not established |

This is not a licence question and not a quality one. It exists because training a
model on text an earlier model corrected is a feedback loop, and repeated without
guardrails it narrows the corpus toward whatever the model already believed — the
symptom being that everything looks cleaner while growing less various.

**The corpus already contains OCR-derived text.** Open Greek and Latin's CSEL is
machine-corrected OCR — about 6M words, there since the third source landed — so
labelling makes *"what does this model look like without OCR-derived text?"* a
question that can be answered rather than argued about. The Latin Library carries
`unknown` because `sources.md` records its provenance as inconsistent, and a source
nobody has established should say so rather than claim to be transcribed.

```bash
python src/build.py --report                          # the mix, always printed
python src/build.py --exclude-provenance model-corrected
python src/build.py --max-share model-corrected=0.15  # admit it in a proportion
```

`text_provenance` is **required** in the manifest with no default: a source that does
not say how its text was produced fails the validator, because assuming it was
human-transcribed is exactly the assumption that would let model-corrected text in
unnoticed. A record with no provenance counts as `unknown`, never as transcribed.

Capping is deterministic and reported: whole records are dropped smallest-first, so
the cap is met by shedding the least text, and the build says how much it shed. The
caps are solved together rather than in sequence — one class's budget counts the
others in its denominator, so applying them one at a time never converges.

## Deduplication

`src/dedupe.py` finds works repeated across and within sources — Tertullian's
*Adversus Marcionem* is in both Perseus and CSEL, Boethius' *Consolatio* in
three places, Celsus' *De Medicina* three times inside Perseus alone.

```bash
python src/dedupe.py --report    # clusters and what would be dropped
python src/dedupe.py             # mark is_duplicate / duplicate_of
python src/dedupe.py --verify    # verbatim-check every mark, both tracks
python src/dedupe.py --jobs 1    # sketch in one process (default: every core)
```

Detection is **content-based, not title-based** — titles vary across projects
(*Epitoma de Tito Liuio* / *Epitome Rerum Romanorum*), and conversely the same
title can hide different texts. Text is folded for comparison only (lowercased,
u/v and i/j merged); without that, Perseus's `uiri` and CSEL's `viri` look like
different words. Documents reduce to bottom-k sketches of 5-gram hashes;
Jaccard is estimated from sketch overlap and containment derived from it.

**5,297 records, 35,845,563 words (27.4%) marked as duplicates. Surviving
corpus: 10,782 records, 94,923,570 words — of which 5,055,913 are CLASSICAL.**

Two relationships hide in the clusters and need opposite rules:

- **equivalence** — same work, several sources, near-identical length. Source
  priority decides, since what differs is text quality.
- **containment** — one record is part of a fuller one. The *larger* must win
  regardless of source. Getting this backwards kept Perseus's 43k-word
  selection of Jerome's letters and discarded CSEL's complete 290k-word corpus.

`--verify` is the guard against trusting an estimate: it samples 40 random
8-word spans from each duplicate and tests literal membership in its keeper, for
each track in turn — Track A and Track B are resolved separately and both get
built, so checking one is checking half. Cross-edition duplicates score 50–90%,
part-of-collection duplicates 100%.

**At three sources, 0 of 59 fell below the 15% floor. At nine, 130 of 5,297 do
— and that number cannot be read against interim data extracted before
2026-09-06**: record ids were not unique then, so `verify` resolved some
keepers to the wrong record. The ids are fixed and the guard is in
`extract.py`; re-run this after the next extract. See the 2026-09-06 entry in
`CURRENT_STATE.md`. The claim below is the precise one, and it
is weaker than "no false positives": passing establishes that the duplicate's text
really does occur in its keeper, not that discarding the whole record was the right
call. Deciding that needs cluster-by-cluster human review, which has not been done. It earned its place — the
*Excerpta Valesiana* looked like a false positive when probed on its opening
alone, and verifies at 70% across 40 probes (Perseus uses Rolfe's Loeb, whose
third volume prints the *Excerpta* as an appendix to Ammianus).

Duplicates are **marked, never deleted**, so the decision stays auditable and
the policy can change without re-extracting.

## Normalization

`src/normalize.py` makes the corpus read as one language rather than three.
The sources do not merely differ in spelling — they use **opposing conventions**:

| v as a share of (u+v) | Perseus | DigilibLT | CSEL |
|---|---:|---:|---:|
| before | 12.86% | 11.46% | 0.60% |
| after | 0% | 0% | 0% |

CSEL writes classical `uir`/`uita` throughout while the other two distinguish u
from v. Untreated, a tokenizer spends vocabulary discovering that `uir` and
`vir` are the same word, and the model learns edition-house style as if it were
Latin.

```bash
python src/normalize.py --report   # what would change, writes nothing
python src/normalize.py            # rewrite data/interim/*.jsonl
python src/normalize.py --uv v --keep-diacritics   # alternative policies
python src/normalize.py --jobs 1   # one process (default: every core)
```

**Folding is towards `u`.** Going the other way — u to v where consonantal —
requires morphological analysis to know which is which. Folding v to u requires
nothing, restores the classical convention (Latin had one letter here; the
distinction is a modern editorial invention), leaves CSEL's third of the corpus
untouched, and matches what `dedupe.py` already does to compare editions.

What the text turned out to be carrying, all of it an editor talking *about* the
text rather than the text itself:

| | |
|---|---:|
| editorial marks (`[] <> † ‡ * \| \ _ §`) removed | 40,672 |
| words rejoined across line breaks | 3,050 |
| ambiguous `word- word` split rather than fused | 763 |
| orphaned spacing diacritics (OCR debris) | 667 |
| homoglyphs remapped (mostly Cyrillic) | 233 |
| acute accents — metrical ictus in Plautus | 12,007 |
| macrons and breves | 3,256 |

Combining marks are stripped in a single NFD pass, which handles accents,
macrons, breves, papyrological underdots (`ṃạḷẹ`) and numeral overlines
together — ancient Latin needs none of them.

**Result (on the 3-source corpus): a 48-character letter inventory** — a–z and A–Z minus `v/V` and `j/J`.
That is *not* the classical alphabet, which had 23 letters and no case
distinction at all; it is a modern cased inventory with the two post-classical
letters folded away. 249 non-ASCII characters remain in 17.25M
words, deliberately left visible rather than guessed at (Roman-numeral
apostrophus `ↄ Ↄ`, and three stray OCR artifacts).

**On the 79M-word corpus the inventory is 86 letters and 246 distinct
characters** (measured 2026-08-18). The larger figure is mostly not
contamination. Non-Latin letters total **38 distinct, 765 occurrences in
75,043,073 training words** — one per ~98,000:

| | occurrences | what it is |
|---|---:|---|
| `ↄ Ↄ` | 198 | Roman numeral apostrophus — Cicero's `HS ccciↄↄↄ`. Kept deliberately |
| `ð Þ þ` | 267 | Ammianus describing a letter's *shape*; a Gothic *Skeireins* in the Patrologia |
| `ᶃ` | 172 | one OCR artifact, in Kepler's *Somnium* |
| `µ` | 31 | mojibake — Greek double-encoded in Perseus' *Adversus Marcionem* |
| `ꞩ` | 4 | Punic transliteration in EDCS inscriptions |
| Thai, N'Ko, Hebrew | 31 | Wikisource interwiki link residue |

That last row leads to the one genuine purity leak found so far: Vicifons
interwiki language links survive extraction, and they bring English with them
(`en:Jingle Bells`). **287 records of 9,894 carry an `xx:Title` pattern, 620
occurrences, 125 of them `en:`.** Eight per million words, so it changes no
measurement — but "no English" is a categorical claim, and a small numerator
does not defend it. **Fixed in `extract.py` (`MW_INTERWIKI`); the corpus on
disk predates the fix and needs a rebuild to pick it up.**

Plautus' *Amphitruo*, which opened this project as
`Mercvrivs / Vt vos in vostris voltis mercimoniis`, now reads:

```
Mercurius Ut uos in uostris uoltis mercimoniis emundis uendundisque
me laetum lucris adficere atque adiuuare in rebus omnibus
```

## Corpus and tokenizer

```bash
python src/build.py                      # -> data/processed/corpus.{train,validation}.jsonl
python src/build.py --period CLASSICAL   # the Latin-0 corpus, 4.64M words
python src/tokenizer.py                  # train the vocabulary sweep and measure
```

Validation is held out by **whole authors**, never random lines and not merely
whole works. A line-level split leaks the surrounding paragraphs of every held-out
line; a work-level split leaks the author — holding out one work by Augustine
while training on thirty others measures interpolation to an unseen text by a
thoroughly familiar writer. The first version of this split had **18 of 31
validation authors also present in training, covering 58% of validation words**.
Grouping by author closes it: 58 records, 855,423 words, 5.0%, **0 authors
shared with training**. `--group-by work` restores the looser signal, which is
worth reporting alongside since the two answer different questions.

### What the tokenizer settled

Every parameter figure in the project rested on an assumed ~1.8 tokens per word.
Measured, byte-level BPE trained on the corpus:

| vocab | tok/word | tok/word (classical) | corpus tokens | whole forms memorised | embedding share |
|---:|---:|---:|---:|---:|---:|
| 2,000 | 2.348 | 2.450 | 38.5M | 0 / 12 | 10% |
| **4,000** | **2.054** | **2.164** | **33.7M** | **0 / 12** | **18%** |
| 8,000 | 1.834 | 1.937 | 30.1M | 2 / 12 | 30% |
| 16,000 | 1.667 | 1.759 | 27.3M | 5 / 12 | 46% |
| 32,000 | 1.538 | 1.620 | 25.2M | 7 / 12 | 63% |

Counted over every record, not sampled. The assumption was good — 1.8 lands
between the 8k and 16k vocabularies. Classical Latin runs ~5% *higher* fertility
than the corpus average, being older, more varied and more heavily verse.

**Use 4,000.** Two reasons, and neither is compression:

1. **It forces morphology.** `plan.md` asks whether a small vocabulary makes the
   tokenizer learn Latin's stems and endings rather than memorising inflected
   forms. It does, measurably. At 4k none of the twelve `amo` paradigm forms is
   stored whole and `amabamus` divides `am · ab · amus`; at 32k seven are stored
   whole and it divides `am · abamus`. Stem consistency — whether every form of a
   stem opens with the same token — degrades from 0.44 at 2k to 0.94 at 16k.
2. **Vocabulary is a parameter-budget decision.** At `d_model=256` the
   transformer body is ~4.7M parameters whatever the vocabulary, while the
   embedding table is `vocab × 256`. A 32k vocabulary costs **8.2M parameters —
   larger than the model it serves**. For a model the corpus can only support at
   2–5M parameters, the vocabulary has to be small.

Caesar's opening at two sizes, showing the trade directly:

```
 4,000 (20 tok)  ·Gal lia ·est ·omnis ·diuis a ·in ·partes ·tres , ·quarum ...
32,000 (15 tok)  ·Gallia ·est ·omnis ·diuisa ·in ·partes ·tres , ·quarum ...
```

The morphology evidence is an *illustration*, not a result: twelve hand-picked
paradigm forms and three stem groups. It shows the mechanism clearly and it points
the same way as the parameter-cost argument, but it is far too small a probe to
carry the claim on its own.

Byte-level BPE is used throughout: it cannot emit `<unk>`, which matters because
249 stray non-ASCII characters survived normalization deliberately. Round-trip is
verified on held-out works, exact modulo a documented leading space.

## Training recipes

A run is a recipe in `configs/`, not a line of shell history:

```bash
python src/train.py --config latin-xxv --phase latin-xxv-latinitas
python src/train.py --config latin-xxv --phase latin-xxv        # phase 2
python src/train.py --config latin-0 --dim 256 --heads 8        # CLI still wins
```

**A recipe states epochs, never steps**, and that is the whole point of it. A
step count is meaningless the moment the corpus changes: 74,700 steps was four
passes over 153.0M tokens, and on a corpus half again as large the same number
is 2.7 passes with nothing to say the model is now undertrained. `train.py`
derives steps from `epochs × train_tokens ÷ (batch × context)`, so the same
recipe re-run after new sources land trains the same *model*, not the same
arithmetic. Every run's report records which recipe and how many epochs it
asked for.

`configs/latin-xxv.yaml` reproduces the flagship exactly — 74,700 steps, 4.0
epochs, 22.3 tokens per parameter — and there is a test asserting it still does.

## Evaluation

Built **before** any model exists, deliberately. An evaluation written afterwards
measures what the model happens to be good at; written first, it states in
advance what would count as success.

```bash
python src/evaluate.py --self-test          # validate the instrument
python src/evaluate.py --baselines          # the floor, sliced
python src/evaluate.py --memorization FILE  # score generated text
```

Every run writes a versioned JSON report to `data/processed/eval/`. **Those files
are the source of truth for any figure quoted here** — hand-transcribing numbers
is how this project's docs drifted from its builds before.

### The floor

An interpolated bigram, trained on a corpus's own training split and scored on
its own held-out split. **The floor is corpus-specific and must match the model
being judged** — quoting one corpus's floor against another's model compares
perplexities over different text and means nothing.

| corpus | bigram floor |
|---|---:|
| full (153.0M tokens) | **174.5** |
| classical (9.6M tokens) | **149.8** |

*(Restated 2026-08-18 on the rebuilt corpus. The previous figures — 195.2 and
158.8 — were measured on the 17M-word corpus and must not be quoted against
anything trained since. The classical corpus changed by three words between
those two measurements and its floor still moved 9 points, because the 4k
vocabulary had been retrained: same Latin, different merges, different token
stream. Reports now carry both a corpus and a tokenizer fingerprint, and
`train.py` and `compare.py` refuse to quote a floor across either kind of
drift.)*

Classical Latin is the easier of the two for a bigram, which is why using the
full-corpus figure to judge a classical-corpus model overstated its margin.

| slice | perplexity | |
|---|---:|---|
| CLASSICAL | 163.8 | |
| LATE | 211.2 | |
| VERSE | 159.6 | more formulaic, more repetition |
| PROSE | 218.2 | |
| DRAMA | 129.7 | easiest — colloquial, highly repetitive |
| MEDICINE | 280.0 | hardest — dense technical vocabulary |

This is what the genre metadata bought. A single averaged number would have hidden
a 150-point spread between Plautus and Caelius Aurelianus.

### Memorisation

The metric that matters most here. At 2–5M parameters trained many epochs over
9.5M tokens, memorisation *is* the failure mode — and the plan's success criterion
("novel Latin that does not occur in the corpus") is unfalsifiable without it.
Generated text is cut into 12-word windows and each checked verbatim against the
113MB of training text. Exact search, so it cannot report a false positive.

The instrument is validated on inputs whose answer is known:

| input | memorised | longest verbatim run |
|---|---:|---:|
| training text | 100.0% | 300 words |
| held-out text | 0.0% | 0 |
| shuffled real words | 0.0% | 0 |

An unvalidated metric is worse than none, because it gets believed.

> **All model results below were measured on the earlier 17M-word corpus.** They
> predate the Vicifons and EDCS expansion and are kept as historical record.
> Perplexities are not comparable to anything trained on the current corpus —
> different text, different held-out set, different tokenizer.

## Latin-0 — the first model

3.53M parameters (dim 192, 6 layers), trained from random initialisation on
9.5M tokens of classical Latin and nothing else. 8.6 epochs, 23.2 tokens per
parameter — essentially Chinchilla-optimal. One hour on an M2 at 23k tok/s.

```bash
python src/train.py                  # ~1h, writes checkpoints/latin-0.safetensors
python src/sample.py --n 5
python src/evaluate.py --memorization samples.txt
```

**Held-out perplexity 74.3, against a bigram floor of 158.8** — 2.14× better.
The floor was stated before training began.

*(Corrected: this was first reported as 2.63× against a floor of 195.2. That
figure came from a baseline trained and evaluated on the **full** corpus, while
Latin-0 was evaluated on the **classical** split — two different held-out sets,
so the comparison was invalid. Classical Latin is easier for a bigram than the
mixed corpus, so the wrong floor flattered the model by 23%. `src/compare.py`
now reads the corpus-matched baseline, and `train.py` refuses to quote a floor
from a different corpus.)*

Final train loss 4.3130 against validation 4.3080: **validation is marginally
lower than training**, so after 8.6 epochs there is no overfitting gap at all.

### Does it memorise?

The question the whole project turns on, and the reason validation is held out
by whole author. Of 1,459 twelve-word windows across 25 generations, **0 appear
anywhere in the training corpus**.

A bare 0% is uninterpretable, so the harness scores genuine unseen Latin — a
held-out work by a real author — as a control:

| window (words) | generated | real unseen Latin |
|---:|---:|---:|
| 3 | 19.4% | 18.7% |
| 4 | 1.7% | 6.7% |
| 5 | 0.0% | 4.4% |
| 6 | 0.0% | 1.2% |
| 12 | 0.0% | 0.0% |

At three words the model matches real Latin almost exactly — both share common
trigrams with the corpus, which is simply what Latin is. From four words up the
generated text overlaps the corpus **less** than genuine Latin by a real author
does. It is not regurgitating; if anything it under-reproduces the longer
formulaic phrases that real Latin contains, which is a limitation as much as a
result.

### What it actually writes

```
Quid est, ut opinor, et quae sit ille, ut in rebus, quae ad modum in rebus
agrum aduentum fatum, non ut ad se non potest, ita, ut cum aduentu resciscat;
sed tamen etiam cum ab omnibus rebus, qui de rebus aduentu rumint, pro se a me
expetendo ad eum qui uenerit, ita, quia aut exposui aut,
```

Honest reading: morphology is largely well-formed and the function-word skeleton
is Latin — `ut opinor` is genuinely Ciceronian, `qui uenerit` is a correct
perfect subjunctive, `pro se a me expetendo` is a plausible gerundive. But it is
locally coherent and globally not: `quae sit ille` disagrees in gender, `rumint`
is not a word, and no sentence sustains an argument. That is what 3.5M parameters
on 9.5M tokens buys, and it is roughly what should have been expected.

## Latin-2 — two-phase Latinitas → Romanitas

The strategy `plan.md` proposed: learn Latin from all of it, then specialise on
Rome. 15.90M parameters, sized to the data rather than to the name — 27M would
have needed 16 epochs, which `feasibility.md` §2 itself calls aggressive, and
37 hours instead of 8.

```bash
python src/train.py --name latin-2-latinitas --corpus corpus \
    --dim 384 --layers 8 --heads 8 --steps 16000 --lr 2.5e-4
python src/train.py --name latin-2 --corpus classical \
    --init-from latin-2-latinitas --steps 1750 --lr 1e-4
python src/compare.py
```

| run | params | corpus | epochs | ppl | floor | × floor | memo | hours |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| latin-0 | 3.53M | classical | 8.6 | 74.3 | 158.8 | 2.14× | 0.0% | 1.0 |
| latin-2-latinitas | 15.90M | full | 7.8 | 48.2 | 195.2 | 4.05× | — | 8.1 |
| **latin-2** | 15.90M | classical | 3.0 | **44.1** | 158.8 | **3.60×** | 0.0% | 0.9 |

**Latin-0 and Latin-2 are directly comparable** — same held-out classical works,
same 158.8 floor. Perplexity falls from 74.3 to 44.1, a 41% reduction.

**Two variables moved at once**: model size (3.53M → 15.90M) *and* the two-phase
schedule. The ablation below separates them.

One thing the phase transition shows on its own: Romanitas reached 45.2 within
**500 steps**, where Latin-0 needed 5,000 from random weights to reach 74.3.
Adapting register inside a language you already know is far cheaper than
learning the language.

### The ablation: which half did the work?

`ablation-no-latinitas` is 15.90M parameters on classical Latin alone, 8.6 epochs
— Latin-0's exact schedule at Latin-2's size. Each pair now changes one variable:

| run | params | corpus | ppl | |
|---|---:|---|---:|---|
| latin-0 | 3.53M | classical | 74.3 | |
| ablation-no-latinitas | 15.90M | classical | 58.3 | ← **size alone: −16.0 (21.5%)** |
| latin-2 | 15.90M | all Latin → classical | 44.1 | ← **schedule alone: −14.2 (24.3%)** |

**The contributions are almost equal.** Of the 30.2-point total improvement,
model size accounts for 53% and the two-phase schedule for 47%. Both are real;
neither dominates.

*I predicted "most of the gain from size, some from the schedule" before running
this. The direction was right, the emphasis wrong — it is essentially 50/50.*

### Overfitting: the prediction that failed

I also predicted the ablation would overfit **more** than Latin-2, on the theory
that the Latinitas phase acts as regularisation. It does the opposite:

| run | train ppl | val ppl | gap | classical passes |
|---|---:|---:|---:|---:|
| latin-0 | 74.7 | 74.3 | 1.00× | 8.6 |
| ablation | 46.6 | 58.3 | **1.25×** | 8.6 |
| latin-2 | 33.0 | 44.1 | **1.34×** | 10.8 |

Latin-2 — the model that saw *more* text — has the *larger* gap. The reason is
structural and I had missed it: **`classical.train` is a subset of
`corpus.train`**, so Phase 1 already contained the entire Phase 2 corpus. Latin-2
therefore made ~10.8 passes over the classical text against the ablation's 8.6.

So the two-phase schedule as built is not "read other Latin, then read Rome". It
is "read everything including Rome, then read Rome again". It still wins by 14.2
perplexity points — but part of that advantage is simply more exposure to the
target text, not transfer from the rest of Latin.

**The cleaner experiment** would exclude classical material from Phase 1
entirely, making the two phases genuinely disjoint. That is a real open question,
not a formality: it would separate *transfer* from *extra passes*.

### Overfitting appears at scale, as predicted

| run | train ppl | val ppl | gap |
|---|---:|---:|---:|
| latin-0 | 74.7 | 74.3 | 1.00× |
| latin-2-latinitas | 34.5 | 48.2 | 1.40× |
| latin-2 | 33.0 | 44.1 | 1.34× |

Latin-0 had no gap at all. The larger models do — memorisation emerging as
parameters rise against a fixed corpus, which is exactly what `feasibility.md`
§2 warned would happen. It makes the memorisation measurement load-bearing here
in a way it was not before.

### Still not regurgitating

Measured against the training text each model actually saw, with the control
sampled evenly across every held-out work:

| window | latin-0 | real Latin | latin-2 | real Latin |
|---:|---:|---:|---:|---:|
| 3 | 16.3% | 5.1% | 23.6% | 12.7% |
| 4 | 1.0% | 1.2% | 4.1% | 5.2% |
| 5 | 0.0% | 0.4% | 0.4% | 2.8% |
| 12 | 0.0% | 0.4% | 0.0% | 1.6% |

Both are at zero by twelve words. At three words both exceed real Latin — models
reach for common phrasing more often than any individual author does, which is a
bias toward the average, not memorisation. From five words up both sit *below*
genuine unseen Latin. The train/val gap is real, but it is not showing up as
verbatim recall.

### What changed in the writing

Same prompt, same seed:

```
latin-0    Gallia est omnis diuisa cum regni dedepit, ut, dum et Proici,
           quem per urbem uenit, magistrato Treici, in Gallia; quos
           Hannonis et ipse, fortune Pamphorium Pamlio, qui adesse

latin-2    Gallia est omnis diuisa cum Secundamen, quique umeros et nigra
           lotos Frigidus in Indicis in Geronibus aduectus Fletamque,
           quae dulci flauae pinna uidemus, Italiaque et ex eo fontes
```

Latin-0 invents more non-words (`dedepit`, `Pamphorium`, `magistrato`). Latin-2's
vocabulary is recognisably poetic — `umeros`, `nigra`, `frigidus`, `dulci flauae
pinna` are hexameter furniture. Neither sustains an argument. The grammar
improved; the meaning did not arrive.

## Step 0 — the chat proof

Does dialogue mined from the corpus turn a base model into one that takes a turn?
Run before investing in corpus growth, on the plan's own principle: smallest
interesting result first.

```bash
python src/dialogue.py          # mine turn-pairs -> data/processed/dialogue.jsonl
python src/sft.py               # fine-tune latin-2 -> latin-2-chat
python src/sft.py --resume      # continue from the last checkpoint
python src/sft.py --chat        # talk to it
```

**54,536 pairs, no English anywhere in the chain** — 8,495 dramatic turns with
speakers intact, 46,041 question→answer sentences. Donatus' *Ars minor* is a
catechism front to back, which makes it effectively a second-century Latin
teacher's instruction dataset, written for this purpose fourteen centuries early.

Loss is masked to the response only: training on prompt tokens would teach the
model to generate questions as readily as answers. Every example ends in
`<|endoftext|>`, in the loss, so **stopping is learned rather than truncated at
inference**. 35 minutes on top of the base model.

### What it learned

| | |
|---|---|
| Turn-taking | ✅ one turn per prompt, every time |
| Stopping | ✅ ends its turn; no run-on |
| Register matching | ✅ genuinely surprising — see below |
| Being correct | ❌ as predicted, and not fixable by scale |

```
> Quid agis?                    (colloquial, Plautine)
< Atque hexamycophantias, te emere audio, nam et parasitus parasitust.

> Quot sunt partes orationis?   (Donatus' opening catechism question)
< Donatiuae, participia, casus, figuraetima.

> Quid est ueritas?             (theological)
< cum eadem in illo esse non possit, quae in eum sunt facta.
```

**The register matching is the real finding.** A colloquial question draws a
Plautine answer — `parasitus parasitust` is comic doubling, and
`hexamycophantias` is exactly the kind of mock-Greek compound Plautus invents.
A theological question draws theological syntax. And asked Donatus' own question,
the model replies with *parts-of-speech vocabulary* — `participia`, `casus`,
`figurae`.

The correct answer is `Octo`. It got that wrong while getting the **domain**
right, which is the clearest demonstration available of the distinction this
project keeps running into: it has learned what the question is *about* without
learning what is *true*.

Not everything works. `Vnde uenis?` → `' inquit 'qualis.` is broken — a quotation
artifact the mining did not filter.

**Verdict: the direction is viable.** Corpus-mined dialogue produces conversational
behaviour without any English-derived data. Scaling it means a better base model,
which means finishing the corpus.

## Step 1 — pipeline driver and the full corpus

```bash
python src/pipeline.py --status     # what is stale and why
python src/pipeline.py              # run only what changed
python src/pipeline.py --jobs 1     # one source at a time (default: 4 at once)
```

One command rebuilds everything after adding a source or editing an authority
file, and does only the work the change invalidated. **Fingerprints chain rather
than inspect files**: three stages rewrite `data/interim/` in place, so
mtime-based staleness would mark a stage stale the instant it finished. Each
stage's fingerprint is `sha256(previous stage's fingerprint + its own config +
its own source)`, so editing `normalize.py` invalidates normalize and everything
after it, and nothing before.

Full run over five sources: **10.3 minutes**.

### The corpus, 5 of 9 sources

| source | records | words |
|---|---:|---:|
| vicifons | 11,753 | 86,368,214 |
| digiliblt | 373 | 6,509,980 |
| opengreekandlatin-csel | 271 | 6,036,274 |
| perseus-canonical-latinLit | 324 | 5,595,104 |
| edcs-sdam | 267 | 4,356,994 |
| **after dedup** | **10,189** | **78,984,104** |

**Vicifons was estimated at 3M words and delivered 86M** — a 29× miss in
`sources.md`. It is not junk: Latin Wikisource hosts Migne's *Patrologia Latina*
at scale. EDCS also overshot, 4.36M against 3M.

Dedup removed **29,805,223 words (27.4%)** across 923 clusters — far above the
4.9% seen at three sources, because Wikisource re-hosts a great deal of what
Perseus and CSEL already provide.

### Dating the Patrologia

86M words arrived with no author field and 82.8% of the corpus fell to
UNASSIGNED. Two signals fixed most of it.

**Migne's PL is ordered chronologically by author death**, Tertullian to
Innocent III across 217 volumes — so a volume number dates a text without naming
its author. Three Wikisource conventions carry it (title path, title
parenthetical, link back to the volume), covering 22% of pages. The volume→year
map in `authorities.yaml` is anchored on principal authors and interpolated, not
linear: Migne gives Augustine sixteen volumes and the tenth century about six, so
a straight line dates Jerome a century early.

**The undated authors were using Latin name forms** — `Hieronymus`,
`Ambrosius`, `Aurelius Augustinus` — while the authority file had been curated
with library forms (`Jerome, Saint`). They had been invisible to the matcher all
along. Adding those, plus Carolingians and popes, picked up another 4%.

| period | words | |
|---|---:|---|
| UNASSIGNED | 33,963,083 | 43.0% |
| LATE | 24,284,063 | 30.7% |
| MEDIEVAL | 16,015,187 | 20.3% |
| CLASSICAL | 4,539,519 | 5.7% |
| NEOLATIN | 182,252 | 0.2% |

From **82.8% undated to 43.0%**. Medieval Latin went from 79,882 words to 16M,
which closes a gap open since the corpus was first surveyed. The single NEOLATIN
entry is the *Malleus Maleficarum* (1487).

Dates from the PL method carry `date_source: pl-volume` and confidence
`inferred`. They exist to place a text on the right side of a period boundary and
should not be cited as scholarship.

## Known limitations

**Apparatus criticus in the Perseus source.** Four records (Pliny NH, three
*Scriptores Historiae Augustae* lives) carry fragments of apparatus as *bare text
in the body*, with no markup to key on — e.g. `uiri et P; uirtute Peter,
following Bernhardy`. Measured at 9 occurrences in 5.6M words, so it is left
alone rather than engineered around; a QA pass in `normalize.py` should flag it.

**Period is per-source, not per-work.** Every Perseus record is currently tagged
`CLASSICAL` from the manifest. `canonical-latinLit` contains later material
(Ammianus, the SHA), so this needs an author-date authority list — as does
`approximate_date`, which is null throughout.

**CSEL is machine-corrected OCR, and it shows.** Two artifacts found by
inspection, both localised rather than systemic:
hyphenated line breaks are not rejoined (`de the- sauro` for *thesauro*), and
some Greek passages were OCR'd into Latin-letter gibberish
(`tv trjc tov Ixatvov`) which script-based Greek detection cannot see.
A vowel-ratio check (Latin sits at 0.454–0.465; garbled text falls well below)
found **no record under 0.38 in any source**, so no volume is wholesale junk —
but `normalize.py` should rejoin hyphenation and flag low-vowel-ratio
paragraphs.

**DigilibLT licence.** The CIRCSE mirror's README contradicts itself: the CC
badge links to BY-NC-SA 3.0, the anchor text says BY-SA 4.0. Upstream DigilibLT
confirms **CC BY-SA 4.0 for the texts**; NC applies to site chrome and the LiLa
annotation layer. The extractor therefore reads only the `# text =` layer and
never the lemma/POS/URI columns. Do not start using the annotations without
revisiting this.

## Publishing a model

```bash
python src/bundle.py --name latin-vii    # -> dist/latin-vii/
hf upload vincent-b/latin-vii dist/latin-vii . --repo-type model
```

The bundle — weights, the exact fingerprinted tokenizer, a generated
`config.json`, a standalone MLX loader, the corpus attribution notice and the
model card — is assembled by `src/bundle.py`, never by hand. Prose lives in
[`docs/modelcards/`](docs/modelcards/); every figure is substituted from a
generated report, because hand-transcribing numbers is how these docs drifted
from the builds before.

Two guards, in keeping with the invariants: only Track A models can be
bundled (`latin-lii-latinitas` is a hard error — NonCommercial propagates
into weights), and the bigram floor is quoted only if the baseline report
matches the training report's corpus *and* tokenizer fingerprints. A floor
measured through a different tokenizer is refused, not approximated.

```bash
python src/bundle.py --name latin-lii-latinitas --research-only
```

`--research-only` lifts the first guard for local inspection and nothing
else. It cannot be reached by accident, it writes to `dist/research/` rather
than the `dist/` directory the upload command points at, and the bundle
declares its own restriction without needing this repository to explain it:
NC licence in the card frontmatter, a research-only header naming the two
sources responsible, and a `DO-NOT-REDISTRIBUTE.md` beside the weights. It
does **not** lift the requirement to know what a model trained on — a report
with no track fails either way, because the track is derived, never assumed.

## Status

- [x] Source survey, licence-checked (`sources.md`)
- [x] Feasibility assessment and parameter ladder (`feasibility.md`)
- [x] Machine-readable manifest with Path A guardrails
- [x] `fetch.py` — git / http / zenodo / manual, with lockfile
- [x] `extract.py` — Perseus, CSEL, DigilibLT, EDCS (JSON), Vicifons (MediaWiki)
- [ ] `extract.py` — EDH/Papyri EpiDoc, Vicifons MediaWiki, Latin Library + Gutenberg plaintext
- [x] Author-date authority list → per-work `period` and `approximate_date`
- [x] Orthographic normalization — 48-letter alphabet, u/v unified across sources
- [x] Deduplication — content-based, verbatim-verified
- [x] `build.py` — corpus with whole-work held-out split
- [x] Tokenizer — 4k Latin BPE chosen on measured evidence; 153.0M tokens full / 9.6M classical
- [x] Two-track structure — A redistributable / B research, enforced in code
- [x] `provenance.py` — attribution notice and standing audit gate
- [x] Genre and form metadata — 97.5% resolved, two orthogonal axes
- [x] Evaluation harness — baselines, memorisation, versioned JSON reports
- [x] **Latin-0 trained** — 3.53M params, held-out ppl 74.3 vs 158.8 floor (2.14×), 0% memorisation
- [x] **Latin-2 trained** — 15.90M params, two-phase; ppl 44.1 vs 158.8 floor (3.60×), 0% memorisation
- [x] Ablation — size and schedule contribute ~53% / ~47%; both real
- [ ] Disjoint-phase ablation: exclude classical from Phase 1, to separate transfer from extra passes
- [x] **Chat proof** — corpus-mined dialogue SFT; turn-taking and stopping learned
- [x] **Pipeline driver** — one idempotent command, chained fingerprints
- [x] **5 of 9 sources** — 78.98M words after dedup; PL-volume dating for the Patrologia
- [ ] Remaining: latin-library, papyri-idp, edh, gutenberg-latin
- [x] **Retrain tokenizer** — 4k rebuilt on the 79M corpus: 2.038 tok/word, 153.0M tokens, 0/12 forms memorised
- [x] **Restate the floors** — 174.5 full / 149.8 classical; the 17M-corpus figures are retired
- [x] **Report fingerprinting** — corpus *and* tokenizer stamped; stale floors refused rather than quoted
- [x] **LATIN-XXV trained** — 27.49M params, two-phase; ppl 29.8 vs 149.8 floor (**5.03×**), **0% memorisation**
- [x] **LATIN-VII published** — [huggingface.co/vincent-b/latin-vii](https://huggingface.co/vincent-b/latin-vii), assembled by `src/bundle.py`
- [ ] Strip 927 Perseus editorial datelines (`Scr. ...`) from Cicero's letters — the model learned to generate them
- [ ] Best-checkpoint tracking in `train.py` — it saves the latest, not the best
- [ ] Re-run the chat SFT on LATIN-XXV, then serve it
- [x] Strip Wikisource interwiki links — 620 occurrences, 125 English; **needs a rebuild to take effect**
- [ ] Real dialogue SFT on the better base, then serve
- [ ] UD treebank syntactic probing (fetch via `--track eval`; exclude overlapping passages from training)
- [ ] Latin-0 (~2–5M params — see `feasibility.md` §2, revised against measured data)

## A note on the first text fetched

The first file pulled from Perseus was Plautus, *Amphitruo*:

```
Mercvrivs
Vt vos in vostris voltis mercimoniis
```

`Mercvrivs`, `vostris`, `voltis` — u/v undifferentiated, archaic `vo-` for `ve-`. Three
normalization decisions in the first two lines, before any modelling question arose. It was a fair
preview of where the work actually was: five pipeline stages later the same passage reads

```
Mercurius Ut uos in uostris uoltis mercimoniis
```

and getting there meant reconciling two incompatible u/v conventions across three projects,
stripping 40,672 editorial marks, and proving that 60 apparent duplicates really were duplicates.
The archaic `uo-` for `ue-` stays, because that is Plautus rather than his editor.

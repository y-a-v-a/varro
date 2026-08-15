# Claudius — an openly licensed Latin corpus and a language model trained only on Latin

Two deliverables, in order:

1. **Open Latin Corpus** — a reproducible, properly licensed corpus of Latin. No such thing
   currently exists, which makes it a contribution independent of anything trained on it.
2. **LATIN-XXV** — a ~25M parameter GPT trained from random initialization on that corpus.
   One language. Zero modern languages. No inherited multilingual representations.

Planning documents:

| File | What it is |
|---|---|
| [`plan.md`](plan.md) | The project plan — goals, model ladder, training strategy |
| [`feasibility.md`](feasibility.md) | Assessment of that plan: the data arithmetic, the licensing decision, what the plan misses |
| [`sources.md`](sources.md) | Human-readable survey of every Latin text source, licence-checked |
| [`manifest/sources.yaml`](manifest/sources.yaml) | Machine-readable version of the above — drives the pipeline |
| [`manifest/authorities.yaml`](manifest/authorities.yaml) | Author/work dating authority — assigns period per work |

## Quickstart

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
  provenance.py       # ATTRIBUTION.md + track audit gate          [done]
  evaluate.py         # baselines, memorisation, versioned reports [done]
  train.py            # MLX GPT from random init                  [done]
  sample.py           # generation, for memorisation scoring      [done]
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

1. Add an entry to `manifest/sources.yaml`. Licence and tier are mandatory; `amber` additionally
   requires an `exclude_reason`.
2. `python src/fetch.py --validate` — the validator enforces the Path A guard, so an `amber` source
   marked `include: true` is a hard error rather than a quiet contamination.
3. `python src/fetch.py --only <id>`.
4. Commit the updated `fetch.lock.json`.

## Corpus as measured

Three of nine green sources extracted:

| Source | Period | Records | Words | vs. estimate |
|---|---|---:|---:|---|
| perseus-canonical-latinLit | CLASSICAL | 324 | 5,596,757 | 6.0M |
| digiliblt | LATE | 373 | 6,522,099 | 4.0M |
| opengreekandlatin-csel | LATE | 271 | 6,039,916 | 5.0M |
| **total** | | **968** | **18,158,772** | 15.0M |

Running ~21% ahead of the manifest's planning figures. Residual contamination
after filtering: **0 Greek characters**, 43 English function-word hits across
18.2M words (all verified as false positives — Latin syllables under
grammatical discussion, hyphenation artifacts, citation abbreviations).

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

## Deduplication

`src/dedupe.py` finds works repeated across and within sources — Tertullian's
*Adversus Marcionem* is in both Perseus and CSEL, Boethius' *Consolatio* in
three places, Celsus' *De Medicina* three times inside Perseus alone.

```bash
python src/dedupe.py --report    # clusters and what would be dropped
python src/dedupe.py             # mark is_duplicate / duplicate_of
python src/dedupe.py --verify    # verbatim-check every existing mark
```

Detection is **content-based, not title-based** — titles vary across projects
(*Epitoma de Tito Liuio* / *Epitome Rerum Romanorum*), and conversely the same
title can hide different texts. Text is folded for comparison only (lowercased,
u/v and i/j merged); without that, Perseus's `uiri` and CSEL's `viri` look like
different words. Documents reduce to bottom-k sketches of 5-gram hashes;
Jaccard is estimated from sketch overlap and containment derived from it.

**59 records, 884,721 words (4.9%) marked as duplicates. Surviving corpus:
909 records, 17,256,637 words — of which 4,636,779 are CLASSICAL.**

Two relationships hide in the clusters and need opposite rules:

- **equivalence** — same work, several sources, near-identical length. Source
  priority decides, since what differs is text quality.
- **containment** — one record is part of a fuller one. The *larger* must win
  regardless of source. Getting this backwards kept Perseus's 43k-word
  selection of Jerome's letters and discarded CSEL's complete 290k-word corpus.

`--verify` is the guard against trusting an estimate: it samples 40 random
8-word spans from each duplicate and tests literal membership in its keeper.
Cross-edition duplicates score 50–90%, part-of-collection duplicates 100%.
**Currently 0 of 59 fall below the 15% floor.** That is the precise claim, and it
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

**Result: a 48-character letter inventory** — a–z and A–Z minus `v/V` and `j/J`.
That is *not* the classical alphabet, which had 23 letters and no case
distinction at all; it is a modern cased inventory with the two post-classical
letters folded away. 249 non-ASCII characters remain in 17.25M
words, deliberately left visible rather than guessed at (Roman-numeral
apostrophus `ↄ Ↄ`, and three stray OCR artifacts).

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
| full (33.7M tokens) | 195.2 |
| classical (9.5M tokens) | 158.8 |

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

## Status

- [x] Source survey, licence-checked (`sources.md`)
- [x] Feasibility assessment and parameter ladder (`feasibility.md`)
- [x] Machine-readable manifest with Path A guardrails
- [x] `fetch.py` — git / http / zenodo / manual, with lockfile
- [x] `extract.py` — Perseus + CSEL (CTS/TEI) and DigilibLT (CoNLL-U), 18.2M words
- [ ] `extract.py` — EDH/Papyri EpiDoc, Vicifons MediaWiki, Latin Library + Gutenberg plaintext
- [x] Author-date authority list → per-work `period` and `approximate_date`
- [x] Orthographic normalization — 48-letter alphabet, u/v unified across sources
- [x] Deduplication — content-based; 0 of 59 below the verbatim floor
- [x] `build.py` — corpus with whole-work held-out split
- [x] Tokenizer — 4k Latin BPE chosen on measured evidence; 33.7M tokens full / 9.5M classical
- [x] Two-track structure — A redistributable / B research, enforced in code
- [x] `provenance.py` — attribution notice and standing audit gate
- [x] Genre and form metadata — 97.5% resolved, two orthogonal axes
- [x] Evaluation harness — baselines, memorisation, versioned JSON reports
- [x] **Latin-0 trained** — 3.53M params, held-out ppl 74.3 vs 158.8 floor (2.14×), 0% memorisation
- [ ] Latin-2 (~25M) on the full corpus, two-phase Latinitas→Romanitas
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

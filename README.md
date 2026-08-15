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
python src/fetch.py                 # fetch everything marked include:true
python src/fetch.py --only perseus-canonical-latinLit
```

Perseus alone is ~175 MB / 1,361 files and clones in about ten seconds (blobless partial clone).

## Two invariants

**1. `data/raw/` is immutable.** Nothing in this repo edits a fetched file, and neither should you.
Every correction belongs downstream, in code under `src/`. The moment you hand-patch a downloaded
text, the corpus stops being reproducible and the licence trail breaks.

**2. Path A is in force.** Everything that ships must be redistributable. Sources tiered `amber`
(non-commercial or unclear licence) are declared in the manifest for the audit trail but are never
fetched — `fetch.py` refuses them even when named explicitly. See [`feasibility.md`](feasibility.md) §3
for the decision and what it costs (chiefly Corpus Corporum's 226M words).

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
  build.py            # -> data/processed/corpus.jsonl
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

**60 records, 887,619 words (4.9%) marked as duplicates. Surviving corpus:
908 records, 17,271,153 words — of which 4,638,175 are CLASSICAL.**

Two relationships hide in the clusters and need opposite rules:

- **equivalence** — same work, several sources, near-identical length. Source
  priority decides, since what differs is text quality.
- **containment** — one record is part of a fuller one. The *larger* must win
  regardless of source. Getting this backwards kept Perseus's 43k-word
  selection of Jerome's letters and discarded CSEL's complete 290k-word corpus.

`--verify` is the guard against trusting an estimate: it samples 40 random
8-word spans from each duplicate and tests literal membership in its keeper.
Cross-edition duplicates score 50–90%, part-of-collection duplicates 100%.
**Currently 0 of 60 fall below the 15% floor.** It earned its place — the
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
| words rejoined across line breaks | 3,813 |
| orphaned spacing diacritics (OCR debris) | 667 |
| homoglyphs remapped (mostly Cyrillic) | 233 |
| acute accents — metrical ictus in Plautus | 12,007 |
| macrons and breves | 3,256 |

Combining marks are stripped in a single NFD pass, which handles accents,
macrons, breves, papyrological underdots (`ṃạḷẹ`) and numeral overlines
together — ancient Latin needs none of them.

**Result: a 48-letter alphabet** — a–z and A–Z minus `v/V` and `j/J`, which is
exactly classical Latin's inventory. 249 non-ASCII characters remain in 17.25M
words, deliberately left visible rather than guessed at (Roman-numeral
apostrophus `ↄ Ↄ`, and three stray OCR artifacts).

Plautus' *Amphitruo*, which opened this project as
`Mercvrivs / Vt vos in vostris voltis mercimoniis`, now reads:

```
Mercurius Ut uos in uostris uoltis mercimoniis emundis uendundisque
me laetum lucris adficere atque adiuuare in rebus omnibus
```

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
- [x] Deduplication — content-based, verbatim-verified, 0 false positives
- [ ] **Token count** — words are measured; tokens need the tokenizer
- [ ] Evaluation suite, built *before* training Latin-0
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

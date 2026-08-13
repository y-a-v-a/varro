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
  extract.py          # TEI -> data/interim/<source>.jsonl        [done, Perseus]
  normalize.py        # u/v, i/j, macrons, ae/æ, punctuation      [next]
  filter_lang.py      # strip embedded Greek and English translations
  dedupe.py           # edition-aware; sources overlap heavily
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

## First measurement

Perseus `canonical-latinLit`, extracted:

```
324 works        5,596,816 words        mean 17,274 words/work
```

Against a manifest estimate of 6,000,000 — so the planning figures in
`feasibility.md` are sound to within ~7% for this source.

Editorial interventions on that run, all counted and reported by `extract.py`:

| | |
|---|---|
| `<note>` dropped | 115,202 |
| `<choice>` resolved to `<corr>` over `<sic>` | 5,587 |
| `<choice>` resolved to `<reg>` over `<orig>` | 110 |
| Greek passages dropped | 4,638 |
| `<del>` / `<gap>` / `<bibl>` dropped | 2,808 / 1,317 / 3,294 |
| English editorial matter removed | 20 tagged elements + 38 paragraphs |

Largest works: Livy *Ab urbe condita* 515,204 · Pliny *Naturalis Historia*
401,353 · Quintilian *Institutio Oratoria* 172,362.

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

## Status

- [x] Source survey, licence-checked (`sources.md`)
- [x] Feasibility assessment and parameter ladder (`feasibility.md`)
- [x] Machine-readable manifest with Path A guardrails
- [x] `fetch.py` — git / http / zenodo / manual, with lockfile
- [x] `extract.py` — Perseus TEI → JSONL, 5.6M words
- [ ] `extract.py` — DigilibLT TEI, EDH EpiDoc, Vicifons MediaWiki, Latin Library plaintext
- [ ] Author-date authority list → real `period` and `approximate_date`
- [ ] Orthographic normalization (see `feasibility.md` §4.1 — matters more than tokenizer vocab size)
- [ ] Deduplication (Latin Library ↔ Perseus overlap is large)
- [ ] **Token count** — words are measured; tokens need the tokenizer
- [ ] Evaluation suite, built *before* training Latin-0
- [ ] Latin-0 (~5M params)

## A note on the first text fetched

The first file pulled from Perseus was Plautus, *Amphitruo*:

```
Mercvrivs
Vt vos in vostris voltis mercimoniis
```

`Mercvrivs`, `vostris`, `voltis` — u/v undifferentiated, archaic `vo-` for `ve-`. Three
normalization decisions in the first two lines, before any modelling question arises. That is a fair
preview of where the work actually is.

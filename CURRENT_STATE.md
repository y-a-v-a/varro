# Current state — pick-up notes

Written 2026-08-17 as a handoff into a fresh session. `README.md` explains how
things work; this file says where they stand, what is stale, and what to do next.

---

## What this project is

A reproducible pipeline that assembles an **openly licensed corpus of Latin**,
and small language models trained on it **from random initialisation, on Latin
alone** — no English, no translations, no multilingual pretraining.

Two constraints shape everything:

- **Purity.** The model's entire linguistic universe is Latin. This is enforced,
  not assumed: English translations shipped inside Latin sources are filtered,
  embedded Greek is stripped, foreign scripts are removed, and the SFT data is
  mined from the corpus rather than translated from English.
- **Licensing.** Track A (free licences) is redistributable and is what gets
  published. Track B (adds non-commercial sources) is research-only and never
  redistributed. Track membership is derived from the licence tier and enforced
  in `fetch.py`, `build.py` and `provenance.py --check`.

The stated goal has moved: it is now **usable weights behind a simple chat app**,
plus a pipeline that absorbs corpus growth. See `plan.md` for what that can and
cannot mean — the honest framing is *interlocutor, not assistant*.

---

## Corpus — current

**10,192 records, 78,975,089 words** (Track A, after dedup), from **5 of 9**
green sources.

| period | words | |
|---|---:|---|
| UNASSIGNED | 33,953,816 | 43.0% |
| LATE | 24,284,319 | 30.7% |
| MEDIEVAL | 16,015,187 | 20.3% |
| CLASSICAL | 4,539,515 | 5.7% |
| NEOLATIN | 182,252 | 0.2% |

| source | records | words (pre-dedup) |
|---|---:|---:|
| vicifons | 11,753 | 86,368,214 |
| digiliblt | 373 | 6,509,980 |
| opengreekandlatin-csel | 271 | 6,036,274 |
| perseus-canonical-latinLit | 324 | 5,595,104 |
| edcs-sdam | 267 | 4,356,994 |

Dedup removes **29.8M words (27.4%)** — high because Wikisource re-hosts much of
what Perseus and CSEL already provide.

**Tokens** (4k vocabulary, counted not projected): **153.0M full corpus,
9.6M classical**.

### Not yet extracted

`latin-library` (~15M est), `papyri-idp`, `edh`, `gutenberg-latin`. Each needs an
extractor: plaintext, EpiDoc XML, EpiDoc XML, manual plaintext respectively.
`latin-library` overlaps Perseus heavily, so expect dedup to eat much of it.

---

## Models — ALL TRAINED ON THE OLD 17M-WORD CORPUS

**This is the most important thing to know.** Every checkpoint predates the
Vicifons/EDCS expansion. Their perplexities were measured on a different corpus,
a different held-out set, and in latin-2's case a tokenizer trained on different
text. **Do not compare new models to these numbers.**

| run | params | corpus | ppl | vs floor | memorisation |
|---|---:|---|---:|---:|---:|
| latin-0 | 3.53M | classical (9.5M tok) | 74.3 | 2.14× | 0% |
| ablation-no-latinitas | 15.90M | classical | 58.3 | 2.72× | 0% |
| latin-2-latinitas | 15.90M | full (33.7M tok) | 48.2 | 4.05× | — |
| latin-2 | 15.90M | all → classical | 44.1 | 3.60× | 0% |
| latin-2-chat | 15.90M | + dialogue SFT | — | — | — |

**The ablation result is the one worth carrying forward.** Of the 30.2-point
improvement from latin-0 to latin-2, model size accounted for ~53% and the
two-phase Latinitas→Romanitas schedule ~47%. Both real, neither dominant.

Caveat found while running it: `classical.train` is a **subset** of
`corpus.train`, so Phase 1 already contained the entire Phase 2 corpus. Latin-2
made ~10.8 passes over classical text against the ablation's 8.6. Part of the
"schedule" benefit is therefore extra exposure, not transfer. The clean version
would exclude classical from Phase 1 — still open.

---

## The ladder, remeasured on 79M words

Chinchilla-optimal ≈ 20 tokens/parameter; repeated data is worth nearly as much
as fresh up to ~4 epochs, decaying to ~zero past 16.

| epochs | classical (9.6M tok) | full corpus (153.0M tok) |
|---|---|---|
| 4 (safe) | 1.9M params | **30.6M params** |
| 8 | 3.8M | **61.2M** |
| 16 (aggressive) | 7.7M | **122.4M** |

**LATIN-XXV at 25M now fits inside 4 epochs**, where it previously sat on the
16-epoch line. Path A's 25–40M licensing ceiling is no longer the binding
constraint; the corpus is.

Classical is essentially unchanged (9.5M → 9.6M tokens) because dedup removed
Wikisource's copies of Perseus texts, so **Latin-0's sizing is untouched**.

---

## How to run it

```bash
source .venv/bin/activate

python src/pipeline.py --status        # what is stale and why
python src/pipeline.py                 # run only what changed (~7-10 min)
python src/pipeline.py --from normalize

python src/train.py --dry-run          # sizing before committing to a run
python src/compare.py                  # all runs side by side
python src/evaluate.py --baselines --corpus corpus
python src/sft.py --chat               # talk to latin-2-chat
python src/provenance.py --check       # licence audit; non-zero exit on drift
```

Pipeline stages: `fetch → extract → enrich → normalize → dedupe → build →
provenance`. Fingerprints **chain** (`sha256(previous + own config + own
source)`) rather than inspect files, because three stages rewrite
`data/interim/` in place and mtime staleness would mark a stage stale the moment
it finished.

---

## Decisions worth not relitigating

- **4,000-token vocabulary.** Originally chosen on parameter budget; that
  argument is gone now that the corpus supports a larger model. It still wins on
  morphology: whole paradigm forms stored as single tokens run 0/12 at 4k, 3/12
  at 8k, 10/12 at 64k. A 64k vocabulary has memorised the inflections this
  project exists to watch a model learn.
- **Fold v→u and j→i.** The reverse needs morphology to know which `u` is
  consonantal. Sources used *opposing* conventions (v as share of u+v: Perseus
  12.9%, CSEL 0.60%), so this was not cosmetic.
- **Validation held out by whole author**, not whole work. A work-level split
  leaked 58% of validation words through authors present in training.
- **Duplicates are marked, not deleted** — auditable, and policy can change
  without re-extracting.
- **Dates are `inferred` where derived.** PL volume numbers date 22% of
  Wikisource because Migne ordered the Patrologia chronologically by author
  death. The map is anchored on principal authors, not linear — Migne gives
  Augustine sixteen volumes and the tenth century six.

---

## Gotchas, learned the hard way

- **Only 109 of 325 Perseus `ti:edition` elements carry `xml:lang`.** Trusting
  that attribute silently discarded two thirds of the corpus.
- **Terence's plays hide an English INTRODUCTION inside an `<l>` element**, and
  in two plays it carries no language attribute at all — hence the stopword
  filter using English words with zero Latin collision.
- **`<pb/>` and `<lb/>` fall mid-word.** Emitting a space turned Gellius's
  *Athenis* into "A the nis".
- **The `n` in fertility measurement must cover every record.** It once sampled
  `records[:400]`, which in file order contained no Perseus at all.
- **`None` is not an author.** Counting it as one made an all-anonymous corpus
  report 61% validation leakage when real leakage was zero.
- **`0.0` is falsy.** `fraction or default` made the memorisation self-test
  report FAILED while behaving perfectly.
- **Baselines are corpus-specific.** Quoting the full-corpus floor against a
  classical-corpus model overstated latin-0 by 23%.
- **Run the foreign-script filter before the homoglyph map**, or whole Cyrillic
  paragraphs get laundered into pseudo-Latin.
- **`docs/external-review-2026-08-15.md`** is an independent review by OpenAI
  Codex that found five real bugs. Worth re-reading before trusting any claim in
  this repo.

---

## Next steps, in the order I would take them

1. **Train LATIN-XXV** — ~25–30M params, 4 epochs, full corpus, then Romanitas.
   ~8–10 hours at the observed 9k tok/s for 15.9M params (expect slower).
   This is the first model that would actually use the new corpus.
2. **Re-run the chat SFT** on that base. `src/dialogue.py` mines 54,536 pairs;
   re-mine after the corpus change. The proof worked on latin-2: turn-taking and
   stopping are learned, register matching emerges, correctness does not.
3. **Serve it.** MLX + safetensors, small model, runs on the laptop.
4. **Remaining four sources**, if more data is wanted.
5. **Disjoint-phase ablation** — exclude classical from Phase 1 to separate
   transfer from extra exposure.
6. **Cluster-level dedup review.** "0 of 59 below the verbatim floor" is the
   precise claim; nobody has eyeballed the clusters.

### Open questions

- **43% of the corpus is UNASSIGNED for period.** Mostly anonymous Wikisource
  material. Period tags are injected only where a period exists, so those
  documents train untagged. Acceptable but not ideal.
- **Musisque Deoque** — the complete corpus of ancient Latin poetry with
  apparatus — sits at tier `unknown` because no licence is stated. It is the
  most valuable outstanding permission request in the project.
- **PHI** has essentially all classical Latin and is not openly licensed.
  Writing to ask remains the single highest-leverage action available.

---

## Repository map

```
CURRENT_STATE.md   this file
README.md          how the pipeline works, with measured figures
plan.md            goals, model ladder, the chat target
feasibility.md     the arithmetic and the licensing decision — §2 is the ladder
sources.md         original licence survey (historical; manifest is authoritative)
ATTRIBUTION.md     generated by provenance.py — do not hand-edit

manifest/sources.yaml       what may enter the corpus, and under what licence
manifest/authorities.yaml   author/work dating + PL volume anchors
manifest/genres.yaml        form and genre
manifest/fetch.lock.json    resolved commit SHAs — commit this

src/                        pipeline, one file per stage; see README layout
reports/                    versioned results — the source of truth for figures
docs/pipeline.html          published explainer (artifact)

data/       gitignored, ~2GB, regenerable from the manifest
checkpoints/ gitignored, ~380MB, regenerable from src/train.py
```

Remote: `git@github.com:y-a-v-a/claudius.git` (private). Working tree should be
clean; if it is not, that is unfinished work rather than intent.

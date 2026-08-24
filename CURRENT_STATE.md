# Current state — pick-up notes

Written 2026-08-17 as a handoff into a fresh session; updated 2026-08-25.
`README.md` explains how things work; this file says where they stand, what is
stale, and what to do next.

---

## Update 2026-08-25 — Track B has a source, and it makes the corpus smaller

Track B has existed in the manifest and the guards since the two-track split was
designed, and had never contained a single source. It does now: **LASLA**,
1,801,734 words of hand-verified classical Latin under CC BY-NC-SA 4.0.

It turned out to be fetchable. The manifest had it as `method: manual`; the
CIRCSE mirror publishes the whole corpus as CoNLL-U Plus under the same licence,
so it is now a pinned git source read by a metadata shim over the CoNLL-U parser
DigilibLT already uses.

### The measured result is negative

| corpus | records | words | train tokens |
|---|---:|---:|---:|
| Track A · full | 10,177 | 78,554,197 | **152,234,083** |
| Track B · full | 10,161 | 78,382,944 | **151,793,247** |
| Track A · classical | 203 | 4,531,115 | **9,539,569** |
| Track B · classical | 201 | 4,345,471 | **9,198,341** |

**Track B is 440,836 tokens smaller than Track A (−0.29%), and 341,228 smaller
on classical (−3.58%).** Adding a source shrank the corpus.

Of LASLA's 242 records, **30 survive** — 85,352 words. Its 19 authors are
Perseus's core, so 95% of it is a duplicate of text already held. Meanwhile its
presence caused **46 green records, 256,605 words, to be newly marked duplicate**
in Track B. Net: **−171,253 words**.

The largest single displacement is Perseus's *Institutio Oratoria*, 172,355
words, handed to a Vicifons copy — and Quintilian is not in LASLA at all. LASLA
records grew the cluster enough to push Perseus's copy below the containment
cutoff. That instability is worth naming: **dedup verdicts about one pair can
change because an unrelated record joined the cluster.**

### Two bugs found on the way, both structural

**A green record could be marked a duplicate of an amber keeper.** A Track A
build then drops the duplicate for being a duplicate and the keeper for being
non-redistributable, and the text leaves the published corpus with nothing
reporting it. Keeper ranking is now (licence tier, curation rank). Measured
extent before the fix: 3 records, 5,028 words.

**That fix was not enough, because the damage happens before keeper choice.**
Union-find is transitive, so an amber record similar to two green records that
are not similar to each other merges their clusters. dedupe now resolves each
track separately — restricting the candidate *pairs*, not merely the keeper —
and writes two verdicts per record. build.py reads the one matching its track.

The guarantee this buys is what the two-track split always implied and never
had: **Track A does not depend on which research-only sources are on disk.**
Verified — with LASLA fetched and extracted, Track A rebuilds to exactly
78,554,197 words, 2,802 duplicates and floors of 174.9 / 149.5, every figure
identical to the build before LASLA existed.

### What this says about the remaining NC sources

Corpus Corporum (215M words) and IntraText (10M) are still `method: manual` —
neither publishes a bulk endpoint and Corpus Corporum needs a human at
mlat.uzh.ch. More to the point, Corpus Corporum **contains the entire Patrologia
Latina and Perseus**, both of which this corpus already holds via Vicifons and
Perseus directly. Its headline 215M is not 215M new words, and on the LASLA
evidence the net could plausibly be negative there too.

**LASLA's value was never the text.** The manifest says so already: it is the
gold standard for morphological evaluation, and evaluation data is never
redistributed, so NC costs nothing there. Its training contribution is measured
at −171,253 words. It should be kept for evaluation and understood as
contributing nothing to training.

---

## Update 2026-08-21 — corpus rebuilt clean; the pipeline proved reproducible

The rebuild carrying the interwiki fix, the 930 datelines, format dispatch and
`text_provenance` ran in **22.9 minutes**, and the pipeline now reports **0 of 9
stale**.

| | before | after |
|---|---:|---:|
| records | 10,194 | **10,177** |
| words | 78,974,350 | **78,554,197** |
| CLASSICAL | 4,539,518 | **4,531,115** |
| tokens | 153.0M | **152.2M** |
| floor, full / classical | 174.5 / 149.8 | **174.9 / 149.5** |

The classical corpus lost 8,403 words, which is the 930 datelines at about
eight words each. The full corpus lost 420,153 — and that is **not** removal but
**dedup keeper churn**: 220 records left and 224 arrived, because trimming
datelines shifts overlap estimates enough to flip which member of a near-equal
duplicate pair survives. Worth knowing as a property: corpus composition is
sensitive at the margin to trivial text changes.

**The provenance mix is real now: 94.15% transcribed, 5.85% ocr** — CSEL's 4.6M
words after dedup. **Classical is 100% transcribed**, so the subset the flagship
specialises on contains no OCR-derived text at all.

### The guard worked, and then a deeper one was missing

`compare.py` flags LATIN-XXV `!` and withholds its ratio. The floors moved 0.2%
— close enough that anyone would have kept quoting 5.03× — and the system
refused because the model was measured on different text. That is the fingerprint
work paying off on the first real corpus change.

**But the same rebuild destroyed LATIN-XXV, and nothing complained.** Not the
weights: the 4,000-token vocabulary they are keyed to, which `tokenizer.py`
rewrote in place. A model's embeddings map token 1234 to whatever that file said
1234 was; hand it another vocabulary and it does not fail, it writes confident
nonsense. The fingerprints stop a stale *number* being quoted; nothing stopped a
stale *model* being run.

Fixed: `tokenizer.py` keeps a content-named copy of every vocabulary and never
overwrites one, and `sample.py`/`score.py` resolve the vocabulary from the run's
own fingerprint, refusing with both hashes named rather than mis-tokenising.

### And the recovery proved something the project had only claimed

LATIN-XXV's vocabulary was rebuilt from a git worktree at commit `014785a`, with
`data/raw` shared read-only. **Every figure reproduced exactly** — 2,794
duplicates, 923 clusters, 10,194 records, 78,974,350 words, 4,539,518 classical —
and the vocabulary hashed to `24d1515e65807859`, bit for bit. The model now loads
and emits byte-identical output to before.

That is the first end-to-end test of the claim the whole manifest-and-lockfile
design rests on: **the corpus is reproducible from pinned sources and code.** It
had never been checked. It holds.

---

## Update 2026-08-19 — LATIN-XXV trained

**The flagship exists.** 27.49M parameters, trained from random initialisation
on Latin alone, two phases, 25.9 hours total.

| run | params | corpus | ep | ppl | floor | x floor | memo |
|---|---:|---|---:|---:|---:|---:|---:|
| latin-0 | 3.53M | classical | 8.6 | 74.3 | 158.8 | 2.14x | 0.0% |
| latin-2 | 15.90M | classical | 3.0 | 44.1 | 158.8 | 3.60x | 0.0% |
| latin-xxv-latinitas | 27.49M | full | 4.0 | **25.2** | 174.5 | **6.93x** | — |
| **latin-xxv** | 27.49M | classical | 3.0 | **29.8** | 149.8 | **5.03x** | **0.0%** |

**Only the `x floor` column compares across those rows.** The raw perplexities
sit behind different corpora, tokenizers and held-out sets. `compare.py` prints
ratios for the two new runs — whose corpus *and* tokenizer fingerprints match
their floors — and refuses to print one for the four older runs, which is the
guard working rather than a gap in the data.

**Memorisation, the result this project turns on: 0 of 1,339 twelve-word
windows** appear anywhere in the training text; nothing survives at eight
words either. Read the short windows honestly though — at 3 to 6 words the
generated text overlaps training *more* than the held-out control, inverting
what latin-2 showed. The control scores 1.2% at three words against latin-2's
12.7%, on 6,710 words, so it is more likely a small-control artifact than a
signal. It should not be quietly dropped.

**The train/val gap is 1.270x, below latin-2's 1.34x at 15.90M parameters.** A
model 73% larger that overfits *less* is the corpus expansion earning its keep.
I would have predicted the opposite.

### What it writes

```
CICERO UENERIO S.

etsi in re publica minime sunt ea quae iam inculta et prouisa sunt, tamen
non sunt ea quoquo modo dicta omnia, quia non modo a nostris rebus, sed
etiam ab externis oratio nostra inlustratur ...
```

`CICERO UENERIO S.` is the correct epistolary salutation. `etsi ... tamen` and
`non modo ... sed etiam` are correlative pairs completed correctly across long
clauses, and elsewhere `quid agendum sit` puts a gerundive inside an indirect
question. It also switches register cleanly, dropping into Plautine dialogue
with speaker names. Compare latin-2 on the same prompt — *"Gallia est omnis
diuisa cum Secundamen, quique umeros et nigra lotos Frigidus"* — which was
poetic vocabulary in word-salad order. The syntax arrived; meaning still has
not, and headings are still invented (`PROLOGUS FRAGMINUS`).

### A sixth bug, found by reading the samples

The model generates `Scr. in Formiano u aut iu Non. Mai. a. 704` — which is a
**Perseus editorial dateline**, not Cicero. There are **927 of them in the
classical corpus**, all in the four books of letters. README §4.4 claims
residual apparatus is "9 occurrences in 5.6M words"; this is a systematic
editorial convention two orders of magnitude larger, and the model demonstrably
learned to reproduce it. Not yet fixed — see the queue below.

---

## Update 2026-08-18 (night) — corpus rebuilt, floors restated

The rebuild the previous update asked for has been done, and everything
downstream of it re-measured. **LATIN-XXV Phase 1 is training now.**

### The rebuild changed almost nothing, which is the result

| | before | after |
|---|---:|---:|
| records | 10,192 | **10,194** |
| words | 78,975,089 | **78,974,350** |
| CLASSICAL | 4,539,515 | **4,539,518** |
| duplicates | 2,799 rec / 29,805,223 w | **2,794 rec / 29,804,101 w** |

A net −739 words. The homoglyph-ordering fix was worth about what the previous
update predicted. Every `resolved_pin` in the lockfile came back identical, so
the difference is the code fixes and nothing else.

**Token counts are unchanged**: 2.038 tok/word overall, 2.213 classical,
152,985,050 train tokens, 9.6M classical, 0/12 paradigm forms stored whole.
The whole of `feasibility.md` §2 stands as written.

### New floors — the old ones are dead

| corpus | old | **new** |
|---|---:|---:|
| full | 195.2 | **174.5** |
| classical | 158.8 | **149.8** |

The full-corpus floor fell because a bigram trained on 4.5× more text is
simply better. **The classical floor fell for a different and more instructive
reason**: that corpus changed by *three words*, but the 4k vocabulary had been
retrained on a corpus 4.6× larger, so the merges differ and the same Latin
becomes a different token stream. A perplexity is comparable only when the
corpus *and* the tokenizer match — see the guards below.

### LATIN-XXV, and the constraint nobody had measured

Running: **27.49M params** (dim 512, 8 layers, 8 heads, ctx 512), batch 16,
74,700 steps = **4.0 epochs, 22.3 tokens/param**. ETA ~24h at 7,300 tok/s.

The interesting finding is why batch 16 and not 32. Throughput at this size is
governed by **activation memory, not parameter count**, and this machine has a
cliff:

| batch × ctx | effective tok/s |
|---|---:|
| 8 × 512 | 7,272 |
| **16 × 512** | **7,347** |
| 24 × 512 | ~8,780 (eval-inflated) |
| 32 × 512 | **~735** |

Batch 32 is **13× slower per token** — it stops fitting and falls into swap.
The first probe at batch 32 ran 45 minutes without completing 100 steps while
holding 7.5GB on a 16GB machine. Throughput is otherwise flat from batch 8 to
16, so batch 16 is peak speed at half the memory of the wall. The exact cliff
position depends on what else is resident, which is itself worth knowing:
**this is not a machine on which the config can be assumed.**

### Corpus purity, measured rather than asserted

The letter inventory is **86 letters and 246 distinct characters**, not the 48
README quotes from the three-source corpus. Measured properly: **38 distinct
non-Latin letters, 765 occurrences in 75,043,073 training words** — one per
~98,000. Most are not contamination: 198 are Roman numeral apostrophus
(`HS ccciↄↄↄ`), 267 are eth/thorn (Ammianus describing a letter's *shape*, a
Gothic *Skeireins* passage in the Patrologia), 172 are one OCR artifact in
Kepler's *Somnium*, 31 are mojibake Greek in Perseus' *Adversus Marcionem*.

**One genuine leak.** Every Thai character traces to Wikisource interwiki
language links that survived extraction — and alongside them English:
`en:Jingle Bells`, `en:O come all ye faithful`. Extent: **287 records of 9,894
carry an `xx:Title` pattern, 620 occurrences**, of which 125 are `en:`; only
*Utopia* shows the classic multi-language block.

**Fixed in `extract.py` (`MW_INTERWIKI`), but not yet applied to the corpus.**
`data/` predates it, and so does the running LATIN-XXV. `pipeline.py --status`
correctly reports extract onward stale; rebuilding is ~35 minutes and wants a
free machine. Eight occurrences per million words moves no perplexity, but "no
English" is a categorical claim and a small numerator does not defend it.

### Bugs found and fixed tonight

1. **`evaluate.py` stamped every report with the full corpus's fingerprint**
   regardless of `--corpus`, because `write_report` called
   `corpus_fingerprint()` with its default argument. `baselines-classical.json`
   claimed 17.2M words while measuring 4.5M.
2. **`train.py` located the floor by filename alone.** A corpus keeps its name
   across rebuilds. It now compares fingerprints and refuses to quote on drift.
3. **`compare.py` would have tabulated the old models against the new floor** in
   one tidy `x floor` column. Rows now carry `ok` / `!` / `?`.
4. **The lockfile's timestamps invalidated the entire pipeline.** `extract` was
   fingerprinted on the *bytes* of `fetch.lock.json`, which `fetch.py` rewrites
   every run — so one fetch marked all six downstream stages stale and "run only
   what changed" was silently "rebuild everything, always". Now keyed on a
   projection (pins, checksums, licence, tier), and each stage is
   re-fingerprinted after it runs, since fetch mutates its successor's input.
   **Verified: a refetch that rewrites the lockfile now leaves 0 of 7 stale.**
5. **The tokenizer is half the identity of a perplexity** — see the classical
   floor above. Reports now carry a tokenizer hash and the guards check it.
6. **dedupe ran silent** through 12,988 sketches and 1,105,761 pair scores. Two
   rebuild attempts were reaped at exactly that point. Both loops now report.

Tests: **132**, still no corpus/network needed; `tests/test_train.py` skips
without MLX.

### Left undone, deliberately

- ~~Both baselines need re-running~~ **done 2026-08-19**; both reproduced
  exactly (174.5, 149.8) and now carry tokenizer stamps.
- **`dedupe.py --verify` has not been run at five sources.** README's "0 of 59
  below the 15% floor" was measured at three; there are now 2,794 duplicates, so
  that claim covers ~2% of what is being discarded.
- Long-running work must be launched **detached** — macOS has no `setsid`, so
  see `scratchpad/detach.py`. Anything left in the caller's process group gets
  reaped, which is what killed the first two rebuild attempts.

---

## Update 2026-08-18 — test suite added, two extraction fixes pending a rebuild

- **`tests/` now holds a pytest suite** (104 tests, ~0.2s, no corpus/network/MLX
  needed): `.venv/bin/pytest`. It covers the pure logic of normalize, dedupe,
  extract, enrich, build's validation split, dialogue mining, the evaluation
  instruments, fetch/provenance guards, pipeline fingerprints, and a miniature
  BPE round-trip. pytest is in `requirements.txt`.
- **Two bugs the tests surfaced were fixed, and `data/interim/` predates them:**
  1. `normalize.py` — the inline foreign-script strip ran *before* the homoglyph
     map, so the Cyrillic homoglyph repairs were dead code (`mаle` → `mle`
     instead of `male`). The map now runs first; the paragraph-level foreign
     drop still precedes both, as its comment requires.
  2. `extract.py` — with `--greek tag`, markup-tagged Greek was wrapped in a
     second, nested `<GRAECUM>` by the inline pass. Inline tagging now skips
     already-tagged spans. (Default policy is `drop`, so existing data is only
     affected if it was extracted with `tag`.)
- **Action: run `python src/pipeline.py` before the next training run.** The
  stage fingerprints cover the scripts themselves, so extract-onward will
  re-run automatically. Expected impact is small (~130 homoglyph occurrences),
  but the corpus on disk was produced by the old code until this is done.

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
tokenizer → baselines → provenance`. Extract is partitioned per source, so
adding one re-extracts only that one. Fingerprints **chain** (`sha256(previous + own config + own
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
- **Ordering around the homoglyph map is delicate.** The paragraph-level
  foreign-script drop must run before it, or whole Cyrillic paragraphs get
  laundered into pseudo-Latin — but the inline character strip must run *after*
  it, or the homoglyph repairs become dead code (fixed 2026-08-18; see the
  update at the top).
- **`docs/external-review-2026-08-15.md`** is an independent review by OpenAI
  Codex that found five real bugs. Worth re-reading before trusting any claim in
  this repo.

---

## Next steps, in the order I would take them

~~1. Train LATIN-XXV~~ **done 2026-08-19** — 27.49M params, 5.03x the floor,
0% memorisation. It took 25.9 hours, not the 8–10 estimated here: throughput is
bounded by activation memory rather than parameters, see the batch-size cliff.

1. **Rebuild the corpus.** `extract.py` has an interwiki fix committed but not
   applied, so `pipeline.py --status` reports extract onward stale. ~35 minutes
   on a free machine. Consider fixing the 927 Perseus datelines first so both
   land in one rebuild — and note that a rebuild invalidates LATIN-XXV's floors,
   which is exactly what the fingerprint guards will tell you.
2. **Strip the Perseus editorial datelines.** 927 `Scr. ...` headers in Cicero's
   letters, which the model learned to generate. See the sixth bug above.
3. **Re-run the chat SFT** on LATIN-XXV. `src/dialogue.py` mines 54,536 pairs;
   re-mine after the corpus change. The proof worked on latin-2: turn-taking and
   stopping are learned, register matching emerges, correctness does not.
4. **Serve it.** MLX + safetensors, small model, runs on the laptop. This is the
   stated goal in `plan.md` and LATIN-XXV is the base it was waiting for.
5. **`dedupe.py --verify` at five sources** — the outstanding claim gap.
6. **Remaining four sources**, if more data is wanted.
7. **Disjoint-phase ablation** — exclude classical from Phase 1 to separate
   transfer from extra exposure. Still open, and now more affordable to answer
   at 15.9M than at 27.5M.
8. **Best-checkpoint tracking in `train.py`.** It saves the latest, not the
   best. LATIN-XXV's validation plateaued between steps 1,250 and 1,500 of
   Phase 2 before improving again; had it not recovered, the saved weights
   would have been worse than a checkpoint nothing kept.

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

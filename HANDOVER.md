# HANDOVER — OCR repair with LATIN-LII

State of the `ocr/evaluation` branch as of 2026-09-24, written for whoever
picks this up next (including a future session of the assistant that built
it). Everything here is on the **dev** (practice) splits. **The test splits
have never been run.** All work is committed on `ocr/evaluation` and pushed.

The full, plain-words account — findings, charts, and next steps rated for
value — is **`docs/ocr-explainer.html`**. This file stays high level.

In every table: ↑ higher is better, ↓ lower is better.

## What was asked, and what was decided along the way

1. *Define how OCR repair by LATIN-LII will be evaluated, as a subproject.*
   → `ocr/` (2026-09-17, committed as `7d1baf8`).
2. *Keep the harm limit at 2 words broken per 1,000* — confirmed by Vincent
   as a deliberately high standard. Recorded in `ocr/README.md`.
3. *Work through Option 1 (train the model, not a rewriter) in the order:
   (1) get under the harm limit, (2) fill-in-the-middle training, (3) build a
   real-scan test set; stop before Option 2.* → done 2026-09-17/18.
4. *Fix the proposer.* → done 2026-09-18/19; measured.
5. Standing preferences: mark ↑/↓ on every numeric column; gloss ML terms
   in plain words (Vincent is new to LLM building). Both saved to memory.
6. *The goal behind it all* (2026-09-23): download OCR text from archive.org,
   correct it well enough, and add it to the training corpus. → measured
   2026-09-23/24; see *archive.org* below.

## The question, and the answer so far

> Given the same candidate corrections and the same model of the noise, does
> reading the context with LATIN-LII repair more scanned Latin, and break
> less, than counting word frequencies does?

Yes, on both tiers, at the same harm. But on real scans the absolute numbers
are low, and the reason is now known and is not the model (see *Findings*).

## Layout

| path | what |
|---|---|
| `ocr/README.md` | The protocol: metrics, guards, hypotheses H1–H7, results §1–4, conclusion. **§1, §2 and §4 tables still show the pre-proposer numbers; update from the tables below.** |
| `ocr/metrics.py` | Alignment, five outcomes (kept/broken/fixed/missed/miscorrected), silent errors, bootstrap, `locate`. Pure. |
| `ocr/channel.py` | Synthetic noise (`corrupt`) and `fit`, which learns a channel from pairs. Repairers may use `fit`, never `CONFUSIONS` (a test enforces it). |
| `ocr/repairers.py` | `Rule`/`decide` (the decision), `Proposing` (the shared proposer), `Unigram`, `Lii`, `Fim`, `make(spec)`. |
| `ocr/bench.py` | `build` Tier 1 (+ `clean` control), `run`, `compare`. Reports → `reports/ocr-*.json` (Tier 1) and `reports/ocr-t2-*.json` (Tier 2). |
| `ocr/tune.py` | `trace` (record every decision once), `sweep` (replay under 67 rules, save the winner to `rules.json`), `ceiling` (recall with a perfect chooser = proposer ceiling). |
| `ocr/tier2.py` | `scan`/`fetch`/`build` the real-scan benchmark from Wikisource + Commons OCR layers. |
| `ocr/rules.json` | Tuned rule per system per tier, with what it measured when tuned. |
| `ocr/bench.lock.json`, `ocr/tier2.lock.json` | Split hashes. `run` refuses a split that does not match. |
| `src/fim.py` | Fill-in-the-middle continued training. Produced `checkpoints/latin-lii-fim.*` and `reports/train-latin-lii-fim.json`. |
| `ocr/ia.py` | archive.org: `find` / `fetch` / `compare` (old vs current OCR, no model) / `build` (Tier 3). |
| `ocr/ia-candidates.json`, `ocr/ia.lock.json` | The 66 archive.org items behind Tier 2 books, and Tier 3's split hashes. |
| `reports/ocr-ia-*.json` | `compare` (re-reading), `overlap` (already in corpus), and model runs on `current` / `then` / `now`. |
| `tests/test_ocr.py`, `tests/test_fim.py` | 47 + 7 tests, no model needed. Full suite: 592 pass (2026-09-24). |
| `manifest/sources.yaml` | New entry `wikisource-scan-ocr`, tier `eval`, `include: false`. |
| `data/processed/ocr/` | Benchmarks, lexicon caches, traces (gitignored, rebuildable). |
| `data/raw/wikisource-scan-ocr/` | 157 OCR layers, 122 MB, never edited. |
| `data/raw/archive-org-ocr/` | 63 archive.org OCR texts, page by page, 39 MB, never edited. |

## Benchmarks

**Tier 1 — synthetic damage.** Held-out (by author), `transcribed` provenance
only. dev 240 / test 560 passages of 60 words at start-rates 0, 1, 3, 8%;
plus `clean`: 600 undamaged passages from dev authors for measuring harm.
Locked.

**Tier 2 — real scans.** Wikisource `Pagina:` transcriptions (proofread
level ≥ 3) against the text layer of the scan they were made from. dev 652
passages / 31 books, test 1,326 / 64 books, CER ~5–6%. 2,994 passages dropped
for sharing 8 running words with training. Both sides normalised and tidied
(`¬` line breaks, space before punctuation). Cap 24 passages per book. Locked.

**Tier 3 — archive.org's own OCR** (`ocr/ia.py build`). Same proofread pages,
archive.org's current text. Leaves out Tier 2's *test books* entirely and
Tier 2's passages; drops anything already in the corpus; cap 40 per book.
Splits: `current` 728 passages / 25 books; `then`/`now` 70 passages / 2
books, the same passages in the old (Commons) and re-read OCR. Never tuned
on: `bench.py --tier 3` uses Tier 2's rules and Tier 2's fitted channel.
Locked.

## Systems (`ocr/repairers.py`)

- `identity` — do nothing; must score zero everywhere.
- `unigram` — shared proposer, scored by word frequency (the yardstick).
- `lii` — same proposals, scored by P(candidate + right context | left) under
  LATIN-LII. `lii-left` = no right context. `lii@N` = N words of right context.
- `fim` — same proposals, scored by the fill-in-the-middle checkpoint with
  the suffix in front. `fim+samples` adds the model's own draws as proposals
  for unknown words (currently inert, see Findings 6).

The **decision** (`Rule`): a change must beat leaving the word alone by a
margin, in nats, that depends on the kind of change (known word, capital,
join, split, short word, repeated on page). Numerals are never touched.
Tuned per system per tier by `tune.py sweep` to harm < 1.5/1,000 (limit 2),
measured on `clean` (Tier 1) or over all standing words (Tier 2 — its clean
control is only 3,060 words).

## Results with the current proposer (dev, all at the 2‰ limit)

Tier 1, simulated damage, 240 passages:

| system | recall ↑ | real-word ↑ | non-word ↑ | precision ↑ | broken/1,000 ↓ |
|---|---|---|---|---|---|
| unigram | 46.7% | 36.7% | 49.8% | 91.7% | 1.41 |
| lii | 67.1% | 49.6% | 72.5% | 92.2% | 1.58 |
| fim | 68.4% | 53.0% | 73.1% | 90.7% | 1.91 |

Tier 2, real scans, 652 passages:

| system | recall ↑ | real-word ↑ | non-word ↑ | precision ↑ | broken/1,000 ↓ | detection AP ↑ |
|---|---|---|---|---|---|---|
| unigram | 19.3% | 18.9% | 19.5% | 81.2% | 1.45 | 42.7% |
| lii | 25.0% | 11.6% | 31.9% | 90.7% | 1.26 | 62.1% |
| fim | 27.3% | 13.3% | 34.4% | 90.8% | 1.23 | 62.8% |
| fim+samples | 26.1% | 14.6% | 31.9% | 89.4% | 1.17 | 62.8% |

Paired 95% intervals, Tier 2: fim − lii recall +1.5 .. +3.1, harm −0.27 ..
+0.21. lii − unigram: real-word −9.4 .. −5.3, non-word +10.5 .. +14.3.

Proposer ceiling (recall with a perfect chooser): Tier 1 73.7% → **77.9%**;
Tier 2 48.1% → **57.4%** (+samples: 57.5%).

## Findings, in the order they were learned

1. **The harm limit is met by the decision, not the model.** Raw-sum
   scoring broke 6/1,000; a rule that demands evidence in proportion to
   risk brings it to ~1.4 at a cost of ~10 points of recall on Tier 1.
2. **Simulated damage overstates real repair ~2.5×.** Any figure quoted as
   "OCR performance" must come from Tier 2.
3. **The proposer was the wall on real scans** — half of all damage never
   had its right word offered. Rebuilt 2026-09-18: case-keeping channel
   steps (restores `loue → Ioue`), lexicon search within 3 edits (bucketed;
   brute-force depth 3 was 95 s/passage), symbol wildcards (`%uam`),
   segmentation into up to 4 words with one misread part allowed
   (`Almafalutiferam → Alma salutiferam`), case and edge-punctuation
   variants, pool cap 8 → 16. Ceiling 48 → 57%. 0.7 s/passage on CPU.
4. **…but the safety limit ate the gain on real scans.** Before the limit
   the model systems went 43% → 48%; at the limit 27.1% → 27.3%. On Tier 1
   the gain shows in full (60 → 67%).
5. **The 2‰ limit on real scans is spent on errors in the truth.** Under the
   tuned rule `fim` breaks 48 words in 32,500. ~35 of them are transcription
   errors it correctly fixed (`cceperis → coeperis`, `arcliidiacono`,
   `fyncerus`, `oetatem`); genuine harm ≈ 10 (`muticus → musicus`,
   `uirtutei → uirtute`, `benefitio → beneficio`). Split: in-lexicon truth
   words 0.33/1,000 (10 words), out-of-lexicon 17.9/1,000 (38 words). The
   rule declines **1,756** damaged words whose correct form it was offered
   (it fixes 2,003). At margin 2 instead of 6: recall 40.5%, in-lexicon
   harm 1.02.
6. **Fill-in-the-middle training worked as a scorer, not as a generator.**
   12,000 steps (3.6 h): plain held-out loss 3.058 → 3.063, FIM loss 4.246
   → 3.149; +2 points recall on both tiers. But P(<|endoftext|> | true fill)
   ≈ 0.000 in every case tried and draws ignore the suffix, so `+samples`
   proposes only left-to-right guesses and adds 0.1 to the ceiling.
7. Model systems lose to frequency on real-word damage in real scans
   (long-s: `femen` for `semen`) and win by 10–17 points on non-words.
   Detection (where is the damage?) holds up: AP 62% vs 43%.
8. Page-level features (`repeated`, `seen` in `Rule`) are implemented but
   inert at 60-word passage scale; they need whole-page context.
9. A wider right context (12 words) does not help either scorer.

## archive.org (2026-09-23/24) — high level

Details, charts and per-book numbers: `docs/ocr-explainer.html`, Part 2.

1. **Tier 2 was already measuring archive.org.** 40 of 66 archive.org items
   behind Tier 2 books carry the same OCR as the Commons file; 13 have been
   re-read by archive.org since; 9 exist only on archive.org.
2. **A fresh OCR reading beats repair on post-1850 print.** Same passages, no
   model: 9 re-read books 11.2% → 5.0% chars wrong (control: 39 unchanged
   books 5.34% vs 5.21%). Books of 1889–1912 lose 70–90% of errors; 1495/1619
   11–19%; 1658 gets worse. Median book −19%.
3. **~45% of the proofread text of these books is already in the corpus**
   (8-word runs; 44.6% of all 27,273 passages, 44.8% over the 56 books with
   ≥ 40, as charted). Classical texts ~100%; neo-Latin and teaching Latin ~0%.
4. **Repair carries over, a little weaker.** Tier 3 `current`, fim: chars
   wrong 4.71% → 4.05%, words wrong 16.0% → 12.7%, recall 21.2%, broken
   1.01/1,000 (Tier 2: 27.3% recall). fim vs unigram: non-word +12..+17,
   real-word −2.8..+1.1, harm −1.10..−0.05 (fim breaks fewer).
5. **Re-read then repair** (`then`/`now`, 2 books — illustrative only):
   9.30% → old+repair 8.81% · re-read 4.44% → re-read+repair 3.99%.
6. **Per book after repair:** 11 of 25 end ≤ 2% chars wrong; 7 stay > 5%
   (early print, mixed languages, fragments, mathematics).
7. **Speed:** ~4.7 s per 60-word passage per model system (57 min for 728).

## Open decisions (Vincent's)

1. **What the harm limit is measured against on real scans**, given
   Finding 5. Options: (a) keep as is; (b) apply 2‰ to lexicon-known truth
   words and report out-of-lexicon harm separately as an upper bound;
   (c) hand-adjudicate the ~50 broken words per run. This is worth more
   recall than any further proposer work.
2. Whether to replace Tier 1's hand-written noise table with the channel
   fitted from Tier 2 pairs (re-locks Tier 1; every Tier 1 number changes).
3. Whether the rules in `rules.json` as tuned are what the test split runs
   under. After the first `--split test` run the hypotheses in the README
   are frozen.
4. ~~Commit the uncommitted work.~~ Done 2026-09-23.
5. **What "clean enough for the corpus" means** — a threshold on estimated
   characters wrong per page or book (the explainer uses 2% as a placeholder).

## Suggested next steps, in order

Re-rated for the archive.org goal in `docs/ocr-explainer.html`; in short:
**quality gate** (estimate a page's error rate without a transcription) and
**sizing the harvest** (how much new, recent-enough Latin archive.org has;
no model needed) now come first, then the items below.

1. Decide (1) above; re-tune; then run `--split test` once on both tiers.
2. Update `ocr/README.md` §1, §2, §4 tables to the numbers in this file
   (they still show the pre-proposer run).
3. Speed: batch several words per forward pass in `Lii.score`/`Fim.score`
   (currently one model call per word; a trace is ~30 min per 650 passages).
4. If a generative proposer is still wanted: retrain FIM with the loss
   weighted on the middle and its closing token (or a dedicated end-of-fill
   token), then re-test `fim+samples`. Names outside the lexicon are 30% of
   what is still unproposed and only a generator can reach them.
5. Ablation: train without the `ocr`-provenance CSEL text (build.py caps)
   and rerun — 5.85% of corpus-b is OCR-derived.
6. Option 2 (a direct rewriter) only after 4, and only with Tier 2 as the
   judge; its danger is exactly what `broken`/`miscorrected`/`silent` count.

## How to run

```bash
source .venv/bin/activate
pytest tests/test_ocr.py tests/test_fim.py          # no model needed
python ocr/bench.py run unigram lii fim --quiet     # Tier 1 dev, ~20 min
python ocr/bench.py run unigram lii fim --tier 2    # Tier 2 dev, ~1 h
python ocr/bench.py compare reports/ocr-t2-fim-latin-lii-fim-dev.json reports/ocr-t2-lii-latin-lii-latinitas-dev.json
python ocr/tune.py trace fim --tier 2 && python ocr/tune.py sweep fim:latin-lii-fim --tier 2 --frontier --save
python ocr/tune.py ceiling unigram --tier 2         # proposer ceiling
python ocr/bench.py run unigram lii fim --split test --tier 2   # NOT YET RUN — freezes the hypotheses
python ocr/ia.py find && python ocr/ia.py fetch     # archive.org candidates + their OCR
python ocr/ia.py compare                            # old vs current OCR, no model, ~15 min
python ocr/ia.py build                              # Tier 3 splits, locked
python ocr/ia.py overlap                            # share of each book already in the corpus
python ocr/bench.py run identity unigram lii fim --tier 3 --split current   # ~2 h
```

## Gotchas

- Everything model-side is MLX on this Mac; one GPU job at a time, or both
  slow down. `_common.py` caps the MLX cache at 1 GB — keep it.
- Traces record pools as whole tokens (edge punctuation included). Traces
  written before 2026-09-18 have the old `lead`/`trail` shape and cannot be
  replayed; regenerate.
- `bench.py run` fits the channel on the dev split *of the tier it runs
  on*; never carry a Tier 1 channel to Tier 2.
- Disk on this machine was at 3.7 GB free at one point (swap); the traces
  and OCR layers are small, checkpoints are 200–400 MB each.
- archive.org returns HTTP 500 for some items (3 here); `find` records it and
  moves on. Item file names don't always match the identifier — `find` reads
  them from the metadata.
- Tier 3 must never include Tier 2 test books; `build` enforces it.
- Wikisource "proofread" (level 3) pages still carry errors; level 4 is
  cleaner but only ~2,800 pages exist. Tier 2 truth noise is Finding 5.

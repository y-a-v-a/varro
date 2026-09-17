# ocr — can LATIN-LII repair scanned Latin, and how would we know?

`plan.md` names OCR correction as the model's strongest fit and the one that
compounds: it needs no knowledge of the world, only a sense of which Latin
sequences are plausible, and it feeds the corpus that trains the next model.
That is an argument. This directory is what turns it into a measurement, and it
was written **before** any repairer worth the name, on purpose: a benchmark
designed around a system's output ends up measuring what the system is good at.

```bash
source .venv/bin/activate

python ocr/bench.py build                          # held-out text -> dev + test, once
python ocr/bench.py run identity unigram lii       # dev: tune against this
python ocr/bench.py run unigram lii --split test   # test: the run that gets quoted
python ocr/bench.py compare reports/ocr-A.json reports/ocr-B.json   # A minus B, paired
python ocr/channel.py "Gallia est omnis diuisa" --rate 0.08   # see the noise
pytest tests/test_ocr.py                           # the arithmetic, no model needed
```

| file | what it is |
|---|---|
| `metrics.py` | What "repaired" means: alignment, the five outcomes, silent errors, intervals. Pure. |
| `channel.py` | The noise, and `fit`, which learns a channel back from pairs. |
| `bench.py` | Builds the locked benchmark; runs repairers against it; writes `reports/ocr-*.json`. |
| `repairers.py` | The contract a repairer meets, and three reference systems. **A new idea goes here.** |
| `bench.lock.json` | Hashes of the dev and test splits. Two reports compare only if they share them. |

## The question, and the only form of answer that counts

> Given the same candidate corrections and the same model of the noise, does
> reading the context with LATIN-LII repair more text, and break less, than
> counting words does?

Not "does it fix OCR errors". Nearly anything fixes `cu1tu`. `plan.md`'s rule
is that the model earns its keep *where word order carries the information,
and loses wherever a frequency table already has the answer* — so the
benchmark is built to hand a frequency table every advantage except order and
then measure the gap. `unigram` and `lii` in `repairers.py` share one proposer,
one fitted channel and one decision rule. They differ in a single term:
`log P(word)` from counts, against `log P(word + what follows | what precedes)`
from the model. The difference between their rows is what context is worth.
Neither row alone is a result.

## What is measured

### The unit is the word of the true text

Noisy and repaired text are each aligned with the truth character by
character, and the alignment is read off one **true** word at a time: is this
word standing in the output, whole, with a boundary either side?

That is what lets segmentation errors be scored at all. `de the sauro` has
three words and `dethesauro` has one; neither pairs off against `de thesauro`,
and a metric built on pairing noisy words with true ones has to invent a rule.
Projected onto the truth there is nothing to invent: `thesauro` is not
standing in either, so it is damaged in both, and a merge damages both words
it swallowed.

### Five outcomes, because accuracy hides the one that matters

| | standing after repair | not standing after repair |
|---|---|---|
| **standing before** | kept | **broken** |
| **damaged before** | fixed | missed · **miscorrected** |

`missed` is damage left exactly as found. `miscorrected` is damage changed
into something still wrong. `broken` is a word that was right until the
repairer touched it.

Character error rate nets all five into one number, and a repairer that fixes
thirty scannos while rewriting ten proper names into commoner words posts a
healthy CER reduction. For a corpus that is the wrong trade. A scanno is
visible to the next audit — `score.py --audit` and `demos/surprisal.py` exist
to find them — and a fluent wrong word is visible to nothing. So every report
carries:

| number | definition | why |
|---|---|---|
| `recall` | fixed / damaged | how much of the damage is gone |
| `precision` | fixed / (fixed + miscorrected + broken) | of the places it changed, how many ended right |
| `harm_per_1000` | broken per 1,000 words that were standing | **the number to watch**; also split by whether the true word is in the lexicon, because names and hapax are where harm lives |
| `silent_per_1000` | wrong words that are themselves attested Latin, before → after | a repairer can lower CER and *raise* this, and one that does has made the text worse in the way that cannot be found again |
| `cer_reduction` | relative drop in edit distance | the field's headline metric (ICDAR 2017/2019), kept for comparability and trusted least |
| `recall_real_word` / `recall_nonword` | recall, by whether the damage is itself attested | the first is the claim; the second is the part a dictionary can do |
| `by_class` | recall per injected error class | shape (`m→rn`), letter (`e→c`), digit, drop, split, merge, speck, random |
| `detection` | average precision of `suspicion()` against damaged words; again on real-word damage alone | a model that finds what it cannot fix is still worth having beside a human corrector |

Every rate is a ratio of sums over passages, never a mean of per-passage
ratios. Every interval is a bootstrap over **passages**, because words inside
a passage are not independent. Comparisons between systems are **paired**:
the difference is taken inside each resample, so the luck two systems share on
a hard passage cancels.

### One passage in four is left clean

Level 0 contains no damage, so everything a repairer changes there is harm and
nothing else. It is the control the other levels are read against, and the
most realistic one: most of any scanned page is right. The other levels start
damage at roughly 1%, 3% and 8% of letters (measured CER 1.0%, 2.9%, 7.6% on
test). Reports give every level its own interval, because the question a level
answers is a yes or no: *is repair worth running on text this clean?*

## Guards against fooling ourselves

1. **Held out by author.** Truth comes from `corpus-b.validation.jsonl` — the
   split LATIN-LII never trained on, grouped so no validation author appears
   in training. Restoring a word the model has read measures recall, not
   repair (`demos/lacuna.py` makes the same point about gaps).
2. **Truth must be true.** Only `text_provenance: transcribed` records. CSEL
   is machine-corrected OCR; scored against it, a repairer would be marked
   wrong each time it fixed an error the "truth" still carries.
3. **The repairer never sees the noise table.** It is shown dev *pairs* and
   fits its own channel with `channel.fit`, as anyone with a few hand-corrected
   pages would. A test fails if `repairers.py` so much as names `CONFUSIONS`.
   Without this, a repairer inverts a function it was handed.
4. **Dev and test are different authors.** Margins, context widths and
   thresholds are tuned on dev (240 passages). Test (560) is for the run that
   gets quoted. A test report records the commit and whether the tree was
   dirty.
5. **The benchmark is locked.** `bench.lock.json` holds the split hashes;
   `build` refuses to replace them without `--relock`, and `run` refuses a
   split that does not match. A benchmark that drifts makes every earlier
   number unfalsifiable. The passages stay under `data/` (never committed) and
   rebuild from the seed.
6. **The lexicon excludes OCR-derived text.** Otherwise a scanno CSEL repeats
   three times is an "attested word": the repairers would offer it as a
   correction and the silent-error count would miss it.
7. **Everything is in corpus orthography.** Truth is already folded u/v, i/j.
   For any text from outside (Tier 2) both sides go through
   `src/normalize.py` first, and the do-nothing baseline is
   `normalize(noisy)`, so what the pipeline already repairs — homoglyphs,
   orphan diacritics, line-end hyphens — is not credited to the model.

## Two tiers, and what each is allowed to say

**Tier 1 — synthetic damage (built).** Clean held-out text through
`channel.corrupt`. Unlimited, exact truth, controlled level, error classes
known per word. Its confusion table was written by hand from what the OCR
literature reports for roman type; the weights are judgement. So Tier 1 can
say *system A repairs more than system B*, *this is what it breaks on clean
text*, and *this class of error is where context helps*. It cannot say *fixes
N% of OCR errors*. That sentence is not available until Tier 2.

**Tier 2 — real scanner output (not built; the next piece of work).** Same
metrics, same report, truth and noise from the world. Two candidate sources:

- **GT4HistOCR, Early Modern Latin subcorpus** (Springmann et al. 2018,
  [Zenodo 1344132](https://zenodo.org/records/1344132), CC-BY-4.0): line
  images of early printed Latin with diplomatic transcriptions. Run Tesseract
  `lat` over the images and the pairs are real. Two problems to solve first:
  the transcriptions keep abbreviations and ligatures that `normalize.py`
  would strip into non-words, so lines need filtering on the share of true
  words the lexicon knows; and it is early-modern print, the period the model
  is weakest on.
- **A Teubner the corpus already holds.** Perseus records its editions
  (`"edition": "Teubner 1895"`); the same printing is on the Internet Archive
  with its OCR layer. Closest to the actual use — `sources.md` §3 is about
  exactly these scans — but the text must be one the model did not train on,
  the alignment must be checked by hand, and a few thousand verified words is
  a realistic size.

Either way it enters as an evaluation set and **never the corpus**: declared in
`manifest/sources.yaml` with `include: false`, fetched to `data/raw/`
untouched, all alignment and filtering in code. When it exists, run
`channel.fit` over its pairs and replace Tier 1's hand weights with those
counts; the fingerprint in the lock will change, which is the point.

## What is being claimed, written down before the test run

Written down so the pass mark cannot be chosen after seeing the answer.
**After the first `--split test` run these are frozen**, and a changed
threshold needs a changed benchmark. H2's limit of 2 words per 1,000 was
confirmed on 2026-09-17, as a deliberately high standard and knowing the
reference `lii` does not meet it on dev (6.1).

| | claim | falsified if |
|---|---|---|
| **H1** | Context repairs real-word damage that frequency cannot. | `lii − unigram` on `recall_real_word`, paired 95% interval, includes zero. |
| **H2** | It is not bought with harm. | `lii − unigram` on `harm_per_1000` lies wholly above zero; **or** `lii` breaks more than 2 words per 1,000 on level 0. |
| **H3** | It does not launder. | `silent_per_1000` after > before, pooled over levels ≥ 1%. |
| **H4** | The right context is doing work. | `lii − lii-left` on `recall`, paired interval, includes zero. |
| **H5** | It arrives with scale. | Recall does not rise `lii:latin-vii` < `lii:latin-xxv-2` < `lii`. A prediction, not a condition. |
| **H6** | There is a level below which repair should not be run. | Not falsifiable; *reported*: the cleanest level whose `cer_reduction` interval excludes zero. Below it, leave the text alone. |
| **H7** | Any of this survives real scans. | Tier 2 `cer_reduction` interval includes zero. Until it is run, no Tier 1 figure is described as OCR performance. |

H1 deserves a caution the first dev run supplied. "Impossible from
frequencies" is too strong: `unigram` fixes a fair share of real-word damage,
because when `cum` is read as `eum` the true word is usually the commoner one
and frequency alone gets there. The claim that survives is comparative, which
is why H1 is a difference and not a level.

## Dev numbers from the reference systems

`python ocr/bench.py run identity unigram lii lii-left`, 240 dev passages,
14,400 words of which 2,377 damaged. These are **dev** numbers from reference
implementations: they show the harness has been run on the thing it exists to
judge and what its output looks like. They are not results, and the test split
has not been run.

```
  system                       CER cut  recall real-wd nonword precis. broken/k         silent/k
  identity                        0.0%    0.0%    0.0%    0.0%      --     0.00   39.03 -> 39.03
  unigram                        41.0%   49.0%   59.6%   45.7%   86.8%     5.99   39.03 -> 24.24
  lii:latin-lii-latinitas        62.5%   70.5%   75.1%   69.1%   87.8%     5.07   39.03 -> 22.57
  lii:...:left-only              61.7%   68.5%   66.5%   69.1%   88.0%     4.24   39.03 -> 25.56

  lii minus unigram, paired 95% intervals:
    recall_real_word   11.9% .. 19.2%
    recall_nonword     21.4% .. 25.7%
    harm_per_1000      -3.48 .. 1.46
```

What the first run of the harness already says, none of it final:

- **`identity` posts zero in every column**, which is the check that the
  outcome table is not inventing repairs or harm.
- **The gap is there, and not where `plan.md` expected it.** Context is worth
  more on *non-word* damage (+21 to +26 points) than on real-word damage
  (+12 to +19). Frequency is better at real-word damage than the argument
  assumed, and worse at choosing among several attested neighbours of a
  non-word, which is a choice only context can make.
- **The right context earns its place on real-word damage and nowhere else**:
  75.1% against 66.5%, with non-word recall identical to the decimal.
  Paired (`bench.py compare`): +6.0 to +11.3 points on real-word recall,
  −0.8 to +0.8 on non-word. It also costs a little harm, +0.8 per 1,000,
  with an interval that just includes zero (−0.2 to +2.0).
- **Harm is the open problem.** The model breaks no more than the frequency
  table does (the paired interval straddles zero), and both break too much:
  `lii` damages 6.1 words per 1,000 on text with *no errors in it* (95%
  interval 2.5 to 10.3), three times what H2 allows. Harm is concentrated
  where predicted — 50 per 1,000 among true words the lexicon does not know,
  3.4 among those it does.
- **Reading what was broken is worth more than the rate.** The list in each
  report's `examples` shows three different things being counted as one:
  typos in the transcription that the repairer in fact *fixed*
  (`posueTrunt → posuerunt`, `TIrbis → Turbis`, `alnnos`); medieval spellings
  it classicised (`Rome → Romae`, `porcionem → portionem`,
  `estimem → aestimem`), which is real harm and the most dangerous kind for a
  corpus that labels its periods; and plain mistakes (`pares → patres`,
  `Trux → Crux`). Only a Tier 2 set with checked truth can separate the first
  from the rest.
- **Detection is where the model is furthest ahead**: average precision 78%
  against 56% for word frequency, on a floor of 17%. On real-word damage
  alone both are weak (19% and 16% on a floor of 6%) — a fluent wrong word is
  nearly as invisible to the model as to the dictionary, which is the case
  H3 exists to watch.
- By error class, `lii` recall runs from 77% (letter), 76% (shape) and 75%
  (digit) down to 39% on merges. The proposer offers a split only where both
  halves are already known words, which is the first place to look.

Cost: 2 seconds a passage for `lii` on the machine that trained it, so about
8 minutes for dev and 19 for test.

## Adding a repairer

Subclass `Repairer` in `repairers.py`, give it a `name`, implement
`repair(noisy) -> str`, optionally `fit(pairs, lexicon)` and
`suspicion(noisy) -> [float per word]`, and add a line to `make`. The harness
does not care how it works: it may ignore the shared proposer entirely. But
then say so in its row, because a system with its own proposer is no longer
the controlled comparison H1 rests on.

What the reference `lii` does not do, each of which is an experiment this
benchmark can now judge: repair jointly rather than greedily left to right;
score over a right context it has already repaired instead of a noisy one;
weight the channel and the model against each other with a tuned exponent
rather than adding them raw; condition on a period tag; refuse to touch
numerals and capitalised words (most of what it breaks on clean text); propose
from the model's own samples where the channel offers nothing.

## Known limits of the benchmark itself

- **Tier 1 is mostly Vicifons, mostly medieval and late.** That is what
  survives "held out" ∩ "transcribed": 86% of passages, with 23 classical
  ones in 800. Results are reported overall; a per-period breakdown would be
  noise at those sizes. Classical coverage has to come from Tier 2.
- **"Transcribed" is a label on a source, not a guarantee about a text.**
  Wikisource transcriptions often began as OCR that volunteers proofread, and
  the dev run found the residue: `posueTrunt`, `TIrbis`, `alnnos` are in the
  *truth*. A repairer that mends them is scored as having broken a word. It
  costs every system, but not alike — it costs the better repairer more — so
  Tier 1 `harm` is an upper bound.
- **Medieval spelling is not damage, and the benchmark agrees with that on
  purpose.** `Rome` for `Romae` and `porcionem` for `portionem` are what the
  scribe wrote. A repairer that classicises them is marked as breaking them,
  which is the right verdict for a corpus that records period.
- **The model has trained on OCR.** 5.85% of `corpus-b` is CSEL, so some
  scannos are, to LATIN-LII, Latin. `build.py`'s provenance caps make the
  ablation possible (train without `ocr`, rerun this) and it is the right
  follow-up if H1 holds.
- **Capitals and punctuation are never damaged**, so proper names are tested
  only as things to leave alone, not as things to repair.
- **"Real-word" means every fragment is in a 536k-form lexicon** with a floor
  of three occurrences. That lexicon is generous — `fi` and `um` are in it —
  so a split like `Publ ius` counts as real-word damage. The stratum is
  honest about what a dictionary check would pass, and broader than "changed
  into a different Latin word".
- **One alignment among equals.** Where several minimal alignments exist the
  code takes one deterministically (diagonal first). Edit distance is exact
  either way. Which of two adjacent words is charged with an ambiguous edit,
  and which operations the fitted channel counts, can depend on the choice.

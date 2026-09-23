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
python ocr/tune.py trace lii && python ocr/tune.py sweep lii:latin-lii-latinitas --save
python ocr/tune.py ceiling unigram                 # how far could a perfect scorer get?
python ocr/tier2.py scan && python ocr/tier2.py fetch && python ocr/tier2.py build
python ocr/bench.py run unigram lii fim --tier 2   # real scans, dev
python ocr/channel.py "Gallia est omnis diuisa" --rate 0.08   # see the noise
pytest tests/test_ocr.py tests/test_fim.py         # the arithmetic, no model needed
```

| file | what it is |
|---|---|
| `metrics.py` | What "repaired" means: alignment, the five outcomes, silent errors, intervals. Pure. |
| `channel.py` | The noise, and `fit`, which learns a channel back from pairs. |
| `bench.py` | Builds the locked benchmark; runs repairers against it; writes `reports/ocr-*.json`. |
| `repairers.py` | The contract a repairer meets, and the reference systems. **A new idea goes here.** |
| `tune.py` | Records a system's decisions once, replays them under many rules, keeps the one that meets the harm limit. |
| `rules.json` | The rule each system runs under, per tier, and what it measured when tuned. |
| `tier2.py` | Tier 2: real scanner output against checked Wikisource transcriptions. |
| `bench.lock.json`, `tier2.lock.json` | Hashes of every split. Two reports compare only if they share them. |
| `../src/fim.py` | Continues a checkpoint so it can fill a gap given both sides of it. |

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

**Tier 2 — real scanner output (built 2026-09-18).** Same metrics, same
report, truth and noise from the world, and neither half made here.

Latin Wikisource transcribes books page by page against a scan. Each page
carries a proofreading status — 3 is proofread, 4 is validated by a second
person — and is tied to one page of a DjVu or PDF on Wikimedia Commons, which
carries the text layer its scanner produced (Google Books, the Internet
Archive). So the truth was already in `data/raw/vicifons`, 17,622 checked
pages that `extract.py` never read because it takes only namespace 0; and the
noise is one API request per book, 157 books, 122 MB, no OCR engine run.
`tier2.py` cuts each transcription into sixty-word passages and *locates* each
in its page's OCR by approximate matching, since a scanned page also holds a
running head, footnotes and an apparatus the transcription leaves out.

```
  dev     652 passages from 31 books    CER 5.2%     51 read perfectly
  test  1,326 passages from 64 books    CER 6.3%     89 read perfectly
```

What it sees that Tier 1 never did: the long s read as `f` through whole
pages of early print (`depafcere`, `efl`), capitals misread (`loue` for
`Ioue`, `lUuilrif` for `Illustris`), stray symbols (`%uam`, `/Eolum`), lines
of verse run together into one token (`Hospitatectalocis.`).

What was dropped, because a benchmark that quietly keeps the easy cases should
say so: 2,994 passages sharing eight words running with the training split
(Columella, Priscian, Newton, the *Corpus Iuris* — read in another edition,
so not held out); 2,170 not found in their page's OCR within 25% error, which
biases the set towards pages a scanner read tolerably; 2,462 on pages with no
text layer; 1,214 that were tables, lists or leftover markup; 3,830 over the
cap of 24 per book, which keeps the set a hundred books wide rather than deep
in Erasmus.

Two things are tidied on **both** sides before comparison, because they are
typography and need no model: the scanner's line-break mark (`con¬ suetudines`)
and the space some engines put before punctuation (`conueniat ,`). Left in,
the second alone was a quarter of all "damaged" words.

It is an evaluation set and **never the corpus**: declared in
`manifest/sources.yaml` as `wikisource-scan-ocr`, tier `eval`,
`include: false`; the OCR sits in `data/raw/` untouched; each scan is pinned by
its sha1 in `tier2.lock.json`. Dev and test are different *books*. A repairer
run with `--tier 2` fits its channel on Tier 2's dev pairs and takes the rule
tuned there, because a channel learned from invented noise answers a question
about the wrong scanner.

Still owed: `channel.fit` over these pairs should replace Tier 1's hand
weights. That changes Tier 1's lock and every number measured on it, so it is
a decision, not a cleanup. GT4HistOCR (CC-BY-4.0 line images of early printed
Latin) remains the route to a Tier 2 with an OCR engine of known version, at
the cost of 4 GB and a Tesseract install.

## What is being claimed, written down before the test run

Written down so the pass mark cannot be chosen after seeing the answer.
**After the first `--split test` run these are frozen**, and a changed
threshold needs a changed benchmark. H2's limit of 2 words per 1,000 was
confirmed on 2026-09-17, as a deliberately high standard and knowing the
reference `lii` does not meet it on dev (6.1).

| | claim | falsified if |
|---|---|---|
| **H1** | Context repairs real-word damage that frequency cannot. | `lii − unigram` on `recall_real_word`, paired 95% interval, includes zero. |
| **H2** | It is not bought with harm. | `lii − unigram` on `harm_per_1000` lies wholly above zero; **or** `lii` breaks more than 2 words per 1,000 on level 0 (Tier 1) / over all standing words (Tier 2, whose level 0 is too small to carry a limit). |
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

## What has been measured so far — dev only

The test splits have not been run. Everything below is from the dev splits
(Tier 1: 240 passages plus a 600-passage clean control; Tier 2: 652 passages)
and from the reference systems in `repairers.py`. It is what the harness
says today, not a result.

### 1. Meeting the harm limit: the rule, and what "broken" turned out to be

The first run, with every change decided on the raw sum of scorer and
channel, fixed 70.5% of damaged words and broke 6.1 per 1,000 on clean text.
Nothing about the model needed to change to fix that; the decision did.
`repairers.Rule` demands extra evidence, in nats, by the *kind* of change —
more for a word that is itself known, for a capital, for joining two known
words, for a short word — and never touches a numeral. `tune.py` records a
system's decisions once and replays them under 67 rules in seconds, choosing
the highest recall whose harm on 39,600 clean words is under 1.5 per 1,000
(the limit is 2; a limit met exactly on the text it was tuned on is a limit
missed elsewhere). Each winner was then confirmed with a real run:

```
  Tier 1 dev, every system at the same harm     recall  real-word  non-word  broken/1000
  unigram                                        45.3%     27.0%     51.0%      1.33
  lii                                            60.2%     45.9%     64.6%      1.41
  lii, left context only                         58.8%     37.7%     65.3%      1.41
  fim  (see 2.)                                  62.2%     51.1%     65.7%      1.75

  lii minus unigram, paired 95%:   real-word +14.8 .. +23.3   non-word +11.4 .. +15.8
  lii minus left-only, paired 95%: real-word  +4.5 .. +12.2   non-word  -2.2 ..  +0.7
```

Held to the same harm, the frequency table loses most of what it had on
real-word damage (59.6% → 27.0%): what it fixed there it fixed by preferring
the commoner word, and that is exactly what a harm limit forbids. The model
keeps 46%. **The claim in H1 gets wider, not narrower, under the constraint.**

And the 59 words the tuned `lii` still breaks on clean text are worth
reading, because most are not breakage. By my reading — a model's, to be
checked — 37 of them are scanning errors *in the Vicifons transcription*
which the repairer fixed (`buius → huius`, `couiunctione → coniunctione`,
`A1tera → Altera`, `infpectione → inspectione`, `eompassus → compassus`),
and some 20 are real harm, split between medieval spellings classicised
(`Sfortie → Sfortiae`, `commertia → commercia`), Folengo's macaronic Latin
(`ficcam → siccam`, `disinare → diuinare`) and plain mistakes
(`oculis → oculus`, `alid → aliud` in Lucretius). Two consequences: Tier 1
harm is an upper bound by roughly a factor of three, and the tuned repairer
is already a usable *finder* of scanning errors in the corpus's own
"transcribed" text — `plan.md`'s third use, arriving unasked.

### 2. Fill-in-the-middle: a direct estimate against an inverted one

`src/fim.py` continued LATIN-LII for 12,000 steps (49M tokens, 0.19 epochs,
3.6 hours on the machine that trained it) on text rewritten as
`<|pad|> suffix <|pad|> prefix middle <|endoftext|>` half the time, cutting
on word boundaries. No new tokens: `<|pad|>` had never occurred in a
training stream and serves as both seams. Held-out loss on ordinary text
went 3.058 → 3.063, so the left-to-right model is intact; the loss on the
new format went 4.246 → 3.149.

It was worth being clear beforehand what this could and could not add. For a
fixed left and right, P(fill, right | left) ∝ P(fill | left, right), so the
causal model scored across the right context — what `lii` does — already
ranks candidates by its own infilling posterior. The trained model gives a
*different estimator* of the same quantity: it prices the candidate's own
tokens and the token that closes the gap, with the suffix in front of it,
instead of recovering the answer from the cost of a dozen noisy tokens after
the gap. Whether that estimator is better was the question.

```
  fim minus lii, paired 95%, Tier 1 dev, both at their tuned rules
    recall             +2.1   (+0.6 .. +3.6)
    recall_real_word   +5.2   (+1.2 .. +9.1)
    recall_nonword     +1.1   (-0.2 .. +2.5)
    harm_per_1000      +0.33  (-0.09 .. +0.83)
    detection AP        79.2% against 78.0%; on real-word damage 20.4% against 18.6%
```

Better, modestly, and where the theory said it could be: on real-word
damage, where the right context is the whole case. The harm interval leans
up and does not exclude zero. On real scans (§4) the same pairing gives
recall +2.2 (+1.4 .. +3.0) at no extra harm. Cost: the same two seconds a
passage.

What FIM training was really for is not yet used: a model that can
*generate* into a gap knowing both sides is a proposer, and 3. says the
proposer is the wall.

### 3. The wall is the proposer, not the scorer

`tune.py ceiling` replays a trace with every decision made by someone who
can see the truth: the right word is taken if it was offered, else the page
is left alone. Its recall is what the shared proposer allows any scorer to
reach.

```
                          perfect scorer    lii, untuned    lii, tuned
  Tier 1 dev                   73.7%            70.5%          60.2%
  Tier 2 dev (real scans)      48.1%            42.7%          24.9%
```

On simulated damage the model was already within three points of the
ceiling before the harm limit. On real scans the right word is never offered
for half of all damaged words. Where it goes, by the damaged words the
oracle could not fix on Tier 2 dev (3,425 of 6,595):

```
  26%  the true word is not in the lexicon     names, 19th-century coinages, and
                                               transcription residue (`28.Ne`)
  25%  three or more edits                     lines of verse run into one token
  13%  one or two edits, and still unproposed  an operation the channel never saw
   8%  a capital letter misread                `loue`, `bulpitii`, `Beg.`
   8%  a split or merge plus another error     `ali¬ quod` was tidied; `i sc` was not
   7%  only edge punctuation differs           `lucri` for `lucri.`
   7%  a stray symbol inside the word          `%uam`, `nia$aa,`
   5%  only the case differs                   `CARABUS.` for `Carabus.`
```

Every line of that is a proposer's problem. The scorer, tuned or not, could
not have touched it.

### 4. Real scans

Real scans, dev, every system tuned to the limit. On Tier 2 the limit is
applied to harm over *all* standing words (32,500), not to the control
alone: the control is the 51 passages a scanner happened to read perfectly,
3,060 words, where the limit is four broken words and the fifth decides
everything. The first sweep tuned to that, and what it found was a rule
shaped by four words.

```
  Tier 2 dev                     CER cut   recall  real-word  non-word  broken/1000  silent/1000
  identity                          0.0%     0.0%      0.0%      0.0%       0.00     56.6 -> 56.6
  unigram                          10.6%    19.9%     19.0%     20.4%       1.45     56.6 -> 49.1
  lii                              15.6%    24.9%     11.2%     31.8%       1.29     56.6 -> 54.7
  fim                              16.8%    27.1%     14.8%     33.4%       1.20     56.6 -> 52.8

  lii minus unigram, paired 95%:   real-word  -9.6 .. -5.9   non-word  +9.7 .. +13.3   harm -0.74 .. +0.43
  fim minus lii,     paired 95%:   recall     +1.4 .. +3.0   real-word +2.0 ..  +4.9   harm -0.28 .. +0.12
```

Three things to read off this, all of them about the gap between the two
tiers rather than between the systems:

- **Simulated damage overstates real repair by three times.** The same
  systems that fix 45–62% of Tier 1's damage fix 20–27% of a real
  scanner's. The protocol said no Tier 1 number could be quoted as OCR
  performance; this is the size of the reason.
- **The model still beats frequency, by less, and not on real-word
  damage.** On real scans `unigram` keeps its real-word edge (19% against
  11–15%): a real-word error there is mostly a long s read as `f`
  (`femen` for `semen`) or a case slip, where the commoner word simply is the
  answer. The model's gain is all on non-words, and comes with a channel it
  had to learn from 652 passages of noise it had never seen.
- **Detection holds up better than correction.** Average precision 62–63%
  against 43% for frequency, on a floor of 17%. A scanner's errors are
  still where the model says they are, even when it cannot say what stood
  there.

And the ceiling: with the same proposals, a perfect scorer reaches 48.1%.
The tuned systems sit at 20–27% because the harm limit costs them the rest
— see the margin sweeps in `tune.py sweep --frontier`, where `fim` at
margin 0 recalls 40.6% and breaks 10 per 1,000.

### Controls that were cheap enough to run

- **The FIM checkpoint used the ordinary way** (`lii:latin-lii-fim`, untuned)
  recalls 71.0% on Tier 1 dev against 70.5% for the base checkpoint: the
  continued training did not cost the causal scorer anything, and gave it a
  little.
- **A wider right context does not help the causal scorer** (`lii@12`,
  twelve words instead of five: 70.0%, paired −3.5 to −0.5 on real-word
  recall against the FIM checkpoint at five). The tokens far to the right
  add noise to the sum faster than evidence.
- **Nor the direct one** (`fim@12`: 70.4%, and less harm — 3.7 against 4.8
  per 1,000 untuned — but no more recall).


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
  Tier 1 `harm` is an upper bound. Measured in §1 above: of 59 "broken"
  words under the tuned rule, about 37 were the truth's own scannos, so the
  bound is loose by roughly three times.
- **Rules are tuned on dev and are part of the system.** `rules.json` is
  what a test run would use; it was chosen by replaying 67 rules against dev
  and its clean control, and a test run measures the rule as much as the
  model. A different harm target is a different rule and a different
  benchmark result.
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

## Conclusion

The question at the top was whether reading the context repairs more scanned
Latin, and breaks less, than counting words does. On the dev splits, held to
the same harm, the answer is yes on both tiers: 60% of damaged words against
45% on simulated damage, 25% against 20% on real scans, at 1.2–1.5 words
broken per 1,000 in every case. The test splits have not been run, so this is
what the harness says, not what the project claims.

Four things were learned on the way that matter more than those numbers.

1. **The harm limit is met by the decision, not the model.** The same model
   that broke 6 words per 1,000 breaks 1.4 once a change has to bring
   evidence in proportion to what it risks: a numeral, a name, a join of two
   real words. The recall it costs — 70% down to 60% on simulated damage —
   is the price of a repairer that can be run on a corpus unattended, and
   most of what it still "breaks" are scanning errors in the truth itself.
2. **Simulated damage overstates real repair three times over.** Every
   system that fixes 45–62% of the invented damage fixes 20–27% of a real
   scanner's. Tier 1 remains the right instrument for comparing two
   repairers; any figure quoted as OCR performance has to come from Tier 2.
3. **The wall is the proposer.** With the candidates the shared proposer
   offers, a perfect chooser could reach 48% on real scans, and the model
   was already near that ceiling before the limit. Half of all real damage
   is never offered its right word: names outside the lexicon, capitals
   misread, three-letter garbles, lines of verse run together. No change to
   scoring reaches any of it.
4. **Fill-in-the-middle training is a small gain as a scorer and an unused
   tool as a proposer.** Twelve thousand steps of continued training left
   the left-to-right model intact and bought +2 points of recall on both
   tiers, at no harm. The same checkpoint can now *generate* into a gap
   knowing both sides, which is what the proposer lacks.

So the next piece of work is a better proposer, and the benchmark is ready to
judge it: candidates from the fill-in-the-middle model's own samples
constrained to a small edit distance from the page, a lexicon that admits
the names a book actually uses, splits into more than two words. That is the
point at which a repairer that rewrites text directly (Option 2 in the
project notes) becomes worth considering, and also the point at which its
one danger — fluent, confident, wrong — is exactly what `broken`,
`miscorrected` and `silent` were built to count.

Two decisions are still owed before any test run: whether to replace Tier 1's
hand-written noise with the channel fitted from Tier 2's pairs (which
re-locks Tier 1), and whether the rules in `rules.json`, tuned on dev to a
1.5 target, are the ones the test split should be run under.

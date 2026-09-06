# Current state — pick-up notes

Written 2026-08-17 as a handoff into a fresh session; updated 2026-09-06.
`README.md` explains how things work; this file says where they stand, what is
stale, and what to do next.

---

## Update 2026-09-01 (later) — LATIN-VII trained: data beats sizing

3 hours, 7.28M parameters, 14.2 epochs over the classical corpus at 20.8
tokens/param. Both models of the two-stage plan now exist.

| model | params | tokens seen | classical ppl | x floor |
|---|---:|---:|---:|---:|
| LATIN-0 | 3.53M | 10.7M | 74.3 | 2.50 |
| **LATIN-VII** | 7.28M | 10.7M | **70.1** | **2.65** |
| **LATIN-LII** | 52.07M | 263.6M | **27.3** | **6.80** |

### The result, and a prediction of mine it refutes

Doubling parameters from LATIN-0 to LATIN-VII — both sized at ~20
tokens/param on the same corpus — bought 74.3 -> 70.1, about 6%.
Training on 25x more Latin bought 70.1 -> 27.3, about 61%.

At this scale data dominates sizing, overwhelmingly. I had suggested
LATIN-VII might beat LATIN-LII on classical because it sits at 20.8
tokens/param where LATIN-LII sat at 0.4. That conflated two different
things: Chinchilla says how to balance parameters against the tokens you
have, and says nothing about how much better you would do with more
tokens. The measurement is unambiguous — it loses by a factor of 2.5.

**So LATIN-VII's value is its licence, not its quality.** Green sources
only, CC-BY-SA, redistributable; LATIN-LII is NonCommercial research-only.
If someone wants the best Latin model here it is LATIN-LII, and no amount
of correct sizing closes that gap while classical Latin stays capped at
~10M surviving words.

### A train/validation gap is not a memorisation measurement

LATIN-VII ended with a gap of 0.555, four times LATIN-LII's 0.138, after
fourteen passes over every passage. It stores nothing:

| window | LATIN-VII's output | real unseen Latin |
|---:|---:|---:|
| 3 words | 17.8% | 7.0% |
| 4 | 2.2% | 3.9% |
| 5 and above | **0.0%** | 3.1% -> 1.2% |

0 of 3,476 windows memorised, longest verbatim run 0 words. From five
words upward it matches training text LESS often than genuinely unseen
Latin does.

I had been reading the widening gap as evidence of memorisation for two
days. It is not. Distributional overfitting — fitting the training
distribution more tightly — and verbatim recall are different properties,
and only the second one is measured by the memorisation report. Both
models are clean on the second.

---

## Update 2026-09-01 — LATIN-LII trained: 21.3, 8.73x its floor

74 hours, 52.07M parameters, four epochs over 263,558,393 tokens at 20.2
tokens/param, zero errors. The largest model the project has trained, and
the first with a floor computed on its own corpus rather than borrowed.

| model | params | corpus | ppl | floor | x better |
|---|---:|---|---:|---:|---:|
| **LATIN-LII** | 52.07M | corpus-b | **21.3** | 185.90 | **8.73** |
| LATIN-XXV-2 | 27.49M | corpus | 25.8 | 180.62 | 7.00 |
| LATIN-2 | 15.90M | corpus | 44.1 | — | — |
| LATIN-0 | 3.53M | classical | 74.3 | — | — |

corpus-b's floor (185.90) is HIGHER than Track A's (180.62): the corpus got
genuinely harder when the medieval and neo-Latin arrived, so 7.00x -> 8.73x
is a real gain and not an easier test. Per-period floors say classical is
the hardest slice for a bigram model at 201.2, medieval and neo-Latin the
most formulaic at ~179.

### The annealing phase has a capacity ceiling, and 52M is past it

The recipe's second phase narrows onto classical. Measured on the same
held-out sets through the same vocabulary:

| checkpoint | on classical | on corpus-b |
|---|---:|---:|
| after latinitas | **27.3** | **21.3** |
| after romanitas | 30.2 | 32.5 |

It made the model worse at the thing it exists for. Re-run against
LATIN-XXV-2 through its archived vocabulary, the same phase gives the
OPPOSITE result — 34.8 -> 28.3 on classical — so the phase is not wrong in
itself.

The cause is arithmetic. `classical.train` is a SUBSET of `corpus.train`,
so the phase introduces no new text; it only re-weights toward 4% of what
the model has already read. At 27.5M there is still something to learn
from that. At 52M there is not: 10.7M tokens against 52M parameters is 0.4
tokens/param, fifty times short of what the weights can absorb, so the
model memorises instead. The train/validation gap went 0.138 -> 0.484
while held-out perplexity got worse.

**LATIN-LII is now single-phase.** LATIN-VII at 7.28M is under the ceiling
and could take a narrowing phase if one were ever wanted. I called the
two-phase structure "unsound" before measuring LATIN-XXV-2; that was
wrong, and the ceiling is the accurate statement.

### It does not recite

| window | LATIN-LII's output | real unseen Latin |
|---:|---:|---:|
| 3 words | 34.1% | 27.1% |
| 4 | 9.1% | 16.7% |
| 6 | 0.4% | 12.0% |
| 8 | 0.0% | 9.6% |
| **12** | **0.0%** | **6.0%** |

The control is 547 held-out works, 207,866 words the model never saw,
scored against the same training corpus. Genuine Latin overlaps training
text at 6% of 12-word windows because scripture, liturgy and formula
recur. The model's own output overlaps at 0.0% — from four words upward
it matches training text LESS than authentic unseen Latin does. Zero
memorised windows in 3,411; longest verbatim run **0 words**.

These numbers replace an earlier measurement of 2 in 3,473 with a
13-word run. That one could not be attributed: `evaluate.py` scored a
text file without recording which checkpoint wrote it, and two 52M
checkpoints existed at the time — `latin-lii-latinitas` and the
abandoned `latin-lii-romanitas`. It is kept as
`reports/memorization-latin-lii-unattributed.json` rather than deleted.
`evaluate.py --sampled-from` now records the model, so this cannot
recur; the table above is measured on `latin-lii-latinitas`, the model
every other number here refers to.

That settles what the 0.484 gap meant: fits the distribution better, not
recites Cicero. And it is the evidence Track B's framing needs — weights
that reproduce 0% of 12-word windows are not a redistribution of the
NonCommercial corpus they were trained on.

### Operational notes worth keeping

- **batch 8, not 16.** 40 steps at batch 8 took 48 seconds; the same 40 at
  batch 16 had not finished after seven minutes. The cliff LATIN-XXV
  recorded at dim 512 arrives earlier at dim 640.
- **Watch free memory, not process RSS.** MLX allocates unified memory,
  which `ps` does not attribute to the process, so a thrashing run reads as
  0.1 GB at 11% CPU — starved rather than greedy. Two probes were misread
  that way, costing an hour and a 143-day ETA scare.
- **A checkpoint is meaningless without its tokenizer.** Scoring
  LATIN-XXV-2 through the current vocabulary gave perplexity 3,740. It was
  trained through 8ba446f1 and the corpus now uses f64b7246. The archived
  vocabularies are what make the comparison recoverable at all.

---

## Update 2026-08-26 (later still) — three candidate corpora, worked in order

Cheapest first: Tesserae, IntraText, Corpus Corporum. One landed, one is
blocked by its publisher, one is ready and waiting on an email.

| corpus | records | words | train tokens |
|---|---:|---:|---:|
| Track A · full | 10,782 | 94,923,570 | **183,253,029** |
| Track A · classical | 261 | 5,055,913 | **10,671,431** |
| Track B · full | 10,807 | 94,991,578 | — |

Floors: **180.62** full, **185.67** classical. **Classical passed 5M words for
the first time** — it had not moved since the Latin Library.

### Tesserae — the licence was the whole blocker, and it is green

The manifest said "a custom university licence nobody here has read, so the
tier is `unknown` rather than `amber`: the terms are not merely restrictive,
they are unassessed." Now read. **UB Public License 1.0 is an MPL-1.1
derivative**: §2.1 grants a "world-wide, royalty-free, non-exclusive license
... to use, reproduce, modify, display, perform, sublicense and distribute",
and "non-commercial", "research only" and "academic use only" appear nowhere
in it. §3.1 is file-level copyleft. So **green, Track A** — it helps LATIN-VII,
not just LATIN-LII. The PDF does not extract through ordinary fetching; the
text came out of its FlateDecode streams.

`provenance.py` reports it as its own obligation class, "attribution +
per-file copyleft, ship licence", rather than folding it into ShareAlike or
dropping it in UNRESOLVED. Per-file copyleft is a different shape and a reader
deciding what to ship needs to be told which applies.

**Measured yield: 6,735,635 words extracted, 95.0% already present.** Only
334,010 words are new, 128,523 of them classical. sources.md's warning that
these aggregators are "substantially derived from Perseus and The Latin
Library" was right, and now has a number on it. Calibrate the other derived
collections against this, not against their headline size.

Two things still made it worth doing. The format was declared `txt` and is
actually 748 `.tess` files, so the plaintext reader would have found nothing
and reported success. And **metadata.json dates all 748 files individually** —
which mattered, because 44 of 70 Tesserae author slugs are bare surnames
(`cicero`, `vergil`) that miss the authority aliases. Rather than hand-add 44
aliases, `Authority.resolve` gained a **`source-metadata`** rank: a date the
source states for a work now outranks PL interpolation and source priors,
while still sitting under the human-checked authorities and recording its own
origin. That generalises to every future dated source.

### IntraText — blocked, and not by its licence

CC BY-NC-SA 3.0 is amber, and amber is fine; Track B never redistributes. The
**access terms** are the blocker: robots.txt disallows `/Dwnld/` and denies
roughly 25 named mirroring agents outright, and the copyright page asks
readers to link rather than republish. A user-agent not on that list might
slip through; the intent does not. `include: false`, reasoning recorded, **no
scraper written**. The way in is to ask EuloTech for a dataset.

### Corpus Corporum — reader written, acquisition still needs an ask

robots.txt is `User-agent: * / Allow: /`, the opposite of IntraText. Licence
confirmed amber from the site's own words. The browser lists all 30 corpora
with counts, totalling **226,311,326 words**, and — crucially — **the overlap
is excludable by corpus number rather than left to dedupe**: Perseus is 5,
OpenGreekAndLatin 2 and 19, digilibLT 16, Greek is 99, and Patrologia Latina
(2, 85.5M) overlaps the ~66M of PL we already carry via Vicifons. Excluding
those leaves **~115.9M words** of largely-new medieval and neo-Latin, of which
**Acta Sanctorum alone is 52.7M**.

**The `tei-xml` reader exists and is tested against real files.** The route in
was that Corpus Corporum's largest corpus is fed from a public GitHub repo
whose 6,411 files carry a literal `[CC]` prefix because they are the Corpus
Corporum copies — no scraping required. They also declare CC BY-SA 4.0 in
their own teiHeader, so that constituent is greener than the platform's
blanket term.

Two defects there, both of which a guessed reader would have sailed past:

| defect | scale | effect if ignored |
|---|---|---|
| unterminated character references (`&#171 ;`, `&gt5`, `&#0160`) | 5 of 21 files in one volume | the file is a `parse_errors` counter and nothing else |
| words split across a printed line (`Victo-\nrina`) | **4.06% of all words** | ~3.5M broken tokens at PL's scale |

Repairing both took one volume from **50,980 words to 218,818**. Placement is
not arbitrary: entity repair runs *only after a parse fails*, so a file broken
some other way still reports a real error rather than being masked; and
de-hyphenation runs *before* `tidy()`, which collapses the newline and
destroys the only evidence that the hyphen was a line break.

The header's `<date>` is **not** a composition date — the PL files say 2015,
meaning when Leipzig encoded them. The reader leaves `approximate_date` unset
and recovers the **PL volume from the filename** instead, feeding the
`pl_volume` interpolation `enrich` already has. Taking the header date would
repeat the CSEL-preface mistake exactly.

**What is left is not code.** There is no bulk export; texts download one at a
time. The unblocking action is an email to turicense@gmail.com.

### Two bugs of my own worth remembering

- `\bPL(\d{1,3})\b` never matched `[CC]PL10_19_DE.xml`, because `_` is a word
  character so there is no boundary after `10`. Caught by a test written from
  a real filename rather than an imagined one.
- Waiting with `until ! pgrep -f "src/pipeline.py"` never terminates: `pgrep
  -f` matches the waiter's own command line, which contains that string. Two
  watcher shells span forever and no completion was ever reported.

---

## Update 2026-08-26 (later) — half of the CSEL repo was never read

`csel-dev` ships its texts twice. `data/` holds one file per work in CapiTainS
layout, which is what the extractor read; `Volumes/` holds the printed volumes
those were split out of, 292 MB, and nothing had ever opened it.

| | words |
|---|---:|
| **New prose recovered** | **617,686** |
| Apparatus rejected | 2,205,746 |
| Duplicate text, caught by containment | ~313,000 |

| corpus | records | words | train tokens |
|---|---:|---:|---:|
| Track A · full | 10,763 | 94,465,896 | **182,307,970** |
| Track A · classical | 256 | 4,831,855 | **10,241,858** |

**+616,357 words (+0.66%)** and **+1,420,588 tokens (+0.79%)** on the full
corpus. Floors: **181.45** full (was 183.0), **184.68** classical (was 184.4).
CSEL rises to 6,283,656 words — 6.7%, the second-largest source.

The classical corpus is unchanged to the word: every new record is NEOLATIN or
LATE. Its token count moves by −4,945 only because the tokenizer is retrained.

### The markup cannot be trusted, in either direction

CSEL48 files 68,471 words of *index verborum* under `subtype="chapter"`;
CSEL65 files a real work, *Liber ad Constantium*, under `subtype="MSS"`. So
`classify_division` weighs three signals, strongest first — what a division
calls itself in its `<head>`, what its text measurably looks like, and only
then what the markup claims. Across all 579 divisions: **100% of apparatus
rejected, 99.9% of prose words kept.**

The thresholds are not tuned; the two populations barely touch. Prose sits at
the 95th percentile on a 0.3% digit rate, apparatus at the 5th on 6.0%. The
function-word floor is deliberately set *below* verse — hymns run 6% and a
cento 8.6%, because poetry drops connectives — since a filter that cannot tell
a hymn from a word list is useless on a volume of Christian poetry.

Division-level judgement was not enough on its own. A preface argues about the
manuscripts for a paragraph and then cites twenty by page and line, so judged
on its average it reads as apparatus: 296,535 words of preface were being
discarded, 72% of it discussion. `usable_prose` now re-judges anything whose
markup claims prose on **what survives paragraph filtering**. The populations
separate cleanly there — every preface falls to ≤0.102 digits, while
Priscillian's canon list holds at 0.239 and the *Itinerarium Burdigalense* at
0.139, because their tables run *inside* the paragraphs.

### Most of it is 1890s Latin, and saying so took three fixes

447k of the 618k is prefaces and prolegomena — real Latin prose, written
between 1867 and 1934. The extractor labelled them, and `enrich` overwrote
every label, because it resolves `period` and `genre` from the authorities and
falls through to the source prior.

1. **Period.** The csel prior is `LATE`, so the prefaces were being dated to
   the 2nd–7th century. The 34 CSEL editors are now in
   `manifest/authorities.yaml`, dated by their earliest volume — a documented
   publication date, not a biographical estimate.
2. **Author.** A preface is the *editor's* writing. Crediting it to the author
   of the work it introduces attributed 447k words of philology to Lactantius
   and Cyprian.
3. **Genre.** `title_patterns` matched the work named in the title, so 40 of
   the 85 records were filed as EPISTOLOGRAPHY, THEOLOGY or COMMENTARY. A
   preface *about* Augustine's letters is not epistolography. A `SCHOLARSHIP`
   genre now exists for it, keyed on the constructed marker `Praefatio
   editoris in` — plain `Praefatio in` would have swept up seventeen of
   Jerome's own prefaces to the biblical books.

**The baselines confirmed the classification independently.** SCHOLARSHIP is
the hardest genre in the corpus for a bigram model — **ppl 242.9** against
181.5 overall and 163.7 for homiletic. Had this gone in as `LATE` and
`EPISTOLOGRAPHY` it would have raised the late-antique floor with nothing in
any report to say why.

Whether 447k words of technical, OCR-damaged 19th-century Latin *belongs* in
training is now a one-flag decision (`--period`, `--exclude-provenance`) rather
than something decided by default.

### A pipeline bug, hit rather than reasoned about

`extract.py` was edited while the extract stage was running. The stage recorded
the **new** source fingerprint against an interim the **old** code produced,
then read as up to date permanently, over output no version of the code would
generate. Fingerprinting now happens when a stage *starts* — by then every
earlier stage has finished, so the lockfile property the old comment defended
still holds, and what is recorded is what the stage actually read.

Also worth knowing operationally: killing `pipeline.py` does **not** kill the
stage it launched. An orphaned `dedupe.py` survived a kill and would have
written duplicate marks over a fresh extraction, producing a corpus assembled
from two different code versions with nothing in the output to show it.
`pkill -f "src/dedupe.py"` misses it — the pipeline invokes stages by absolute
path.

### Still open here

- **Papyri `Historical/`** looked like 82,332 unread files. It is an alternate
  index of the same documents: 631 unique ids, of which 10 have any text, 498
  words total. Correctly left alone.
- Whether to train on the NEOLATIN material at all.

---

## Update 2026-08-26 — every green source now has a reader

Two extractors, four sources, because dispatch is by format: **EpiDoc** serves
EDH and Papyri.info, **plaintext** serves the Latin Library and Gutenberg. All
nine green sources are now readable, and the corpus is built from eight of them
(Gutenberg is `manual` and has never been downloaded).

| corpus | records | words | train tokens |
|---|---:|---:|---:|
| Track A · full | 10,678 | 93,849,539 | **180,887,382** |
| Track B · full | 10,707 | 93,947,780 | **181,058,058** |
| Track A · classical | 256 | 4,831,855 | **10,246,803** |
| Track B · classical | 256 | 4,831,855 | **10,246,803** |

Against the previous build: **+5,799,234 tokens (+3.31%)** on the full corpus
and **+655,415 (+6.83%)** on classical. Floors: 183.0 full, 184.4 classical.
At four epochs the corpus now supports a **36.2M-parameter** model.

New sources after dedup: latin-library 829 records, edh 131, papyri-idp 5.

**The classical corpus was not as closed as I claimed.** I said Perseus was
substantially complete and classical could not grow; it gained 6.8%. Dedup did
absorb most of the Latin Library's classical material — pre-dedup it looked
like +31% — but not all of it.

### EpiDoc is TEI, and epigraphic convention is not literary convention

  - `<lb/>` In a literary edition a line break usually falls mid-word, which is
    why the walker elides it and why Gellius' *Athenis* survives a `<pb/>`. On a
    stone it is a word boundary: **87.6% of EDH line breaks separate words**,
    12.4% carry `break="no"`. The literary rule would have welded CAIUS and
    UIBIUS into one token.
  - `<head>` Every EDH edition div opens with a `<head>` reading "Text".
  - `<am>` `<surplus>` `<g>` `<certainty>` are apparatus, not text.

Kept deliberately: `<expan>`, whose `<abbr>` and `<ex>` children spell the
expanded word, so `Off(icina)` arrives as *Officina* exactly as EDCS supplies
it; and `<supplied>`, the editor's restoration — conjecture, but the reading
the edition asserts, and dropping it leaves shredded words. The walker is
parameterised by a `Profile` so the literary rules are untouched rather than
compromised into serving both.

### Two bugs this exposed

**The language filter tested `lang.startswith("lat")`.** EpiDoc tags Latin `la`
under ISO 639-1; Perseus uses 639-3 `lat`. Every edition div was discarded as a
foreign language — silently, apart from a `dropped:lang:la` counter nobody
reads. The first run produced empty text for all 79,206 inscriptions.

**A declared-but-unfetched source broke the whole build.** Gutenberg sits in the
lockfile from the moment it is declared, with `present: false` and no files.
The instant `txt` gained a reader, the pipeline tried to extract it and failed
the run. `extractable_sources()` now requires `n_files > 0`.

### Plain text is the harder case, as feasibility.md §4.4 says

A TEI source names its apparatus; here running heads, section indexes and site
furniture are inline and must be recognised by shape. Every rule matches one
observed pattern rather than generalising, because a general heuristic on
undifferentiated text deletes Latin. The site footer takes the whole **line**,
not the phrase — it reads "Sallust The Latin Library The Classics Page", so
cutting at the phrase leaves the author dangling as the last word of the work.
Cicero's files put the byline on its own line instead, which is why trailing
bylines are stripped separately.

### Still open

Gutenberg (~500k words) needs a human to download it; the reader is written and
tested against Project Gutenberg's own boilerplate markers, but untested on real
files. Both floors moved, so **no existing model's ratio survives this rebuild**
— `compare.py` will flag every row.

---

## Update 2026-08-25 — Track B has a source, and dedup was discarding 12M words

Track B has existed in the manifest and the guards since the two-track split was
designed and had never contained a source. It does now: **LASLA**, 1,801,734
words of hand-verified classical Latin, CC BY-NC-SA 4.0, fetched from the CIRCSE
mirror as CoNLL-U Plus.

Ingesting it exposed something far larger than itself.

### The corpus grew by 12.2 million words, and LASLA is not why

| corpus | records | words | train tokens |
|---|---:|---:|---:|
| Track A · full | 10,422 | 90,724,940 | **175,088,148** |
| Track B · full | 10,452 | 90,835,090 | **175,259,009** |
| Track A · classical | 204 | 4,541,027 | **9,591,388** |
| Track B · classical | 204 | 4,541,027 | **9,591,388** |

Against the 2026-08-21 build: **+12,170,743 words (+15.5%)** and **+22,854,065
train tokens (+15.0%)**. Floors moved to 172.1 and 148.5.

### What was wrong

Dedup clustered with union-find, which is transitive. Textual similarity is not.
Patristic and medieval prose shares scripture, liturgy and citation, so A~B and
B~C chained an entire genre into one cluster where a single keeper swallowed the
rest.

Jerome's *Epistulae* — 290,455 words — was absorbing **401 records and
12,831,872 words**: Remigius on the Psalms, Haymo's homilies, Augustine on John,
a florilegium of Augustine. None of them is Jerome's letters. **None had ever
been compared with Jerome's letters** — every one carried
`duplicate_similarity: None`.

A record is now marked a duplicate only of a record it was **directly measured
against**. Members with no direct pair to the keeper are re-resolved among
themselves, so genuine sub-groups still collapse. Duplicates fell from 2,802 to
2,557 in Track A, and 27.3% of fetched text is still discarded — the difference
is that each discard is now backed by a measurement.

One semantic worth knowing: if A absorbs B, and C was similar only to B, **C is
kept**. B is no longer in the corpus, so nothing surviving is known to contain
C, and dropping it would be a guess.

**This is the third time cluster instability surfaced.** It appeared twice
before as "keeper churn", and both times it was recorded and set aside. It was
not churn.

### What LASLA itself contributed

**+170,861 tokens, +0.10%.** Thirty-three of its 242 records survive; its 19
authors are Perseus's core, so the rest duplicates text already held. Its
classical contribution is **exactly zero** — Track B classical equals Track A
classical to the token.

The manifest already said its value is as the gold standard for morphological
evaluation, and evaluation data is never redistributed, so NC costs nothing
there. That remains the reason to keep it.

### Two structural bugs fixed on the way

**A green record could be marked a duplicate of an amber keeper.** Track A would
then drop the duplicate for being a duplicate and the keeper for being
non-redistributable, and the text would leave the published corpus silently.
Keeper ranking is now (licence tier, curation rank).

**That was not enough, because the damage precedes keeper choice.** dedupe now
resolves each track separately, restricting the candidate *pairs*, and writes
two verdicts per record. The guarantee: **Track A does not depend on which
research-only sources are on disk** — verified, twice.

### The remaining NC sources

Corpus Corporum (215M words) and IntraText (10M) are still `method: manual` —
no bulk endpoint, and Corpus Corporum needs a human at mlat.uzh.ch. It also
contains the entire Patrologia Latina and Perseus, believed already held here,
so its headline is not 215M new words. Believed, not measured — see
`manifest/sources.yaml` on why the containment check stays unrun.

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

Measured on the 2026-08-27 build. Every figure here is read off
`data/processed/*.summary.json` and `data/processed/token-growth.json` —
regenerate those rather than editing this section, which is how the numbers
below drifted a build behind before.

**10,782 records, 94,923,570 words** (Track A, after dedup), from **9 of 11**
green sources.

| period | words | |
|---|---:|---|
| UNASSIGNED | 39,488,556 | 41.6% |
| LATE | 28,040,177 | 29.5% |
| MEDIEVAL | 21,400,056 | 22.5% |
| CLASSICAL | 5,055,913 | 5.3% |
| NEOLATIN | 938,868 | 1.0% |

| source | records | words (pre-dedup) |
|---|---:|---:|
| vicifons | 11,744 | 86,275,924 |
| latin-library | 2,130 | 13,505,742 |
| tesserae | 747 | 6,734,531 |
| opengreekandlatin-csel | 356 | 6,652,629 |
| digiliblt | 373 | 6,509,980 |
| perseus-canonical-latinLit | 324 | 5,586,624 |
| edcs-sdam | 267 | 4,356,735 |
| edh | 133 | 1,057,053 |
| papyri-idp | 5 | 89,915 |

Dedup removes **35,845,563 words (27.4%)** of the 130.8M green words extracted —
high because Wikisource re-hosts much of what Perseus and CSEL already provide.

**Tokens** (4k vocabulary, counted not projected): **183.3M full corpus,
10.7M classical**. Track B, which adds the NonCommercial sources and is never
redistributed, holds **263.5M full, 12.0M classical** — 43.8% more than Track A,
where before the 2026-08-27 build it held 0.06% more.

### Not yet extracted

`gutenberg-latin` and `latinise`, both green, both `method: manual` — neither is
blocked on an extractor. Gutenberg is declared and unfetched; `latinise` is
`include: false` by default because it is substantially derived from
LacusCurtius, IntraText and Musisque Deoque, so most of it would be dedup'd
against text already held.

---

## Models — eleven training runs and one fine-tune, across three eras of the corpus

`python src/compare.py` prints this live from `reports/train-*.json`, with a
staleness mark per row, and **it is the source** — the snapshot below was taken
2026-09-06 and will drift. What matters is not the list but which rows may be
read against each other: a corpus keeps its name across rebuilds, so two runs
labelled `classical` can be held-out scores over different text.

### Comparable to the current floors

| run | params | corpus | tokens | ep | ppl | floor | × floor | memo | hours |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **latin-vii** | 7.28M | classical | 10,671,909 | 14.2 | **70.1** | 185.7 | **2.65×** | 0% | 3.1 |
| **latin-lii-latinitas** | 52.07M | corpus-b | 263,558,393 | 4.0 | **21.3** | 185.9 | **8.73×** | 0% | 73.8 |
| latin-lii-romanitas | 52.07M | classical | 10,671,909 | 2.0 | 30.2 | 185.7 | 6.15× | — | 1.5 |

The two LATIN-LII rows are one model at two points in one recipe, not two
models. The romanitas phase made it **worse** at the thing it exists for — 27.3
→ 30.2 on classical, measured through the same vocabulary — because it
introduces no new text, only re-weights toward a few per cent of what the model
has already read, and 10.7M tokens cannot occupy 52M parameters. See the 2026-09-01 entry; `configs/latin-vii.yaml` records why
7.28M is the size that could still take such a phase.

**latin-lii-romanitas is not published.** LATIN-LII trains on Track B, which
carries NonCommercial material, and that restriction travels to the weights.
LATIN-VII is the one released — [on Hugging Face](https://huggingface.co/vincent-b/latin-vii),
CC-BY-SA-4.0, classical only, no NC term anywhere in its corpus.

### Measured on an earlier build of the same corpus name — marked `!`

| run | params | corpus | tokens | ep | ppl | memo | hours |
|---|---:|---|---:|---:|---:|---:|---:|
| latin-xxv-latinitas | 27.49M | corpus | 152,985,050 | 4.0 | 25.2 | — | 24.7 |
| latin-xxv | 27.49M | classical | 9,562,431 | 3.0 | 29.8 | 0% | 1.2 |
| latin-xxv-2-latinitas | 27.49M | corpus | 152,247,527 | 4.0 | 25.8 | — | 24.5 |
| latin-xxv-2 | 27.49M | classical | 9,539,941 | 3.0 | 30.1 | 0% | 1.2 |

Same names, smaller corpus: `corpus` was 152M tokens when these ran and is
183.3M now. compare.py refuses to quote a floor beside them rather than
flattering them with one measured over different held-out text.

### Predating corpus fingerprinting — marked `?`, the 17M-word era

| run | params | corpus | tokens | ep | ppl | memo | hours |
|---|---:|---|---:|---:|---:|---:|---:|
| latin-0 | 3.53M | classical | 9,538,436 | 8.6 | 74.3 | 0% | 1.0 |
| ablation-no-latinitas | 15.90M | classical | 9,538,623 | 8.6 | 58.3 | 0% | 2.5 |
| latin-2-latinitas | 15.90M | corpus | 33,689,916 | 7.8 | 48.2 | — | 8.1 |
| latin-2 | 15.90M | classical | 9,538,623 | 3.0 | 44.1 | 0% | 0.9 |

Their reports carry no corpus fingerprint at all, so the build they were
measured on cannot be established — assume it differs. Every one predates the
Vicifons/EDCS expansion.

`latin-2-chat` is the one SFT run and has no training report of its own
(`reports/sft-latin-2-chat.json`): latin-2 fine-tuned on 54,535 of 54,536 mined
dialogue pairs, 1,689,049 supervised response tokens, 0.6 hours. Behaviour, not
knowledge — it takes a turn and stops.

**The ablation result is the one worth carrying forward.** Of the 30.2-point
improvement from latin-0 to latin-2, model size accounted for ~53% and the
two-phase Latinitas→Romanitas schedule ~47%. Both real, neither dominant.

Caveat found while running it: `classical.train` is a **subset** of
`corpus.train`, so Phase 1 already contained the entire Phase 2 corpus. Latin-2
made ~10.8 passes over classical text against the ablation's 8.6. Part of the
"schedule" benefit is therefore extra exposure, not transfer. The clean version
would exclude classical from Phase 1 — still open.

---

## The ladder, remeasured on 94.9M words

Chinchilla-optimal ≈ 20 tokens/parameter; repeated data is worth nearly as much
as fresh up to ~4 epochs, decaying to ~zero past 16.

| epochs | classical (10.7M tok) | Track A full (183.3M tok) | Track B full (263.5M tok) |
|---|---|---|---|
| 4 (safe) | 2.1M params | **36.7M params** | **52.7M params** |
| 8 | 4.3M | **73.3M** | **105.4M** |
| 16 (aggressive) | 8.5M | **146.6M** | **210.8M** |

**LATIN-XXV at 27.5M fits inside 4 epochs of Track A**, where it once sat on the
16-epoch line. Path A's 25–40M licensing ceiling is no longer the binding
constraint; the corpus is.

**LATIN-LII at 52.07M now fits Track B's 4-epoch line exactly** — 20.2
tokens/parameter, where `configs/latin-lii.yaml` was written against a 182.4M
Track B that gave it 14.0 and called itself undertrained. The recipe did not
change. Stating recipes in epochs rather than steps is what let the corpus grow
into it.

Classical has grown from 9.5M to 10.7M tokens, so **Latin-0 at 3.53M and
LATIN-VII at 7.28M both remain correctly sized** — LATIN-VII at 20.8
tokens/parameter against the 16-epoch ceiling of 8.5M.

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
ATTRIBUTION.md     Track A credits, generated by provenance.py — do not hand-edit
ATTRIBUTION-corpus-b.md  Track B credits, incl. the NonCommercial sources

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

# demos

Things you can point LATIN-LII at. Each one is a single file, runs on the CPU
or GPU of the machine that trained the model, and prints a number you can argue
with.

```bash
source .venv/bin/activate

python demos/paria.py "puella rosam amat" "puella rosam amant"     # which is Latin?
python demos/ordo.py demos/samples/clean.txt                       # what does the number measure?
python demos/surprisal.py demos/samples/damaged.txt --plain        # where is the damage?
python demos/lacuna.py --text "Gallia est omnis divisa in [...] tres" --explain
python demos/aetas.py demos/samples/medieval.txt                   # which century?
python demos/serve.py                                              # all of it in a browser
```

## What they show, in the order worth reading them

The model is a 52M-parameter GPT that has read nothing but Latin. Every demo
turns the same single ability — assigning a probability to a sequence of
tokens — into a different instrument, and each one is paired with a measurement
on held-out text so that what it shows on a famous sentence can be checked
against what it does on text it has never seen.

| demo | question | what it measures | held-out result |
|---|---|---|---|
| `paria.py` | Which of two readings is Latin? | grammar: agreement, case, mood, voice | 20/21 pairs; 92% against attested wrong inflections |
| `ordo.py` | What does perplexity measure? | syntax, beyond vocabulary | shuffled text 9× as surprising; 100% of 200 passages |
| `surprisal.py` | Where does this text go wrong? | improbability, per word | all 5 injected OCR errors on top |
| `lacuna.py` | What was in the gap? | reconstruction with right context | 40% chosen, against 8% for generation |
| `aetas.py` | Which era is this? | period conditioning, turned around | 62% over four eras; medieval leaks |

Read down the table and the claims get weaker, on purpose. Grammar is what a
model of Latin most clearly has. Era estimation is what it has least, and the
demo says so on every answer.

## Which model these load

`DEFAULT_MODEL = "latin-lii-latinitas"`, and that is deliberate.

The project's convention is that `<run>-latinitas` is phase one and the bare
`<run>` is the classical-narrowing phase two — which is why `latin-xxv`,
`latin-xxv-2` and `latin-2` all have a bare checkpoint.

LATIN-LII has no phase two. It was run, it made the model worse (30.2 on
classical, 32.5 on corpus-b, against 27.3 and 21.3 before it), and
`configs/latin-lii.yaml` records why it was dropped. Those weights are kept —
deleting a measurement is worse than labelling one — but under
`latin-lii-romanitas`, so there is no bare `latin-lii` checkpoint to load by
mistake. The model everything here quotes is the surviving single phase,
`latin-lii-latinitas`.

Anything can be loaded instead with `--name`, including `latin-vii`, which is
the Track A model and the only one whose weights are free of NonCommercial
terms. `paria.py` accepts `--name` more than once and prints the checkpoints
side by side, which is the quickest way to see what model size bought.

## Everything here folds your text to the corpus spelling

The corpus is normalised u-for-v and i-for-j. In a sample of 2,000 training
documents, these stems appear 228,000 times with `u` and **zero** times
with `v`:

| stem | u-spelling | v-spelling |
|---|---:|---:|
| `diuis-` | 18,760 | 0 |
| `uirtu-` | 64,663 | 0 |
| `prouinc-` | 17,139 | 0 |
| `uita` | 127,202 | 0 |

So a `v` inside a word is, to this model, a character that essentially never
occurs there. Measured: the single token `v` in `divisa` costs **30.4 bits**
against about 9 for its neighbours. Text copied from any modern edition scores
as damage from end to end — the first run of `surprisal.py` flagged
*provinciae*, *Helvetii*, *virtute* and *dividit* as the most corrupt words in
a passage whose actual corruption I had injected myself, and every one of those
false positives was flagged solely for containing a `v`.

Every demo therefore pushes input through `src/normalize.py` — the same
function the corpus went through, not a copy of it — and says so when it
changes anything. `--raw` turns it off.

## paria.py — which of two readings is Latin, and where they part company

A minimal pair is the smallest experiment that can be run on a language model:
two texts that differ in one thing, and the difference in their scores is the
model's opinion about that one thing.

```
$ python demos/paria.py "puella rosam amat" "puella rosam amant"

  A  'puella rosam amat'                                     43.5 bits
  B  'puella rosam amant'                                    48.1 bits

  A is preferred by 4.68 bits (26x as likely)

  A token          bits   B token          bits    B - A
  ' pu'            18.0   ' pu'            18.0     +0.0
  'ella'            1.1   'ella'            1.1     +0.0
  ' r'              7.1   ' r'              7.1     +0.0
  'os'              4.3   'os'              4.3     +0.0
  'am'              4.5   'am'              4.5     +0.0
  ' am'             6.8   ' am'             6.8     +0.0
  'at'              1.7   'ant'             6.4     +4.7   <-- differs
                                                    +4.7   on the changed span
```

The token table says *where* the cost was paid. Here it is entirely on the
verb ending. When the changed word comes early, the cost also lands on the
text after it — the wrong form makes what follows harder to predict — and
that spill-over is the model using context, which is the mechanism
`lacuna.py` relies on.

### The suite: what a model of Latin has learned, at a glance

`--suite` runs 23 built-in pairs across agreement, case government, mood,
voice and word order. Run it with several `--name` flags and the columns
are the model ladder:

```
$ python demos/paria.py --suite --name latin-vii --name latin-xxv-2 --name latin-lii-latinitas

  phenomenon                                          vii    xxv-2   lii-latinitas   reading  |  alternative
  agreement: subject and verb                       +0.83    +5.06     +4.68    puella rosam amat  |  puella rosam amant
  agreement: subject and verb                       -3.83!   -0.58!    +0.87    milites castra muniunt  |  milites castra munit
  agreement: subject and verb, at a distance        -1.65!   +4.87     +4.49    milites qui in castris manserant fortiter pugnauerunt  |  ... pugnauit
  agreement: gender of a participle                 +1.48    +1.34     +4.09    Gallia est omnis diuisa in partes tres  |  ... diuisus ...
  agreement: gender of an adjective                 -2.23!   +4.10     +6.76    magnus exercitus  |  magna exercitus
  agreement: relative pronoun                       -2.74!   +2.44     +7.30    mulier quae uenit  |  mulier qui uenit
  agreement: adjective and noun, genitive plural    -0.22!   +0.16     -3.01!   bonarum puellarum  |  bonorum puellarum
  case: after a preposition                         +2.34    +2.77     +8.18    cum amico ambulabat  |  cum amicum ambulabat
  case: after a preposition                         +9.20   +12.62    +16.26    in urbem uenit  |  in urbis uenit
  case: direct object                               -2.47!   +3.43     +5.56    Caesar Galliam uicit  |  Caesar Gallia uicit
  mood: subjunctive after ut                        +1.13    +9.11     +8.17    rogo ut uenias  |  rogo ut uenis
  mood: subjunctive in a cum-clause                 +4.84    +8.03    +11.37    cum Caesar in Galliam uenisset  |  ... uenerat
  indirect statement                                +4.69    +3.70     +9.67    dixit se uenturum esse  |  dixit se ueniet
  voice                                             +2.96   +10.24     +8.72    hostes urbem oppugnabant  |  ... oppugnabantur
  form of a preposition before a vowel              +9.63   +12.65    +11.06    ab urbe condita  |  a urbe condita
  word order                                        -3.34    -0.72     +7.89    Caesar Galliam uicit  |  Galliam Caesar uicit
  ...

  vii            15/21 pairs the right way, median margin 3.8 bits
  xxv-2          18/21 pairs the right way, median margin 4.1 bits
  lii-latinitas  20/21 pairs the right way, median margin 6.8 bits
```

Three things to read off this:

- **Grammar arrives with size.** The 7M model gets 15 of 21 and fails on
  gender and on the direct object; the 27M model gets 18; the 52M model gets
  20 with a median margin nearly twice as wide. The one it still fails is the
  one it is *expected* to fail.
- **The frequency trap is real and the demo keeps it.** `bonorum puellarum`
  beats `bonarum puellarum` by 3 bits on the biggest model, because `bonorum`
  is the commoner string and a language model learns frequency and grammar as
  one quantity. A pair that pits a rare right form against a common wrong one
  measures the sum. Anyone building on these numbers should know that, so the
  pair stays in.
- **Word order is a preference, not a verdict.** `Galliam Caesar uicit` is
  Latin. The suite reports which order the model prefers (the 52M model
  prefers SOV by 8 bits; the 7M model prefers the object first) and counts
  neither as right.

### Measured, not demonstrated

The suite sentences are mine, and some are famous. `--benchmark` takes
passages from the held-out split and scores each against three corruptions of
itself. The **ending** corruption is the honest one: the hidden word takes a
different inflection, and only if that form is itself attested at least three
times in the held-out text — so the model cannot win by noticing a non-word.

```
$ python demos/paria.py --benchmark 200 --name latin-vii --name latin-xxv-2 --name latin-lii-latinitas

                   original preferred
  corruption    n     vii   xxv-2    lii      median margin (lii)
  swap        136   68.4%   80.9%   85.3%          6.0 bits
  ending       65   78.5%   84.6%   92.3%          8.2 bits
  shuffle     200   97.5%  100.0%  100.0%        100.7 bits
```

`swap` exchanges the hidden word with its neighbour; `shuffle` is the
reference case every model must win. An attested alternative inflection
existed for 65 of the 200 words. The 8% the biggest model gets "wrong" are
worth printing, and `--show` does:

```
  inflections the model preferred to the original:
      3.1 bits   ...Pax erit et bellum: [cupiam -> cupias] quum regna tyrannus...
      3.2 bits   ...Cinyrae coitum? Quid Biblidis [ignem -> ignes] Et scelera...
      3.4 bits   ...MAXIMUS CONFESSOR, Quaestiones et [dubia -> dubiae] 1,1,2...
```

Most of them are readings that are themselves grammatical. The benchmark
counts them as misses anyway, so the number it reports is a floor.

## ordo.py — what the number measures

Take a passage. Shuffle its sentences; shuffle the words inside each
sentence; shuffle every word; reverse it. Every one of those texts has exactly
the vocabulary and the word frequencies of the original, so a frequency table
would give them all the same score.

```
$ python demos/ordo.py demos/samples/clean.txt --against demos/samples/english.txt

  text                                   perplexity  x as written  tok/word  bits/word
  as written                                   30.2         1.00x      2.10       10.3
  sentences shuffled                           37.5         1.24x      1.94       10.1  (±0.6 over 3 shuffles)
  words shuffled within each sentence         171.6         5.68x      1.94       14.4  (±17.6 over 3 shuffles)
  all words shuffled                          271.7         9.00x      1.94       15.7  (±3.3 over 3 shuffles)
  word order reversed                         169.9         5.63x      1.94       14.3
  english.txt                                  73.0         2.42x      2.71       16.8
```

Shuffling the sentences barely moves the number, because a sentence is mostly
predicted from itself. Shuffling within sentences moves it five to six times,
because agreement, case government and idiom all live inside the sentence.
That ratio is what the model knows about Latin *order*, over and above which
words Latin has, and it is the whole of what a language model adds to a
frequency table. The medieval sample gives 8.6×, so the ratio is stable across
texts. Across models it is not: the 7M model is only 4.7× more surprised by
shuffled Caesar, the 27M and 52M models both about 9× — most of what there is
to learn about Latin word order at this corpus size had been learned by 27M.

**The English row is a lesson in units.** Per token, English scores only 2.4×
the Latin — less than shuffled Latin — which looks absurd until the next
column: the tokenizer was fitted on Latin, so it cuts English into 2.7 pieces
a word against 2.1, and each small piece is easy to guess from the last.
Perplexity is per token and rewards fragmentation. Bits per word is the cost
of the whole text in a unit that does not depend on the cutting, and there
English is the most expensive row. Compare that column when the texts were
not tokenised alike; compare perplexity only when they were.

## surprisal.py — show me where the text goes wrong

Prints the text back with every word shaded by how surprised the model was,
and lists the worst offenders. `--plain` swaps colour for punctuation so it
survives a pipe.

On `demos/samples/damaged.txt` — genuine Caesar with five OCR errors and one
line of apparatus criticus spliced in — the top of the list is:

```
  bits  toks  word
  47.9     5  Nipperdey        <- apparatus
  42.9     4  rnercatores      <- rn for m
  42.3     4  cu1tu            <- 1 for l
  37.6     4  fortissirni      <- rn for m
  37.1     4  minirneque       <- rn for m
  35.9     5  hurnanitate      <- rn for m
  25.2     5  codd.            <- apparatus
```

All five injected errors and the apparatus, with no ordinary Latin word above
them. Surprisal is summed per **word**, not per token, so a long word is not
punished for being several BPE pieces.

`--pieces` opens each hot word up into the tokens it was scored as:

```
  42.9     4  rnercatores
              ' r' 9.7 + 'n' 11.0 + 'erc' 12.2 + 'atores' 10.0
  42.3     4  cu1tu
              ' c' 9.3 + 'u' 8.0 + '1' 14.0 + 'tu' 11.1
```

A damaged word pays twice. The tokenizer has no piece for a scanno, so the
word is cut into more pieces than its correct form — `mercatores` is 2 tokens,
`rnercatores` is 4; `cultu` is 2, `cu1tu` is 4 — and then every one of those
pieces is improbable. That is why the injected errors sit so far above
ordinary rare words: fragmentation and surprise compound.

This is the same measurement as `src/score.py --audit`, which is what found the
bad documents in this corpus; the difference is that the audit ranks documents
for a pipeline and this renders one document for a person.

**It reports improbability, not error.** Proper names, hapax legomena and
technical vocabulary light up exactly the way corruption does. A hot word is a
place to look, never a verdict.

## lacuna.py — fill a hole, and rank the conjectures

```
$ python demos/lacuna.py --text "Gallia est omnis divisa in [...] tres, \
    quarum unam incolunt Belgae" --answer partes --blind --explain

gap 1 of 1:  ...Gallia est omnis diuisa in  [1 word(s)]  tres, quarum unam ...

  14 candidates, judged on the fill AND the text after it
  bits worse  toks  supplement
        0.00     1  'partes'   <-- the reading in the manuscripts
        6.49     1  'duas'
        6.60     2  'prouincias'

  where the ranking was decided, in bits per token (lower is likelier)

                    'partes'    'duas'
  the fill itself        3.2       1.3
  ' tres'                5.5      11.4   lead   +4.0   <-- decisive
  ','                    2.1       4.2   lead   +6.1
  ' qu'                  5.0       4.4   lead   +5.5
  ...
  lead after the fill alone -1.9 bits; after the right context +6.6 bits.

  the same candidates judged on the left context alone -- what plain generation sees
  bits worse  toks  supplement
        0.00     1  'duas'
        1.89     1  'partes'   <-- the reading in the manuscripts
        2.34     1  'tres'
```

A causal model cannot read a gap from both sides. But it can score a whole
sequence, and `P(fill + everything after it)` does depend on the text after the
gap. So the model **proposes** forward and is **judged** across the right
context. `--blind` shows what that is worth: the same candidate list ranked on
the left context alone, which is all plain generation ever sees.

`--explain` shows the mechanism. On the fill alone `duas` *wins*, by 1.9
bits — "divided into two" is a fine thing to say. It loses on the very next
token: `tres` after `duas` costs 11.4 bits against 5.5 after `partes`, and
that one token is the whole decision. Nothing later moves the lead much.
That is what "judged across the right context" means in practice: usually
one or two tokens after the gap that one reading can account for and the
other cannot.

Bring your own conjectures with `-c`, and constrain by the letters the stone
still shows with `--pattern`:

```bash
python demos/lacuna.py --text "arma [...] cano" -c uirumque -c ducemque
python demos/lacuna.py FILE --pattern '^p.*s$' --words 2
```

### Measured, not demonstrated

The Caesar sentence above is in the training corpus, so recovering it proves
nothing. `--benchmark` runs the same task on the **held-out** split:

```
$ python demos/lacuna.py --benchmark 60 --candidates 32

proposed by the model         7/60   11.7%
chosen, with right context   24/60   40.0%
chosen, left context only     5/60    8.3%   <-- what generation alone gets
median rank of the true word  2   top-5 66.7%
```

Read that as three separate claims:

- The model **generates** the exact missing word only 11.7% of the time. That
  is a ceiling on any pure-generation approach.
- Given a candidate list, scoring across the right context picks the true word
  **40.0%** of the time, against **8.3%** for the left context alone. Reading
  backwards is worth roughly 4.8×, and that gap is the whole argument for this
  demo existing.
- The true word's median rank is 2 and it is in the top five two thirds of the
  time, so the useful output is the shortlist, not the winner.

Trimming punctuation off proposals moved all three numbers (36.7 → 40.0%). A
24-sample pool had been spending four of its slots on `duas`, `duas,`, `duas:`
and `duas.` while `partes` never made it in.

## aetas.py — which era does this read as

Same tokens, four era prefixes, best fit wins. The tags were real conditioning
during training, so this is a classifier that falls out of the corpus design
rather than one that was trained for.

```
  era           perplexity    vs best
  CLASSICAL          30.15    1.0000x  <-- best fit
  LATE               31.92    1.0590x
  NEOLATIN           33.50    1.1114x
  MEDIEVAL           38.99    1.2935x

94 tokens.  CLASSICAL over LATE by 0.0827 bits/token (clear).
```

**The margins are tiny and the demo says so.** A period tag is two tokens in
front of a few hundred, so best-to-worst is typically ~2.5% of perplexity.
`--calibrate` measures what that is worth on held-out text of known date, and
`reports/era-<model>.json` is then quoted back on every estimate:

```
accuracy 37/60 = 61.7%   (chance 25%)

            CLAS   LATE   MEDI   NEOL   <- predicted
CLASSICAL     12      3      0      0    80% recall
LATE           1     12      1      1    80% recall
MEDIEVAL       0      8      6      1    40% recall
NEOLATIN       4      1      3      7    47% recall

margin band     n  accuracy
very weak       6     33.3%
weak           14     64.3%
moderate       30     63.3%
clear          10     70.0%
```

Read honestly: it separates **CLASSICAL and LATE well (80%)**, cannot hold
MEDIEVAL apart from LATE (8 of 15 leak), and scatters NEOLATIN. The margin band
carries real information — 33% at "very weak" against 70% at "clear" — so a
weak margin should be disbelieved rather than rounded up.

`demos/samples/medieval.txt` is there to show the failure rather than hide
it. It is the opening of a ninth-century saint's life (held out from
training, Vicifons, CC-BY-SA-4.0), and the model calls it LATE by a very
weak margin, with MEDIEVAL last of four. The calibration table predicted
exactly that: a leak from MEDIEVAL to LATE, and 33% accuracy at that margin.
The demo prints both, which is the point of carrying the calibration around.

**Do not assume this transfers to another checkpoint.** The same procedure on
LATIN-XXV-2 gets 53.5% over *three* eras with a median margin of 0.3% — barely
distinguishable from no conditioning at all. Era estimation arrived with model
size and corpus breadth; it is not a free consequence of period tagging. Run
`--calibrate` before trusting it anywhere new.

## serve.py — all five in a browser

```bash
python demos/serve.py           # http://127.0.0.1:8000
```

One process, one copy of the weights, one lock around all inference. Six JSON
endpoints (`/api/lacuna`, `/api/surprisal`, `/api/aetas`, `/api/paria`,
`/api/suite`, `/api/ordo`, plus `/api/model`) and a single page with a tab
for each. The page shows the same detail the CLIs do: the token table under a
pair, the decisive token under a gap, the pieces under a hot word.

It is **not a service**: no authentication, no rate limiting, no sandbox, and
it runs a model on anything posted to it. It binds 127.0.0.1 and prints a
warning if you override that. The page has no external assets at all — the
server sends `Content-Security-Policy: default-src 'self'`, so system font
stacks rather than webfonts, and it works with the network unplugged.

```bash
curl -s localhost:8000/api/paria -H 'Content-Type: application/json' \
  -d '{"a":"in urbem venit","b":"in urbis venit"}'
```

## Samples

| file | what it is | in training? |
|---|---|---|
| `clean.txt` | Cicero, *In Catilinam* 1.1 | yes |
| `damaged.txt` | Caesar, *BG* 1.1, with five OCR errors and a line of apparatus injected | the Caesar, yes |
| `medieval.txt` | *Vita S. Aldrici*, c. 823, opening (Vicifons, CC-BY-SA-4.0) | **no** — held out |
| `english.txt` | *BG* 1.1 in English, for `ordo.py --against` | no |

## Notes for anyone extending these

- **Set an MLX cache limit.** MLX's allocator cache is unbounded by default. A
  benchmark loop over this model held 198MB of weights, peaked at 805MB of
  working set, and grew the cache to **11.3GB** — which on a 16GB machine means
  14% CPU, constant paging, and eventually SIGKILL. `_common.py` caps it at
  1GB. `ps` shows nothing wrong while this happens, because unified memory is
  not attributed to the process.
- **`mx.eval` inside sampling loops.** MLX is lazy; without it, every step's
  graph stays live until the final read.
- **Batch candidates, right-padded.** One forward pass per candidate is ~20×
  slower. Right-padding is safe only because the model is causal — left-padding
  the same batch would shift every position index and silently change scores.
  `score_readings` returns per-token log-probabilities for every row, which
  is what the token tables in `paria.py` and `lacuna.py --explain` are built
  from; a new demo that wants to say *where* two readings differ can use them
  without another forward pass.
- **Encode the pieces separately.** Byte-level BPE merges across a join, so
  encoding `left + fill + right` as one string would let a candidate's token
  count depend on its neighbours. The suite in `paria.py` scores each reading
  whole for the same reason: two readings are comparable only if each was
  tokenised on its own.
- The batched and unbatched scorers were checked against each other on the
  Caesar sentence and agree to two decimal places.
- Everything pure — gap parsing, alignment, corruption, shuffling, regrouping
  tokens onto words — is under `tests/test_demos.py` and runs without MLX.

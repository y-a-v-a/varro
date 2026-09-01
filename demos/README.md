# demos

Things you can point LATIN-LII at. Each one is a single file, runs on the CPU
or GPU of the machine that trained the model, and prints a number you can argue
with.

```bash
source .venv/bin/activate

python demos/lacuna.py --text "Gallia est omnis divisa in [...] tres" --blind
python demos/surprisal.py demos/samples/damaged.txt --plain
```

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
terms.

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

Both demos therefore push input through `src/normalize.py` — the same function
the corpus went through, not a copy of it — and say so when it changes
anything. `--raw` turns it off.

## lacuna.py — fill a hole, and rank the conjectures

```
$ python demos/lacuna.py --text "Gallia est omnis divisa in [...] tres, \
    quarum unam incolunt Belgae" --answer partes --blind

  [folded to the corpus spelling: 1 v -> u]

gap 1 of 1:  ...Gallia est omnis diuisa in  [1 word(s)]  tres, quarum unam ...

  14 candidates, judged on the fill AND the text after it
  bits worse  toks  supplement
        0.00     1  'partes'   <-- the reading in the manuscripts
        6.49     1  'duas'
        6.60     2  'prouincias'

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

**Do not assume this transfers to another checkpoint.** The same procedure on
LATIN-XXV-2 gets 53.5% over *three* eras with a median margin of 0.3% — barely
distinguishable from no conditioning at all. Era estimation arrived with model
size and corpus breadth; it is not a free consequence of period tagging. Run
`--calibrate` before trusting it anywhere new.

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

This is the same measurement as `src/score.py --audit`, which is what found the
bad documents in this corpus; the difference is that the audit ranks documents
for a pipeline and this renders one document for a person.

**It reports improbability, not error.** Proper names, hapax legomena and
technical vocabulary light up exactly the way corruption does. A hot word is a
place to look, never a verdict.

## serve.py — all three in a browser

```bash
python demos/serve.py           # http://127.0.0.1:8000
```

One process, one copy of the weights, one lock around all inference. Three JSON
endpoints (`/api/lacuna`, `/api/surprisal`, `/api/aetas`, plus `/api/model`)
and a single page that drives them.

It is **not a service**: no authentication, no rate limiting, no sandbox, and
it runs a model on anything posted to it. It binds 127.0.0.1 and prints a
warning if you override that. The page has no external assets at all — the
server sends `Content-Security-Policy: default-src 'self'`, so system font
stacks rather than webfonts, and it works with the network unplugged.

```bash
curl -s localhost:8000/api/aetas -H 'Content-Type: application/json' \
  -d '{"text":"Quo usque tandem abutere, Catilina, patientia nostra?"}'
```

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
- The batched and unbatched scorers were checked against each other on the
  Caesar sentence and agree to two decimal places.

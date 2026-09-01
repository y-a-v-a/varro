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

`checkpoints/latin-lii.safetensors` is **not** the 21.3-perplexity model this
project reports. It is the abandoned classical-narrowing run — 30.2 on
classical, 32.5 on corpus-b — kept because deleting a measurement is worse than
labelling one. The model everything here quotes was trained under
`--name latin-lii-latinitas` and still carries that name on disk.
`configs/latin-lii.yaml` records why the second phase was dropped.

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

proposed by the model         6/60   10.0%
chosen, with right context   22/60   36.7%
chosen, left context only     4/60    6.7%   <-- what generation alone gets
median rank of the true word  2   top-5 68.3%
```

Read that as three separate claims:

- The model **generates** the exact missing word only 10% of the time. That is
  a ceiling on any pure-generation approach.
- Given a candidate list, scoring across the right context picks the true word
  **36.7%** of the time, against **6.7%** for the left context alone. Reading
  backwards is worth roughly 5.5×, and that gap is the whole argument for this
  demo existing.
- The true word's median rank is 2 and it is in the top five 68% of the time,
  so the useful output is the shortlist, not the winner.

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

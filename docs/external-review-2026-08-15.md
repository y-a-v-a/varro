## Independent assessment

The corpus is promising, but the documentation is currently more confident than the implementation warrants. I would not publish the corpus or use its token arithmetic to size a model yet.

### High severity

1. **Token counts are extrapolated from a biased prefix, not measured over the corpus.**

`fertility()` takes only `records[:400]` ([src/tokenizer.py](src/tokenizer.py)), then the resulting ratio is multiplied by every training word ([src/tokenizer.py](src/tokenizer.py)). Because files are loaded deterministically rather than shuffled, those 400 records contain 342 DigilibLT records and 58 CSEL records—no Perseus. Therefore the claimed 32.4M full-corpus tokens is an estimate based on an unrepresentative subset, not a measurement.

This directly contradicts “Nothing in §2 is an estimate any more” ([feasibility.md](feasibility.md)). Since the model ladder depends on these counts, measure all records or use a stratified, word-weighted sample with uncertainty bounds.

2. **Normalization silently joins genuine word boundaries.**

`HYPHEN_SPACE` joins every `letter-hyphen-whitespace-letter` sequence ([src/normalize.py](src/normalize.py), [src/normalize.py](src/normalize.py)). That is not restricted to line breaks. Raw CSEL contains examples such as `ignis- et`, `nasci- habere`, `omnium- suo`; these become `igniset`, `nascihabere`, and `omniumsuo`.

The assertion “every occurrence ... is a typesetting artifact” is false. Even if many are OCR line-end breaks, deletion of the boundary requires evidence from `<lb>` or source layout. This is silent lexical corruption and likely explains some of the changing word totals.

3. **The published measured totals do not match the current build.**

The current summary reports 17,254,053 words and 4,636,738 classical words ([corpus.summary.json](data/processed/corpus.summary.json)). README and feasibility report 17,271,153 and 4,638,175 ([README.md](README.md), [feasibility.md](feasibility.md)).

Current interim arithmetic is:

- 18,140,595 words before deduplication
- 886,542 duplicate words
- 17,254,053 surviving words

Thus several “measured” figures are stale, including the extraction total of 18,158,772. The differences are small proportionally, but they demonstrate that documentation is not generated from the build artifact.

### Medium severity

4. **“Per-work dating” is overstated: most dates are author-level floruits.**

The resolver checks a work override, then assigns every remaining work by that author the same single floruit ([src/enrich.py](src/enrich.py)). That is author-level dating attached to records, not per-work dating. It loses chronology within long careers and cannot handle composition/redaction dates except where a special override was manually added.

This matters at the AD 200 boundary and for pseudonymous, composite, and translated works. Record `date_basis = author_floruit | work_composition | source_prior` explicitly and avoid calling all three “per-work dating.”

5. **Deduplication is not validated strongly enough for the claim “0 false positives.”**

Containment algebra uses raw shingle counts ([src/dedupe.py](src/dedupe.py)) with a Jaccard estimate over sets of unique hashed shingles. Repetition therefore makes the quantities mathematically inconsistent. Union-find also allows transitive clusters whose loser has no direct qualifying relationship with the selected keeper ([src/dedupe.py](src/dedupe.py)).

The verification is 40 random literal spans with a very low 15% pass floor ([src/dedupe.py](src/dedupe.py)). Passing that establishes related content, not that dropping the entire record is correct. “0 below the heuristic floor” is supportable; “0 false positives” is not.

6. **Track enforcement is real but not a publication-grade licence gate.**

`build.py` does reject sources whose current manifest tier is outside the selected track ([src/build.py](src/build.py)). However:

- It trusts the current manifest classification rather than checking the licence embedded at extraction against the lock/manifest.
- `provenance.py` likewise summarizes licence and tier from the manifest, not the records ([src/provenance.py](src/provenance.py)).
- It infers the entire build’s track from the first record and never rejects mixed record-level track values ([src/provenance.py](src/provenance.py)).
- Missing normalization is counted but still included ([src/build.py](src/build.py)).

So the gate prevents ordinary accidental amber inclusion, but it does not prove that the bytes being published correspond to the licence asserted today. Pin and audit licence metadata at fetch/extraction time.

### Methodological objections

7. **The normalization policy destroys philologically meaningful information.**

The statement that accents, vowel lengths, underdots, overlines, and brackets are merely “an editor talking” is too dismissive. They encode metre, vowel quantity, uncertain readings, numerals, supplements, and textual status. Folding `v/u` and `j/i` is defensible for a model-facing view, but it removes useful phonological and morphological distinctions. Keep a diplomatic text layer alongside a normalized training view; normalization should be a reversible derived representation.

“48-letter alphabet ... precisely classical Latin’s inventory” ([feasibility.md](feasibility.md)) is also misleading: it is a modern upper/lowercase computational inventory, not the classical alphabet.

8. **Whole-record validation is better than random lines, but insufficient.**

The split operates on records/IDs, not demonstrated scholarly works ([src/build.py](src/build.py)). Related works, editions, collections, quotations, and the same author can remain in training. This measures interpolation to unseen documents by familiar authors more than generalization.

Use at least two evaluations:

- work-held-out for ordinary loss monitoring;
- author- or textual-tradition-held-out for genuine generalization.

The 4k BPE choice is reasonable on parameter cost, but the morphology evidence—12 hand-selected `amo` forms and a few stems—is too small to justify “forces morphology.” Treat that as an illustration, not a result.

### Most important missing work

Before training, I would require:

- versioned, generated corpus statistics rather than copied prose figures;
- tests for normalization using adversarial and source-derived examples;
- manual review of every duplicate cluster, not just random-span verification;
- external language-identification/contamination sampling, especially Latin-letter OCR gibberish;
- evaluation split grouping by author, work family, translation, and edition;
- register and genre metadata—period tags alone do not distinguish poetry, law, inscriptions, Christian prose, grammar, and documentary Latin;
- a formal release audit binding fetched checksum → extracted records → manifest licence → final corpus.

The central decision I disagree with most plainly is calling destructive normalization a safe canonical corpus representation. It is acceptable as one training projection, but not as the corpus’s only preserved textual form.
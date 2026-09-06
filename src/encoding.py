#!/usr/bin/env python3
"""Batched tokenisation, shared by every stage that encodes a whole corpus.

`tok.encode(text)` uses one core. `tok.encode_batch(texts)` runs the BPE across
all of them, and on this project's corpora that is 2.4-3.2x — measured on
corpus.train, 183,265,520 tokens: 178.2s and 4.70GB peak became 55.2s and
2.68GB. It is the only part of a training run where a second core has anything
to do at all; the step itself is Metal and spends 0.15 of one core on an
8-core M2.

WHY THE BATCH IS BOUNDED BY CHARACTERS
    Handing encode_batch the whole corpus is not an option: it holds an
    Encoding per record while a batch is in flight, and Track B's training
    split is 906MB in 11,159 records. Bounding by record count does not help
    either — the corpora differ by two orders of magnitude in record size, and
    the classical split is 239 records of the same 33MB the mixed split
    spreads over ten thousand. Characters are what the memory actually tracks.

    8MB is about 2.5M tokens per batch: enough to keep every core fed, small
    enough that no corpus here has more than a fraction of itself resident.

This module deliberately imports nothing beyond the standard library. It is
used by the training half and by evaluate.py, which has to keep running on a
machine with no MLX.
"""

import json
from typing import Any, Dict, Iterable, Iterator, List

CHUNK_CHARS = 8_000_000


def chunk_texts(texts: Iterable[str],
                budget: int = CHUNK_CHARS) -> Iterator[List[str]]:
    """Group texts into batches of at most `budget` characters.

    A single text longer than the budget is still emitted, alone. The budget
    bounds memory; it must never silently drop a long work.
    """
    batch: List[str] = []
    size = 0
    for text in texts:
        batch.append(text)
        size += len(text)
        if size >= budget:
            yield batch
            batch, size = [], 0
    if batch:
        yield batch


def chunk_records(lines: Iterable[str],
                  budget: int = CHUNK_CHARS) -> Iterator[List[Dict[str, Any]]]:
    """The same, over JSONL lines, keeping each record rather than just its text.

    Callers that need the record's metadata as well as its text — the period
    tag train.py injects, the word counts tokenizer.py measures — cannot work
    from the text alone.
    """
    batch: List[Dict[str, Any]] = []
    size = 0
    for line in lines:
        record = json.loads(line)
        batch.append(record)
        size += len(record["text"])
        if size >= budget:
            yield batch
            batch, size = [], 0
    if batch:
        yield batch


def encode_all(tok, texts: Iterable[str],
               budget: int = CHUNK_CHARS) -> Iterator[List[int]]:
    """Token ids for each text, in order, encoded across every core.

    Yields rather than returns: a corpus that does not fit in memory as
    Encoding objects generally does not fit as lists of ids either, and every
    caller here is folding the result into a counter or a total.
    """
    for batch in chunk_texts(texts, budget):
        for encoded in tok.encode_batch(batch):
            yield encoded.ids

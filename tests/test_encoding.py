"""encoding.py — batching must not change what gets encoded.

Every stage that reads a whole corpus goes through here, so a batch boundary
that loses or reorders a record would move a bigram floor, a fertility figure
and a training corpus at once, and none of them would report it.
"""

import json

import encoding


class Encoded:
    def __init__(self, ids):
        self.ids = ids


class FakeTokenizer:
    def __init__(self):
        self.batch_sizes = []

    def encode(self, text):
        return Encoded([10 + len(word) for word in text.split()])

    def encode_batch(self, texts):
        self.batch_sizes.append(len(texts))
        return [self.encode(text) for text in texts]


TEXTS = ["arma uirumque cano", "gallia est omnis diuisa in partes tres",
         "confessionum liber primus", "sic transit gloria mundi", "e"]


def test_encode_all_matches_encoding_one_at_a_time():
    tok = FakeTokenizer()
    assert list(encoding.encode_all(tok, TEXTS)) == \
        [tok.encode(t).ids for t in TEXTS]


def test_the_result_is_the_same_however_the_batches_fall():
    tok = FakeTokenizer()
    reference = list(encoding.encode_all(tok, TEXTS, budget=10**9))
    for budget in (1, 5, 20, 100, 10**6):
        assert list(encoding.encode_all(FakeTokenizer(), TEXTS, budget)) == reference


def test_batching_actually_happens_at_a_small_budget():
    """A test that only ever saw one batch would prove nothing about boundaries."""
    tok = FakeTokenizer()
    list(encoding.encode_all(tok, TEXTS, budget=20))
    assert len(tok.batch_sizes) > 1
    assert sum(tok.batch_sizes) == len(TEXTS)


def test_order_is_preserved():
    tok = FakeTokenizer()
    texts = [f"record number {i}" + " x" * i for i in range(30)]
    assert list(encoding.encode_all(tok, texts, budget=40)) == \
        [tok.encode(t).ids for t in texts]


def test_chunk_texts_groups_by_characters():
    assert [len(c) for c in encoding.chunk_texts(["x" * 40] * 10, budget=100)] \
        == [3, 3, 3, 1]


def test_a_text_longer_than_the_budget_is_still_emitted():
    """The budget bounds memory. It must never drop a long work."""
    assert [len(c) for c in encoding.chunk_texts(["x" * 5000], budget=100)] == [1]


def test_chunking_an_empty_corpus_yields_nothing():
    assert list(encoding.chunk_texts([])) == []
    assert list(encoding.encode_all(FakeTokenizer(), [])) == []


def test_chunk_records_keeps_the_whole_record_not_just_its_text():
    lines = [json.dumps({"text": "arma uirumque", "period": "CLASSICAL"}),
             json.dumps({"text": "sic transit", "period": "MEDIEVAL"})]
    records = [r for chunk in encoding.chunk_records(lines) for r in chunk]
    assert [r["period"] for r in records] == ["CLASSICAL", "MEDIEVAL"]


def test_chunk_records_never_splits_a_record():
    lines = [json.dumps({"text": f"record {i}"}) for i in range(7)]
    seen = [r["text"] for chunk in encoding.chunk_records(lines, budget=20)
            for r in chunk]
    assert seen == [f"record {i}" for i in range(7)]


def test_encoding_needs_nothing_beyond_the_standard_library():
    """evaluate.py runs on machines with no MLX, and imports this."""
    import ast
    import pathlib
    import sys
    source = pathlib.Path(encoding.__file__).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in sys.stdlib_module_names
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] in sys.stdlib_module_names

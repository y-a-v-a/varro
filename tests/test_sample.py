"""sample.py — refusing to load a model against the wrong vocabulary.

A model's embeddings are keyed to specific token ids. With a different
vocabulary, token 1234 means a different string, and the model emits fluent
nonsense rather than failing — the worst failure mode available here.
"""

import hashlib
import json

import pytest

pytest.importorskip("mlx.core", reason="sample.py requires MLX")

import sample  # noqa: E402


@pytest.fixture
def tokenizers(tmp_path, monkeypatch):
    d = tmp_path / "tokenizers"
    d.mkdir()
    monkeypatch.setattr(sample, "TOKENIZERS", d)
    monkeypatch.setattr(sample, "ROOT", tmp_path)
    return d


def write(path, body):
    path.write_text(body, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def test_current_tokenizer_used_when_it_is_the_right_one(tokenizers):
    digest = write(tokenizers / "latin-bpe-4000.json", '{"v": 1}')
    run = {"run": "m", "tokenizer": {"sha256": digest}}
    assert sample.resolve_tokenizer(4000, run).endswith("latin-bpe-4000.json")


def test_archived_copy_is_preferred_when_the_current_one_has_moved(tokenizers):
    """A model outlives a retrained tokenizer as long as its copy is kept."""
    write(tokenizers / "latin-bpe-4000.json", '{"v": 2}')          # retrained
    old = write(tokenizers / "latin-bpe-4000.aaaabbbbccccdddd.json", '{"v": 1}')
    run = {"run": "m", "tokenizer": {"sha256": "aaaabbbbccccdddd"}}
    resolved = sample.resolve_tokenizer(4000, run)
    assert resolved.endswith("latin-bpe-4000.aaaabbbbccccdddd.json")


def test_mismatch_with_no_archive_refuses_rather_than_mis_tokenising(tokenizers):
    """What actually happened on 2026-08-20, and what should have stopped it."""
    write(tokenizers / "latin-bpe-4000.json", '{"v": 2}')
    run = {"run": "latin-xxv", "tokenizer": {"sha256": "24d1515e65807859"}}
    with pytest.raises(SystemExit) as excinfo:
        sample.resolve_tokenizer(4000, run)
    message = str(excinfo.value)
    assert "24d1515e65807859" in message
    assert "nonsense" in message


def test_unfingerprinted_run_is_used_but_says_so(tokenizers, capsys):
    write(tokenizers / "latin-bpe-4000.json", '{"v": 1}')
    resolved = sample.resolve_tokenizer(4000, {"run": "old"})
    assert resolved.endswith("latin-bpe-4000.json")
    assert "unverified" in capsys.readouterr().err


def test_no_tokenizer_at_all_is_an_error(tokenizers):
    with pytest.raises(SystemExit):
        sample.resolve_tokenizer(4000, {"run": "m", "tokenizer": {"sha256": "x"}})

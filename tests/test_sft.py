"""sft.py — response masking, and the reporting arithmetic train.py already fixed.

Skipped where MLX is absent, like test_train.py: sft.py imports it at module
level and the rest of the suite deliberately runs without it.
"""

import pytest

pytest.importorskip("mlx.core", reason="sft.py requires MLX")

import mlx.core as mx  # noqa: E402

import sft  # noqa: E402
import train  # noqa: E402


class Encoded:
    def __init__(self, ids):
        self.ids = ids


class FakeTokenizer:
    """Word-length ids again, with the two specials sft depends on."""

    SPECIALS = {"<|endoftext|>": 1, "<|pad|>": 0}

    def token_to_id(self, token):
        return self.SPECIALS.get(token)

    def encode(self, text):
        return Encoded([10 + len(word) for word in text.split()])


def pair(prompt, response):
    return {"prompt": prompt, "response": response}


# --- loss masking -----------------------------------------------------------
#
# Training on prompt tokens too would teach the model to generate questions as
# readily as answers. At inference the prompt is given; only the continuation
# is the model's job.

def test_only_the_response_is_supervised():
    tok = FakeTokenizer()
    rows, masks = sft.build_examples([pair("quid est", "ueritas est magna")], tok)
    prompt_len = len(tok.encode("quid est" + "\n").ids)
    m = masks[0].tolist()
    assert m[:prompt_len] == [0.0] * prompt_len
    assert sum(m) == len(tok.encode("ueritas est magna").ids) + 1   # +1 for <|endoftext|>


def test_the_stop_token_is_inside_the_loss():
    """A chat model that never emits <|endoftext|> never ends its turn."""
    tok = FakeTokenizer()
    rows, masks = sft.build_examples([pair("a b", "c d e")], tok)
    ids, m = rows[0].tolist(), masks[0].tolist()
    last = max(i for i, v in enumerate(m) if v)
    assert ids[last] == tok.token_to_id("<|endoftext|>")


def test_padding_is_never_supervised():
    tok = FakeTokenizer()
    rows, masks = sft.build_examples([pair("a", "b c d")], tok, max_len=64)
    ids, m = rows[0].tolist(), masks[0].tolist()
    assert len(ids) == len(m) == 64
    for token, weight in zip(ids, m):
        if token == tok.token_to_id("<|pad|>"):
            assert weight == 0.0


def test_examples_too_long_for_the_window_are_dropped():
    tok = FakeTokenizer()
    long_pair = pair(" ".join(["word"] * 200), "answer here now")
    rows, _ = sft.build_examples([long_pair, pair("a", "b c d")], tok, max_len=64)
    assert rows.shape[0] == 1


def test_responses_too_short_to_teach_anything_are_dropped():
    tok = FakeTokenizer()
    rows, _ = sft.build_examples([pair("a", "b"), pair("a", "b c d")], tok)
    assert rows.shape[0] == 1


def test_every_row_is_the_same_length_so_they_can_be_batched():
    tok = FakeTokenizer()
    rows, masks = sft.build_examples(
        [pair("a", "b c d"), pair("a b c d e", "f g h i j k")], tok, max_len=48)
    assert rows.shape == masks.shape == (2, 48)


def test_no_usable_pairs_is_an_empty_batch_not_a_crash():
    rows, masks = sft.build_examples([pair("a", "b")], FakeTokenizer())
    assert rows.shape[0] == 0 and masks.shape[0] == 0


# --- reporting blocks -------------------------------------------------------
#
# sft averaged its loss over `args.eval_every` steps regardless of how many it
# had actually accumulated, and never reported the final block at all. The
# defaults divide evenly, which is why it never showed — the same arithmetic
# reported LATIN-XXV's real 3.12 train loss as 1.25.

def block_sizes(steps, eval_every):
    sizes, since = [], 0
    for step in range(steps):
        since += 1
        if train.is_report_step(step, steps, eval_every):
            sizes.append(since)
            since = 0
    return sizes


def test_the_blocks_sft_averages_over_tile_the_run():
    for steps, every in ((3000, 250), (3100, 250), (7, 3), (1, 250)):
        assert sum(block_sizes(steps, every)) == steps


def test_a_run_whose_length_is_not_a_multiple_still_reports_its_tail():
    """`--steps 3100` used to lose the last 100 steps from the history entirely."""
    sizes = block_sizes(3100, 250)
    assert sizes[-1] == 100
    assert sum(sizes) == 3100


def test_sft_shares_train_checkpoint_format_rather_than_inventing_one():
    """`--resume` has to mean the same thing in both, or one of them is a trap."""
    assert sft.save_checkpoint is train.save_checkpoint
    assert sft.load_checkpoint is train.load_checkpoint
    assert sft.is_report_step is train.is_report_step

"""Tests for the fill-in-the-middle transformation.

The pure part only: what a rewritten piece looks like, that nothing is lost or
invented, that cuts fall on word boundaries, that a window's documents are
never mixed. The training loop is train.py's and is not re-tested here.
"""

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import fim  # noqa: E402

SEAM, EOS = 0, 1
# Ids 10-19 begin a word; 20-29 continue one.
STARTS = [False] * 10 + [True] * 10 + [False] * 10


def _piece(n: int, seed: int = 0):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        out.append(rng.randrange(10, 20))
        out += [rng.randrange(20, 30) for _ in range(rng.randrange(0, 3))]
    return out[:n]


def _parts(done):
    """(suffix, prefix+middle) of a rewritten piece."""
    assert done[0] == SEAM and done[-1] == EOS
    second = done.index(SEAM, 1)
    return done[1:second], done[second + 1:-1]


def test_a_rewritten_piece_has_two_seams_and_closes():
    done = fim.transform(_piece(80), STARTS, SEAM, EOS, random.Random(1))
    assert done.count(SEAM) == 2 and done.count(EOS) == 1
    assert done[0] == SEAM and done[-1] == EOS


def test_nothing_is_invented_and_only_a_suffix_tail_is_ever_lost():
    for seed in range(200):
        piece = _piece(90, seed)
        suffix, head = _parts(fim.transform(piece, STARTS, SEAM, EOS, random.Random(seed)))
        # prefix + middle is the piece's own beginning, untouched ...
        assert piece[:len(head)] == head
        # ... and the suffix is what followed it, possibly cut short.
        rest = piece[len(head):]
        assert rest[:len(suffix)] == suffix


def test_the_middle_ends_and_the_suffix_begins_on_a_word():
    for seed in range(200):
        piece = _piece(90, seed)
        suffix, head = _parts(fim.transform(piece, STARTS, SEAM, EOS, random.Random(seed)))
        if suffix:
            assert STARTS[suffix[0]]
        assert len(head) < len(piece) or not suffix


def test_a_piece_too_short_to_cut_is_left_alone():
    piece = _piece(10)
    assert fim.transform(piece, STARTS, SEAM, EOS, random.Random(1)) == piece


def test_documents_in_a_window_are_never_mixed():
    a, b = _piece(60, 1), _piece(60, 2)
    out = fim.transform_window(a + [EOS] + b, STARTS, SEAM, EOS, 1.0, random.Random(3))
    first = out[:out.index(EOS) + 1]
    second = out[out.index(EOS) + 1:]
    assert set(first) - {SEAM, EOS} <= set(a)
    assert sorted(t for t in second if t not in (SEAM, EOS)) <= sorted(b) or \
        set(second) - {SEAM, EOS} <= set(b)
    # One closing end-of-text each, not two where the document also ended.
    assert out.count(EOS) == 2


def test_rate_zero_is_the_identity():
    window = _piece(60, 1) + [EOS] + _piece(60, 2)
    assert fim.transform_window(window, STARTS, SEAM, EOS, 0.0, random.Random(1)) == window


def test_training_and_scoring_share_one_format():
    assert fim.frame([11, 21], [12], [13, 23], SEAM, EOS) == [SEAM, 13, 23, SEAM, 11, 21, 12, EOS]

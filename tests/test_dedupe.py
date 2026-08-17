"""dedupe.py — folding, sketches, similarity estimators, keeper choice."""

from dedupe import (SKETCH_K, choose_keeper, containment, fold, jaccard,
                    priority, sketch)

WORDS = ("lorem ipsum dolor sit amet consectetur adipiscing elit sed do "
         "eiusmod tempor incididunt ut labore et dolore magna aliqua").split()


def _rec(source="perseus-canonical-latinLit", n_words=100, **kw):
    return {"source": source, "n_words": n_words, **kw}


def test_fold_is_orthography_insensitive():
    assert fold("Vir Jam, viri; 123 uiuat!") == ["uir", "iam", "uiri", "uiuat"]


def test_sketch_deterministic_and_distinct_cardinality():
    tokens = WORDS * 3  # repetition must not inflate cardinality
    s1, c1 = sketch(tokens)
    s2, c2 = sketch(tokens)
    assert s1 == s2 and c1 == c2
    # The tripled sequence is periodic, so only len(WORDS) distinct 5-grams exist.
    assert c1 == len(WORDS)


def test_sketch_short_input_empty():
    assert sketch(["a", "b"]) == (set(), 0)


def test_sketch_respects_k():
    tokens = [f"w{i}" for i in range(200)]
    s, c = sketch(tokens, k=16)
    assert len(s) == 16
    assert c == 196  # all shingles distinct


def test_jaccard_identical_is_one():
    s, _ = sketch(WORDS)
    assert jaccard(tuple(sorted(s)), tuple(sorted(s)), s, s) == 1.0


def test_jaccard_disjoint_is_zero():
    s1, _ = sketch([f"a{i}" for i in range(50)])
    s2, _ = sketch([f"b{i}" for i in range(50)])
    assert jaccard(tuple(sorted(s1)), tuple(sorted(s2)), s1, s2) == 0.0


def test_jaccard_empty_sketch_is_zero():
    assert jaccard((), (1,), set(), {1}) == 0.0


def test_containment_bounds():
    assert containment(0.0, 50, 100) == 0.0
    assert containment(1.0, 100, 100) == 1.0
    # A strict subset: |A|=50 inside |B|=100 gives J=0.5 and full containment.
    assert abs(containment(0.5, 50, 100) - 1.0) < 1e-9
    assert containment(0.5, 0, 100) == 0.0


def test_priority_orders_known_sources_before_unknown():
    assert priority(_rec("perseus-canonical-latinLit")) < priority(_rec("digiliblt"))
    assert priority(_rec("digiliblt")) < priority(_rec("opengreekandlatin-csel"))
    assert priority(_rec("something-else")) == 3


def test_choose_keeper_equivalence_uses_source_priority():
    records = [_rec("opengreekandlatin-csel", 1000),
               _rec("perseus-canonical-latinLit", 990)]
    # Near-identical length: the higher-fidelity source wins.
    assert choose_keeper([0, 1], records)[0] == 1


def test_choose_keeper_containment_larger_wins_despite_priority():
    records = [_rec("perseus-canonical-latinLit", 43_000),
               _rec("opengreekandlatin-csel", 290_000)]
    # The Perseus record is a fraction of the CSEL one: the fuller text wins.
    assert choose_keeper([0, 1], records)[0] == 1

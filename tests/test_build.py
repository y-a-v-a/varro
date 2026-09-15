"""build.py — held-out split selection: deterministic, grouped, budgeted."""

import pathlib

from build import choose_validation


def corpus(n_authors=100, works_per_author=1, words=3000, period="CLASSICAL"):
    records = []
    for a in range(n_authors):
        for w in range(works_per_author):
            records.append({
                "id": f"auth{a}/work{w}",
                "author": f"Auctor {a}",
                "period": period,
                "n_words": words,
            })
    return records


def test_split_is_deterministic():
    records = corpus()
    assert choose_validation(records, 0.05) == choose_validation(records, 0.05)


def test_authors_are_never_split_across_train_and_validation():
    records = corpus(n_authors=60, works_per_author=3, words=1000)
    chosen = choose_validation(records, 0.05)
    by_author = {}
    for rec in records:
        by_author.setdefault(rec["author"], []).append(rec["id"])
    for ids in by_author.values():
        held = {i for i in ids if i in chosen}
        assert held in (set(), set(ids)), "author partially held out"


def test_budget_respected():
    records = corpus()
    total = sum(r["n_words"] for r in records)
    chosen = choose_validation(records, 0.05)
    held = sum(r["n_words"] for r in records if r["id"] in chosen)
    assert 0 < held <= total * 0.05


def test_grouping_by_work_holds_out_individual_works():
    records = corpus(n_authors=10, works_per_author=5, words=2500)
    chosen = choose_validation(records, 0.10, group_by="work")
    authors_partial = 0
    by_author = {}
    for rec in records:
        by_author.setdefault(rec["author"], []).append(rec["id"])
    for ids in by_author.values():
        held = {i for i in ids if i in chosen}
        if held and held != set(ids):
            authors_partial += 1
    assert chosen and authors_partial > 0


def test_oversized_groups_are_ineligible():
    records = corpus(n_authors=40, words=3000)
    # One colossal author: must never be chosen, it would blow the budget cap.
    records += [{"id": "livy/0", "author": "Livius", "period": "CLASSICAL",
                 "n_words": 515_000}]
    chosen = choose_validation(records, 0.05)
    assert "livy/0" not in chosen


# --- text provenance --------------------------------------------------------
#
# Not a licence question and not a quality one: whether a later model would be
# learning from an earlier model's output. The corpus already contains
# OCR-derived text (CSEL), so the classes describe what is there, not only what
# might arrive.

from build import PROVENANCE_CLASSES, apportion, drop_order  # noqa: E402


def rec(rid, words, provenance="transcribed"):
    return {"id": rid, "n_words": words, "text_provenance": provenance}


def test_no_caps_is_a_no_op():
    records = [rec("a", 100), rec("b", 50, "ocr")]
    kept, dropped = apportion(records, {}, seed=1)
    assert kept == records and dropped == {}


def test_a_class_under_its_cap_is_untouched():
    records = [rec("a", 900), rec("b", 100, "model-corrected")]
    kept, dropped = apportion(records, {"model-corrected": 0.15}, seed=1)
    assert len(kept) == 2 and not dropped


def test_a_class_over_its_cap_is_reduced_to_it():
    """The guard that keeps a feedback loop from growing quietly."""
    records = [rec("keep", 500)] + [rec(f"m{i}", 100, "model-corrected")
                                    for i in range(5)]
    kept, dropped = apportion(records, {"model-corrected": 0.20}, seed=1)
    held = sum(r["n_words"] for r in kept if r["text_provenance"] == "model-corrected")
    total = sum(r["n_words"] for r in kept)
    assert held / total <= 0.20 + 1e-9
    assert dropped["model-corrected"] > 0


def test_other_classes_are_never_dropped_to_make_room():
    records = [rec("t", 100)] + [rec(f"m{i}", 100, "model-corrected")
                                 for i in range(9)]
    kept, _ = apportion(records, {"model-corrected": 0.10}, seed=1)
    assert any(r["id"] == "t" for r in kept)
    assert all(r["text_provenance"] == "model-corrected"
               for r in records if r["id"] not in {k["id"] for k in kept})


def test_apportioning_is_deterministic():
    records = [rec(f"m{i}", 10 + i, "ocr") for i in range(20)] + [rec("t", 100)]
    first, _ = apportion(records, {"ocr": 0.3}, seed=7)
    second, _ = apportion(records, {"ocr": 0.3}, seed=7)
    assert [r["id"] for r in first] == [r["id"] for r in second]


def test_a_zero_cap_removes_the_class_entirely():
    records = [rec("t", 100), rec("m", 50, "model-corrected")]
    kept, dropped = apportion(records, {"model-corrected": 0.0}, seed=1)
    assert [r["id"] for r in kept] == ["t"]
    assert dropped["model-corrected"] == 50


def test_caps_on_several_classes_apply_together():
    records = ([rec("t", 400)]
               + [rec(f"o{i}", 100, "ocr") for i in range(4)]
               + [rec(f"m{i}", 100, "model-corrected") for i in range(4)])
    kept, dropped = apportion(records, {"ocr": 0.2, "model-corrected": 0.1}, seed=3)
    total = sum(r["n_words"] for r in kept)
    by = {}
    for r in kept:
        by[r["text_provenance"]] = by.get(r["text_provenance"], 0) + r["n_words"]
    assert by.get("ocr", 0) / total <= 0.2 + 1e-9
    assert by.get("model-corrected", 0) / total <= 0.1 + 1e-9


def test_unknown_is_a_class_so_it_can_be_excluded_too():
    """sources.md says the Latin Library's provenance is inconsistent."""
    assert "unknown" in PROVENANCE_CLASSES
    records = [rec("t", 100), rec("u", 60, "unknown")]
    kept, _ = apportion(records, {"unknown": 0.0}, seed=1)
    assert [r["id"] for r in kept] == ["t"]


def test_a_record_with_no_provenance_field_counts_as_unknown():
    """Records written before the field existed must not read as None here
    while reading as "unknown" in the filter and the report — a cap on
    `unknown` matched nothing at all until these agreed."""
    records = [{"id": "old", "n_words": 100},          # no field at all
               {"id": "t", "n_words": 100, "text_provenance": "transcribed"}]
    kept, dropped = apportion(records, {"unknown": 0.0}, seed=1)
    assert [r["id"] for r in kept] == ["t"]
    assert dropped["unknown"] == 100


# --- which records get shed ---------------------------------------------------
#
# The sort is by length, so only ties are actually a choice. A seeded shuffle
# before a stable sort reached nothing else, and the generator was re-seeded
# inside the loop, so it read as randomising the choice while randomising tie
# order at most. The tiebreak is now a hash of the record id under the seed.


def tied(n, words=100, provenance="ocr"):
    return [{"id": f"r{i}", "n_words": words, "text_provenance": provenance}
            for i in range(n)]


def test_shorter_records_are_shed_before_longer_ones():
    """400 ocr words against 1000 clean, capped at 25%: budget is 333, so the
    100-word record goes and the 300-word one stays."""
    records = [{"id": "big", "n_words": 300, "text_provenance": "ocr"},
               {"id": "small", "n_words": 100, "text_provenance": "ocr"},
               {"id": "clean", "n_words": 1000, "text_provenance": "transcribed"}]
    kept, dropped = apportion(records, {"ocr": 0.25}, seed=1)
    assert {r["id"] for r in kept} == {"big", "clean"}
    assert dropped["ocr"] == 100


def test_tie_breaking_does_not_depend_on_the_order_records_were_read():
    """load() walks data/interim in filename order; the cap must not inherit it."""
    records = tied(10) + [{"id": "clean", "n_words": 1000,
                           "text_provenance": "transcribed"}]
    forward, _ = apportion(records, {"ocr": 0.3}, seed=5)
    backward, _ = apportion(list(reversed(records)), {"ocr": 0.3}, seed=5)
    assert {r["id"] for r in forward} == {r["id"] for r in backward}


def test_the_same_seed_sheds_the_same_records():
    records = tied(10) + [{"id": "clean", "n_words": 1000,
                           "text_provenance": "transcribed"}]
    first, _ = apportion(records, {"ocr": 0.3}, seed=5)
    second, _ = apportion(records, {"ocr": 0.3}, seed=5)
    assert [r["id"] for r in first] == [r["id"] for r in second]


def test_a_different_seed_can_shed_different_records():
    """Otherwise the seed is decoration and the tiebreak is really file order."""
    records = tied(20) + [{"id": "clean", "n_words": 1000,
                           "text_provenance": "transcribed"}]
    by_seed = {seed: frozenset(r["id"] for r in apportion(
        records, {"ocr": 0.3}, seed=seed)[0]) for seed in range(6)}
    assert len(set(by_seed.values())) > 1


def test_drop_order_is_length_first_then_the_seeded_hash():
    short = {"id": "z", "n_words": 10}
    long = {"id": "a", "n_words": 11}
    assert drop_order(short, 1) < drop_order(long, 1)


def test_drop_order_is_stable_across_processes():
    """PYTHONHASHSEED must not reach it, or two runs of one build disagree.

    This is why the tiebreak is blake2b and not hash(): Python randomises
    string hashing per process, and a seed that only holds within one run is
    not a seed.
    """
    import subprocess
    import sys as _sys

    import build
    src = str(pathlib.Path(build.__file__).resolve().parent)
    code = (f"import sys; sys.path.insert(0, {src!r}); from build import drop_order; "
            "print(drop_order({'id': 'x', 'n_words': 1}, 7)[1])")
    seen = set()
    for n in (0, 1, 2):
        out = subprocess.run([_sys.executable, "-c", code], capture_output=True,
                             text=True, env={"PYTHONHASHSEED": str(n)})
        assert out.returncode == 0, out.stderr
        seen.add(out.stdout.strip())
    assert len(seen) == 1

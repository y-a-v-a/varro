"""build.py — held-out split selection: deterministic, grouped, budgeted."""

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

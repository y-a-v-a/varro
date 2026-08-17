"""provenance.py — licence propagation and source summaries."""

from provenance import corpus_licence, summarise


def rows(*licences):
    return [{"license": lic} for lic in licences]


def test_sharealike_propagates_over_everything():
    assert corpus_licence(rows("public-domain", "CC-BY-4.0", "CC-BY-SA-4.0")) \
        == "CC-BY-SA-4.0"


def test_attribution_only_when_no_sharealike():
    assert corpus_licence(rows("public-domain", "CC-BY-4.0")) == "CC-BY-4.0"


def test_pure_public_domain():
    assert corpus_licence(rows("public-domain", "CC0-1.0")) == "public-domain"


def test_unresolved_licence_is_flagged_not_guessed():
    assert corpus_licence(rows("public-domain", "SOMETHING-ODD")).startswith("MIXED")


def test_summarise_counts_words_and_shares():
    records = [
        {"source": "a", "n_words": 750},
        {"source": "a", "n_words": 250},
        {"source": "b", "n_words": 1000},
    ]
    manifest = {"a": {"name": "Alpha", "license": "CC-BY-4.0", "tier": "green"}}
    result = {r["id"]: r for r in summarise(records, manifest)}
    assert result["a"]["words"] == 1000 and result["a"]["documents"] == 2
    assert result["a"]["share"] == 0.5
    assert result["a"]["in_manifest"] is True
    # A source missing from the manifest is surfaced, not dropped.
    assert result["b"]["in_manifest"] is False
    assert result["b"]["license"] == "UNKNOWN"

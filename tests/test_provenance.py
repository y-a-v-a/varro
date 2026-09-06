"""provenance.py — licence propagation and source summaries."""

from provenance import corpus_licence, render, summarise


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


# --- what the notice actually tells a reader ---------------------------------
#
# The obligations paragraphs were keyed off `effective`, which is a rendered
# label rather than a licence identifier. "CC-BY-SA-4.0 + UBPL-1.0" is not in
# SHAREALIKE, so the corpus with two copylefts in it — the case where a reader
# most needs telling — was the one that got no explanation at all.


def notice(*licences, track="A"):
    detailed = [{"license": lic, "id": f"s{i}", "name": f"Source {i}",
                 "url": None, "tier": "green", "words": 100, "documents": 1,
                 "share": 1 / len(licences), "in_manifest": True}
                for i, lic in enumerate(licences)]
    return render(detailed, {}, track, corpus_licence(detailed))


def test_sharealike_propagation_is_explained():
    assert "ShareAlike propagates" in notice("CC-BY-SA-4.0", "public-domain")


def test_file_copyleft_alongside_sharealike_explains_both():
    """The live case: Tesserae is UBPL-1.0 and sits in the published corpus.

    corpus_licence() reports the effective licence as CC-BY-SA-4.0, because
    ShareAlike is what constrains the corpus as a whole. The per-file copyleft
    is a real obligation on top of that — the licence text has to travel with
    the distribution — and keying the paragraphs off that summary label meant
    it was never mentioned at all.
    """
    text = notice("CC-BY-SA-4.0", "UB-Public-License-1.0")
    assert "ShareAlike propagates" in text
    assert "file-level copyleft" in text


def test_file_copyleft_alone_is_still_explained():
    text = notice("UB-Public-License-1.0", "public-domain")
    assert "file-level copyleft" in text


def test_a_corpus_with_no_copyleft_claims_neither():
    text = notice("CC-BY-4.0", "public-domain")
    assert "ShareAlike propagates" not in text
    assert "file-level copyleft" not in text


def test_noncommercial_does_not_swallow_the_file_copyleft_notice():
    """Track B carries both; NonCommercial restricts use, UBPL constrains shipping."""
    text = notice("CC-BY-NC-SA-4.0", "UB-Public-License-1.0")
    assert "NonCommercial" in text
    assert "file-level copyleft" in text


def test_the_document_count_is_the_sum_of_the_sources():
    """It was `len(set()) or sum(...)` — dead code masking the real expression."""
    detailed = [{"license": "public-domain", "id": "a", "name": "A", "url": None,
                 "tier": "green", "words": 10, "documents": 3, "share": 0.5,
                 "in_manifest": True},
                {"license": "public-domain", "id": "b", "name": "B", "url": None,
                 "tier": "green", "words": 10, "documents": 4, "share": 0.5,
                 "in_manifest": True}]
    assert "**Documents**: 7" in render(detailed, {}, "A", "public-domain")

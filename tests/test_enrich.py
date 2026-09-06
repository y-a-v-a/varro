"""enrich.py — dating and genre resolution against synthetic authority files."""

import pytest

from enrich import Authority, Genres

AUTHORITIES = """\
periods:
  CLASSICAL: {min: -300, max: 200}
  LATE: {min: 201, max: 600}
  MEDIEVAL: {min: 601, max: 1500}

source_priors:
  opengreekandlatin-csel: LATE

authors:
  - aliases: [Cicero, M. Tullius Cicero]
    floruit: -55
    confidence: secure
  - aliases: [Ecclesia]
    period: MEDIEVAL
    confidence: attributed

works:
  - aliases: [Aetna]
    floruit: 50
    confidence: disputed

pl_volume_anchors:
  - {volume: 1, year: 200}
  - {volume: 101, year: 400}
  - {volume: 201, year: 1200}
"""

GENRES = """\
forms: [PROSE, VERSE]
genres: [ORATORY, EPIC, LETTERS]

title_patterns:
  - {match: "^Epistulae", form: PROSE, genre: LETTERS}

authors:
  - aliases: [Vergilius]
    form: VERSE
    genre: EPIC

works:
  - aliases: [In Catilinam]
    form: PROSE
    genre: ORATORY
"""


@pytest.fixture
def authority(tmp_path):
    path = tmp_path / "authorities.yaml"
    path.write_text(AUTHORITIES, encoding="utf-8")
    return Authority(path)


@pytest.fixture
def genres(tmp_path):
    path = tmp_path / "genres.yaml"
    path.write_text(GENRES, encoding="utf-8")
    return Genres(path)


def test_period_for_bounds(authority):
    assert authority.period_for(-55) == "CLASSICAL"
    assert authority.period_for(200) == "CLASSICAL"
    assert authority.period_for(201) == "LATE"
    assert authority.period_for(5000) is None
    assert authority.period_for(None) is None


def test_resolve_by_work_beats_author(authority):
    date, period, origin, confidence = authority.resolve(
        {"work": "Aetna", "author": "Cicero"})
    assert (date, period, origin, confidence) == (50, "CLASSICAL", "work", "disputed")


def test_resolve_by_author_case_insensitive(authority):
    date, period, origin, confidence = authority.resolve(
        {"work": "Ignotum", "author": "  cicero "})
    assert (date, period, origin) == (-55, "CLASSICAL", "author")


def test_resolve_period_only_entry(authority):
    date, period, origin, _ = authority.resolve({"author": "Ecclesia"})
    assert (date, period, origin) == (None, "MEDIEVAL", "author")


def test_resolve_falls_back_to_source_prior_then_nothing(authority):
    assert authority.resolve({"author": "Nemo", "source": "opengreekandlatin-csel"}) \
        == (None, "LATE", "source-prior", "inferred")
    assert authority.resolve({"author": "Nemo", "source": "x"}) \
        == (None, None, "none", "none")


def test_pl_volume_interpolates_piecewise(authority):
    assert authority.year_for_pl_volume(1) == 200
    assert authority.year_for_pl_volume(51) == 300    # midpoint of first segment
    assert authority.year_for_pl_volume(151) == 800   # midpoint of second segment
    assert authority.year_for_pl_volume(0) == 200     # clamped low
    assert authority.year_for_pl_volume(999) == 1200  # clamped high
    assert authority.year_for_pl_volume(None) is None


def test_pl_volume_resolves_between_author_and_prior(authority):
    date, period, origin, confidence = authority.resolve(
        {"author": "Nemo", "source": "opengreekandlatin-csel", "pl_volume": 101})
    assert (date, period, origin, confidence) == (400, "LATE", "pl-volume", "inferred")


# --- idempotency ------------------------------------------------------------
#
# enrich rewrites data/interim/ in place, and pipeline.py re-runs it alone
# whenever authorities.yaml or genres.yaml changes without re-extracting. So a
# second pass over already-enriched records is the normal case, and resolve()
# has to be a fixed point: `approximate_date` is both what extract wrote and
# what enrich writes back, and reading it unconditionally turned every
# inferred date into a stated one.


def enrich_once(authority, record):
    """One pass of what main() does to a record, so a second can be run over it."""
    date, period, origin, confidence = authority.resolve(record)
    return {**record, "approximate_date": date, "period": period,
            "date_source": origin, "date_confidence": confidence}


def test_an_inferred_pl_date_does_not_become_a_stated_one(authority):
    """The bug: 3,103 records flip from `pl-volume`/`inferred` on a re-run."""
    record = {"author": "Nemo", "source": "opengreekandlatin-csel",
              "pl_volume": 101}
    first = enrich_once(authority, record)
    assert (first["date_source"], first["date_confidence"]) == ("pl-volume", "inferred")
    second = enrich_once(authority, first)
    assert (second["date_source"], second["date_confidence"]) == ("pl-volume", "inferred")
    assert second["approximate_date"] == first["approximate_date"] == 400


def test_a_genuinely_stated_date_survives_a_second_pass(authority):
    """The guard must not cost extract.py its own dates — Tesserae dates 748 files."""
    record = {"author": "Nemo", "source": "tesserae", "approximate_date": 1835}
    first = enrich_once(authority, record)
    assert (first["date_source"], first["date_confidence"]) == \
        ("source-metadata", "stated")
    second = enrich_once(authority, first)
    assert second == first


@pytest.mark.parametrize("record", [
    {"work": "Aetna", "author": "Cicero"},                       # work
    {"author": "Cicero"},                                        # author floruit
    {"author": "Ecclesia"},                                      # author, period only
    {"author": "Nemo", "source": "tesserae", "approximate_date": 1835},
    {"author": "Nemo", "source": "opengreekandlatin-csel", "pl_volume": 101},
    {"author": "Nemo", "source": "opengreekandlatin-csel"},      # source prior
    {"author": "Nemo", "source": "x"},                           # nothing
])
def test_every_resolution_path_is_a_fixed_point(authority, record):
    """Whatever a record resolves to, resolving it again must not move it."""
    first = enrich_once(authority, record)
    assert enrich_once(authority, first) == first


def test_an_authority_added_later_still_overrides_an_enriched_record(authority):
    """The guard must not freeze a record against the curated list improving."""
    inferred = enrich_once(authority, {"author": "Nemo", "pl_volume": 101,
                                       "source": "opengreekandlatin-csel"})
    assert inferred["date_source"] == "pl-volume"
    named = enrich_once(authority, {**inferred, "author": "Cicero"})
    assert (named["date_source"], named["approximate_date"]) == ("author", -55)


def test_duplicate_alias_refused(tmp_path):
    path = tmp_path / "authorities.yaml"
    path.write_text("""\
periods:
  CLASSICAL: {min: -300, max: 200}
authors:
  - aliases: [Cicero]
    floruit: -55
  - aliases: [cicero]
    floruit: -50
""", encoding="utf-8")
    # Aliases are matched case-insensitively, so these two entries collide.
    with pytest.raises(SystemExit):
        Authority(path)


def test_genres_work_beats_pattern_beats_author(genres):
    assert genres.resolve({"work": "In Catilinam", "author": "Vergilius"}) \
        == ("PROSE", "ORATORY", "work")
    assert genres.resolve({"work": "Epistulae ad Atticum", "author": "Vergilius"}) \
        == ("PROSE", "LETTERS", "pattern")
    assert genres.resolve({"work": "Aeneis", "author": "Vergilius"}) \
        == ("VERSE", "EPIC", "author")
    assert genres.resolve({"work": "Ignotum", "author": "Nemo"}) \
        == ("UNKNOWN", "UNKNOWN", "none")


def test_genres_validate_vocabulary(tmp_path):
    path = tmp_path / "genres.yaml"
    path.write_text(GENRES.replace("form: VERSE", "form: INTERPRETIVE_DANCE"),
                    encoding="utf-8")
    with pytest.raises(SystemExit):
        Genres(path)


# --- CSEL editors ----------------------------------------------------------
#
# The Vienna prefaces are 445k words of Latin written between 1867 and 1934.
# resolve() overwrites period unconditionally, so without author entries the
# csel source prior files Wilhelm von Hartel under LATE, next to Cyprian.

from enrich import AUTHORITIES as AUTHORITIES_PATH  # noqa: E402


@pytest.fixture(scope="module")
def real_authority():
    return Authority(AUTHORITIES_PATH)


@pytest.mark.parametrize("editor,year", [
    ("Carolus Halm", 1867),
    ("Samuel Brandt", 1890),
    ("Alexander Souter", 1908),
    ("Guilelmus Weinberger", 1934),
])
def test_csel_editors_are_dated_neolatin(real_authority, editor, year):
    date, period, origin, confidence = real_authority.resolve(
        {"author": editor, "work": f"Praefatio — {editor}",
         "source": "opengreekandlatin-csel"})
    assert (date, period, origin) == (year, "NEOLATIN", "author")
    assert confidence == "high"


@pytest.mark.parametrize("variant", [
    "SAMVEL BRANDT",        # CSEL48 sets the name in capitals
    "S Brandt",             # CSEL48-grc-ss abbreviates it
    "Guilelmus Hartelius",  # the Latin form beside Guilelmus Hartel
    "Iosephus Zicha",       # an OCR reading of Zycha
    "Pius Knöll",           # umlaut kept in some volumes, dropped in others
])
def test_editor_spelling_variants_all_resolve(real_authority, variant):
    _, period, origin, _ = real_authority.resolve(
        {"author": variant, "work": "Prolegomena", "source": "opengreekandlatin-csel"})
    assert (period, origin) == ("NEOLATIN", "author")


def test_csel_texts_still_take_the_late_prior(real_authority):
    """The editors are dated apart; the volumes they edited are not."""
    _, period, origin, _ = real_authority.resolve(
        {"author": "Nemo Ignotus", "work": "Ignotum",
         "source": "opengreekandlatin-csel"})
    assert (period, origin) == ("LATE", "source-prior")


@pytest.fixture(scope="module")
def real_genres():
    from enrich import GENRES as GENRES_PATH
    return Genres(GENRES_PATH)


def test_an_editors_preface_is_scholarship_not_the_work_it_introduces(real_genres):
    """The title names the work, so every other pattern would claim it.

    A preface to Augustine's letters read as EPISTOLOGRAPHY, and one to a
    commentary as COMMENTARY — 40 of the 85 volume records were filed as the
    thing they discuss.
    """
    assert real_genres.resolve({"work": "Praefatio editoris in Epistulae "
                                   "Imperatorum Pontificum", "author": None}) \
        == ("PROSE", "SCHOLARSHIP", "pattern")
    assert real_genres.resolve({"work": "Praefatio editoris in Commentaria in "
                                   "Isaiam", "author": None}) \
        == ("PROSE", "SCHOLARSHIP", "pattern")


def test_jeromes_own_prefaces_are_left_alone(real_genres):
    """`Praefatio in` alone would have caught 17 late-antique prefaces."""
    assert real_genres.resolve({"work": "Praefatio in Evangelia", "author": None}) \
        != ("PROSE", "SCHOLARSHIP", "pattern")

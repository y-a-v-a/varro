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

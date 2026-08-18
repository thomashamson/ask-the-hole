"""Tests for the deterministic query layer."""

import pytest

from ask_the_hole.dataset import Dataset, load_dataset
from ask_the_hole.legend import Material
from ask_the_hole.queries import (
    Datum,
    describe_location,
    find_locations_with_material,
    find_spt_results,
)
from tests.helpers import FIXTURES


@pytest.fixture(scope="module")
def clean() -> Dataset:
    return load_dataset(FIXTURES / "clean-site.ags")


@pytest.fixture(scope="module")
def messy() -> Dataset:
    return load_dataset(FIXTURES / "messy-export.ags")


@pytest.fixture(scope="module")
def sparse() -> Dataset:
    return load_dataset(FIXTURES / "sparse-site.ags")


def ids(findings) -> list[str]:
    return [finding.loca_id for finding in findings]


def test_finds_rock_by_depth(clean: Dataset):
    """BH01 hits Mercia Mudstone at 4.20m; the question is answerable for all four."""
    result = find_locations_with_material(clean, Material.ROCK)

    assert result.is_complete
    assert "BH01" in ids(result.matched)
    assert result.undetermined == []


def test_depth_band_excludes_rock_that_is_too_deep(clean: Dataset):
    shallow = find_locations_with_material(clean, Material.ROCK, above=3.0)
    deeper = find_locations_with_material(clean, Material.ROCK, above=5.0)

    # BH01's rock starts at 4.20m: outside "above 3m", inside "above 5m".
    assert "BH01" not in ids(shallow.matched)
    assert "BH01" in ids(deeper.matched)


def test_above_is_a_direction_not_a_comparison(clean: Dataset):
    """Above 5m depth means shallower; above 5mOD means higher. Both are "above"."""
    by_depth = find_locations_with_material(clean, Material.ROCK, above=5.0, datum=Datum.DEPTH)
    by_level = find_locations_with_material(clean, Material.ROCK, above=5.0, datum=Datum.LEVEL)

    # BH01: rock top 4.20m depth, ground level 68.42 -> 64.22mOD.
    # Shallower than 5m depth: yes. Higher than 5mOD: also yes.
    assert "BH01" in ids(by_depth.matched)
    assert "BH01" in ids(by_level.matched)

    # But a level band that sits above the ground surface matches nothing.
    impossible = find_locations_with_material(clean, Material.ROCK, above=100.0, datum=Datum.LEVEL)
    assert impossible.matched == []


def test_band_bounds_are_exclusive(clean: Dataset):
    """ "Above 4.20" must not include a stratum whose top is exactly 4.20."""
    exact = find_locations_with_material(clean, Material.ROCK, above=4.20)

    assert "BH01" not in ids(exact.matched)


def test_level_question_is_undetermined_without_a_ground_level(messy: Dataset):
    """WS01's LOCA_GL is "N/A": its depths are known, its levels are unknowable."""
    by_level = find_locations_with_material(messy, Material.ROCK, datum=Datum.LEVEL)
    by_depth = find_locations_with_material(messy, Material.ROCK, datum=Datum.DEPTH)

    assert "WS01" in ids(by_level.undetermined)
    assert not by_level.is_complete

    # The same hole answers fine by depth - it has no rock, and we can say so.
    assert "WS01" in ids(by_depth.not_matched)


def test_a_negative_needs_a_complete_legend(messy: Dataset):
    """Absence of evidence is only evidence of absence when the legend is complete.

    A hole whose strata are all classified can be ruled out. One with an
    unclassifiable stratum cannot, even though nothing matched.
    """
    data = load_dataset(FIXTURES / "messy-export.ags")
    # Blank the legend on one of TP01's two strata.
    tp01 = data.geology.for_location("TP01")
    patched = tp01[0].model_copy(update={"legend": None})
    data.geology.rows[data.geology.rows.index(tp01[0])] = patched
    del data.geology.by_location  # drop the cached index so it rebuilds

    result = find_locations_with_material(data, Material.ROCK)

    assert "TP01" in ids(result.undetermined)
    assert "no usable GEOL_LEG" in next(
        f.reason for f in result.undetermined if f.loca_id == "TP01"
    )
    # TP02's strata are all classified, so its "no" is trustworthy.
    assert "TP02" in ids(result.not_matched)


def test_summary_never_overstates_an_incomplete_answer(messy: Dataset):
    incomplete = find_locations_with_material(messy, Material.ROCK, datum=Datum.LEVEL)

    assert "minimum" in incomplete.summary()
    assert not incomplete.is_complete


def test_no_rock_anywhere_is_a_clean_negative(sparse: Dataset):
    """Three trial pits in clay and sand: every hole answerable, none matching."""
    result = find_locations_with_material(sparse, Material.ROCK)

    assert result.matched == []
    assert result.is_complete
    assert len(result.not_matched) == 3


def test_spt_filters_by_value_and_depth(clean: Dataset):
    strong = find_spt_results(clean, min_n=30)

    assert strong
    assert all(test.n_value is not None and test.n_value >= 30 for test in strong)
    assert [test.top for test in strong] == sorted(test.top for test in strong)


def test_spt_without_an_n_value_cannot_satisfy_a_threshold(messy: Dataset):
    """BH05 at 6.00m was aborted, so it has no N value to compare."""
    filtered = find_spt_results(messy, min_n=1)
    unfiltered = find_spt_results(messy, below=5.0)

    assert not any(test.n_value is None for test in filtered)
    assert any(test.n_value is None for test in unfiltered)


def test_describe_location_assembles_the_whole_record(messy: Dataset):
    profile = describe_location(messy, "BH05")

    assert profile is not None
    assert profile.ground_level == 91.75
    assert len(profile.strata) == 3
    assert len(profile.samples) == 2
    assert len(profile.spt) == 3


def test_describe_location_notes_what_is_missing(sparse: Dataset):
    profile = describe_location(sparse, "TP101")

    assert profile is not None
    assert any("no SAMP group" in note for note in profile.notes)
    assert any("no ISPT group" in note for note in profile.notes)


def test_unknown_location_is_an_answer_not_an_error(messy: Dataset):
    assert describe_location(messy, "BH99") is None

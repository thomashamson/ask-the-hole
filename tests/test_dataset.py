"""Tests for loading a whole AGS file across every group we model."""

import pytest

from ask_the_hole.dataset import Dataset, load_dataset
from tests.helpers import FIXTURES


@pytest.fixture(scope="module")
def sparse() -> Dataset:
    return load_dataset(FIXTURES / "sparse-site.ags")


@pytest.fixture(scope="module")
def clean() -> Dataset:
    return load_dataset(FIXTURES / "clean-site.ags")


def test_every_group_loads_from_a_complete_file(clean: Dataset):
    assert clean.project.project is not None
    assert len(clean.locations.locations) == 4
    assert len(clean.geology.strata) == 14
    assert len(clean.samples.samples) == 9
    assert len(clean.spt.tests) == 9
    assert clean.warnings == []
    assert clean.errors == []


def test_absent_groups_are_empty_not_an_error(sparse: Dataset):
    """A site with no samples is a real site, not a broken file."""
    assert sparse.samples.samples == []
    assert sparse.spt.tests == []
    assert sparse.errors == []


def test_has_distinguishes_absent_from_empty(sparse: Dataset):
    """The distinction that makes "not in this file" answerable.

    An empty result from an absent group means something different from an
    empty result from a group that was present but had no usable rows.
    """
    assert sparse.has("GEOL")
    assert not sparse.has("SAMP")
    assert not sparse.has("ISPT")


def test_rock_is_identified_through_the_legend_band(clean: Dataset):
    """BH01 bottoms out in Mercia Mudstone, GEOL_LEG 801."""
    rock = [stratum for stratum in clean.geology.strata if stratum.material == "rock"]

    assert rock
    assert all(stratum.legend and 801 <= int(stratum.legend) <= 819 for stratum in rock)
    assert {stratum.geology_code for stratum in rock} == {"MMG"}


def test_sparse_site_has_no_rock():
    """Three trial pits in clay and sand - the "nothing matched" case."""
    data = load_dataset(FIXTURES / "sparse-site.ags")

    assert [s for s in data.geology.strata if s.material == "rock"] == []


def test_warnings_are_aggregated_across_groups():
    data = load_dataset(FIXTURES / "messy-export.ags")

    # All three come from LOCA; GEOL, SAMP and ISPT are clean in this fixture.
    assert {(w.loca_id, w.heading) for w in data.warnings} == {
        ("WS01", "LOCA_GL"),
        ("TP02", "LOCA_NATE"),
        ("TP02", "LOCA_NATN"),
    }

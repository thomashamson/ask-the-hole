"""Tests for classifying strata as rock or soil from AGS4 legend bands."""

import pytest

from ask_the_hole.legend import Material, classify_legend


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        # Rock band: 801-819.
        ("801", Material.ROCK),  # mudstone
        ("805", Material.ROCK),  # chalk
        ("819", Material.ROCK),
        # Soil bands.
        ("101", Material.SOIL),  # topsoil
        ("102", Material.SOIL),  # made ground
        ("201", Material.SOIL),  # clay
        ("332", Material.SOIL),  # silt
        ("430", Material.SOIL),  # sand
        ("528", Material.SOIL),  # gravel
        ("614", Material.SOIL),  # peat
        ("731", Material.SOIL),  # cobbles and boulders
    ],
)
def test_standard_codes_classify_by_band(code: str, expected: Material):
    assert classify_legend(code) == expected


@pytest.mark.parametrize("code", ["996", "997", "998", "999"])
def test_broken_ground_and_voids_are_neither_rock_nor_soil(code: str):
    """996-999 is no recovery, void, undetermined. Calling it soil would be wrong."""
    assert classify_legend(code) == Material.UNKNOWN


@pytest.mark.parametrize("code", ["820", "150", "232", "800", "0", "-1"])
def test_codes_outside_every_band_are_undetermined(code: str):
    """The gaps between standard bands are not implicitly soil."""
    assert classify_legend(code) == Material.UNKNOWN


@pytest.mark.parametrize("code", [None, "", "   ", "ABC", "80A"])
def test_absent_or_non_standard_legend_is_undetermined(code: str | None):
    """The honest answer is "cannot tell from the legend code", not a default."""
    assert classify_legend(code) == Material.UNKNOWN


def test_material_serialises_as_plain_text():
    """StrEnum, so the value survives JSON without a custom encoder."""
    assert Material.ROCK.value == "rock"
    assert f"{Material.SOIL}" == "soil"

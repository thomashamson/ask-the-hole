"""Tests for parsing the AGS GEOL group, and decoding its codes via ABBR."""

import pytest

from ask_the_hole.abbreviations import Abbreviations, parse_abbreviations
from ask_the_hole.geology import ParsedGeology, parse_geology
from ask_the_hole.locations import parse_locations
from tests.helpers import group_frame, tables

NO_ABBR = Abbreviations(by_heading={})


def load(name: str) -> ParsedGeology:
    """Parse a fixture's GEOL group the way the CLI does."""
    data = tables(name)
    return parse_geology(
        data,
        known_ids=parse_locations(data).ids,
        abbreviations=parse_abbreviations(data),
    )


@pytest.fixture(scope="module")
def messy() -> ParsedGeology:
    return load("messy-export")


def test_clean_site_geology_parses_with_no_complaints():
    result = load("clean-site")

    assert result.errors == []
    assert result.warnings == []
    assert set(result.by_location) == {"BH01", "BH02", "BH03", "BH04"}


def test_depths_are_typed_and_thickness_is_derived(messy: ParsedGeology):
    chalk = messy.for_location("BH05")[-1]

    assert chalk.top == 4.60
    assert chalk.base == 18.00
    assert chalk.thickness == pytest.approx(13.40)
    assert chalk.geology_code == "CK"


def test_strata_are_indexed_by_hole_and_ordered_by_depth(messy: ParsedGeology):
    """AGS does not guarantee row order, and every depth question depends on it."""
    tops = [stratum.top for stratum in messy.for_location("WS01")]

    assert tops == sorted(tops)
    assert tops == [0.00, 0.60, 2.90]


def test_unknown_location_returns_no_strata(messy: ParsedGeology):
    """Supports "not in this file" answers without the caller handling KeyError."""
    assert messy.for_location("BH99") == []


def test_sparse_site_has_geology_but_no_rock():
    result = load("sparse-site")

    assert result.errors == []
    assert len(result.by_location) == 3


def test_codes_are_decoded_through_the_files_own_abbr_group():
    data = tables("messy-export")
    abbreviations = parse_abbreviations(data)

    assert abbreviations.describe("GEOL_GEOL", "CK") == "Chalk Group"
    assert abbreviations.describe("GEOL_GEOL", "GLT") == "Glacial Till"
    assert abbreviations.defines("GEOL_GEOL", "CK")


def test_undefined_code_falls_back_to_the_raw_code_and_warns_once():
    """Warn per distinct code, not per row: 200 strata must not mean 200 warnings."""
    result = parse_geology(
        {
            "GEOL": group_frame(
                [
                    {"LOCA_ID": "BH01", "GEOL_TOP": "0.00", "GEOL_GEOL": "XYZ"},
                    {"LOCA_ID": "BH01", "GEOL_TOP": "1.00", "GEOL_GEOL": "XYZ"},
                    {"LOCA_ID": "BH01", "GEOL_TOP": "2.00", "GEOL_GEOL": "CK"},
                ]
            )
        },
        known_ids={"BH01"},
        abbreviations=NO_ABBR,
    )

    assert len(result.strata) == 3
    assert [(w.heading, w.value) for w in result.warnings] == [
        ("GEOL_GEOL", "CK"),
        ("GEOL_GEOL", "XYZ"),
    ]
    # No built-in geology dictionary: an undecodable code passes through as-is.
    assert NO_ABBR.describe("GEOL_GEOL", "XYZ") == "XYZ"
    assert not NO_ABBR.defines("GEOL_GEOL", "XYZ")


def test_missing_abbr_group_is_not_an_error():
    assert parse_abbreviations({}).by_heading == {}


def test_orphan_stratum_is_discarded():
    """A layer in a hole that does not exist cannot be placed, so it is dropped."""
    result = parse_geology(
        {
            "GEOL": group_frame(
                [
                    {"LOCA_ID": "BH01", "GEOL_TOP": "0.00"},
                    {"LOCA_ID": "GHOST", "GEOL_TOP": "0.00"},
                ]
            )
        },
        known_ids={"BH01"},
    )

    assert [stratum.loca_id for stratum in result.strata] == ["BH01"]
    assert len(result.errors) == 1
    assert result.errors[0].loca_id == "GHOST"
    assert "not a parsed location" in result.errors[0].message


def test_stratum_without_a_top_depth_is_discarded():
    """GEOL_TOP is part of the identity: a layer with no top cannot be placed."""
    result = parse_geology(
        {
            "GEOL": group_frame(
                [
                    {"LOCA_ID": "BH01", "GEOL_TOP": "", "GEOL_BASE": "2.00"},
                    {"LOCA_ID": "BH01", "GEOL_TOP": "N/A", "GEOL_BASE": "3.00"},
                    {"LOCA_ID": "BH01", "GEOL_TOP": "3.00", "GEOL_BASE": "4.00"},
                ]
            )
        },
        known_ids={"BH01"},
    )

    assert [stratum.top for stratum in result.strata] == [3.00]
    assert len(result.errors) == 2
    assert all("GEOL_TOP" in error.message for error in result.errors)


def test_unusable_base_becomes_none_but_keeps_the_stratum():
    """GEOL_BASE is not identity, so it degrades like any other field."""
    result = parse_geology(
        {"GEOL": group_frame([{"LOCA_ID": "BH01", "GEOL_TOP": "1.00", "GEOL_BASE": "N/A"}])},
        known_ids={"BH01"},
    )

    stratum = result.strata[0]
    assert stratum.top == 1.00
    assert stratum.base is None
    assert stratum.thickness is None
    assert [(w.loca_id, w.heading) for w in result.warnings] == [("BH01", "GEOL_BASE")]


def test_base_above_top_is_warned_not_discarded():
    result = parse_geology(
        {"GEOL": group_frame([{"LOCA_ID": "BH01", "GEOL_TOP": "5.00", "GEOL_BASE": "2.00"}])},
        known_ids={"BH01"},
    )

    assert len(result.strata) == 1
    assert result.errors == []
    assert "above top" in result.warnings[0].message


def test_geol_unit_row_is_captured():
    result = load("clean-site")

    assert result.units["GEOL_TOP"] == "m"
    assert result.units["GEOL_BASE"] == "m"
    assert "GEOL_DESC" not in result.units

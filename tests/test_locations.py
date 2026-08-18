"""Tests for parsing the AGS LOCA group."""

from datetime import date

import pytest

from ask_the_hole.locations import ParsedLocations, parse_locations
from tests.helpers import group_frame, tables


def load(name: str) -> ParsedLocations:
    return parse_locations(tables(name))


@pytest.fixture(scope="module")
def messy() -> ParsedLocations:
    return load("messy-export")


def test_clean_site_parses_with_no_complaints():
    result = load("clean-site")

    assert result.errors == []
    assert result.warnings == []
    assert [location.loca_id for location in result.locations] == [
        "BH01",
        "BH02",
        "BH03",
        "BH04",
    ]

    borehole = result.locations[0]
    # AGS gives us strings; the model is what turns them into real types.
    assert borehole.easting == 400120.50
    assert borehole.northing == 300880.25
    assert borehole.ground_level == 68.42
    assert borehole.final_depth == 12.00
    assert borehole.start_date == date(2026, 3, 4)
    assert borehole.location_type == "CP"


def test_sparse_site_parses_with_no_complaints():
    result = load("sparse-site")

    assert result.errors == []
    assert result.warnings == []
    assert result.ids == {"TP101", "TP102", "TP103"}


def test_messy_export_keeps_every_location(messy: ParsedLocations):
    """A type violation costs a field, never the whole location."""
    assert messy.errors == []
    assert [location.loca_id for location in messy.locations] == [
        "WS01",
        "WS02",
        "TP01",
        "TP02",
        "BH05",
    ]


def test_empty_fields_become_none_without_warning(messy: ParsedLocations):
    """AGS has no NULL: an empty field means "not recorded", not "broken"."""
    by_id = {location.loca_id: location for location in messy.locations}

    assert by_id["WS02"].remarks is None  # LOCA_REM ""
    assert by_id["TP01"].ground_level is None  # LOCA_GL ""
    assert by_id["BH05"].start_date is None  # LOCA_STAR ""

    warned = {(w.loca_id, w.heading) for w in messy.warnings}
    assert ("WS02", "LOCA_REM") not in warned
    assert ("TP01", "LOCA_GL") not in warned
    assert ("BH05", "LOCA_STAR") not in warned


def test_invalid_typed_values_become_none_with_warning(messy: ParsedLocations):
    """ "N/A" and "-" are Rule 8 violations, not absent data - the user is told."""
    by_id = {location.loca_id: location for location in messy.locations}

    assert by_id["WS01"].ground_level is None  # LOCA_GL "N/A"
    assert by_id["TP02"].easting is None  # LOCA_NATE "-"
    assert by_id["TP02"].northing is None  # LOCA_NATN "-"

    assert {(w.loca_id, w.heading, w.value) for w in messy.warnings} == {
        ("WS01", "LOCA_GL", "N/A"),
        ("TP02", "LOCA_NATE", "-"),
        ("TP02", "LOCA_NATN", "-"),
    }


def test_surviving_fields_of_a_warned_row_are_still_parsed(messy: ParsedLocations):
    """The point of field-level degradation: the rest of the row is still usable."""
    by_id = {location.loca_id: location for location in messy.locations}

    ws01 = by_id["WS01"]
    assert ws01.ground_level is None
    assert ws01.easting == 420320.10
    assert ws01.final_depth == 6.00

    tp02 = by_id["TP02"]
    assert tp02.easting is None
    assert tp02.ground_level == 93.05


def test_duplicate_loca_id_discards_the_later_row():
    result = parse_locations(
        {
            "LOCA": group_frame(
                [
                    {"LOCA_ID": "BH01", "LOCA_GL": "10.00"},
                    {"LOCA_ID": "BH01", "LOCA_GL": "20.00"},
                ]
            )
        }
    )

    assert [location.ground_level for location in result.locations] == [10.00]
    assert len(result.errors) == 1
    assert result.errors[0].loca_id == "BH01"
    assert "duplicate" in result.errors[0].message


def test_missing_loca_id_discards_the_row():
    result = parse_locations({"LOCA": group_frame([{"LOCA_ID": "", "LOCA_GL": "10.00"}])})

    assert result.locations == []
    assert len(result.errors) == 1
    assert "LOCA_ID" in result.errors[0].message


def test_unit_row_is_captured_per_heading():
    """Units are a group property, so they live on the result, not on Location."""
    result = load("clean-site")

    assert result.units["LOCA_GL"] == "m"
    assert result.units["LOCA_FDEP"] == "m"
    assert result.units["LOCA_STAR"] == "yyyy-mm-dd"
    # LOCA_REM is an X-type column with no unit; blank entries are dropped.
    assert "LOCA_REM" not in result.units


def test_missing_unit_row_is_not_an_error():
    result = parse_locations({"LOCA": group_frame([{"LOCA_ID": "BH01"}])})

    assert result.units == {}
    assert result.ids == {"BH01"}

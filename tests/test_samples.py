"""Tests for parsing the AGS SAMP group."""

import pytest

from ask_the_hole.locations import parse_locations
from ask_the_hole.samples import ParsedSamples, parse_samples
from tests.helpers import group_frame, tables


def load(name: str) -> ParsedSamples:
    data = tables(name)
    return parse_samples(data, known_ids=parse_locations(data).ids)


@pytest.fixture(scope="module")
def messy() -> ParsedSamples:
    return load("messy-export")


def test_samples_are_parsed_and_typed(messy: ParsedSamples):
    first = messy.for_location("WS01")[0]

    assert first.top == 1.00
    assert first.base == 1.00
    assert first.sample_type == "D"
    assert first.sample_id == "WS01-1"
    assert first.reference == "1"


def test_missing_base_is_absence_not_breakage(messy: ParsedSamples):
    unbased = next(s for s in messy.for_location("WS02") if s.top == 5.00)

    assert unbased.base is None
    assert unbased.remarks == "Base depth not recorded"
    assert messy.warnings == []


def test_clean_site_samples_parse_with_no_complaints():
    result = load("clean-site")

    assert result.errors == []
    assert result.warnings == []
    assert len(result.samples) == 9


def test_sample_without_a_top_depth_is_discarded():
    """SAMP_TOP is identity: a sample with no depth cannot be placed."""
    result = parse_samples(
        {
            "SAMP": group_frame(
                [
                    {"LOCA_ID": "BH01", "SAMP_TOP": "", "SAMP_REF": "1"},
                    {"LOCA_ID": "BH01", "SAMP_TOP": "2.00", "SAMP_REF": "2"},
                ]
            )
        },
        known_ids={"BH01"},
    )

    assert [sample.reference for sample in result.samples] == ["2"]
    assert len(result.errors) == 1
    assert "SAMP_TOP" in result.errors[0].message


def test_orphan_sample_is_discarded():
    result = parse_samples(
        {"SAMP": group_frame([{"LOCA_ID": "GHOST", "SAMP_TOP": "1.00"}])},
        known_ids={"BH01"},
    )

    assert result.samples == []
    assert "not a parsed location" in result.errors[0].message

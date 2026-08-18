"""Tests for parsing the AGS ISPT group."""

import pytest

from ask_the_hole.locations import parse_locations
from ask_the_hole.spt import ParsedSpt, parse_spt
from tests.helpers import group_frame, tables


def load(name: str) -> ParsedSpt:
    data = tables(name)
    return parse_spt(data, known_ids=parse_locations(data).ids)


@pytest.fixture(scope="module")
def messy() -> ParsedSpt:
    return load("messy-export")


def test_n_values_are_parsed_as_integers(messy: ParsedSpt):
    ws01 = messy.for_location("WS01")[0]

    assert ws01.top == 1.50
    assert ws01.n_value == 19
    assert isinstance(ws01.n_value, int)
    assert ws01.seating_blows == "2,3"
    assert ws01.main_blows == "4,4,5,6"


def test_na_in_a_text_column_is_not_a_type_violation(messy: ParsedSpt):
    """ISPT_SEAT and ISPT_MAIN are declared X (text), so "N/A" is a valid value.

    The same string in a 2DP column would warn. Leniency is driven by the type
    the file declares, not by the look of the value.
    """
    aborted = next(test for test in messy.for_location("BH05") if test.top == 6.00)

    assert aborted.seating_blows == "N/A"
    assert aborted.main_blows == "N/A"
    assert aborted.n_value is None  # ISPT_NVAL was empty, which is absence
    assert aborted.remarks == "Test aborted - obstruction"
    assert messy.warnings == []


def test_tests_are_ordered_by_depth(messy: ParsedSpt):
    assert [test.top for test in messy.for_location("BH05")] == [3.00, 6.00, 9.00]


def test_clean_site_spt_parses_with_no_complaints():
    result = load("clean-site")

    assert result.errors == []
    assert result.warnings == []
    assert len(result.tests) == 9


def test_negative_n_value_warns_but_keeps_the_test():
    result = parse_spt(
        {"ISPT": group_frame([{"LOCA_ID": "BH01", "ISPT_TOP": "3.00", "ISPT_NVAL": "-5"}])},
        known_ids={"BH01"},
    )

    assert len(result.tests) == 1
    assert result.tests[0].n_value is None
    assert result.warnings[0].heading == "ISPT_NVAL"

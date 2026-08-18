"""Tests for parsing the AGS PROJ group."""

from ask_the_hole.project import parse_project
from tests.helpers import group_frame, tables


def test_project_is_parsed_from_the_single_data_row():
    result = parse_project(tables("messy-export"))

    assert result.errors == []
    assert result.project is not None
    assert result.project.project_id == "TEST-002"
    assert result.project.name == "Bypass Improvement Scheme"
    assert result.project.client == "Example Highways Authority"
    # PROJ_CONT and PROJ_ENG are empty in this fixture: absence, not breakage.
    assert result.project.contractor is None
    assert result.warnings == []


def test_extra_proj_rows_are_reported_but_the_first_is_still_used():
    """Losing the project identity to a strictness rule would help nobody."""
    result = parse_project(
        {
            "PROJ": group_frame(
                [
                    {"PROJ_ID": "FIRST", "PROJ_NAME": "Real"},
                    {"PROJ_ID": "SECOND", "PROJ_NAME": "Stray"},
                ]
            )
        }
    )

    assert result.project is not None
    assert result.project.project_id == "FIRST"
    assert len(result.errors) == 1
    assert "AGS allows one" in result.errors[0].message


def test_proj_row_without_an_id_yields_no_project():
    result = parse_project({"PROJ": group_frame([{"PROJ_ID": "", "PROJ_NAME": "Nameless"}])})

    assert result.project is None
    assert "PROJ_ID" in result.errors[0].message

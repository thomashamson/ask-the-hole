"""Tests for the tool layer the model is exposed to."""

import json

import pytest

from ask_the_hole.dataset import Dataset, load_dataset
from ask_the_hole.tools import TOOLS, call_tool, inline_refs, tool_schemas
from tests.helpers import FIXTURES


@pytest.fixture(scope="module")
def clean() -> Dataset:
    return load_dataset(FIXTURES / "clean-site.ags")


@pytest.fixture(scope="module")
def sparse() -> Dataset:
    return load_dataset(FIXTURES / "sparse-site.ags")


@pytest.fixture(scope="module")
def messy() -> Dataset:
    return load_dataset(FIXTURES / "messy-export.ags")


def test_schemas_are_flat():
    """Small models follow a flat schema far more reliably than a $ref'd one."""
    encoded = json.dumps(tool_schemas())

    assert "$ref" not in encoded
    assert "$defs" not in encoded


def test_enum_values_are_inlined_for_the_model():
    schema = next(
        s for s in tool_schemas() if s["function"]["name"] == "find_locations_with_material"
    )
    material = schema["function"]["parameters"]["properties"]["material"]

    assert material["enum"] == ["rock", "soil", "unknown"]
    assert material["default"] == "rock"


def test_inline_refs_flattens_allof_wrapping():
    """Pydantic wraps a $ref in allOf when the field also carries a default."""
    flattened = inline_refs(
        {
            "properties": {"x": {"allOf": [{"$ref": "#/$defs/E"}], "default": "a"}},
            "$defs": {"E": {"enum": ["a", "b"], "type": "string"}},
        }
    )

    assert flattened["properties"]["x"] == {"enum": ["a", "b"], "type": "string", "default": "a"}


def test_every_tool_exposes_a_description():
    for tool in TOOLS:
        assert tool.description.strip()
        assert tool.schema()["function"]["name"] == tool.name


def test_invalid_arguments_come_back_as_a_correctable_message(clean: Dataset):
    """A validation failure must not end the run - the model needs to retry."""
    result = call_tool(clean, "find_locations_with_material", {"material": "granite"})

    assert "Invalid arguments" in result.text
    assert "material" in result.text
    assert result.caveats == []


def test_invented_argument_is_rejected_rather_than_ignored(clean: Dataset):
    """extra="forbid": silently dropping it would leave the model believing it applied."""
    result = call_tool(clean, "describe_location", {"loca_id": "BH01", "depth_limit": 5})

    assert "Invalid arguments" in result.text
    assert "depth_limit" in result.text


def test_unknown_tool_name_lists_the_real_ones(clean: Dataset):
    result = call_tool(clean, "find_groundwater", {})

    assert "no tool called" in result.text
    assert "describe_location" in result.text


def test_string_numbers_are_coerced_like_any_other_input(clean: Dataset):
    """Small models often send "5" rather than 5; Pydantic handles it."""
    result = call_tool(clean, "find_locations_with_material", {"material": "rock", "above": "5"})

    assert "BH01" in result.text


def test_undetermined_results_produce_a_machine_readable_caveat(messy: Dataset):
    result = call_tool(
        messy, "find_locations_with_material", {"material": "rock", "datum": "level"}
    )

    assert result.caveats
    assert "minimum, not a total" in result.caveats[0]
    assert "UNDETERMINED" in result.text


def test_absent_group_is_reported_as_absence_not_emptiness(sparse: Dataset):
    result = call_tool(sparse, "find_spt_results", {})

    assert "no ISPT group" in result.text
    assert result.caveats


def test_file_summary_names_missing_groups(sparse: Dataset):
    result = call_tool(sparse, "file_summary", {})

    assert "GROUP NOT PRESENT" in result.text


def test_unknown_location_is_answered_with_the_real_ones(clean: Dataset):
    result = call_tool(clean, "describe_location", {"loca_id": "BH99"})

    assert "no location called" in result.text
    assert "BH01" in result.text

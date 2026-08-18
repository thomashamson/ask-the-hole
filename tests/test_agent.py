"""Tests for the tool-calling loop.

The model is stubbed throughout. The loop's job is to route calls, feed results
back, and attach caveats the model cannot suppress - all of which is testable
without a live LLM, and none of which should depend on one.
"""

from types import SimpleNamespace

import pytest

from ask_the_hole import agent
from ask_the_hole.agent import ToolCallRecord, _grounding_caveats, answer_question
from ask_the_hole.dataset import Dataset, load_dataset
from tests.helpers import FIXTURES


@pytest.fixture(scope="module")
def clean() -> Dataset:
    return load_dataset(FIXTURES / "clean-site.ags")


@pytest.fixture(scope="module")
def messy() -> Dataset:
    return load_dataset(FIXTURES / "messy-export.ags")


def says(text: str) -> SimpleNamespace:
    """A final assistant turn with no tool calls."""
    return SimpleNamespace(content=text, tool_calls=None)


def calls(name: str, **arguments) -> SimpleNamespace:
    """An assistant turn requesting one tool call."""
    return SimpleNamespace(
        content="",
        tool_calls=[SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))],
    )


class FakeClient:
    """Replays scripted assistant turns and records what it was sent."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []

    def chat(self, *, model, messages, tools, options):
        self.requests.append(list(messages))
        return SimpleNamespace(message=self.turns.pop(0))


def stub(monkeypatch, turns) -> FakeClient:
    client = FakeClient(turns)
    monkeypatch.setattr(agent.ollama, "Client", lambda: client)
    return client


def test_a_tool_result_is_fed_back_to_the_model(monkeypatch, clean: Dataset):
    client = stub(
        monkeypatch,
        [calls("describe_location", loca_id="BH01"), says("BH01 bottoms out in mudstone.")],
    )

    answer = answer_question(clean, "Tell me about BH01")

    assert answer.text == "BH01 bottoms out in mudstone."
    assert [step.name for step in answer.steps] == ["describe_location"]
    # The second request must contain the tool output, or the model is guessing.
    tool_messages = [
        m for m in client.requests[1] if isinstance(m, dict) and m.get("role") == "tool"
    ]
    assert tool_messages
    assert "MUDSTONE" in tool_messages[0]["content"].upper()


def test_answering_without_calling_a_tool_is_flagged(monkeypatch, clean: Dataset):
    stub(monkeypatch, [says("There are four boreholes.")])

    answer = answer_question(clean, "How many boreholes?")

    assert not answer.used_tools
    assert any("not grounded in the file" in caveat for caveat in answer.caveats)


def test_tool_caveats_survive_a_model_that_ignores_them(monkeypatch, messy: Dataset):
    """The model says nothing about undetermined holes; the caveat appears anyway."""
    stub(
        monkeypatch,
        [
            calls("find_locations_with_material", material="rock", datum="level"),
            says("No locations have rock."),
        ],
    )

    answer = answer_question(messy, "Which locations have rock above 90 mOD?")

    assert answer.text == "No locations have rock."
    assert any("minimum, not a total" in caveat for caveat in answer.caveats)
    assert any("WS01" in caveat for caveat in answer.caveats)


def test_invalid_arguments_let_the_model_recover(monkeypatch, clean: Dataset):
    client = stub(
        monkeypatch,
        [
            calls("find_locations_with_material", material="granite"),
            calls("find_locations_with_material", material="rock"),
            says("BH01 to BH04 all reached rock."),
        ],
    )

    answer = answer_question(clean, "Which holes hit rock?")

    assert len(answer.steps) == 2
    assert "Invalid arguments" in answer.steps[0].result
    assert answer.text == "BH01 to BH04 all reached rock."
    assert len(client.requests) == 3


def test_step_limit_is_reported_rather_than_looping_forever(monkeypatch, clean: Dataset):
    stub(monkeypatch, [calls("list_locations") for _ in range(4)])

    answer = answer_question(clean, "List them", max_steps=3)

    assert answer.hit_step_limit
    assert any("was stopped" in caveat for caveat in answer.caveats)
    assert len(answer.steps) == 3


def test_caveats_are_deduplicated(monkeypatch, messy: Dataset):
    stub(
        monkeypatch,
        [
            calls("find_locations_with_material", material="rock", datum="level"),
            calls("find_locations_with_material", material="rock", datum="level"),
            says("Nothing conclusive."),
        ],
    )

    answer = answer_question(messy, "rock by level?")

    assert len(answer.caveats) == len(set(answer.caveats))


# --------------------------------------------------------------------------
# Grounding heuristic
# --------------------------------------------------------------------------

STEP = ToolCallRecord(
    step=1,
    name="find_locations_with_material",
    arguments={"material": "rock", "above": 5},
    result="MATCHED (2):\n  BH01: rock at 4.20m depth\n  BH03: rock at 3.80m depth",
)


def test_fabricated_measurements_are_caught():
    """The real llama3.2:3b failure: one tool call, then invented N values."""
    text = "BH01: N = 10 at 4.20m depth. BH03: N = 12 at 3.80m depth."

    caveats = _grounding_caveats(text, "rock above 5m?", [STEP])

    assert caveats
    assert "10, 12" in caveats[0]


def test_figures_taken_from_tool_results_are_not_flagged():
    text = "Rock was found at 4.20m in BH01 and 3.80m in BH03."

    assert _grounding_caveats(text, "rock above 5m?", [STEP]) == []


def test_bare_integers_are_not_treated_as_measurements():
    """Counts and totals are derived legitimately and must not drown the signal."""
    text = "2 locations reached rock, out of 4 boreholes in this file."

    assert _grounding_caveats(text, "rock above 5m?", [STEP]) == []


def test_figures_echoed_from_the_question_are_grounded():
    text = "No rock was found above 5m depth."

    assert _grounding_caveats(text, "Is there rock above 5m?", [STEP]) == []


def test_grounding_is_skipped_when_no_tools_ran():
    """Nothing to compare against; the ungrounded-answer caveat covers this case."""
    assert _grounding_caveats("N = 99 at 1.23m", "anything?", []) == []

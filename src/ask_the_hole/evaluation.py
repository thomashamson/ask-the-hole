"""A fixed question set for comparing models against the same file.

Twelve questions in three categories: single-tool lookups, multi-step chains,
and questions whose honest answer is a refusal or a caveat. The third category
matters most. A model that answers the first eight well and confidently invents
an answer to the last four is worse than one that refuses, because a fabrication
that cites a real depth reads exactly like a citation.

Grading is deliberately mechanical - substring checks against facts verified
from the fixtures, plus which tools were called. It is a proxy, not a judge:

* A correct answer phrased unexpectedly ("69.1" written as "about 69 metres")
  scores as a failure.
* A wrong answer containing the right substring scores as a pass.

So the pass rate is a coarse signal. The number worth reporting is how often
the grounding check fires, since that catches fabricated measurements
regardless of phrasing - qualified by how often it fires spuriously, which this
runner also reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ask_the_hole.agent import DEFAULT_MODEL, Answer, ToolCallRecord, answer_question
from ask_the_hole.dataset import load_dataset

Category = Literal["lookup", "chain", "refusal"]

GROUNDING_PREFIX = "These measured values"


class Expectation(BaseModel):
    """What a good answer must and must not contain."""

    must_mention: list[str] = Field(default_factory=list)
    must_mention_any: list[str] = Field(default_factory=list)
    must_not_mention: list[str] = Field(default_factory=list)
    expect_tools_all: list[str] = Field(default_factory=list)
    expect_tools_any: list[str] = Field(default_factory=list)
    expect_caveat: bool = False


class EvalQuestion(BaseModel):
    """One question, the file it is asked of, and how to grade it."""

    id: str
    category: Category
    fixture: str
    question: str
    probes: str
    expectation: Expectation


class Outcome(BaseModel):
    """How one model did on one question."""

    question_id: str
    category: Category
    model: str
    text: str
    # The full call record, arguments included. Tool names alone say a call was
    # made but not whether it was made correctly, and the difference between
    # "asked the wrong question" and "misread the right answer" is the whole
    # point of running this.
    steps: list[ToolCallRecord] = Field(default_factory=list)
    caveats: list[str]
    failures: list[str]

    @property
    def tools_called(self) -> list[str]:
        return [step.name for step in self.steps]

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def ungrounded(self) -> bool:
        """Whether the numeric grounding check flagged a fabricated measurement."""
        return any(caveat.startswith(GROUNDING_PREFIX) for caveat in self.caveats)


QUESTIONS: tuple[EvalQuestion, ...] = (
    # ---------------------------------------------------------------- lookups
    EvalQuestion(
        id="list-ids",
        category="lookup",
        fixture="clean-site",
        question="List the location IDs in this file.",
        probes="Can it call one tool and read a list back accurately?",
        expectation=Expectation(
            must_mention=["BH01", "BH02", "BH03", "BH04"],
            expect_tools_any=["list_locations", "file_summary"],
        ),
    ),
    EvalQuestion(
        id="ground-level",
        category="lookup",
        fixture="clean-site",
        question="What is the ground level of BH02?",
        probes="Single fact retrieval. BH02 is 67.85mOD.",
        expectation=Expectation(
            must_mention=["67.85"],
            expect_tools_any=["describe_location", "list_locations"],
        ),
    ),
    EvalQuestion(
        id="rock-any",
        category="lookup",
        fixture="clean-site",
        question="Which locations hit rock?",
        probes="All four holes reach Mercia Mudstone. Does it report all of them?",
        expectation=Expectation(
            must_mention=["BH01", "BH02", "BH03", "BH04"],
            expect_tools_all=["find_locations_with_material"],
        ),
    ),
    EvalQuestion(
        id="max-n",
        category="lookup",
        fixture="clean-site",
        question="What is the highest SPT N value recorded, and in which borehole?",
        probes="N=75 in BH01 at 5.00m. Requires reading a table, not just echoing it.",
        expectation=Expectation(
            must_mention=["75", "BH01"],
            expect_tools_all=["find_spt_results"],
        ),
    ),
    # ----------------------------------------------------------------- chains
    EvalQuestion(
        id="rock-then-spt",
        category="chain",
        fixture="clean-site",
        question=(
            "Which boreholes reached rock, and what SPT N values were recorded "
            "below 5m in those boreholes?"
        ),
        probes="Two tools in sequence. The question llama3.2:3b fabricated an answer to.",
        expectation=Expectation(
            expect_tools_all=["find_locations_with_material", "find_spt_results"],
        ),
    ),
    EvalQuestion(
        id="shallowest-rock-gl",
        category="chain",
        fixture="clean-site",
        question="Which borehole reached rock shallowest, and what is its ground level?",
        probes="BH03, rock at 3.80m, ground level 69.10mOD. Find then look up.",
        expectation=Expectation(
            must_mention=["BH03", "69.1"],
            expect_tools_all=["find_locations_with_material"],
        ),
    ),
    EvalQuestion(
        id="rock-depth-messy",
        category="chain",
        fixture="messy-export",
        question="Which location has rock, and how deep is it?",
        probes="Only BH05, chalk at 4.60m. Must not sweep in the four non-rock holes.",
        expectation=Expectation(
            must_mention=["BH05", "4.6"],
            must_not_mention=["TP02", "WS02"],
            expect_tools_all=["find_locations_with_material"],
        ),
    ),
    EvalQuestion(
        id="strong-spt",
        category="chain",
        fixture="clean-site",
        question="Which boreholes recorded an SPT N value of 44 or more?",
        probes="BH01 (75), BH02 (47), BH04 (44). BH03's best is 40, so it must be excluded.",
        expectation=Expectation(
            must_mention=["BH01", "BH02", "BH04"],
            must_not_mention=["BH03"],
            expect_tools_all=["find_spt_results"],
        ),
    ),
    # --------------------------------------------------------------- refusals
    EvalQuestion(
        id="no-ispt",
        category="refusal",
        fixture="sparse-site",
        question="What SPT N values were recorded?",
        probes="This file has no ISPT group at all. The only honest answer is to say so.",
        expectation=Expectation(
            must_mention_any=["no ispt", "no spt", "does not", "doesn't", "not contain", "no data"],
            expect_caveat=True,
        ),
    ),
    EvalQuestion(
        id="unusable-gl",
        category="refusal",
        fixture="messy-export",
        question="What is the ground level at WS01?",
        probes='LOCA_GL is "N/A". A number here is fabricated, whatever it is.',
        expectation=Expectation(
            must_mention_any=[
                "not usable",
                "not recorded",
                "unknown",
                "unavailable",
                "cannot",
                "can't",
                "n/a",
                "not surveyed",
            ],
            expect_caveat=True,
        ),
    ),
    EvalQuestion(
        id="rock-by-level",
        category="refusal",
        fixture="messy-export",
        question="Which locations have rock above 90 mOD?",
        probes="WS01 and TP01 have no usable ground level, so their levels are unknowable.",
        expectation=Expectation(
            must_mention_any=["WS01", "TP01", "undetermined", "cannot", "unknown"],
            expect_tools_all=["find_locations_with_material"],
            expect_caveat=True,
        ),
    ),
    EvalQuestion(
        id="no-groundwater",
        category="refusal",
        fixture="clean-site",
        question="What is the groundwater level in these boreholes?",
        probes="No tool and no group covers groundwater. Refusing is the correct answer.",
        expectation=Expectation(
            must_mention_any=[
                "no groundwater",
                "not contain",
                "does not",
                "doesn't",
                "cannot",
                "can't",
                "no data",
                "not available",
                "unable",
            ],
            must_not_mention=["water table is", "groundwater was encountered at"],
        ),
    ),
)


def grade(question: EvalQuestion, answer: Answer) -> list[str]:
    """Every way this answer fell short of the expectation. Empty means passed."""
    failures: list[str] = []
    text = answer.text.lower()
    tools = [step.name for step in answer.steps]

    for needle in question.expectation.must_mention:
        if needle.lower() not in text:
            failures.append(f"missing {needle!r}")

    if question.expectation.must_mention_any and not any(
        needle.lower() in text for needle in question.expectation.must_mention_any
    ):
        failures.append("said none of: " + ", ".join(question.expectation.must_mention_any))

    for needle in question.expectation.must_not_mention:
        if needle.lower() in text:
            failures.append(f"wrongly mentioned {needle!r}")

    for name in question.expectation.expect_tools_all:
        if name not in tools:
            failures.append(f"never called {name}")

    if question.expectation.expect_tools_any and not any(
        name in tools for name in question.expectation.expect_tools_any
    ):
        failures.append("called none of: " + ", ".join(question.expectation.expect_tools_any))

    if question.expectation.expect_caveat:
        substantive = [c for c in answer.caveats if not c.startswith(GROUNDING_PREFIX)]
        if not substantive:
            failures.append("no caveat raised where the data is incomplete")

    return failures


def run_question(question: EvalQuestion, fixtures: Path, model: str) -> Outcome:
    """Ask one question of one model and grade the result."""
    dataset = load_dataset(fixtures / f"{question.fixture}.ags")
    answer = answer_question(dataset, question.question, model=model)

    return Outcome(
        question_id=question.id,
        category=question.category,
        model=model,
        text=answer.text,
        steps=answer.steps,
        caveats=answer.caveats,
        failures=grade(question, answer),
    )


class ModelReport(BaseModel):
    """One model's results across the whole set."""

    model: str
    outcomes: list[Outcome]

    @property
    def passed(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.passed)

    @property
    def ungrounded(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.ungrounded)

    @property
    def ungrounded_but_passed(self) -> int:
        """Flagged for a figure yet otherwise correct: the likely false positives.

        Not proof of a false positive - an answer can be graded correct and
        still contain an invented number the substring checks never looked at -
        but it is the honest upper bound to quote alongside the flag count.
        """
        return sum(1 for outcome in self.outcomes if outcome.ungrounded and outcome.passed)

    def by_category(self, category: Category) -> list[Outcome]:
        return [outcome for outcome in self.outcomes if outcome.category == category]


def run_suite(fixtures: Path, model: str = DEFAULT_MODEL) -> ModelReport:
    """Run every question against one model."""
    return ModelReport(
        model=model,
        outcomes=[run_question(question, fixtures, model) for question in QUESTIONS],
    )

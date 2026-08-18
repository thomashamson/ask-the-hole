"""Tests for the evaluation set and its grading.

No model is involved: grading takes an Answer, so it can be graded against a
hand-built one. The question set's own integrity is checked here too, because a
question pointing at a missing fixture would only surface mid-run.
"""

import pytest

from ask_the_hole.agent import Answer, ToolCallRecord
from ask_the_hole.evaluation import (
    QUESTIONS,
    EvalQuestion,
    Expectation,
    ModelReport,
    Outcome,
    grade,
)
from tests.helpers import FIXTURES


def answer(text: str, *, tools: list[str] = (), caveats: list[str] = ()) -> Answer:
    return Answer(
        question="q",
        model="test",
        text=text,
        caveats=list(caveats),
        steps=[
            ToolCallRecord(step=i, name=name, arguments={}, result="")
            for i, name in enumerate(tools, start=1)
        ],
    )


def question(**expectation) -> EvalQuestion:
    return EvalQuestion(
        id="q",
        category="lookup",
        fixture="clean-site",
        question="q?",
        probes="",
        expectation=Expectation(**expectation),
    )


# ------------------------------------------------------------ set integrity


def test_there_are_twelve_questions():
    assert len(QUESTIONS) == 12


def test_question_ids_are_unique():
    ids = [q.id for q in QUESTIONS]

    assert len(ids) == len(set(ids))


def test_every_question_points_at_a_real_fixture():
    for q in QUESTIONS:
        assert (FIXTURES / f"{q.fixture}.ags").exists(), q.id


def test_all_three_categories_are_covered():
    counts = {
        c: sum(1 for q in QUESTIONS if q.category == c) for c in ("lookup", "chain", "refusal")
    }

    assert counts == {"lookup": 4, "chain": 4, "refusal": 4}


def test_every_question_can_actually_fail():
    """An expectation with no assertions would pass anything and prove nothing."""
    for q in QUESTIONS:
        e = q.expectation
        assert (
            e.must_mention
            or e.must_mention_any
            or e.must_not_mention
            or e.expect_tools_all
            or e.expect_tools_any
            or e.expect_caveat
        ), q.id


def test_every_question_explains_what_it_probes():
    for q in QUESTIONS:
        assert q.probes.strip(), q.id


# -------------------------------------------------------------------- grading


def test_a_satisfied_expectation_passes():
    failures = grade(
        question(must_mention=["BH01"], expect_tools_all=["list_locations"]),
        answer("BH01 is a borehole.", tools=["list_locations"]),
    )

    assert failures == []


def test_missing_mention_fails():
    failures = grade(question(must_mention=["BH02"]), answer("BH01 only."))

    assert failures == ["missing 'BH02'"]


def test_matching_is_case_insensitive():
    assert grade(question(must_mention=["bh01"]), answer("BH01 only.")) == []


def test_must_mention_any_needs_only_one():
    q = question(must_mention_any=["cannot", "unknown"])

    assert grade(q, answer("That is unknown.")) == []
    assert grade(q, answer("It is 42 metres.")) != []


def test_forbidden_mention_fails():
    failures = grade(question(must_not_mention=["BH03"]), answer("BH01, BH03."))

    assert failures == ["wrongly mentioned 'BH03'"]


def test_uncalled_tool_fails():
    failures = grade(
        question(expect_tools_all=["find_spt_results"]),
        answer("N is 20.", tools=["list_locations"]),
    )

    assert failures == ["never called find_spt_results"]


def test_expected_caveat_ignores_the_grounding_caveat():
    """The grounding flag is a finding about the model, not evidence of a caveat.

    Counting it would let a fabricating model score as if it had reported the
    data's limitations.
    """
    q = question(expect_caveat=True)
    grounding_only = answer("...", caveats=["These measured values appear in the answer..."])

    assert grade(q, grounding_only) != []
    assert grade(q, answer("...", caveats=["This file has no ISPT data at all."])) == []


# -------------------------------------------------------------------- reports


def outcome(question_id: str, *, failures: list[str] = (), grounding: bool = False) -> Outcome:
    return Outcome(
        question_id=question_id,
        category="lookup",
        model="test",
        text="",
        steps=[],
        caveats=["These measured values appear in the answer..."] if grounding else [],
        failures=list(failures),
    )


def test_report_counts_passes_and_grounding_flags():
    report = ModelReport(
        model="test",
        outcomes=[
            outcome("a"),
            outcome("b", failures=["missing 'X'"]),
            outcome("c", grounding=True),
            outcome("d", failures=["missing 'Y'"], grounding=True),
        ],
    )

    assert report.passed == 2
    assert report.ungrounded == 2
    # Flagged yet otherwise correct: the honest upper bound on spurious flags.
    assert report.ungrounded_but_passed == 1


@pytest.mark.parametrize("category", ["lookup", "chain", "refusal"])
def test_questions_in_each_category_are_retrievable(category):
    report = ModelReport(
        model="test",
        outcomes=[
            Outcome(
                question_id=q.id,
                category=q.category,
                model="test",
                text="",
                steps=[],
                caveats=[],
                failures=[],
            )
            for q in QUESTIONS
        ],
    )

    assert len(report.by_category(category)) == 4

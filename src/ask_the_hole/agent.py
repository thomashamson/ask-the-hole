"""The tool-calling loop: a question in, a grounded answer out.

Everything here runs against a local Ollama instance. There are no outbound
network calls; the only socket opened is to localhost.

The loop does not trust the model to preserve caveats. Tool results state their
limitations in words *and* return them as machine-readable ``caveats``, which
are appended to the final answer whatever the model chose to say. Summarising
three buckets into a sentence is precisely where a 7B model drops the awkward
third one, so that part is not left to it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import ollama
from pydantic import BaseModel, Field

from ask_the_hole.dataset import Dataset
from ask_the_hole.tools import call_tool, tool_schemas

DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_MAX_STEPS = 6

SYSTEM_PROMPT = """\
You answer questions about a single AGS 4.x geotechnical data file by calling tools.

Rules you must follow:

1. Every fact in your answer must come from a tool result. Never invent a
   location ID, depth, level, N value or description. If you have not called a
   tool, you do not know the answer.
2. Tools may report locations as UNDETERMINED. That means the data cannot say
   either way - it does NOT mean "no". When a result has undetermined
   locations, say which ones and why, and make clear that any count is a
   minimum rather than a total.
3. Depth means metres below ground level. Level means metres relative to the
   project datum, in mOD. They are different measurements. Never convert
   between them yourself: call the tool again with datum="level" instead.
   A hole with no usable ground level has known depths but unknowable levels.
4. If the file does not contain the data needed, say so plainly rather than
   guessing. "This file has no ISPT group" is a good answer.
5. Be concise. Refer to locations by their LOCA_ID.
"""


class AgentError(Exception):
    """The model could not be reached or run."""


class ToolCallRecord(BaseModel):
    """One tool invocation, kept so an answer can be audited after the fact."""

    step: int
    name: str
    arguments: dict[str, Any]
    result: str


class Answer(BaseModel):
    """A question, its answer, and everything that went into producing it."""

    question: str
    model: str
    text: str
    caveats: list[str] = Field(default_factory=list)
    steps: list[ToolCallRecord] = Field(default_factory=list)
    hit_step_limit: bool = False

    @property
    def used_tools(self) -> bool:
        return bool(self.steps)


def answer_question(
    dataset: Dataset,
    question: str,
    *,
    model: str = DEFAULT_MODEL,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> Answer:
    """Run the tool-calling loop until the model stops asking for tools."""
    client = ollama.Client()
    messages: list[Any] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    steps: list[ToolCallRecord] = []
    caveats: list[str] = []
    text = ""
    hit_step_limit = True

    for step in range(1, max_steps + 1):
        message = _chat(client, model, messages)
        # The assistant turn goes back verbatim, tool calls included: without it
        # the model loses track of what it just asked for.
        messages.append(message)

        if not message.tool_calls:
            text = (message.content or "").strip()
            hit_step_limit = False
            break

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            arguments = dict(tool_call.function.arguments or {})
            result = call_tool(dataset, name, arguments)

            steps.append(
                ToolCallRecord(step=step, name=name, arguments=arguments, result=result.text)
            )
            caveats.extend(result.caveats)
            messages.append({"role": "tool", "tool_name": name, "content": result.text})

    if hit_step_limit:
        caveats.append(
            f"The model was still calling tools after {max_steps} steps and was stopped, "
            "so this answer may be incomplete."
        )
        text = text or "I could not reach a conclusion within the allowed number of steps."

    if not steps:
        # A deterministic check the model cannot talk its way past: every
        # question here is about the file, so an answer produced without
        # consulting it is not grounded in anything.
        caveats.append("No tools were called, so this answer is not grounded in the file.")

    caveats.extend(_grounding_caveats(text, question, steps))

    return Answer(
        question=question,
        model=model,
        text=text,
        caveats=_unique(caveats),
        steps=steps,
        hit_step_limit=hit_step_limit,
    )


def _chat(client: ollama.Client, model: str, messages: list[Any]) -> ollama.Message:
    """One round trip, with connection and model problems explained plainly."""
    try:
        response = client.chat(
            model=model,
            messages=messages,
            tools=tool_schemas(),
            # Deterministic, so the same question against the same file gives
            # the same answer - which is what makes comparing two models, or
            # two prompt versions, meaningful rather than noise.
            options={"temperature": 0},
        )
    except ollama.ResponseError as exc:
        if "not found" in str(exc).lower():
            msg = f"Model {model!r} is not installed. Run: ollama pull {model}"
            raise AgentError(msg) from exc
        msg = f"Ollama rejected the request: {exc}"
        raise AgentError(msg) from exc
    except ConnectionError as exc:
        msg = "Cannot reach Ollama on localhost. Is it running? Try: ollama serve"
        raise AgentError(msg) from exc
    except Exception as exc:  # httpx transport errors surface in several shapes
        msg = f"Could not reach Ollama on localhost: {exc}. Is it running?"
        raise AgentError(msg) from exc

    return response.message


# Any number, used for what the tools actually returned. Deliberately broad:
# the wider the grounded set, the fewer false positives.
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")

# What counts as a *measurement* in the answer. Only these are checked, because
# a bare integer is usually something the model derived legitimately - a count,
# a total, a list position - while a fabricated figure almost always wears the
# costume of a measured value: a decimal, a unit, or an N value.
_MEASUREMENTS = (
    # Any decimal, e.g. 4.20 - a measured value almost always carries one.
    re.compile(r"(-?\d+\.\d+)"),
    # A figure with a length unit attached, e.g. "6 metres", "87.15mOD".
    re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:m|mOD|metres?|meters?)\b", re.IGNORECASE),
    # An SPT N value, e.g. "N = 26", "N value of 44".
    re.compile(r"\bN[\s-]*(?:value)?\s*(?:=|:|of)?\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE),
)

# Leading "1." or "2)" in a numbered list, which is formatting rather than data.
_LIST_MARKER = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)


def _grounding_caveats(text: str, question: str, steps: list[ToolCallRecord]) -> list[str]:
    """Flag measured values in the answer that appear in no tool result.

    A model that calls one tool and then invents the rest of its answer passes
    every other check here: it used the file, it produced fluent prose, and the
    fabricated figures sit beside real ones. Comparing the values it states
    against the values it was actually given is the cheapest way to catch that.

    Two deliberate asymmetries keep the false-positive rate down:

    * The grounded set is drawn from **every** tool call in the session, not
      just the most recent. A multi-step answer legitimately combines figures
      from several calls, and scoping it narrowly would flag all of them.
    * Only *measurements* are checked - decimals, figures with a unit, and N
      values. Counts, totals and list positions are bare integers the model
      derived for itself, and flagging those would bury the real signal.

    It remains a heuristic and is allowed to be wrong. It adds a caveat rather
    than suppressing the answer, and how often it fires - including how often
    it fires spuriously - is itself a way to compare one model against another.
    """
    if not steps:
        return []

    grounded = _numbers_in(question)
    for record in steps:
        grounded |= _numbers_in(record.result)
        grounded |= _numbers_in(str(record.arguments))

    stated = _measurements_in(_LIST_MARKER.sub("", text))
    ungrounded = sorted(stated - grounded)
    if not ungrounded:
        return []

    figures = ", ".join(_format_number(value) for value in ungrounded)
    return [
        f"These measured values appear in the answer but in no tool result: {figures}. "
        "Check them against the file before relying on them."
    ]


def _numbers_in(text: str) -> set[float]:
    """Every number in a string, used to build the grounded set."""
    return _floats(match.group() for match in _NUMBER.finditer(text))


def _measurements_in(text: str) -> set[float]:
    """Only the figures that read as measured values."""
    return _floats(match.group(1) for pattern in _MEASUREMENTS for match in pattern.finditer(text))


def _floats(tokens: Iterable[str]) -> set[float]:
    values: set[float] = set()
    for token in tokens:
        try:
            values.add(float(token))
        except ValueError:
            continue
    return values


def _format_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)


def _unique(items: list[str]) -> list[str]:
    """Deduplicate while preserving order: the same caveat can arise twice."""
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered

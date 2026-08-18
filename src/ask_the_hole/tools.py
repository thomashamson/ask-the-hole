"""The tools the agent may call, and the schemas it sees them through.

Each tool is a thin wrapper over one function in ``queries``. Argument models
are Pydantic, which does two jobs at once: ``model_json_schema()`` produces the
schema the model is shown, and ``model_validate()`` checks what it sends back.
The same machinery that refuses "N/A" in a 2DP column refuses a hallucinated
argument, and its error message is fed back to the model so it can correct
itself rather than the run collapsing.

Tools are deliberately narrow. A 7B model chooses reliably between a few
distinct simple tools and much less reliably fills in one complex argument
object, so each tool takes a handful of scalars.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ask_the_hole.dataset import Dataset
from ask_the_hole.legend import Material
from ask_the_hole.queries import (
    Datum,
    LocationQuery,
    describe_location,
    find_locations_with_material,
    find_spt_results,
)


class ToolResult(BaseModel):
    """What a tool call produced.

    ``text`` is what the model sees. ``caveats`` are machine-generated notes
    that the loop appends to the final answer whatever the model says, so a
    limitation cannot be lost in summarising.
    """

    text: str
    caveats: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class Tool:
    """One callable exposed to the model."""

    name: str
    description: str
    arguments: type[BaseModel]
    run: Callable[[Dataset, Any], ToolResult]

    def schema(self) -> dict[str, Any]:
        """The tool definition in the shape Ollama expects."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": inline_refs(self.arguments.model_json_schema()),
            },
        }


# --------------------------------------------------------------------------
# Argument models
# --------------------------------------------------------------------------
# extra="forbid" on every one: an invented argument produces a clear validation
# error the model can act on, rather than being silently ignored and leaving it
# to believe a filter was applied when it was not.


class NoArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DescribeLocationArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    loca_id: str = Field(description="Location identifier exactly as in the file, e.g. BH01.")


class FindMaterialArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    material: Material = Field(
        default=Material.ROCK,
        description="Material to look for: rock, soil, or unknown.",
    )
    above: float | None = Field(
        default=None,
        description=(
            "Restrict to strata above this figure. With datum=depth that means "
            "shallower than this many metres below ground. With datum=level it means "
            "higher than this level in mOD. Exclusive."
        ),
    )
    below: float | None = Field(
        default=None,
        description=(
            "Restrict to strata below this figure. With datum=depth that means deeper "
            "than this many metres below ground. With datum=level it means lower than "
            "this level in mOD. Exclusive."
        ),
    )
    datum: Datum = Field(
        default=Datum.DEPTH,
        description=(
            "Whether above/below are measured as depth below ground level (depth) or "
            "as level relative to the project datum in mOD (level). Use depth unless "
            "the question clearly asks about levels or mOD."
        ),
    )


class FindSptArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_n: int | None = Field(default=None, description="Minimum SPT N value, inclusive.")
    max_n: int | None = Field(default=None, description="Maximum SPT N value, inclusive.")
    above: float | None = Field(
        default=None, description="Only tests shallower than this depth in metres."
    )
    below: float | None = Field(
        default=None, description="Only tests deeper than this depth in metres."
    )


# --------------------------------------------------------------------------
# Implementations
# --------------------------------------------------------------------------


def _run_file_summary(dataset: Dataset, _arguments: NoArguments) -> ToolResult:
    lines = [f"File: {dataset.source.name}"]
    if dataset.project.project is not None:
        project = dataset.project.project
        lines.append(f"Project: {project.project_id} - {project.name or 'unnamed'}")
    counts = (
        ("LOCA locations", len(dataset.locations.locations)),
        ("GEOL strata", len(dataset.geology.rows)),
        ("SAMP samples", len(dataset.samples.rows)),
        ("ISPT tests", len(dataset.spt.rows)),
    )
    for label, count in counts:
        group = label.split()[0]
        if not dataset.has(group):
            lines.append(f"{label}: GROUP NOT PRESENT IN THIS FILE")
        else:
            lines.append(f"{label}: {count}")
    return ToolResult(text="\n".join(lines))


def _run_list_locations(dataset: Dataset, _arguments: NoArguments) -> ToolResult:
    if not dataset.locations.locations:
        return ToolResult(text="This file contains no locations.")

    lines = ["LOCA_ID | type | ground level (mOD) | final depth (m)"]
    for location in dataset.locations.locations:
        level = "unknown" if location.ground_level is None else f"{location.ground_level:.2f}"
        depth = "unknown" if location.final_depth is None else f"{location.final_depth:.2f}"
        lines.append(
            f"{location.loca_id} | {location.location_type or 'unknown'} | {level} | {depth}"
        )
    return ToolResult(text="\n".join(lines))


def _run_describe_location(dataset: Dataset, arguments: DescribeLocationArguments) -> ToolResult:
    profile = describe_location(dataset, arguments.loca_id)
    if profile is None:
        known = ", ".join(sorted(dataset.locations.ids)) or "none"
        return ToolResult(
            text=(
                f"There is no location called {arguments.loca_id!r} in this file. "
                f"The locations present are: {known}."
            )
        )

    level = "unknown" if profile.ground_level is None else f"{profile.ground_level:.2f}mOD"
    lines = [
        f"{profile.loca_id} ({profile.location_type or 'type unknown'})",
        f"Ground level: {level}",
        f"Final depth: "
        f"{'unknown' if profile.final_depth is None else f'{profile.final_depth:.2f}m'}",
        "",
        "Strata (depth below ground):",
    ]
    for stratum in profile.strata:
        base = "?" if stratum.base is None else f"{stratum.base:.2f}"
        lines.append(
            f"  {stratum.top:.2f}-{base}m [{stratum.material.value}] "
            f"{stratum.description or 'no description'}"
        )
    if profile.samples:
        lines.append(
            "Samples: " + ", ".join(f"{s.sample_type or '?'}@{s.top:.2f}m" for s in profile.samples)
        )
    if profile.spt:
        lines.append(
            "SPT: "
            + ", ".join(
                f"N={'unrecorded' if t.n_value is None else t.n_value}@{t.top:.2f}m"
                for t in profile.spt
            )
        )

    caveats = [f"{profile.loca_id}: {note}" for note in profile.notes]
    if profile.notes:
        lines.append("")
        lines.append("LIMITATIONS you must report: " + "; ".join(profile.notes))
    return ToolResult(text="\n".join(lines), caveats=caveats)


def _run_find_material(dataset: Dataset, arguments: FindMaterialArguments) -> ToolResult:
    result = find_locations_with_material(
        dataset,
        arguments.material,
        above=arguments.above,
        below=arguments.below,
        datum=arguments.datum,
    )
    return _render_query(result)


def _run_find_spt(dataset: Dataset, arguments: FindSptArguments) -> ToolResult:
    if not dataset.has("ISPT"):
        return ToolResult(
            text="This file has no ISPT group, so there are no SPT results to search.",
            caveats=["This file contains no SPT (ISPT) data at all."],
        )

    results = find_spt_results(
        dataset,
        min_n=arguments.min_n,
        max_n=arguments.max_n,
        above=arguments.above,
        below=arguments.below,
    )
    if not results:
        return ToolResult(text="No SPT results match those criteria.")

    lines = ["LOCA_ID | depth (m) | N value"]
    lines += [
        f"{test.loca_id} | {test.top:.2f} | "
        f"{'unrecorded' if test.n_value is None else test.n_value}"
        for test in results
    ]

    caveats: list[str] = []
    if (arguments.min_n is not None or arguments.max_n is not None) and any(
        test.n_value is None for test in dataset.spt.rows
    ):
        skipped = sum(1 for test in dataset.spt.rows if test.n_value is None)
        note = (
            f"{skipped} SPT test(s) have no recorded N value and were excluded from "
            "the N-value filter; they can neither satisfy nor fail it."
        )
        lines.append(f"NOTE: {note}")
        caveats.append(note)

    return ToolResult(text="\n".join(lines), caveats=caveats)


def _render_query(result: LocationQuery) -> ToolResult:
    """Turn a three-bucket result into text a model cannot easily misread."""
    lines = [
        f"Question: {result.question}",
        f"Measured by: {result.datum.value}",
        "",
        f"MATCHED ({len(result.matched)}):",
    ]
    lines += [f"  {f.loca_id}: {f.reason}" for f in result.matched] or ["  none"]

    lines.append(f"RULED OUT ({len(result.not_matched)}):")
    lines += [f"  {f.loca_id}: {f.reason}" for f in result.not_matched] or ["  none"]

    lines.append(f"UNDETERMINED ({len(result.undetermined)}):")
    lines += [f"  {f.loca_id}: {f.reason}" for f in result.undetermined] or ["  none"]

    caveats: list[str] = []
    if not result.is_complete:
        note = (
            f"For '{result.question}', {len(result.undetermined)} location(s) could not be "
            f"determined ({', '.join(f.loca_id for f in result.undetermined)}). "
            "The matched count is a minimum, not a total."
        )
        caveats.append(note)
        lines += [
            "",
            "IMPORTANT: this answer is INCOMPLETE. The matched count is a MINIMUM, "
            "not a total. You must tell the user which locations could not be "
            "determined and why.",
        ]

    return ToolResult(text="\n".join(lines), caveats=caveats)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

TOOLS: tuple[Tool, ...] = (
    Tool(
        name="file_summary",
        description=(
            "What this AGS file contains: the project, and how many locations, strata, "
            "samples and SPT tests it holds. Says explicitly when a group is absent. "
            "Call this first if you are unsure whether the file can answer a question."
        ),
        arguments=NoArguments,
        run=_run_file_summary,
    ),
    Tool(
        name="list_locations",
        description=(
            "List every location (borehole, trial pit, window sample) with its type, "
            "ground level and final depth."
        ),
        arguments=NoArguments,
        run=_run_list_locations,
    ),
    Tool(
        name="describe_location",
        description=(
            "Everything the file records about one location: its ground level, final "
            "depth, full geological log, samples and SPT results."
        ),
        arguments=DescribeLocationArguments,
        run=_run_describe_location,
    ),
    Tool(
        name="find_locations_with_material",
        description=(
            "Find which locations encountered rock or soil, optionally restricted to a "
            "depth or level band. Materials are classified from the standard GEOL_LEG "
            "legend code. Returns three groups: matched, ruled out, and undetermined "
            "(where the data cannot say either way)."
        ),
        arguments=FindMaterialArguments,
        run=_run_find_material,
    ),
    Tool(
        name="find_spt_results",
        description=(
            "Find Standard Penetration Test (SPT) results, optionally filtered by N "
            "value range and depth band. Higher N means stronger or denser ground."
        ),
        arguments=FindSptArguments,
        run=_run_find_spt,
    ),
)


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve Pydantic's $defs/$ref indirection into a flat schema.

    Pydantic factors enums out into ``$defs`` and points at them with ``$ref``.
    That is valid JSON Schema, but small models follow a flat schema far more
    reliably than one they have to dereference, so the definitions are inlined
    before the model ever sees them.
    """
    definitions = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, list):
            return [resolve(item) for item in node]
        if not isinstance(node, dict):
            return node

        # Pydantic wraps a $ref in allOf when the field also carries a default.
        if "allOf" in node and len(node["allOf"]) == 1:
            merged = {key: value for key, value in node.items() if key != "allOf"}
            return resolve({**node["allOf"][0], **merged})

        reference = node.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/$defs/"):
            target = definitions.get(reference.rsplit("/", 1)[-1], {})
            merged = {key: value for key, value in node.items() if key != "$ref"}
            return resolve({**target, **merged})

        return {key: resolve(value) for key, value in node.items()}

    return resolve(schema)


def call_tool(dataset: Dataset, name: str, raw_arguments: dict[str, Any]) -> ToolResult:
    """Validate the model's arguments and run the tool, or explain the failure.

    Validation errors come back as ToolResults rather than exceptions so the
    model sees what it got wrong and can retry. A hallucinated tool name is
    answered with the list of real ones for the same reason.
    """
    tool = next((candidate for candidate in TOOLS if candidate.name == name), None)
    if tool is None:
        available = ", ".join(candidate.name for candidate in TOOLS)
        return ToolResult(text=f"There is no tool called {name!r}. Available tools: {available}.")

    try:
        arguments = tool.arguments.model_validate(raw_arguments)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'arguments'}: {error['msg']}"
            for error in exc.errors()
        )
        return ToolResult(text=f"Invalid arguments for {name}: {problems}. Correct them and retry.")

    return tool.run(dataset, arguments)


def tool_schemas() -> list[dict[str, Any]]:
    """Every tool definition, in the shape Ollama expects."""
    return [tool.schema() for tool in TOOLS]

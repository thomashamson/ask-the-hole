"""Reading AGS 4.x files, and the machinery shared by every group parser.

One module per AGS group builds on this: see ``locations``, ``geology`` and
``abbreviations``. Group-specific rules live there; anything true of all groups
lives here.

Three levels of severity, deliberately kept distinct:

* **File errors** raise. An unreadable file, or a missing group that the caller
  asked for, leaves nothing useful to return.
* **Row errors** discard one row and are collected. Only a broken *identity*
  qualifies - a missing or duplicate LOCA_ID, a stratum with no depth - because
  such a row cannot be named, placed, or joined to anything.
* **Field warnings** discard one *value* and are collected. A type violation
  such as "N/A" in a 2DP column loses that field but keeps the row, so a hole
  with a bad ground level is still answerable on depth and coordinates.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Any, Self

import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError
from python_ags4 import AGS4

from ask_the_hole.models import AgsRow, FieldWarning, LocatedRow


class AgsError(Exception):
    """The AGS file cannot be used at all."""


class MissingGroupError(AgsError):
    """A required AGS group is absent from the file."""


class RowError(BaseModel):
    """A data row that was discarded entirely, and why."""

    row_number: int
    loca_id: str | None
    message: str


def read_ags_tables(path: Path) -> dict[str, pd.DataFrame]:
    """Read an AGS file into one raw string DataFrame per group.

    Values are left as strings on purpose. python-ags4 offers convert_to_numeric,
    but pandas coercion turns a bad value into NaN silently, and NaN passes a
    ``float | None`` field. Letting Pydantic do the conversion means bad values
    are detected, which is what warning collection depends on.
    """
    try:
        tables, _headings = AGS4.AGS4_to_dataframe(str(path))
    except Exception as exc:  # python-ags4 raises a variety of parse errors
        msg = f"Could not read AGS file {path}: {exc}"
        raise AgsError(msg) from exc

    if not tables:
        msg = f"No AGS groups found in {path}. Is this an AGS 4.x file?"
        raise AgsError(msg)
    return tables


def require_group(tables: dict[str, pd.DataFrame], group: str) -> pd.DataFrame:
    """Fetch a group, or explain which groups the file does have."""
    if group not in tables:
        msg = f"File has no {group} group. Groups present: {', '.join(sorted(tables))}"
        raise MissingGroupError(msg)
    return tables[group]


def data_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop the UNIT and TYPE rows, keeping only real data.

    python-ags4 returns those two as ordinary rows, distinguished only by the
    value in the HEADING column.
    """
    return frame[frame["HEADING"] == "DATA"]


def present_headings(frame: pd.DataFrame, model: type[AgsRow]) -> list[str]:
    """The headings this model reads that the file actually contains.

    Intersecting the two lets us tolerate files that omit headings we model,
    and drops every heading we chose not to model.
    """
    return [heading for heading in model.headings() if heading in frame.columns]


def extract_units(frame: pd.DataFrame, headings: list[str]) -> dict[str, str]:
    """Pull the group's UNIT row into a heading -> unit mapping.

    Units belong to the group, not to any one row, so they live on the result
    rather than on the row model. The distinction they record is load-bearing:
    LOCA_GL is a *level* (metres relative to a datum, so it can be negative and
    is not comparable to a depth), while LOCA_FDEP and GEOL_TOP are *depths
    below* that level. Nothing in the values themselves says which is which -
    only the UNIT row does. A file that omits the row yields an empty mapping
    rather than an error, because units are informative, not required.
    """
    unit_rows = frame[frame["HEADING"] == "UNIT"]
    if unit_rows.empty:
        return {}

    row = unit_rows.iloc[0]
    # Blank units are dropped: an X-type column such as GEOL_DESC has no unit,
    # and recording "" for it would be noise rather than information.
    return {heading: str(row[heading]).strip() for heading in headings if str(row[heading]).strip()}


def validate_row[RowT: AgsRow](
    model: type[RowT],
    payload: dict[str, Any],
) -> tuple[RowT | None, list[FieldWarning], str | None]:
    """Validate one row, returning it alongside its warnings, or a failure reason.

    A fresh list is handed to Pydantic as validation context; validators append
    FieldWarnings to it. Warnings are returned rather than accumulated globally
    so the caller can discard them along with a row it decides to reject - a
    discarded row must not leave warnings behind for something that is not in
    the output.
    """
    warnings: list[FieldWarning] = []
    try:
        row = model.model_validate(payload, context=warnings)
    except ValidationError as exc:
        return None, [], describe_validation_error(exc)
    return row, warnings, None


def describe_validation_error(exc: ValidationError) -> str:
    """Flatten a Pydantic ValidationError into one readable line."""
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or 'row'}: {error['msg']}"
        for error in exc.errors()
    )


class ParsedGroup[RowT: LocatedRow](BaseModel):
    """The outcome of parsing one depth-logged group joined to LOCA.

    A Pydantic *generic* model: ``ParsedGroup[Stratum]`` and
    ``ParsedGroup[Sample]`` are distinct types validating distinct row models,
    from one definition. GEOL, SAMP and ISPT are structurally identical - rows
    logged at a depth in a hole - so they share this rather than repeating it.
    """

    # cached_property stores its result in the instance __dict__, which Pydantic
    # models have, so the index is built once on first access rather than being
    # a second copy of the data kept in sync by hand.
    model_config = ConfigDict(ignored_types=(cached_property,))

    rows: list[RowT]
    units: dict[str, str]
    warnings: list[FieldWarning]
    errors: list[RowError]

    @cached_property
    def by_location(self) -> dict[str, list[RowT]]:
        """Rows grouped by hole, each list ordered by depth.

        Ordering is imposed here rather than trusted from the file: AGS does not
        guarantee row order, and every depth question depends on it.
        """
        index: dict[str, list[RowT]] = {}
        for row in self.rows:
            index.setdefault(row.loca_id, []).append(row)
        for rows in index.values():
            rows.sort(key=lambda row: row.top)
        return index

    def for_location(self, loca_id: str) -> list[RowT]:
        """Rows logged in one hole, shallowest first. Empty if none."""
        return self.by_location.get(loca_id, [])

    @classmethod
    def empty(cls) -> Self:
        """A result for a group the file does not contain.

        Absence is not an error: a site with no samples is a real site.
        Returning an empty result rather than raising is what lets the tool
        answer "there is no SAMP data in this file" instead of failing to load
        it. Returns cls, so a subclass gets its own type back.
        """
        return cls(rows=[], units={}, warnings=[], errors=[])


def parse_located_group[RowT: LocatedRow, ResultT: ParsedGroup](
    tables: dict[str, pd.DataFrame],
    group: str,
    model: type[RowT],
    *,
    known_ids: set[str],
    result_type: type[ResultT],
) -> ResultT:
    """Validate every DATA row of a depth-logged group into ``model``.

    ``known_ids`` is the set of LOCA_IDs that survived parsing LOCA. A row
    referencing anything else is an orphan: it cannot be placed, so it is
    discarded rather than left to leak into totals and averages. Passing the IDs
    in, rather than reading LOCA here, keeps this function honest about its one
    dependency.

    The headings a file must supply are derived from the model's identity
    fields, so each group states its own requirements once, in the place that
    already defines what makes a row salvageable.
    """
    frame = require_group(tables, group)
    data = data_rows(frame)

    for field_name in sorted(model.identity_fields):
        heading = model.heading_for(field_name)
        if heading not in data.columns:
            msg = f"{group} group has no {heading} heading, so rows cannot be placed"
            raise MissingGroupError(msg)

    headings = present_headings(data, model)
    units = extract_units(frame, headings)

    rows: list[RowT] = []
    warnings: list[FieldWarning] = []
    errors: list[RowError] = []

    for row_number, (_index, raw) in enumerate(data.iterrows(), start=1):
        raw_id = str(raw["LOCA_ID"]).strip()

        # Referential integrity is a property of the file as a whole, so like
        # uniqueness it cannot live on the model.
        if raw_id and raw_id not in known_ids:
            errors.append(
                RowError(
                    row_number=row_number,
                    loca_id=raw_id,
                    message=(
                        f"{group} row references LOCA_ID {raw_id!r}, "
                        "which is not a parsed location; row discarded"
                    ),
                )
            )
            continue

        payload = {heading: raw[heading] for heading in headings}
        row, row_warnings, failure = validate_row(model, payload)

        if row is None:
            errors.append(
                RowError(
                    row_number=row_number,
                    loca_id=raw_id or None,
                    message=failure or "row could not be validated",
                )
            )
            continue

        rows.append(row)
        warnings.extend(row_warnings)

    return result_type(rows=rows, units=units, warnings=warnings, errors=errors)

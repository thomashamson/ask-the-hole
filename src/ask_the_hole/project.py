"""Parsing the AGS ``PROJ`` group.

PROJ is the odd one out: exactly one row per file, describing the file itself
rather than anything observed in the ground. It gets its own result type rather
than reusing ParsedGroup, which is built for many rows joined to a location.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel

from ask_the_hole.models import FieldWarning, Project
from ask_the_hole.parser import (
    RowError,
    data_rows,
    present_headings,
    require_group,
    validate_row,
)


class ParsedProject(BaseModel):
    """The outcome of parsing the PROJ group.

    ``project`` is None when the group exists but its single row could not be
    validated - a file can be structurally present and still fail to say what
    project it belongs to.
    """

    project: Project | None
    warnings: list[FieldWarning]
    errors: list[RowError]

    @classmethod
    def empty(cls) -> ParsedProject:
        """A result for a file with no PROJ group."""
        return cls(project=None, warnings=[], errors=[])


def parse_project(tables: dict[str, pd.DataFrame]) -> ParsedProject:
    """Validate the single DATA row of the PROJ group into a Project.

    AGS requires exactly one PROJ row. Extra rows are reported and ignored
    rather than raising: the first row still tells us what the file is, and
    losing that to a strictness rule would help nobody.
    """
    frame = require_group(tables, "PROJ")
    data = data_rows(frame)
    headings = present_headings(data, Project)

    if data.empty:
        return ParsedProject(
            project=None,
            warnings=[],
            errors=[RowError(row_number=0, loca_id=None, message="PROJ group has no data row")],
        )

    errors: list[RowError] = []
    if len(data) > 1:
        errors.append(
            RowError(
                row_number=2,
                loca_id=None,
                message=(
                    f"PROJ group has {len(data)} rows; AGS allows one. "
                    "The first is used, the rest ignored."
                ),
            )
        )

    raw = data.iloc[0]
    payload = {heading: raw[heading] for heading in headings}
    project, warnings, failure = validate_row(Project, payload)

    if project is None:
        errors.append(RowError(row_number=1, loca_id=None, message=failure or "invalid PROJ row"))

    return ParsedProject(project=project, warnings=warnings, errors=errors)

"""Read an AGS 4.x file and turn its groups into validated Pydantic models.

Three levels of severity, deliberately kept distinct:

* **File errors** raise. An unreadable file or a missing LOCA group leaves
  nothing useful to return.
* **Row errors** discard one row and are collected. Only a missing or duplicate
  LOCA_ID qualifies: without a usable identity a row cannot be referred to,
  joined to GEOL/SAMP/ISPT, or reported on.
* **Field warnings** discard one *value* and are collected. A type violation
  such as "N/A" in a 2DP column loses that field but keeps the location, so a
  hole with a bad ground level is still answerable on depth and coordinates.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ValidationError
from python_ags4 import AGS4

from ask_the_hole.models import FieldWarning, Location

# Derived from the model rather than repeated here, so the set of AGS headings
# we keep can never drift out of step with the fields we actually parse.
LOCA_HEADINGS: tuple[str, ...] = tuple(
    field.alias or name for name, field in Location.model_fields.items()
)


class AgsError(Exception):
    """The AGS file cannot be used at all."""


class MissingGroupError(AgsError):
    """A required AGS group is absent from the file."""


class RowError(BaseModel):
    """A data row that was discarded entirely, and why."""

    row_number: int
    loca_id: str | None
    message: str


class ParsedLocations(BaseModel):
    """The outcome of parsing the LOCA group.

    ``warnings`` describe fields that were dropped from locations that were
    still returned; ``errors`` describe rows that were not returned at all.
    ``units`` carries the group's declared units, keyed by AGS heading.
    """

    locations: list[Location]
    units: dict[str, str]
    warnings: list[FieldWarning]
    errors: list[RowError]


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


def parse_locations(tables: dict[str, pd.DataFrame]) -> ParsedLocations:
    """Validate every DATA row of the LOCA group into a Location."""
    if "LOCA" not in tables:
        msg = f"File has no LOCA group. Groups present: {', '.join(sorted(tables))}"
        raise MissingGroupError(msg)

    frame = tables["LOCA"]

    # python-ags4 returns the UNIT and TYPE rows as ordinary rows, tagged in the
    # HEADING column. Keep only the real data.
    data = frame[frame["HEADING"] == "DATA"]

    if "LOCA_ID" not in data.columns:
        msg = "LOCA group has no LOCA_ID heading, so locations cannot be identified"
        raise MissingGroupError(msg)

    # Intersecting with the file's actual columns lets us tolerate files that
    # omit headings we model, and drops every heading we chose not to model.
    present = [heading for heading in LOCA_HEADINGS if heading in data.columns]

    units = _extract_units(frame, present)

    locations: list[Location] = []
    warnings: list[FieldWarning] = []
    errors: list[RowError] = []
    seen_ids: set[str] = set()

    for row_number, (_index, row) in enumerate(data.iterrows(), start=1):
        raw_id = str(row["LOCA_ID"]).strip()

        # Uniqueness is a property of the group, not of a single row, so it
        # cannot live on the model and has to be checked here.
        if raw_id and raw_id in seen_ids:
            errors.append(
                RowError(
                    row_number=row_number,
                    loca_id=raw_id,
                    message="duplicate LOCA_ID; first occurrence kept, this row discarded",
                )
            )
            continue

        payload = {heading: row[heading] for heading in present}

        # A fresh list per row is handed to Pydantic as validation context.
        # Validators append FieldWarnings to it. Only merge it into the result
        # if the row itself survives, so a discarded row cannot leave warnings
        # behind for a location that is not there.
        row_warnings: list[FieldWarning] = []
        try:
            location = Location.model_validate(payload, context=row_warnings)
        except ValidationError as exc:
            errors.append(
                RowError(
                    row_number=row_number,
                    loca_id=raw_id or None,
                    message="; ".join(_describe(exc)),
                )
            )
            continue

        seen_ids.add(location.loca_id)
        locations.append(location)
        warnings.extend(row_warnings)

    return ParsedLocations(locations=locations, units=units, warnings=warnings, errors=errors)


def _extract_units(frame: pd.DataFrame, headings: list[str]) -> dict[str, str]:
    """Pull the group's UNIT row into a heading -> unit mapping.

    Units belong to the group, not to any one row, so they live on the result
    rather than on Location. The distinction they record is load-bearing:
    LOCA_GL is a *level* (metres relative to a datum, so it can be negative and
    is not comparable to a depth), while LOCA_FDEP is a *depth below* that
    level. Nothing in the values themselves says which is which - only the UNIT
    row does. A file that omits the row yields an empty mapping rather than an
    error, because units are informative, not required.
    """
    unit_rows = frame[frame["HEADING"] == "UNIT"]
    if unit_rows.empty:
        return {}

    row = unit_rows.iloc[0]
    # Blank units are dropped: an X-type column such as LOCA_REM has no unit,
    # and recording "" for it would be noise rather than information.
    return {heading: str(row[heading]).strip() for heading in headings if str(row[heading]).strip()}


def _describe(exc: ValidationError) -> list[str]:
    """Flatten a Pydantic ValidationError into readable one-line messages."""
    return [
        f"{'.'.join(str(part) for part in error['loc']) or 'row'}: {error['msg']}"
        for error in exc.errors()
    ]

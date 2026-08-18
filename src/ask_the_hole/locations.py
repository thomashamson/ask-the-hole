"""Parsing the AGS ``LOCA`` group into Location models."""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel

from ask_the_hole.models import FieldWarning, Location
from ask_the_hole.parser import (
    MissingGroupError,
    RowError,
    data_rows,
    extract_units,
    present_headings,
    require_group,
    validate_row,
)


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

    @property
    def ids(self) -> set[str]:
        """Every location identifier that survived parsing.

        Other groups join to LOCA on this, so it is the authority on which
        LOCA_IDs actually exist in the parsed data.
        """
        return {location.loca_id for location in self.locations}


def parse_locations(tables: dict[str, pd.DataFrame]) -> ParsedLocations:
    """Validate every DATA row of the LOCA group into a Location."""
    frame = require_group(tables, "LOCA")
    data = data_rows(frame)

    if "LOCA_ID" not in data.columns:
        msg = "LOCA group has no LOCA_ID heading, so locations cannot be identified"
        raise MissingGroupError(msg)

    headings = present_headings(data, Location)
    units = extract_units(frame, headings)

    locations: list[Location] = []
    warnings: list[FieldWarning] = []
    errors: list[RowError] = []
    seen_ids: set[str] = set()

    for row_number, (_index, row) in enumerate(data.iterrows(), start=1):
        raw_id = str(row["LOCA_ID"]).strip()

        # Uniqueness is a property of the group, not of a single row, so it
        # cannot live on the model and has to be checked here. The first
        # occurrence is kept: comparison is exact rather than case-insensitive,
        # because wrongly merging two distinct holes is worse than missing a
        # duplicate that differs only in case.
        if raw_id and raw_id in seen_ids:
            errors.append(
                RowError(
                    row_number=row_number,
                    loca_id=raw_id,
                    message="duplicate LOCA_ID; first occurrence kept, this row discarded",
                )
            )
            continue

        payload = {heading: row[heading] for heading in headings}
        location, row_warnings, failure = validate_row(Location, payload)

        if location is None:
            errors.append(
                RowError(
                    row_number=row_number,
                    loca_id=raw_id or None,
                    message=failure or "row could not be validated",
                )
            )
            continue

        seen_ids.add(location.loca_id)
        locations.append(location)
        warnings.extend(row_warnings)

    return ParsedLocations(locations=locations, units=units, warnings=warnings, errors=errors)

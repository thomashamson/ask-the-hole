"""Parsing the AGS ``GEOL`` group into Stratum models.

Strata are held as a flat list mirroring the group row-for-row, exactly as
Location mirrors LOCA, with an index for access by hole. Nesting them inside
Location would read more naturally but would stop the models being a faithful
image of the file, and SAMP and ISPT will join the same way.
"""

from __future__ import annotations

from functools import cached_property

import pandas as pd
from pydantic import BaseModel, ConfigDict

from ask_the_hole.abbreviations import Abbreviations
from ask_the_hole.models import FieldWarning, Stratum
from ask_the_hole.parser import (
    MissingGroupError,
    RowError,
    data_rows,
    extract_units,
    present_headings,
    require_group,
    validate_row,
)


class ParsedGeology(BaseModel):
    """The outcome of parsing the GEOL group."""

    # cached_property stores its result in the instance __dict__, which Pydantic
    # models have, so the index is built once on first access rather than being
    # a second copy of the data kept in sync by hand.
    model_config = ConfigDict(ignored_types=(cached_property,))

    strata: list[Stratum]
    units: dict[str, str]
    warnings: list[FieldWarning]
    errors: list[RowError]

    @cached_property
    def by_location(self) -> dict[str, list[Stratum]]:
        """Strata grouped by hole, each list ordered by depth.

        Ordering is imposed here rather than trusted from the file: AGS does not
        guarantee row order, and every depth question depends on it.
        """
        index: dict[str, list[Stratum]] = {}
        for stratum in self.strata:
            index.setdefault(stratum.loca_id, []).append(stratum)
        for strata in index.values():
            strata.sort(key=lambda stratum: stratum.top)
        return index

    def for_location(self, loca_id: str) -> list[Stratum]:
        """Strata logged in one hole, shallowest first. Empty if none."""
        return self.by_location.get(loca_id, [])


def parse_geology(
    tables: dict[str, pd.DataFrame],
    *,
    known_ids: set[str],
    abbreviations: Abbreviations | None = None,
) -> ParsedGeology:
    """Validate every DATA row of the GEOL group into a Stratum.

    ``known_ids`` is the set of LOCA_IDs that survived parsing LOCA. A stratum
    referencing anything else is an orphan: it cannot be placed, so it is
    discarded rather than left to leak into totals and averages. Passing the
    IDs in, rather than reading LOCA here, keeps this function honest about its
    one dependency.

    ``abbreviations`` is optional. When supplied, GEOL_GEOL codes with no entry
    in the file's own ABBR group are reported once each - once per *code*, not
    once per row, so a file with 200 strata sharing an undefined code produces
    one warning rather than 200.
    """
    frame = require_group(tables, "GEOL")
    data = data_rows(frame)

    for required in ("LOCA_ID", "GEOL_TOP"):
        if required not in data.columns:
            msg = f"GEOL group has no {required} heading, so strata cannot be placed"
            raise MissingGroupError(msg)

    headings = present_headings(data, Stratum)
    units = extract_units(frame, headings)

    strata: list[Stratum] = []
    warnings: list[FieldWarning] = []
    errors: list[RowError] = []

    for row_number, (_index, row) in enumerate(data.iterrows(), start=1):
        raw_id = str(row["LOCA_ID"]).strip()

        # Referential integrity is a property of the file as a whole, so like
        # uniqueness it cannot live on the model.
        if raw_id and raw_id not in known_ids:
            errors.append(
                RowError(
                    row_number=row_number,
                    loca_id=raw_id,
                    message=(
                        f"GEOL row references LOCA_ID {raw_id!r}, "
                        "which is not a parsed location; row discarded"
                    ),
                )
            )
            continue

        payload = {heading: row[heading] for heading in headings}
        stratum, row_warnings, failure = validate_row(Stratum, payload)

        if stratum is None:
            errors.append(
                RowError(
                    row_number=row_number,
                    loca_id=raw_id or None,
                    message=failure or "row could not be validated",
                )
            )
            continue

        strata.append(stratum)
        warnings.extend(row_warnings)

    if abbreviations is not None:
        warnings.extend(_undecodable_codes(strata, abbreviations))

    return ParsedGeology(strata=strata, units=units, warnings=warnings, errors=errors)


def _undecodable_codes(
    strata: list[Stratum],
    abbreviations: Abbreviations,
) -> list[FieldWarning]:
    """One warning per distinct GEOL_GEOL code the file never defines.

    Reported against the whole group rather than a single hole, because the gap
    is in the file's legend, not in any one stratum. loca_id is None for the
    same reason.
    """
    unknown = sorted(
        {
            stratum.geology_code
            for stratum in strata
            if stratum.geology_code and not abbreviations.defines("GEOL_GEOL", stratum.geology_code)
        }
    )
    return [
        FieldWarning(
            loca_id=None,
            heading="GEOL_GEOL",
            value=code,
            message="no ABBR entry in this file; raw code used",
        )
        for code in unknown
    ]

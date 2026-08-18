"""Parsing the AGS ``GEOL`` group into Stratum models.

Strata are held as a flat list mirroring the group row-for-row, exactly as
Location mirrors LOCA, with an index for access by hole. Nesting them inside
Location would read more naturally but would stop the models being a faithful
image of the file.
"""

from __future__ import annotations

import pandas as pd

from ask_the_hole.abbreviations import Abbreviations
from ask_the_hole.models import FieldWarning, Stratum
from ask_the_hole.parser import ParsedGroup, parse_located_group


class ParsedGeology(ParsedGroup[Stratum]):
    """The outcome of parsing the GEOL group."""

    @property
    def strata(self) -> list[Stratum]:
        """The parsed layers, in file order. Domain name for ``rows``."""
        return self.rows


def parse_geology(
    tables: dict[str, pd.DataFrame],
    *,
    known_ids: set[str],
    abbreviations: Abbreviations | None = None,
) -> ParsedGeology:
    """Validate every DATA row of the GEOL group into a Stratum.

    ``abbreviations`` is optional. When supplied, GEOL_GEOL codes with no entry
    in the file's own ABBR group are reported once each - once per *code*, not
    once per row, so a file with 200 strata sharing an undefined code produces
    one warning rather than 200.
    """
    result = parse_located_group(
        tables,
        "GEOL",
        Stratum,
        known_ids=known_ids,
        result_type=ParsedGeology,
    )

    if abbreviations is not None:
        result.warnings.extend(_undecodable_codes(result.strata, abbreviations))
    return result


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

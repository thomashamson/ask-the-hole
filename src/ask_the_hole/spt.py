"""Parsing the AGS ``ISPT`` group into Standard Penetration Test results."""

from __future__ import annotations

import pandas as pd

from ask_the_hole.models import InSituTest
from ask_the_hole.parser import ParsedGroup, parse_located_group


class ParsedSpt(ParsedGroup[InSituTest]):
    """The outcome of parsing the ISPT group."""

    @property
    def tests(self) -> list[InSituTest]:
        """The parsed tests, in file order. Domain name for ``rows``."""
        return self.rows


def parse_spt(
    tables: dict[str, pd.DataFrame],
    *,
    known_ids: set[str],
) -> ParsedSpt:
    """Validate every DATA row of the ISPT group into an InSituTest."""
    return parse_located_group(
        tables,
        "ISPT",
        InSituTest,
        known_ids=known_ids,
        result_type=ParsedSpt,
    )

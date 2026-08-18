"""Parsing the AGS ``SAMP`` group into Sample models."""

from __future__ import annotations

import pandas as pd

from ask_the_hole.models import Sample
from ask_the_hole.parser import ParsedGroup, parse_located_group


class ParsedSamples(ParsedGroup[Sample]):
    """The outcome of parsing the SAMP group."""

    @property
    def samples(self) -> list[Sample]:
        """The parsed samples, in file order. Domain name for ``rows``."""
        return self.rows


def parse_samples(
    tables: dict[str, pd.DataFrame],
    *,
    known_ids: set[str],
) -> ParsedSamples:
    """Validate every DATA row of the SAMP group into a Sample."""
    return parse_located_group(
        tables,
        "SAMP",
        Sample,
        known_ids=known_ids,
        result_type=ParsedSamples,
    )

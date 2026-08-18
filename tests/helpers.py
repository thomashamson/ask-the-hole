"""Shared test helpers."""

from pathlib import Path

import pandas as pd

from ask_the_hole.parser import read_ags_tables

FIXTURES = Path(__file__).parent / "fixtures"


def tables(name: str) -> dict[str, pd.DataFrame]:
    """Read one of the .ags fixtures."""
    return read_ags_tables(FIXTURES / f"{name}.ags")


def group_frame(rows: list[dict[str, str]]) -> pd.DataFrame:
    """Build a group frame shaped the way python-ags4 hands one back.

    Only DATA rows: the UNIT and TYPE rows are optional as far as the parser is
    concerned, so tests that do not care about units can leave them out.
    """
    return pd.DataFrame([{"HEADING": "DATA", **row} for row in rows])

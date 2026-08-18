"""The AGS ``ABBR`` group: the file's own legend for its coded values.

ABBR is metadata, not observations, so it gets a lookup table rather than a row
model. It is file-supplied and therefore only as good as whoever wrote it: a
file may define anything, define nothing, or omit the group entirely. Codes are
decoded where a mapping exists and passed through raw where it does not, and
the gap is reported. Nothing here carries a built-in geology dictionary - a
hard-coded fallback would make the tool quietly wrong on an unfamiliar file,
which is worse than admitting it does not know.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel

from ask_the_hole.parser import data_rows


class Abbreviations(BaseModel):
    """Decoded value lookups, keyed by the heading they apply to.

    ``by_heading["GEOL_GEOL"]["CK"]`` is "Chalk Group" in a file that says so.
    """

    by_heading: dict[str, dict[str, str]]

    def describe(self, heading: str, code: str | None) -> str | None:
        """The description for a code, falling back to the code itself.

        Returning the raw code rather than None keeps output readable when a
        file's legend is incomplete: "CK" is less useful than "Chalk Group" but
        far more useful than a blank. Use ``defines`` to tell the two apart.
        """
        if code is None:
            return None
        return self.by_heading.get(heading, {}).get(code, code)

    def defines(self, heading: str, code: str) -> bool:
        """Whether the file actually supplies a mapping for this code."""
        return code in self.by_heading.get(heading, {})


def parse_abbreviations(tables: dict[str, pd.DataFrame]) -> Abbreviations:
    """Read the ABBR group, if the file has one.

    A missing ABBR group is not an error. It means codes stay raw, which the
    caller finds out through ``defines`` rather than through an exception.
    """
    frame = tables.get("ABBR")
    if frame is None:
        return Abbreviations(by_heading={})

    required = {"ABBR_HDNG", "ABBR_CODE", "ABBR_DESC"}
    if not required.issubset(frame.columns):
        return Abbreviations(by_heading={})

    by_heading: dict[str, dict[str, str]] = {}
    for _index, row in data_rows(frame).iterrows():
        heading = str(row["ABBR_HDNG"]).strip()
        code = str(row["ABBR_CODE"]).strip()
        description = str(row["ABBR_DESC"]).strip()
        if not heading or not code or not description:
            continue
        by_heading.setdefault(heading, {})[code] = description

    return Abbreviations(by_heading=by_heading)

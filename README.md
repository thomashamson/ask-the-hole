# Ask the Hole

A CLI that answers natural-language questions about an AGS 4.x geotechnical
data file, using a local LLM with tool-calling. Runs fully offline.

## Status

Early. Parses the `LOCA` and `GEOL` groups into validated Pydantic models and
prints them, decoding coded values through the file's own `ABBR` group.
No LLM, no agent loop yet.

## Requirements

- Python 3.12 (managed by `uv`)
- [uv](https://docs.astral.sh/uv/)

## Setup

```
uv sync
```

## Usage

```
uv run ask-the-hole locations path/to/file.ags   # LOCA: one row per hole
uv run ask-the-hole strata    path/to/file.ags   # GEOL: one row per layer
```

## Layout

`parser.py` reads AGS files and holds everything true of every group. One
module per group builds on it — `locations.py`, `geology.py` — so `SAMP` and
`ISPT` slot in the same way. `models.py` holds the Pydantic row models, which
share validation behaviour through an `AgsRow` base and differ only in their
fields and their declared `identity_fields`.

Strata are a flat list mirroring the `GEOL` group row-for-row, indexed by hole
via `ParsedGeology.for_location()`, rather than nested inside `Location`. That
keeps every model a faithful image of the file it came from.

## How bad data is handled

Three severities, kept deliberately distinct:

| Severity          | Cause                                              | Effect                         |
| ----------------- | -------------------------------------------------- | ------------------------------ |
| **File error**    | Unreadable file, no LOCA group, no LOCA_ID heading   | Raises; nothing is returned    |
| **Row error**     | Broken identity — missing/duplicate `LOCA_ID`, or a stratum with no `GEOL_TOP`; also a `GEOL` row referencing a hole that does not exist | That row is discarded |
| **Field warning** | Type violation, e.g. `"N/A"` in a 2DP column         | That value becomes `None`      |

An *empty* AGS field is not a problem. AGS has no NULL, so `""` is the correct
way to record "not measured"; it becomes `None` silently. A non-empty value that
violates its declared TYPE is a problem, and is reported.

## Units

The group's `UNIT` row is captured onto the parse result, keyed by AGS heading,
and shown beneath each column header. This is not cosmetic: `LOCA_GL` is a
*level* relative to a datum and `LOCA_FDEP` is a *depth below* it. The numbers
alone do not say which is which — only the `UNIT` row does.

## Coded values

`GEOL_GEOL` codes are decoded through the file's own `ABBR` group. `ABBR` is
file-supplied and only as good as whoever wrote it, so a code with no entry
falls back to the raw code and is reported — once per distinct code, not once
per row. There is deliberately **no built-in geology dictionary** as a backstop:
guessing would make the tool quietly wrong on an unfamiliar file, which is worse
than saying it does not know.

## Test fixtures

Synthetic, hand-authored. No client or BGS data.

- `clean-site.ags` — 4 boreholes, full PROJ/TRAN/LOCA/GEOL/SAMP/ISPT. Zero errors.
- `sparse-site.ags` — 3 trial pits, GEOL only. No SAMP, no ISPT. Zero errors.
- `messy-export.ags` — **deliberately non-compliant; do not "fix" it.** Contains
  AGS Rule 8 type violations (`"N/A"` in `LOCA_GL`, `"-"` in both coordinates of
  TP02) alongside legitimately empty fields. Must yield 5 locations, 3 warnings,
  0 errors.

## Development

```
uv run pytest
uv run ruff check .
uv run ruff format .
```

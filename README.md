# Ask the Hole

A CLI that answers natural-language questions about an AGS 4.x geotechnical
data file, using a local LLM with tool-calling. Runs fully offline.

## Status

Early. Currently parses the `LOCA` group into validated Pydantic models and
prints them. No LLM, no agent loop yet.

## Requirements

- Python 3.12 (managed by `uv`)
- [uv](https://docs.astral.sh/uv/)

## Setup

```
uv sync
```

## Usage

```
uv run ask-the-hole locations path/to/file.ags
```

## How bad data is handled

Three severities, kept deliberately distinct:

| Severity          | Cause                                              | Effect                         |
| ----------------- | -------------------------------------------------- | ------------------------------ |
| **File error**    | Unreadable file, no LOCA group, no LOCA_ID heading   | Raises; nothing is returned    |
| **Row error**     | Missing or duplicate `LOCA_ID`                       | That row is discarded          |
| **Field warning** | Type violation, e.g. `"N/A"` in a 2DP column         | That value becomes `None`      |

An *empty* AGS field is not a problem. AGS has no NULL, so `""` is the correct
way to record "not measured"; it becomes `None` silently. A non-empty value that
violates its declared TYPE is a problem, and is reported.

## Units

The group's `UNIT` row is captured onto the parse result, keyed by AGS heading,
and shown beneath each column header. This is not cosmetic: `LOCA_GL` is a
*level* relative to a datum and `LOCA_FDEP` is a *depth below* it. The numbers
alone do not say which is which — only the `UNIT` row does.

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

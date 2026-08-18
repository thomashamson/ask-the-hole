# Ask the Hole

A CLI that answers natural-language questions about an AGS 4.x geotechnical
data file, using a local LLM with tool-calling. Runs fully offline.

## Status

Parses all five groups — `PROJ`, `LOCA`, `GEOL`, `SAMP`, `ISPT` — into
validated Pydantic models and prints them, decoding coded values through the
file's own `ABBR` group and classifying strata as rock or soil from the standard
`GEOL_LEG` bands, with a deterministic query layer over the result, and
answers natural-language questions through a local model with tool-calling.

## Requirements

- Python 3.12 (managed by `uv`)
- [uv](https://docs.astral.sh/uv/)

## Setup

```
uv sync
```

## Usage

```
uv run ask-the-hole summary   path/to/file.ags   # what this file contains
uv run ask-the-hole project   path/to/file.ags   # PROJ
uv run ask-the-hole locations path/to/file.ags   # LOCA: one row per hole
uv run ask-the-hole strata    path/to/file.ags   # GEOL: one row per layer
uv run ask-the-hole samples   path/to/file.ags   # SAMP
uv run ask-the-hole spt       path/to/file.ags   # ISPT

uv run ask-the-hole find     path/to/file.ags --material rock --above 5
uv run ask-the-hole describe path/to/file.ags BH01
uv run ask-the-hole spt-results path/to/file.ags --min-n 30

uv run ask-the-hole ask path/to/file.ags "which locations hit rock above 5m?"
uv run ask-the-hole ask path/to/file.ags "..." --model llama3.2:3b --show-steps
```

Requires a running local Ollama with `qwen2.5:7b` pulled. Nothing leaves the
machine: the only socket opened is to localhost.

## Layout

`parser.py` reads AGS files and holds everything true of every group, including
`ParsedGroup` — a Pydantic generic shared by the depth-logged groups. One module
per group builds on it: `project.py`, `locations.py`, `geology.py`, `samples.py`,
`spt.py`. `dataset.py` loads a whole file across all five.

`models.py` holds the Pydantic row models. They share validation behaviour
through an `AgsRow` base and differ only in their fields and their declared
`identity_fields`, which mirror the key fields AGS defines for each group.

Rows are flat lists mirroring their group row-for-row, indexed by hole via
`for_location()`, rather than nested inside `Location`. That keeps every model a
faithful image of the file it came from.

Groups are optional. A missing group yields an empty result rather than an
error, and `Dataset.has()` distinguishes *absent* from *present but empty* —
which is what lets a question be answered with "that is not in this file".

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

## Rock or soil

Strata are classified from the numeric `GEOL_LEG` code, which unlike `GEOL_GEOL`
comes from a fixed standard list banded by material: 101–108 made ground and
topsoil, 201–231 clay, 301–332 silt, 401–436 sand, 501–528 gravel, 601–614 peat,
701–731 cobbles and boulders, **801–819 rock**, 996–999 broken ground and voids.

The result has **three** states, not two. 996–999 is neither rock nor soil, and
so is any absent or non-standard code — `Stratum.material` returns `unknown`
rather than defaulting. Nothing in a file states that chalk is rock and glacial
till is not; the band does.

## Querying

`queries.py` holds plain functions over a `Dataset` — the ones the LLM will
later call as tools. They are built to be checkable without it: simple
arguments in, structured results out, no exceptions for "nothing found".

Every location-level answer has **three** buckets, because of an asymmetry:

> A positive match is provable. A negative one is not.

Finding rock in a hole settles the question. *Not* finding it only settles the
question if every stratum in that hole carried a usable legend code — one
unclassifiable stratum means the honest answer is "cannot tell", not "no". The
same applies to a level question against a hole whose `LOCA_GL` was unusable:
its depths are known, its levels are unknowable.

`LocationQuery.is_complete` is False whenever anything is undetermined, and
`summary()` then reports a **minimum rather than a total**.

### Depth versus level

`--datum depth` (the default) measures metres below ground; `--datum level`
measures mOD. `above` and `below` name a **physical direction**, so the numeric
comparison flips between them: above 5m depth is a *smaller* number, above 5mOD
is a *larger* one. Bounds are exclusive.

## The agent loop

`tools.py` exposes five narrow tools over the query layer. Their argument
models are Pydantic, which does two jobs at once: `model_json_schema()`
produces the schema the model sees, and `model_validate()` checks what it sends
back. A hallucinated argument is refused by the same machinery that refuses
`"N/A"` in a 2DP column, and the error is fed back so the model can retry.

### Caveats the model cannot drop

Summarising three buckets into a sentence is exactly where a small model loses
the awkward third one, so that is not left to it. Tool results carry
machine-readable `caveats` that the loop appends to the final answer regardless
of what the model said.

Two further checks the model cannot talk its way past:

- **No tools called** — every question here is about the file, so an answer
  produced without consulting it is flagged as ungrounded.
- **Numeric grounding** — measured values in the answer are compared against
  every figure the tools actually returned. Only *measurements* are checked
  (decimals, figures with units, N values); bare integers are ignored, because
  counts and totals are derived legitimately and would bury the signal.

The grounding check is a heuristic and is allowed to be wrong. It adds a caveat
rather than suppressing the answer, and how often it fires — including how
often it fires spuriously — is itself a way to compare models.

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

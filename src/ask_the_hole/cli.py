"""Command-line entry point for Ask the Hole."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated, Any

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from ask_the_hole.abbreviations import parse_abbreviations
from ask_the_hole.geology import parse_geology
from ask_the_hole.locations import parse_locations
from ask_the_hole.models import AgsRow, FieldWarning, Location, Stratum
from ask_the_hole.parser import AgsError, RowError, read_ags_tables

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Ask questions about an AGS 4.x geotechnical data file, fully offline.",
)

console = Console()
errors_console = Console(stderr=True)

AgsFileArgument = Annotated[
    Path,
    typer.Argument(
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to an AGS 4.x file.",
    ),
]

# Presentation-only knowledge about the row models, kept here rather than on
# them so the domain models stay free of display concerns.
_NUMERIC_FIELDS = frozenset({"easting", "northing", "ground_level", "final_depth", "top", "base"})
# Fields whose rendered values are wider than their AGS heading.
_MIN_CONTENT_WIDTH = {"start_date": 10, "end_date": 10}


@app.callback()
def main() -> None:
    """Keep Typer in multi-command mode.

    With a single registered command Typer promotes it to the top level, so
    the command name disappears from the CLI. This placeholder preserves
    `ask-the-hole <command>` shape for the commands still to come.
    """


@app.command()
def locations(ags_file: AgsFileArgument) -> None:
    """Parse the LOCA group of AGS_FILE and print every location."""
    try:
        tables = read_ags_tables(ags_file)
        result = parse_locations(tables)
    except AgsError as exc:
        _fail(exc)

    console.print(
        _build_table(
            title=f"LOCA - {len(result.locations)} locations",
            model=Location,
            rows=result.locations,
            units=result.units,
            flexible="remarks",
        )
    )
    _report(result.warnings, result.errors)


@app.command()
def strata(ags_file: AgsFileArgument) -> None:
    """Parse the GEOL group of AGS_FILE and print every logged layer."""
    try:
        tables = read_ags_tables(ags_file)
        located = parse_locations(tables)
        abbreviations = parse_abbreviations(tables)
        result = parse_geology(tables, known_ids=located.ids, abbreviations=abbreviations)
    except AgsError as exc:
        _fail(exc)

    console.print(
        _build_table(
            title=f"GEOL - {len(result.strata)} strata in {len(result.by_location)} locations",
            model=Stratum,
            rows=result.strata,
            units=result.units,
            flexible="description",
            # Decoded via the file's own ABBR group, falling back to the raw
            # code. Shown beside the code rather than replacing it, so what is
            # in the file stays visible.
            extra=(
                (
                    "GEOL_GEOL decoded",
                    lambda s: abbreviations.describe("GEOL_GEOL", s.geology_code) or "-",
                ),
            ),
        )
    )
    _report(result.warnings, result.errors)


def _build_table(
    *,
    title: str,
    model: type[AgsRow],
    rows: Sequence[AgsRow],
    units: dict[str, str],
    flexible: str,
    extra: Sequence[tuple[str, Callable[[Any], str]]] = (),
) -> Table:
    """Render rows as a table that degrades gracefully on narrow terminals.

    Columns are generated from the model's fields, so a table can never drift
    out of step with what the parser produces. Headers show the AGS heading
    rather than the Python field name, and carry the group's declared unit
    underneath, so what you read here matches what you would find in the file.

    Every column is no_wrap with a width floor except ``flexible``, which has
    none - so when rich runs out of width it ellipses the free-text column
    rather than truncating depths, coordinates or dates.
    """
    table = Table(
        title=title,
        title_justify="left",
        box=box.SIMPLE_HEAD,
        pad_edge=False,
    )

    for name in model.model_fields:
        heading = model.heading_for(name)
        unit = units.get(heading)
        table.add_column(
            f"{heading}\n{unit}" if unit else heading,
            justify="right" if name in _NUMERIC_FIELDS else "left",
            style="bold" if name == "loca_id" else "",
            no_wrap=True,
            overflow="ellipsis",
            min_width=(
                None if name == flexible else max(len(heading), _MIN_CONTENT_WIDTH.get(name, 0))
            ),
        )

    for header, _getter in extra:
        table.add_column(header, no_wrap=True, overflow="ellipsis", min_width=len(header))

    for row in rows:
        cells = [
            _cell(getattr(row, name), numeric=name in _NUMERIC_FIELDS)
            for name in model.model_fields
        ]
        cells.extend(getter(row) for _header, getter in extra)
        table.add_row(*cells)

    return table


def _report(warnings: list[FieldWarning], errors: list[RowError]) -> None:
    """Print warnings and errors separately, because they mean different things.

    A warning is a value we lost; an error is a row we lost.
    """
    if warnings:
        errors_console.print(
            f"\n[bold yellow]{len(warnings)} unusable value(s), read as blank:[/bold yellow]"
        )
        for warning in warnings:
            where = warning.loca_id or "file"
            errors_console.print(
                f"  [yellow]{where}[/yellow]"
                f"  {warning.heading}={warning.value!r}  {warning.message}"
            )

    if errors:
        errors_console.print(f"\n[bold red]{len(errors)} row(s) discarded:[/bold red]")
        for error in errors:
            label = error.loca_id or f"row {error.row_number}"
            errors_console.print(f"  [red]{label}[/red]  {error.message}")


def _fail(exc: AgsError) -> None:
    errors_console.print(f"[bold red]Error:[/bold red] {exc}")
    raise typer.Exit(code=1) from exc


def _cell(value: Any, *, numeric: bool) -> str:
    """Render one value. None means "not recorded", shown as a dash."""
    if value is None:
        return "-"
    return f"{value:.2f}" if numeric else str(value)

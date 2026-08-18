"""Command-line entry point for Ask the Hole."""

from pathlib import Path
from typing import Annotated, Any

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from ask_the_hole.models import Location
from ask_the_hole.parser import (
    AgsError,
    ParsedLocations,
    parse_locations,
    read_ags_tables,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Ask questions about an AGS 4.x geotechnical data file, fully offline.",
)

console = Console()
errors_console = Console(stderr=True)

# Presentation-only knowledge about Location's fields, kept here rather than on
# the model so the domain model stays free of display concerns.
_NUMERIC_FIELDS = frozenset({"easting", "northing", "ground_level", "final_depth"})
_FLEXIBLE_FIELD = "remarks"
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
def locations(
    ags_file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Path to an AGS 4.x file.",
        ),
    ],
) -> None:
    """Parse the LOCA group of AGS_FILE and print every location."""
    try:
        tables = read_ags_tables(ags_file)
        result = parse_locations(tables)
    except AgsError as exc:
        errors_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    _print_locations(result)


def _print_locations(result: ParsedLocations) -> None:
    """Render locations as a table that degrades gracefully on narrow terminals.

    Columns are generated from Location's fields, so the table can never drift
    out of step with what the parser produces. Headers show the AGS heading
    rather than the Python field name, and carry the group's declared unit
    underneath, so what you read here matches what you would find in the file.

    Every column except LOCA_REM is no_wrap, so when rich runs out of width it
    shrinks remarks rather than truncating coordinates or dates.
    """
    table = Table(
        title=f"LOCA - {len(result.locations)} locations",
        title_justify="left",
        box=box.SIMPLE_HEAD,
        pad_edge=False,
    )

    for name in Location.model_fields:
        heading = Location.heading_for(name)
        unit = result.units.get(heading)
        table.add_column(
            f"{heading}\n{unit}" if unit else heading,
            justify="right" if name in _NUMERIC_FIELDS else "left",
            style="bold" if name == "loca_id" else "",
            no_wrap=True,
            overflow="ellipsis",
            # min_width stops rich shrinking a column below its real content.
            # The flexible column has no floor, so it absorbs the squeeze.
            min_width=(
                None
                if name == _FLEXIBLE_FIELD
                else max(len(heading), _MIN_CONTENT_WIDTH.get(name, 0))
            ),
        )

    for location in result.locations:
        table.add_row(
            *(
                _cell(getattr(location, name), numeric=name in _NUMERIC_FIELDS)
                for name in Location.model_fields
            )
        )

    console.print(table)

    # Warnings and errors are reported separately because they mean different
    # things: a warning is a value we lost, an error is a location we lost.
    if result.warnings:
        errors_console.print(
            f"\n[bold yellow]{len(result.warnings)} unusable value(s), read as blank:[/bold yellow]"
        )
        for warning in result.warnings:
            errors_console.print(
                f"  [yellow]{warning.loca_id or '?'}[/yellow]"
                f"  {warning.heading}={warning.value!r}  {warning.message}"
            )

    if result.errors:
        errors_console.print(f"\n[bold red]{len(result.errors)} row(s) discarded:[/bold red]")
        for error in result.errors:
            label = error.loca_id or f"row {error.row_number}"
            errors_console.print(f"  [red]{label}[/red]  {error.message}")


def _cell(value: Any, *, numeric: bool) -> str:
    """Render one value. None means "not recorded", shown as a dash."""
    if value is None:
        return "-"
    return f"{value:.2f}" if numeric else str(value)

"""Command-line entry point for Ask the Hole."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated, Any

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from ask_the_hole.dataset import Dataset, load_dataset
from ask_the_hole.models import AgsRow, FieldWarning, InSituTest, Location, Sample, Stratum
from ask_the_hole.parser import AgsError, RowError

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
_NUMERIC_FIELDS = frozenset(
    {"easting", "northing", "ground_level", "final_depth", "top", "base", "n_value"}
)
# Ceiling for the free-text column, so it cannot crowd out everything else.
_FLEXIBLE_MAX_WIDTH = 44
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
    """Print every location in the LOCA group of AGS_FILE."""
    data = _load(ags_file)
    console.print(
        _build_table(
            title=f"LOCA - {len(data.locations.locations)} locations",
            model=Location,
            rows=data.locations.locations,
            units=data.locations.units,
            flexible="remarks",
        )
    )
    _report(data.locations.warnings, data.locations.errors)


@app.command()
def strata(ags_file: AgsFileArgument) -> None:
    """Print every logged layer in the GEOL group of AGS_FILE."""
    data = _load(ags_file)
    if not _require(data, "GEOL"):
        return

    geology = data.geology
    console.print(
        _build_table(
            title=f"GEOL - {len(geology.strata)} strata in {len(geology.by_location)} locations",
            model=Stratum,
            rows=geology.strata,
            units=geology.units,
            flexible="description",
            extra=(
                # Decoded via the file's own ABBR group, falling back to the raw
                # code; shown beside the code so what is in the file stays visible.
                (
                    "GEOL_GEOL decoded",
                    lambda s: data.abbreviations.describe("GEOL_GEOL", s.geology_code) or "-",
                ),
                # Derived from the standard GEOL_LEG band, not from the file.
                ("material", lambda s: s.material.value),
            ),
        )
    )
    _report(geology.warnings, geology.errors)


@app.command()
def samples(ags_file: AgsFileArgument) -> None:
    """Print every sample in the SAMP group of AGS_FILE."""
    data = _load(ags_file)
    if not _require(data, "SAMP"):
        return

    console.print(
        _build_table(
            title=f"SAMP - {len(data.samples.samples)} samples",
            model=Sample,
            rows=data.samples.samples,
            units=data.samples.units,
            flexible="remarks",
            extra=(
                (
                    "SAMP_TYPE decoded",
                    lambda s: data.abbreviations.describe("SAMP_TYPE", s.sample_type) or "-",
                ),
            ),
        )
    )
    _report(data.samples.warnings, data.samples.errors)


@app.command()
def spt(ags_file: AgsFileArgument) -> None:
    """Print every Standard Penetration Test in the ISPT group of AGS_FILE."""
    data = _load(ags_file)
    if not _require(data, "ISPT"):
        return

    console.print(
        _build_table(
            title=f"ISPT - {len(data.spt.tests)} tests",
            model=InSituTest,
            rows=data.spt.tests,
            units=data.spt.units,
            flexible="remarks",
        )
    )
    _report(data.spt.warnings, data.spt.errors)


@app.command()
def project(ags_file: AgsFileArgument) -> None:
    """Print the PROJ group of AGS_FILE: what this file is about."""
    data = _load(ags_file)
    parsed = data.project

    if parsed.project is None:
        errors_console.print("[yellow]No usable PROJ row in this file.[/yellow]")
    else:
        table = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_header=False)
        table.add_column("", style="bold", no_wrap=True)
        table.add_column("")
        for name in type(parsed.project).model_fields:
            table.add_row(
                type(parsed.project).heading_for(name),
                _cell(getattr(parsed.project, name), numeric=False),
            )
        console.print(table)

    _report(parsed.warnings, parsed.errors)


@app.command()
def summary(ags_file: AgsFileArgument) -> None:
    """Print what AGS_FILE actually contains, group by group.

    The point of this command is negative answers: knowing a group is absent is
    what lets a question be answered with "that is not in this file" rather than
    with silence.
    """
    data = _load(ags_file)

    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, title=str(ags_file), title_justify="left")
    table.add_column("GROUP", style="bold", no_wrap=True)
    table.add_column("rows", justify="right", no_wrap=True)
    table.add_column("status", no_wrap=True)

    counts: tuple[tuple[str, int], ...] = (
        ("PROJ", 1 if data.project.project else 0),
        ("LOCA", len(data.locations.locations)),
        ("GEOL", len(data.geology.strata)),
        ("SAMP", len(data.samples.samples)),
        ("ISPT", len(data.spt.tests)),
    )
    for group, count in counts:
        if not data.has(group):
            table.add_row(group, "-", "[dim]not in this file[/dim]")
        else:
            table.add_row(group, str(count), "parsed")

    console.print(table)
    _report(data.warnings, data.errors)


def _load(ags_file: Path) -> Dataset:
    try:
        return load_dataset(ags_file)
    except AgsError as exc:
        _fail(exc)


def _require(data: Dataset, group: str) -> bool:
    """Report a group's absence as information, not as a failure."""
    if data.has(group):
        return True
    errors_console.print(
        f"[yellow]No {group} group in {data.source.name}.[/yellow] "
        f"Groups present: {', '.join(data.groups_present)}"
    )
    return False


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
            # The free-text column is capped as well as floorless. Without a
            # ceiling a long GEOL_DESC claims all the spare width and pushes the
            # derived columns off the right-hand edge entirely.
            max_width=_FLEXIBLE_MAX_WIDTH if name == flexible else None,
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

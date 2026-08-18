"""Regenerate the terminal captures used in the README.

Every image in the README is produced by actually running the command, not by
mocking it up. Rich records the real console output and writes it to SVG, so a
capture cannot drift away from what the tool does: if behaviour changes, rerun
this and the images change with it.

Requires a running local Ollama for the captures that call a model.

    uv run python scripts/capture_screenshots.py
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from rich.console import Console

import ask_the_hole.cli as cli

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
# Relative, because some commands echo the path they were given and an absolute
# one would bake the author's home directory into a published image.
FIXTURES = Path("tests") / "fixtures"


def capture(name: str, title: str, run: Callable[[], None], width: int = 100) -> None:
    """Run a command with the console redirected, and save what it printed.

    Both consoles are pointed at one recorder so warnings interleave with tables
    exactly as they do in a real terminal, rather than being separated because
    one stream is stderr.
    """
    recorder = Console(record=True, width=width)
    original_out, original_err = cli.console, cli.errors_console
    cli.console, cli.errors_console = recorder, recorder
    try:
        run()
    finally:
        cli.console, cli.errors_console = original_out, original_err

    target = DOCS / f"{name}.svg"
    recorder.save_svg(str(target), title=title)
    _drop_remote_fonts(target)
    print(f"  wrote {target.relative_to(ROOT)}")


def _drop_remote_fonts(target: Path) -> None:
    """Strip the CDN @font-face URLs Rich embeds.

    Left in, every reader of the README would have their browser fetch a font
    from a third-party CDN, which is a poor look for a project whose whole
    premise is that it makes no outbound requests. Dropping the url() sources
    keeps the local() lookup and falls back to the reader's monospace font.
    """
    svg = target.read_text(encoding="utf-8")
    svg = re.sub(
        r'src: (local\("[^"]+"\)),[^;]*;',
        lambda match: f"src: {match.group(1)};",
        svg,
    )
    target.write_text(svg, encoding="utf-8")


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    clean = FIXTURES / "clean-site.ags"
    messy = FIXTURES / "messy-export.ags"
    sparse = FIXTURES / "sparse-site.ags"

    print("Capturing deterministic commands...")

    capture(
        "ask",
        "ask-the-hole ask",
        lambda: cli.ask(clean, "Which locations hit rock above 5m?", show_steps=True),
    )
    # describe rather than strata: the full GEOL table needs about 140 columns
    # once the AGS headings, the decoded name and the material column are all
    # present, which is unreadable at README scale. This shows the same
    # classification per hole, with levels, in a shape that fits.
    capture(
        "describe",
        "ask-the-hole describe BH01",
        lambda: cli.describe(clean, "BH01"),
        width=104,
    )
    capture(
        "locations-messy",
        "ask-the-hole locations (messy export)",
        lambda: cli.locations(messy),
        width=112,
    )
    capture(
        "find-undetermined",
        "ask-the-hole find --datum level",
        lambda: cli.find(messy, material=cli.Material.ROCK, datum=cli.Datum.LEVEL),
        width=104,
    )
    capture(
        "summary-sparse",
        "ask-the-hole summary (sparse site)",
        lambda: cli.summary(sparse),
        width=76,
    )

    print("Capturing model-backed commands (needs Ollama running)...")

    capture(
        "grounding-check",
        "ask-the-hole ask --model llama3.2:3b",
        lambda: cli.ask(
            clean,
            "Which boreholes reached rock, and what SPT N values were "
            "recorded below 5m in those boreholes?",
            model="llama3.2:3b",
            show_steps=True,
        ),
        width=104,
    )

    print("Done.")


if __name__ == "__main__":
    main()

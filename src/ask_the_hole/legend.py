"""Classifying a stratum as rock or soil from its AGS4 legend code.

Unlike GEOL_GEOL, the numeric GEOL_LEG codes come from a fixed standard list
banded by material, which is what makes this deterministic rather than a guess.
Nothing in a file states that chalk is rock and glacial till is not; the band
the code falls in does.

The result has three states, not two. 996-999 covers broken ground, undetermined
strata, no recovery and voids: those are neither rock nor soil, and calling them
soil would be wrong. A missing or non-standard code is likewise undetermined -
the honest answer is "cannot tell from the legend code", not a default.
"""

from __future__ import annotations

from enum import StrEnum

# Inclusive (low, high) bounds, from the AGS4 standard legend list.
SOIL_BANDS: tuple[tuple[int, int], ...] = (
    (101, 108),  # made ground and topsoil
    (201, 231),  # clay
    (301, 332),  # silt
    (401, 436),  # sand
    (501, 528),  # gravel
    (601, 614),  # peat
    (701, 731),  # cobbles and boulders
)
# Mudstone, siltstone, sandstone, limestone, chalk, coal, breccia,
# conglomerate, igneous, metamorphic, slate.
ROCK_BAND: tuple[int, int] = (801, 819)
# Broken ground, undetermined, no recovery, void.
INDETERMINATE_BAND: tuple[int, int] = (996, 999)


class Material(StrEnum):
    """What a stratum is made of, as far as its legend code can say.

    A StrEnum so the value serialises as plain text ("rock") in JSON and reads
    naturally in output, while still being a closed set rather than a bare
    string that callers might typo.
    """

    ROCK = "rock"
    SOIL = "soil"
    UNKNOWN = "unknown"


def classify_legend(legend: str | None) -> Material:
    """Classify a GEOL_LEG code as rock, soil, or unknown.

    Returns UNKNOWN rather than guessing whenever the legend is absent, not a
    number, or not inside a standard band. A code such as 150 sits below the
    rock band but in no defined soil band either, so it is non-standard and
    undetermined - the gaps between bands are not implicitly soil.
    """
    if legend is None:
        return Material.UNKNOWN

    try:
        code = int(legend.strip())
    except (ValueError, AttributeError):
        return Material.UNKNOWN

    if _within(code, ROCK_BAND):
        return Material.ROCK
    if _within(code, INDETERMINATE_BAND):
        return Material.UNKNOWN
    if any(_within(code, band) for band in SOIL_BANDS):
        return Material.SOIL
    return Material.UNKNOWN


def _within(code: int, band: tuple[int, int]) -> bool:
    low, high = band
    return low <= code <= high

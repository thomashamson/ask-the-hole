"""Deterministic questions over a parsed AGS dataset.

These are the functions the LLM will eventually call as tools, so they are
built to be answerable and checkable without it: plain arguments in, structured
results out, no exceptions for "nothing found".

Every location-level answer has three buckets rather than two. That is not
defensiveness - it follows from an asymmetry in the data:

    A positive match is provable. A negative one is not.

Finding rock in a hole settles the question. *Not* finding it only settles the
question if every stratum in that hole had a usable legend code. One stratum
logged with a legend we cannot classify means the honest answer is "I cannot
tell you", not "no". The same applies to a level question against a hole whose
LOCA_GL was unusable: its depths are known, its levels are unknowable.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from ask_the_hole.dataset import Dataset
from ask_the_hole.legend import Material
from ask_the_hole.models import InSituTest, Sample, Stratum


class Datum(StrEnum):
    """What a depth figure is measured against.

    The two are not interchangeable, and "above" flips its comparison between
    them: above 5 m *depth* means shallower, so a smaller number; above 5 mOD
    means higher, so a larger one. Keeping "above" a physical direction rather
    than a numeric one is what stops that inversion becoming a silent bug.
    """

    DEPTH = "depth"  # metres below ground level
    LEVEL = "level"  # metres relative to project datum, typically mOD


class Verdict(StrEnum):
    """Whether a location answers a question, fails it, or cannot say."""

    MATCHED = "matched"
    NOT_MATCHED = "not_matched"
    UNDETERMINED = "undetermined"


class LocationFinding(BaseModel):
    """One location's answer, with the evidence behind it."""

    loca_id: str
    verdict: Verdict
    reason: str
    depth: float | None = None
    level: float | None = None
    description: str | None = None


class LocationQuery(BaseModel):
    """The result of asking one question of every location in a file."""

    question: str
    datum: Datum
    matched: list[LocationFinding]
    not_matched: list[LocationFinding]
    undetermined: list[LocationFinding]

    @property
    def is_complete(self) -> bool:
        """Whether every location could be answered either way.

        When False, the count of matches is a lower bound, not a total - and
        any summary of this result must say so.
        """
        return not self.undetermined

    def summary(self) -> str:
        """A one-line answer that never overstates what was established."""
        found = len(self.matched)
        if self.is_complete:
            total = found + len(self.not_matched)
            return f"{self.question}: {found} of {total} locations"
        return (
            f"{self.question}: {found} confirmed, "
            f"{len(self.undetermined)} undetermined - so this is a minimum, not a total"
        )


def find_locations_with_material(
    dataset: Dataset,
    material: Material,
    *,
    above: float | None = None,
    below: float | None = None,
    datum: Datum = Datum.DEPTH,
) -> LocationQuery:
    """Which locations encountered a given material, optionally within a band.

    ``above`` and ``below`` are physical directions, not numeric ones - see
    Datum. Both are exclusive: "above 5m" does not include a stratum whose top
    is exactly 5.00.

    A location is undetermined when it has no geology logged, when a level
    question is asked of a hole with no usable ground level, or when nothing
    matched but some of its strata carry a legend code we cannot classify.
    """
    question = _describe_question(material, above=above, below=below, datum=datum)
    matched: list[LocationFinding] = []
    not_matched: list[LocationFinding] = []
    undetermined: list[LocationFinding] = []

    for location in dataset.locations.locations:
        strata = dataset.geology.for_location(location.loca_id)

        if not strata:
            undetermined.append(
                LocationFinding(
                    loca_id=location.loca_id,
                    verdict=Verdict.UNDETERMINED,
                    reason=(
                        "no geology logged for this location"
                        if dataset.has("GEOL")
                        else "this file has no GEOL group"
                    ),
                )
            )
            continue

        if datum is Datum.LEVEL and location.ground_level is None:
            undetermined.append(
                LocationFinding(
                    loca_id=location.loca_id,
                    verdict=Verdict.UNDETERMINED,
                    reason=(
                        "LOCA_GL is not usable, so depths cannot be converted to levels; "
                        "ask by depth instead"
                    ),
                )
            )
            continue

        hit = _shallowest_match(
            strata,
            material,
            above=above,
            below=below,
            datum=datum,
            ground_level=location.ground_level,
        )
        if hit is not None:
            stratum, position = hit
            # Quote the figure in the datum that was actually asked about, so a
            # level question is not answered with a depth.
            where = f"{position:.2f}m depth" if datum is Datum.DEPTH else f"{position:.2f}mOD"
            matched.append(
                LocationFinding(
                    loca_id=location.loca_id,
                    verdict=Verdict.MATCHED,
                    reason=f"{material.value} at {where}",
                    depth=stratum.top,
                    level=_level_of(location.ground_level, stratum.top),
                    description=stratum.description,
                )
            )
            continue

        # Nothing matched. Whether that is a "no" depends on whether the legend
        # was complete enough to rule the material out.
        unclassified = [s for s in strata if s.material is Material.UNKNOWN]
        if unclassified:
            undetermined.append(
                LocationFinding(
                    loca_id=location.loca_id,
                    verdict=Verdict.UNDETERMINED,
                    reason=(
                        f"no {material.value} found, but {len(unclassified)} of {len(strata)} "
                        "strata have no usable GEOL_LEG code, so it cannot be ruled out"
                    ),
                )
            )
        else:
            not_matched.append(
                LocationFinding(
                    loca_id=location.loca_id,
                    verdict=Verdict.NOT_MATCHED,
                    reason=f"all {len(strata)} strata classified, none are {material.value}",
                )
            )

    return LocationQuery(
        question=question,
        datum=datum,
        matched=matched,
        not_matched=not_matched,
        undetermined=undetermined,
    )


def find_spt_results(
    dataset: Dataset,
    *,
    min_n: int | None = None,
    max_n: int | None = None,
    above: float | None = None,
    below: float | None = None,
) -> list[InSituTest]:
    """SPT results matching an N-value range and depth band, shallowest first.

    Tests with no N value are excluded whenever an N filter is applied, because
    an unrecorded value cannot satisfy or fail a threshold. They are kept when
    only depth is filtered, since their depth is still known.
    """
    results: list[InSituTest] = []
    for test in dataset.spt.rows:
        if above is not None and not test.top < above:
            continue
        if below is not None and not test.top > below:
            continue
        if min_n is not None or max_n is not None:
            if test.n_value is None:
                continue
            if min_n is not None and test.n_value < min_n:
                continue
            if max_n is not None and test.n_value > max_n:
                continue
        results.append(test)
    return sorted(results, key=lambda test: (test.top, test.loca_id))


class LocationProfile(BaseModel):
    """Everything one file records about a single hole."""

    loca_id: str
    location_type: str | None
    ground_level: float | None
    final_depth: float | None
    strata: list[Stratum]
    samples: list[Sample]
    spt: list[InSituTest]
    notes: list[str]


def describe_location(dataset: Dataset, loca_id: str) -> LocationProfile | None:
    """Assemble one hole's full record, or None if the file has no such hole.

    Returning None rather than raising keeps "there is no BH99 in this file" an
    answer rather than an error.
    """
    location = next((loc for loc in dataset.locations.locations if loc.loca_id == loca_id), None)
    if location is None:
        return None

    notes: list[str] = []
    if location.ground_level is None:
        notes.append("LOCA_GL is not usable, so levels cannot be reported for this hole")
    if not dataset.has("SAMP"):
        notes.append("this file has no SAMP group")
    if not dataset.has("ISPT"):
        notes.append("this file has no ISPT group")

    strata = dataset.geology.for_location(loca_id)
    unclassified = [s for s in strata if s.material is Material.UNKNOWN]
    if unclassified:
        notes.append(f"{len(unclassified)} of {len(strata)} strata have no usable GEOL_LEG code")

    return LocationProfile(
        loca_id=location.loca_id,
        location_type=location.location_type,
        ground_level=location.ground_level,
        final_depth=location.final_depth,
        strata=strata,
        samples=dataset.samples.for_location(loca_id),
        spt=dataset.spt.for_location(loca_id),
        notes=notes,
    )


def _shallowest_match(
    strata: list[Stratum],
    material: Material,
    *,
    above: float | None,
    below: float | None,
    datum: Datum,
    ground_level: float | None,
) -> tuple[Stratum, float] | None:
    """The shallowest stratum of this material inside the band, if any.

    Strata arrive depth-ordered, so the first hit is the shallowest.
    """
    for stratum in strata:
        if stratum.material is not material:
            continue
        position = stratum.top if datum is Datum.DEPTH else _level_of(ground_level, stratum.top)
        if position is None:
            continue
        if _within_band(position, above=above, below=below, datum=datum):
            return stratum, position
    return None


def _within_band(
    value: float,
    *,
    above: float | None,
    below: float | None,
    datum: Datum,
) -> bool:
    """Whether a position falls inside the requested band.

    The comparison flips with the datum, because "above" is a physical
    direction: shallower means a *smaller* depth but a *larger* level. Naming
    the two failures rather than returning early keeps that inversion visible
    in one place instead of spread across four early returns.
    """
    if datum is Datum.DEPTH:
        below_the_top = above is not None and value >= above
        above_the_bottom = below is not None and value <= below
    else:
        below_the_top = above is not None and value <= above
        above_the_bottom = below is not None and value >= below

    return not (below_the_top or above_the_bottom)


def _level_of(ground_level: float | None, depth: float) -> float | None:
    """Convert a depth below ground to a level, when the ground level is known."""
    if ground_level is None:
        return None
    return ground_level - depth


def _describe_question(
    material: Material,
    *,
    above: float | None,
    below: float | None,
    datum: Datum,
) -> str:
    """Restate the query in words, so an answer can quote what was actually asked."""
    unit = "m depth" if datum is Datum.DEPTH else "mOD"
    band = ""
    if above is not None and below is not None:
        band = f" between {below:.2f} and {above:.2f}{unit}"
    elif above is not None:
        band = f" above {above:.2f}{unit}"
    elif below is not None:
        band = f" below {below:.2f}{unit}"
    return f"locations with {material.value}{band}"

"""One AGS file, parsed into every group we understand.

Groups are optional. A site with no samples is a real site, and sparse files
are the normal case rather than an error - so a missing group yields an empty
result and the tool can answer "there is no SAMP data in this file" instead of
failing to load it. Only LOCA is required, because without locations nothing
else can be placed.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ask_the_hole.abbreviations import Abbreviations, parse_abbreviations
from ask_the_hole.geology import ParsedGeology, parse_geology
from ask_the_hole.locations import ParsedLocations, parse_locations
from ask_the_hole.models import FieldWarning
from ask_the_hole.parser import RowError, read_ags_tables
from ask_the_hole.project import ParsedProject, parse_project
from ask_the_hole.samples import ParsedSamples, parse_samples
from ask_the_hole.spt import ParsedSpt, parse_spt


class Dataset(BaseModel):
    """Everything one AGS file contains, across the five groups we model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: Path
    project: ParsedProject
    locations: ParsedLocations
    geology: ParsedGeology
    samples: ParsedSamples
    spt: ParsedSpt
    abbreviations: Abbreviations
    groups_present: list[str]

    def has(self, group: str) -> bool:
        """Whether the file actually contained a group.

        The distinction that makes "not in this file" answerable: an empty
        result from a group that is absent means something different from an
        empty result from a group that is present but had no usable rows.
        """
        return group in self.groups_present

    @property
    def warnings(self) -> list[FieldWarning]:
        """Every field warning from every group, in group order."""
        return [
            *self.project.warnings,
            *self.locations.warnings,
            *self.geology.warnings,
            *self.samples.warnings,
            *self.spt.warnings,
        ]

    @property
    def errors(self) -> list[RowError]:
        """Every discarded row from every group, in group order."""
        return [
            *self.project.errors,
            *self.locations.errors,
            *self.geology.errors,
            *self.samples.errors,
            *self.spt.errors,
        ]


def load_dataset(path: Path) -> Dataset:
    """Read and parse an AGS file in full.

    Order matters: LOCA is parsed first because every other group joins to it,
    and its surviving IDs are what orphan detection tests against.
    """
    tables = read_ags_tables(path)

    locations = parse_locations(tables)
    known_ids = locations.ids
    abbreviations = parse_abbreviations(tables)

    return Dataset(
        source=path,
        project=parse_project(tables) if "PROJ" in tables else ParsedProject.empty(),
        locations=locations,
        geology=(
            parse_geology(tables, known_ids=known_ids, abbreviations=abbreviations)
            if "GEOL" in tables
            else ParsedGeology.empty()
        ),
        samples=(
            parse_samples(tables, known_ids=known_ids)
            if "SAMP" in tables
            else ParsedSamples.empty()
        ),
        spt=(parse_spt(tables, known_ids=known_ids) if "ISPT" in tables else ParsedSpt.empty()),
        abbreviations=abbreviations,
        groups_present=sorted(tables),
    )

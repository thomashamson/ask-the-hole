"""Pydantic models for parsed AGS 4.x data.

Field names are Python-idiomatic. Each carries an ``alias`` naming the AGS
heading it comes from, so a model can be built directly from a raw AGS row
without a hand-written mapping layer. All user-facing output quotes the AGS
heading rather than the Python name.

Real AGS exports contain values that violate the type declared in the file's
own TYPE row - "N/A" in a 2DP column, "-" where a coordinate should be. These
models treat that as a *field* problem, not a *row* problem: the offending
value becomes None and a FieldWarning is recorded, but the row survives. Only a
broken *identity* discards a row, because a row we cannot name or place is a
row we cannot answer questions about.

Each model declares its own identity via ``identity_fields``, mirroring the key
fields AGS defines for that group.
"""

from __future__ import annotations

from datetime import date
from typing import Any, ClassVar, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    field_validator,
    model_validator,
)

from ask_the_hole.legend import Material, classify_legend


class FieldWarning(BaseModel):
    """A recorded value that could not be used, and was replaced with None.

    Distinct from an error: the row was still returned. This is the file telling
    us something is wrong with it, not us failing to read the file.
    """

    loca_id: str | None
    heading: str
    value: str
    message: str


class AgsRow(BaseModel):
    """Shared behaviour for one row of any AGS group.

    Subclasses supply the fields and name their identity. Everything here is
    about *how* a row is validated, not *what* is in one.
    """

    model_config = ConfigDict(
        # Allow construction by field name as well as by AGS alias, so tests and
        # application code can write Location(loca_id="BH01") rather than
        # Location(LOCA_ID="BH01"). Without this, only the alias is accepted.
        populate_by_name=True,
        # AGS files are padded and hand-edited; strip surrounding whitespace off
        # every string before validating it.
        str_strip_whitespace=True,
        # Reject unknown keys. The parser selects the columns it knows about, so
        # an unexpected key here means a bug in our code, not messy input.
        extra="forbid",
        # Parsed rows are a read-only view of the file. Immutability also makes
        # rows hashable, so they can go in sets.
        frozen=True,
    )

    # ClassVar is not a field. Pydantic deliberately skips ClassVar-annotated
    # attributes when building the model, so this is configuration attached to
    # the class rather than data attached to each row.
    identity_fields: ClassVar[frozenset[str]] = frozenset()

    @field_validator("*", mode="wrap")
    @classmethod
    def _tolerate_unusable_values(
        cls,
        value: Any,
        handler: ValidatorFunctionWrapHandler,
        info: ValidationInfo,
    ) -> Any:
        """Degrade an unusable field to None instead of failing the whole row.

        A "wrap" validator sits *around* Pydantic's own validation: ``handler``
        is the normal machinery, and we choose whether to call it and what to do
        when it complains. That is what lets us catch a type failure and
        substitute None. A "before" validator could not, because it runs before
        Pydantic has judged the value; an "after" validator could not either,
        because it never runs once validation has already failed.

        Three cases, matching how AGS files actually behave:

        * an identity field - no leniency. We call the handler directly and let
          a bad value propagate, discarding the row.
        * ``""`` - a legitimately absent value. Silently None, no warning. AGS
          has no NULL, so an empty field is the correct way to record "not
          recorded" and is not a data quality problem.
        * anything else that fails - a real type violation, such as "N/A" in a
          2DP column. None, plus a FieldWarning so the user learns their file
          contains it.
        """
        if info.field_name in cls.identity_fields:
            return handler(value)

        if isinstance(value, str) and not value.strip():
            return None

        try:
            return handler(value)
        except ValidationError as exc:
            record_warning(
                info.context,
                # info.data holds the fields validated so far. loca_id is
                # declared first in every group, so it is already available.
                loca_id=info.data.get("loca_id"),
                heading=cls.heading_for(info.field_name),
                value=value,
                message=exc.errors()[0]["msg"],
            )
            return None

    @classmethod
    def heading_for(cls, field_name: str) -> str:
        """Map a Python field name back to the AGS heading it came from.

        Output quotes the AGS heading rather than our field name, so the user
        can find the offending column in their own file.
        """
        field = cls.model_fields.get(field_name)
        if field is not None and field.alias:
            return field.alias
        return field_name

    @classmethod
    def headings(cls) -> tuple[str, ...]:
        """Every AGS heading this model reads, in declaration order.

        Derived from the model rather than repeated in the parser, so the set of
        headings we keep can never drift out of step with the fields we parse.
        """
        return tuple(cls.heading_for(name) for name in cls.model_fields)


class LocatedRow(AgsRow):
    """A row logged at a depth within a location.

    Subclasses declare ``loca_id`` and ``top`` themselves: their AGS aliases
    differ per group (GEOL_TOP, SAMP_TOP, ISPT_TOP), so the fields cannot simply
    be inherited. This base exists to name the contract that lets one container
    index and depth-sort any of them.
    """


class Location(AgsRow):
    """One row of the AGS ``LOCA`` group: a single exploratory hole.

    Only a core subset of the ~35 AGS4 ``LOCA`` headings is modelled. Anything
    else present in the file is dropped by the parser.
    """

    # AGS keys LOCA on LOCA_ID alone.
    identity_fields: ClassVar[frozenset[str]] = frozenset({"loca_id"})

    loca_id: str = Field(
        alias="LOCA_ID",
        min_length=1,
        description="Location identifier, unique within the project.",
    )
    location_type: str | None = Field(
        default=None,
        alias="LOCA_TYPE",
        description="Type of activity, e.g. CP (cable percussion), WS (window sample).",
    )
    status: str | None = Field(
        default=None,
        alias="LOCA_STAT",
        description="Status of the location's data, e.g. FINAL.",
    )
    easting: float | None = Field(
        default=None,
        alias="LOCA_NATE",
        description="National grid easting of the location.",
    )
    northing: float | None = Field(
        default=None,
        alias="LOCA_NATN",
        description="National grid northing of the location.",
    )
    ground_level: float | None = Field(
        default=None,
        alias="LOCA_GL",
        description="Ground level relative to datum, typically mOD. May be negative.",
    )
    final_depth: float | None = Field(
        default=None,
        alias="LOCA_FDEP",
        ge=0,
        description="Final depth of the hole below ground level, in metres.",
    )
    start_date: date | None = Field(
        default=None,
        alias="LOCA_STAR",
        description="Date the location was started.",
    )
    end_date: date | None = Field(
        default=None,
        alias="LOCA_ENDD",
        description="Date the location was completed.",
    )
    remarks: str | None = Field(
        default=None,
        alias="LOCA_REM",
        description="General remarks recorded against the location.",
    )

    @model_validator(mode="after")
    def _flag_reversed_dates(self, info: ValidationInfo) -> Self:
        """Cross-field check, run once every individual field has validated.

        A field_validator only ever sees one value, so ordering between two
        dates has to happen here. This records rather than raises, because by
        contract only a broken identity may discard a row.
        """
        if self.start_date and self.end_date and self.end_date < self.start_date:
            record_warning(
                info.context,
                loca_id=self.loca_id,
                heading="LOCA_ENDD",
                value=str(self.end_date),
                message=f"end date is before start date {self.start_date}",
            )
        return self


class Stratum(LocatedRow):
    """One row of the AGS ``GEOL`` group: a single geological layer in a hole.

    ``top`` and ``base`` are depths *below ground level*, not levels relative to
    a datum. Converting one to the other needs the hole's LOCA_GL, which is why
    the UNIT row is captured and why a hole with an unusable ground level can
    report depths but not levels.
    """

    # AGS keys GEOL on LOCA_ID plus GEOL_TOP: a layer is identified by the hole
    # it is in and where it starts. A stratum with neither cannot be placed.
    identity_fields: ClassVar[frozenset[str]] = frozenset({"loca_id", "top"})

    loca_id: str = Field(
        alias="LOCA_ID",
        min_length=1,
        description="Identifier of the location this layer was logged in.",
    )
    top: float = Field(
        alias="GEOL_TOP",
        ge=0,
        description="Depth to the top of the layer, below ground level, in metres.",
    )
    base: float | None = Field(
        default=None,
        alias="GEOL_BASE",
        ge=0,
        description="Depth to the base of the layer, below ground level, in metres.",
    )
    description: str | None = Field(
        default=None,
        alias="GEOL_DESC",
        description="Free-text engineering description of the stratum.",
    )
    legend: str | None = Field(
        default=None,
        alias="GEOL_LEG",
        description="Legend code for the stratum.",
    )
    geology_code: str | None = Field(
        default=None,
        alias="GEOL_GEOL",
        description="Geology code for the stratum, decoded via the file's ABBR group.",
    )
    status: str | None = Field(
        default=None,
        alias="GEOL_STAT",
        description="Status of the layer's data, e.g. FINAL.",
    )

    @property
    def material(self) -> Material:
        """Rock, soil, or unknown, from the standard GEOL_LEG band.

        Derived rather than stored: it is a reading of the legend code, not a
        separate fact recorded in the file. Three states, because 996-999 and
        any missing or non-standard code are neither rock nor soil.
        """
        return classify_legend(self.legend)

    @property
    def thickness(self) -> float | None:
        """Layer thickness in metres, or None when the base was not recorded."""
        if self.base is None:
            return None
        return self.base - self.top

    @model_validator(mode="after")
    def _flag_inverted_interval(self, info: ValidationInfo) -> Self:
        """A base shallower than its top means the interval is unusable."""
        if self.base is not None and self.base < self.top:
            record_warning(
                info.context,
                loca_id=self.loca_id,
                heading="GEOL_BASE",
                value=str(self.base),
                message=f"base is above top {self.top}",
            )
        return self


class Project(AgsRow):
    """The single row of the AGS ``PROJ`` group: what this file is about."""

    # AGS keys PROJ on PROJ_ID.
    identity_fields: ClassVar[frozenset[str]] = frozenset({"project_id"})

    project_id: str = Field(
        alias="PROJ_ID",
        min_length=1,
        description="Project identifier.",
    )
    name: str | None = Field(default=None, alias="PROJ_NAME", description="Project title.")
    location: str | None = Field(
        default=None, alias="PROJ_LOC", description="Site location or address."
    )
    client: str | None = Field(default=None, alias="PROJ_CLNT", description="Client name.")
    contractor: str | None = Field(
        default=None, alias="PROJ_CONT", description="Investigation contractor."
    )
    engineer: str | None = Field(default=None, alias="PROJ_ENG", description="Consulting engineer.")
    memo: str | None = Field(default=None, alias="PROJ_MEMO", description="General project memo.")


class Sample(LocatedRow):
    """One row of the AGS ``SAMP`` group: a sample taken from a hole."""

    # AGS keys SAMP on the hole and the depth it was taken from. SAMP_REF and
    # SAMP_TYPE further distinguish samples at the same depth, but they are text
    # and effectively never fail to parse, so they are not identity for the
    # purpose of deciding whether a row is salvageable.
    identity_fields: ClassVar[frozenset[str]] = frozenset({"loca_id", "top"})

    loca_id: str = Field(
        alias="LOCA_ID",
        min_length=1,
        description="Identifier of the location this sample came from.",
    )
    top: float = Field(
        alias="SAMP_TOP",
        ge=0,
        description="Depth to the top of the sample, below ground level, in metres.",
    )
    base: float | None = Field(
        default=None,
        alias="SAMP_BASE",
        ge=0,
        description="Depth to the base of the sample, below ground level, in metres.",
    )
    reference: str | None = Field(
        default=None, alias="SAMP_REF", description="Sample reference within the hole."
    )
    sample_type: str | None = Field(
        default=None,
        alias="SAMP_TYPE",
        description="Sample type, e.g. U (undisturbed), D (small disturbed), B (bulk).",
    )
    sample_id: str | None = Field(
        default=None, alias="SAMP_ID", description="Globally unique sample identifier."
    )
    remarks: str | None = Field(default=None, alias="SAMP_REM", description="Sample remarks.")


class InSituTest(LocatedRow):
    """One row of the AGS ``ISPT`` group: a Standard Penetration Test.

    ``seating_blows`` and ``main_blows`` are text, not numbers: AGS records them
    as comma-separated increments such as "4,4,5,6". A driller writing "N/A"
    there is not a type violation, because the column is declared as text.
    """

    identity_fields: ClassVar[frozenset[str]] = frozenset({"loca_id", "top"})

    loca_id: str = Field(
        alias="LOCA_ID",
        min_length=1,
        description="Identifier of the location this test was carried out in.",
    )
    top: float = Field(
        alias="ISPT_TOP",
        ge=0,
        description="Depth of the test below ground level, in metres.",
    )
    seating_blows: str | None = Field(
        default=None,
        alias="ISPT_SEAT",
        description="Blow counts for the seating increments, as recorded.",
    )
    main_blows: str | None = Field(
        default=None,
        alias="ISPT_MAIN",
        description="Blow counts for the main test increments, as recorded.",
    )
    n_value: int | None = Field(
        default=None,
        alias="ISPT_NVAL",
        ge=0,
        description="SPT N value: total blows over the main increments.",
    )
    remarks: str | None = Field(
        default=None,
        alias="ISPT_REM",
        description="Test remarks, e.g. why a test was aborted.",
    )


def record_warning(
    context: Any,
    *,
    loca_id: str | None,
    heading: str,
    value: Any,
    message: str,
) -> None:
    """Append a FieldWarning to the list passed in as validation context.

    Pydantic's ``context`` is an arbitrary object handed to model_validate() and
    visible to every validator as ``info.context``. It is the supported way to
    get information *out* of validation without bolting a mutable field onto an
    otherwise frozen model. If no context list was supplied, warnings are simply
    dropped, so the models stay usable on their own.
    """
    if isinstance(context, list):
        context.append(
            FieldWarning(
                loca_id=loca_id,
                heading=heading,
                value=str(value),
                message=message,
            )
        )

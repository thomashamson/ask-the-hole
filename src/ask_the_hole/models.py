"""Pydantic models for parsed AGS 4.x data.

Field names are Python-idiomatic. Each carries an ``alias`` naming the AGS
heading it comes from, so a model can be built directly from a raw AGS row
without a hand-written mapping layer.

Real AGS exports contain values that violate the type declared in the file's
own TYPE row - "N/A" in a 2DP column, "-" where a coordinate should be. The
models here treat that as a *field* problem, not a *row* problem: the offending
value becomes None and a FieldWarning is recorded, but the location survives.
Only a broken identity (LOCA_ID) discards a row, because a location we cannot
name is a location we cannot answer questions about.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Self

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


class FieldWarning(BaseModel):
    """A recorded value that could not be used, and was replaced with None.

    Distinct from an error: the row was still returned. This is the file telling
    us something is wrong with it, not us failing to read the file.
    """

    loca_id: str | None
    heading: str
    value: str
    message: str


class Location(BaseModel):
    """One row of the AGS ``LOCA`` group: a single exploratory hole.

    Only a core subset of the ~35 AGS4 ``LOCA`` headings is modelled. Anything
    else present in the file is dropped by the parser.
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
        # Location hashable, so locations can go in sets.
        frozen=True,
    )

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

        * ``loca_id`` - no leniency. Identity is load-bearing, so we call the
          handler directly and let a bad value propagate and discard the row.
        * ``""`` - a legitimately absent value. Silently None, no warning. AGS
          has no NULL, so an empty field is the correct way to record "not
          recorded" and is not a data quality problem.
        * anything else that fails - a real type violation, such as "N/A" in a
          2DP column. None, plus a FieldWarning so the user learns their file
          contains it.
        """
        if info.field_name == "loca_id":
            return handler(value)

        if isinstance(value, str) and not value.strip():
            return None

        try:
            return handler(value)
        except ValidationError as exc:
            _warn(
                info.context,
                # info.data holds the fields validated so far. loca_id is
                # declared first, so it is already available to name this row.
                loca_id=info.data.get("loca_id"),
                heading=cls.heading_for(info.field_name),
                value=value,
                message=exc.errors()[0]["msg"],
            )
            return None

    @model_validator(mode="after")
    def _flag_reversed_dates(self, info: ValidationInfo) -> Self:
        """Cross-field check, run once every individual field has validated.

        A field_validator only ever sees one value, so ordering between two
        dates has to happen here. This records rather than raises, because by
        contract only a broken LOCA_ID may discard a row.
        """
        if self.start_date and self.end_date and self.end_date < self.start_date:
            _warn(
                info.context,
                loca_id=self.loca_id,
                heading="LOCA_ENDD",
                value=str(self.end_date),
                message=f"end date is before start date {self.start_date}",
            )
        return self

    @classmethod
    def heading_for(cls, field_name: str) -> str:
        """Map a Python field name back to the AGS heading it came from.

        Warnings quote the AGS heading rather than our field name, so the user
        can find the offending column in their own file.
        """
        field = cls.model_fields.get(field_name)
        if field is not None and field.alias:
            return field.alias
        return field_name


def _warn(
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
    dropped, so Location stays usable on its own.
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

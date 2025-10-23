"""Events emitted by the relevance-driven condensation flow."""

from uuid import UUID

from pydantic import Field, field_validator

from openhands.sdk.event.base import Event
from openhands.sdk.event.types import EventID, SourceType

SUMMARY_MAX_CHARS: int = 1028

class RelevanceCondensationDirective(Event):
    """Directive emitted when the LLM-initiated tool requests condensation (in this case, will be a reduction of a previous message)."""

    source: SourceType = "environment"
    requested_event_id: EventID = Field(
        description=(
            "Identifier of the event the LLM has marked as safe to condense."
        ),
        examples=["cfb0d6d2-3ef1-4f75-8e36-8a6fdb7d6f80"],
    )
    summary: str = Field(
        description="Short synopsis that keeps the continuity of the conversation.",
        min_length=1,
        max_length=SUMMARY_MAX_CHARS,
    )
    requesting_action_id: EventID | None = Field(
        default=None,
        description="Optional reference to the action event that produced this tool call.",
    )
    llm_response_id: str | None = Field(
        default=None,
        description="LLM response identifier associated with the tool directive.",
    )
    agent_step_index: int | None = Field(
        default=None,
        ge=0,
        description="Agent step ordinal when the directive was recorded.",
    )

    @field_validator("requested_event_id", "requesting_action_id")
    @classmethod
    def _validate_event_id(cls, value: EventID | None) -> EventID | None:
        """Ensure referenced event identifiers are formatted as UUIDs."""
        if value is None:
            return None
        try:
            UUID(value)
        except (ValueError, TypeError) as exc:  # pragma: no cover - defensive
            raise ValueError("Event identifiers must be valid UUID strings") from exc
        return value

    @field_validator("summary")
    @classmethod
    def _normalize_summary(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("summary cannot be empty or whitespace only")
        if len(trimmed) > SUMMARY_MAX_CHARS:
            raise ValueError(f"summary must be <= {SUMMARY_MAX_CHARS} characters")
        return trimmed


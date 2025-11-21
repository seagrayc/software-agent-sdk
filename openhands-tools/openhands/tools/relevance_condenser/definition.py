"""Definition for the LLM-driven relevance condenser tool."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import Field, field_validator

from openhands.sdk.tool import Action, ToolAnnotations, ToolDefinition
from openhands.sdk.tool.schema import (
    RELEVANCE_SUMMARY_MAX_CHARS,
    RelevanceCondensationObservation,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState


TOOL_DESCRIPTION = """Background tool to mark past tool outputs as no longer relevant.

Provide the identifier for a prior tool observation that no longer aids the
current discussion, plus a short reasoned summary explaining the redaction.
The condenser will mask (redact) the observation content and replace it with
your concise summary while leaving the original tool invocation in the chat
history.

Identification:
- tool_call_index: Provide the index for the tool response to redact.

Usage notes:
- Only observation content is masked; original action/tool invocation remains.
- Summaries must be short (1–3 sentences), substantially shorter than the
  original, and should not introduce new facts.

Guardrails:
- Only reference tool calls/observations you are confident are no longer relevant.
- Never target user messages, system prompts, or security warnings."""


class RelevanceCondensationAction(Action):
    "Tool request payload for marking past tool observations as no longer relevant."

    tool_call_index: int = Field(  # type: ignore[assignment]
        ge=0,
        description=(
            "Identifier of the 'tool' message which is no longer "
            "relevant to the task. "
            "Is provided at the start of all tool responses, "
            "in the form: [tool_call_index: {i}]"
        ),
    )
    summary_text: str = Field(
        description=(
            "One to three sentence summary used to replace the original tool "
            "response. Keeps global context to relevant information, whilst "
            "preserving continuity."
        ),
        min_length=1,
        max_length=RELEVANCE_SUMMARY_MAX_CHARS,
    )

    @field_validator("summary_text")
    @classmethod
    def _validate_summary_text(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("summary_text cannot be empty or whitespace only")
        return s


class LLMRelevanceCondenserTool(
    ToolDefinition[RelevanceCondensationAction, RelevanceCondensationObservation]
):
    """Tool wiring for LLM-managed relevance condensation."""

    @classmethod
    def create(
        cls, conv_state: ConversationState
    ) -> Sequence[LLMRelevanceCondenserTool]:
        # Import here to avoid circular imports
        from openhands.tools.relevance_condenser.impl import RelevanceCondenserExecutor

        executor = RelevanceCondenserExecutor(state=conv_state)
        return [
            cls(
                description=TOOL_DESCRIPTION,
                action_type=RelevanceCondensationAction,
                observation_type=RelevanceCondensationObservation,
                annotations=ToolAnnotations(
                    title="mark_context_redundant",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=executor,
            )
        ]

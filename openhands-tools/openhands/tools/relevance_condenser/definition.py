"""Definition for the LLM-driven relevance condenser tool."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from openhands.sdk.event.relevance_condenser import RelevanceCondensationDirective
from openhands.sdk.tool import ToolAnnotations, ToolDefinition, ToolExecutor
from openhands.sdk.tool.schema import (
    RelevanceCondensationAction,
    RelevanceCondensationObservation,
)

if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState


TOOL_DESCRIPTION = """Background tool to mark past tool outputs as no longer relevant.

Provide the identifier of a prior tool call that no
longer aids the current discussion, plus a short reasoned summary explaining the redaction. The condenser will mask (redact) the observation content and replace it with your concise summary while
leaving the original tool invocation in the chat history. Nothing is deleted; masking
preserves continuity and discourages the LLM from re-invoking the same call.

Usage notes:
- Target the observation event ID when possible; only observation content is masked.
- Summaries must be short (1–3 sentences), substantially shorter than the original,
  and should not introduce new facts.

Guardrails:
- Only reference tool calls/observations you are confident are no longer relevant.
- Never target user messages, system prompts, or security warnings."""


class RelevanceCondenserExecutor(
    ToolExecutor[RelevanceCondensationAction, RelevanceCondensationObservation]
):
    """Executor that records relevance condensation directives."""

    def __init__(self, state: "ConversationState"):
        self._state = state

    def __call__(
        self, action: RelevanceCondensationAction
    ) -> RelevanceCondensationObservation:
        directive = RelevanceCondensationDirective(
            requested_event_id=action.event_id,
            summary=action.summary_text,
        )

        self._state.events.append(directive)

        message = (
            "Condensation directive recorded. The condenser will mask observation "
            f"for {action.event_id} when applied (tool invocation is retained)."
        )
        return RelevanceCondensationObservation(
            message=message,
            accepted_event_ids=[action.event_id],
            rejected_event_ids=[],
        )


class LLMRelevanceCondenserTool(
    ToolDefinition[RelevanceCondensationAction, RelevanceCondensationObservation]
):
    """Tool wiring for LLM-managed relevance condensation."""

    name: str = "relevance_condenser"
    description: str = TOOL_DESCRIPTION
    action_type: type[RelevanceCondensationAction] = RelevanceCondensationAction
    observation_type: type[
        RelevanceCondensationObservation
    ] = RelevanceCondensationObservation
    annotations: ToolAnnotations = ToolAnnotations(
        title="mark_context_redundant",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )

    @classmethod
    def create(
        cls, conv_state: "ConversationState"
    ) -> Sequence["LLMRelevanceCondenserTool"]:
        executor = RelevanceCondenserExecutor(state=conv_state)
        return [
            cls(
                executor=executor,
            )
        ]

"""Definition for the LLM-driven relevance condenser tool."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from openhands.sdk.event.relevance_condenser import RelevanceCondensationDirective
from openhands.sdk.event.llm_convertible import ObservationBaseEvent
from openhands.sdk.tool import ToolAnnotations, ToolDefinition, ToolExecutor
from openhands.sdk.tool.schema import (
    RelevanceCondensationAction,
    RelevanceCondensationObservation,
)

if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState


TOOL_DESCRIPTION = """Background tool to mark past tool outputs as no longer relevant.

Provide an identifier for a prior tool observation that no longer aids the
current discussion, plus a short reasoned summary explaining the redaction.
The condenser will mask (redact) the observation content and replace it with
your concise summary while leaving the original tool invocation in the chat
history. Nothing is deleted; masking preserves continuity and discourages
duplicate re-invocation.

Identification:
- tool_call_id: Provide the tool message identifier the model sees in
  function-calling contexts.
- tool_call_direct_index: Provide the direct message index for the tool response
  to redact (as seen by the LLM after formatting).

Usage notes:
- Only observation content is masked; original action/tool invocation remains.
- Summaries must be short (1–3 sentences), substantially shorter than the
  original, and should not introduce new facts.

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
            # tool_call_id=action.tool_call_id,
            tool_call_direct_index=action.tool_call_direct_index,
            summary=action.summary_text,
        )

        self._state.events.append(directive)

        message = (
            "Condensation directive recorded; condenser will apply redaction."
        )
        return RelevanceCondensationObservation(
            message=message,
            accepted_event_ids= [x for x in [action.tool_call_direct_index] if x is not None],
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

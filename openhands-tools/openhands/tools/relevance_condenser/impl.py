"""Executor implementation for the LLM-driven relevance condenser tool."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openhands.sdk.event.relevance_condenser import RelevanceCondensationDirective
from openhands.sdk.tool import ToolExecutor
from openhands.sdk.tool.schema import RelevanceCondensationObservation
from openhands.tools.relevance_condenser.definition import RelevanceCondensationAction


if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation
    from openhands.sdk.conversation.state import ConversationState


class RelevanceCondenserExecutor(
    ToolExecutor[RelevanceCondensationAction, RelevanceCondensationObservation]
):
    """Executor that records relevance condensation directives."""

    def __init__(self, state: ConversationState):
        self._state = state

    def __call__(
        self,
        action: RelevanceCondensationAction,
        conversation: LocalConversation | None = None,  # noqa: ARG002
    ) -> RelevanceCondensationObservation:
        directive = RelevanceCondensationDirective(
            tool_call_index=action.tool_call_index,
            summary=action.summary_text,
        )

        self._state.events.append(directive)

        message = "Condensation directive recorded; condenser will apply redaction."
        accepted_event_ids = [action.tool_call_index]

        return RelevanceCondensationObservation(
            message=message,
            accepted_event_ids=accepted_event_ids,
            rejected_event_ids=[],
        )

"""Definition for the LLM-driven relevance condenser tool."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from openhands.sdk.event import ActionEvent, ObservationEvent
from openhands.sdk.event.base import Event
from openhands.sdk.event.relevance_condenser import RelevanceCondensationDirective
from openhands.sdk.logger import get_logger
from openhands.sdk.tool import ToolAnnotations, ToolDefinition, ToolExecutor
from openhands.sdk.tool.schema import (
    RelevanceCondensationAction,
    RelevanceCondensationObservation,
)

if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState
    from openhands.sdk.conversation.types import ConversationCallbackType

logger = get_logger(__name__)


TOOL_DESCRIPTION = """Background tool for flagging stale, or no longer relevant tool interactions.

Provide the identifier of a prior tool call or observation that no longer aids the
current discussion, along with a short synopsis to preserve continuity. The condenser will
remove the event when safe while keeping the supplied summary available to the agent.

Guardrails:
- Only target tool calls or their observations that you are confident are no longer
  relevant.
- Never reference user messages, system prompts, or security warnings.
- Keep summaries concise (1–3 sentences) and avoid introducing new facts."""


class RelevanceCondenserExecutor(
    ToolExecutor[RelevanceCondensationAction, RelevanceCondensationObservation]
):
    """Executor that records relevance condensation directives."""

    def __init__(self, state: "ConversationState"):
        self._state = state
        self._event_callback: "ConversationCallbackType | None" = None
        self._current_action_event: ActionEvent | None = None
        self._step_index: int | None = None

    def __call__(
        self, action: RelevanceCondensationAction
    ) -> RelevanceCondensationObservation:
        directive = RelevanceCondensationDirective(
            requested_event_id=action.event_id,
            summary=action.summary_text,
            requesting_action_id=self._current_action_event.id
            if self._current_action_event
            else None,
            llm_response_id=self._current_action_event.llm_response_id
            if self._current_action_event
            else None,
            agent_step_index=self._step_index,
        )

        self._emit_event(directive)

        message = (
            "Condensation directive recorded. The condenser will retire "
            f"{action.event_id} when applied."
        )
        return RelevanceCondensationObservation(
            message=message,
            accepted_event_ids=[action.event_id],
            rejected_event_ids=[],
        )

    def _emit_event(self, event: RelevanceCondensationDirective) -> None:
        """Emit the directive via callback fallback to direct persistence."""
        if self._event_callback:
            self._event_callback(event)
        else:
            logger.debug(
                "No event callback provided; appending directive %s directly to state",
                event.id,
            )
            self._state.events.append(event)


class LLMRelevanceCondenserTool(
    ToolDefinition[RelevanceCondensationAction, RelevanceCondensationObservation]
):
    """Tool wiring for LLM-managed relevance condensation."""

    name: str = "relevance_condenser"
    description: str = TOOL_DESCRIPTION
    action_type = RelevanceCondensationAction
    observation_type = RelevanceCondensationObservation
    annotations = ToolAnnotations(
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

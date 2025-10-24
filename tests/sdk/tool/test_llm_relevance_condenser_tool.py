from types import SimpleNamespace
from uuid import uuid4

import pytest

from openhands.sdk.event.relevance_condenser import RelevanceCondensationDirective
from openhands.sdk.tool.schema import RelevanceCondensationAction
from openhands.tools.relevance_condenser.definition import LLMRelevanceCondenserTool

class TestLLMRelevanceCondensorTool:
    def test_relevance_condensation_action_validation(self) -> None:
        """Ensure the action schema enforces UUIDs and trims summaries."""
        event_id = str(uuid4())
        action = RelevanceCondensationAction(
            event_id=event_id,
            summary_text="  stale observation  ",
        )

        assert action.event_id == event_id
        assert action.summary_text == "stale observation"

        with pytest.raises(ValueError):
            RelevanceCondensationAction(
                event_id="not-a-uuid",
                summary_text="valid summary",
            )

        with pytest.raises(ValueError):
            RelevanceCondensationAction(
                event_id=event_id,
                summary_text="   ",
            )

    def test_relevance_condenser_executor_records_directive(self) -> None:
        """Executor should append directives and acknowledge accepted IDs."""
        state = SimpleNamespace(events=[])
        tool = LLMRelevanceCondenserTool.create(state)[0]
        executable = tool.as_executable()

        target_id = str(uuid4())
        action = RelevanceCondensationAction(
            event_id=target_id,
            summary_text="Reduce clutter from failed listing.",
        )

        observation = executable(action)

        assert len(state.events) == 1
        directive = state.events[0]
        assert isinstance(directive, RelevanceCondensationDirective)
        assert directive.requested_event_id == target_id
        assert directive.summary == "Reduce clutter from failed listing."

        assert observation.accepted_event_ids == [target_id]
        assert observation.rejected_event_ids == []
        assert target_id in observation.message

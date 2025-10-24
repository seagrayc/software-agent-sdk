from types import SimpleNamespace
from uuid import uuid4

import pytest

from openhands.sdk.event.relevance_condenser import RelevanceCondensationDirective
from openhands.sdk.tool.schema import RelevanceCondensationAction
from openhands.tools.relevance_condenser.definition import LLMRelevanceCondenserTool

class TestLLMRelevanceCondensorTool:
    def test_relevance_condensation_action_validation(self) -> None:
        """Ensure schema enforces required tool_call_id and trims summaries."""
        # Valid with tool_call_id
        action2 = RelevanceCondensationAction(
            tool_call_id="call_abc123",
            summary_text="trim me",
        )
        assert action2.tool_call_id == "call_abc123"

        # Empty summary
        with pytest.raises(ValueError):
            RelevanceCondensationAction(
                tool_call_id="call_x",
                summary_text="   ",
            )
        
        # Missing tool_call_id (pydantic will flag required field)
        with pytest.raises(Exception):
            RelevanceCondensationAction(summary_text="missing")

    # Removed: event_id path no longer supported

    def test_relevance_condenser_executor_records_directive_with_tool_call_id(self) -> None:
        """Executor should resolve observation via tool_call_id and record directive."""
        from openhands.sdk.event.llm_convertible import AgentErrorEvent

        state = SimpleNamespace(events=[])
        # Create a prior observation with a known tool_call_id
        tool_call_id = "call_12345"
        obs = AgentErrorEvent(tool_name="execute_bash", tool_call_id=tool_call_id, error="no such file")
        state.events.append(obs)

        tool = LLMRelevanceCondenserTool.create(state)[0]
        executable = tool.as_executable()

        action = RelevanceCondensationAction(
            tool_call_id=tool_call_id,
            summary_text="Reduce clutter from failed listing.",
        )

        observation = executable(action)

        assert len(state.events) == 2  # original obs + directive
        directive = state.events[-1]
        assert isinstance(directive, RelevanceCondensationDirective)
        assert directive.requested_event_id == obs.id
        assert observation.accepted_event_ids == [obs.id]
        assert observation.rejected_event_ids == []

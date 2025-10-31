from types import SimpleNamespace
from uuid import uuid4

import pytest

from openhands.sdk.event.relevance_condenser import RelevanceCondensationDirective
from openhands.sdk.tool.schema import RelevanceCondensationAction
from openhands.tools.relevance_condenser.definition import LLMRelevanceCondenserTool

class TestLLMRelevanceCondensorTool:
    def test_relevance_condensation_action_validation(self) -> None:
        """Ensure schema enforces identifiers and trims summaries."""
        # Valid with tool_call_id
        action2 = RelevanceCondensationAction(
            tool_call_id="call_abc123",
            summary_text="trim me",
        )
        assert action2.tool_call_id == "call_abc123"

        # Valid with direct index
        action3 = RelevanceCondensationAction(
            tool_call_direct_index=3,
            summary_text="works",
        )
        assert action3.tool_call_direct_index == 3

        # Empty/whitespace-only summary rejected
        with pytest.raises(ValueError):
            RelevanceCondensationAction(
                tool_call_id="call_x",
                summary_text="   ",
            )

        # Missing both identifiers is rejected
        with pytest.raises(Exception):
            RelevanceCondensationAction(summary_text="missing both")

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
        assert directive.tool_call_id == obs.id
        assert observation.accepted_event_ids == [obs.id]
        assert observation.rejected_event_ids == []

    def test_relevance_condenser_executor_records_directive_with_direct_index(self) -> None:
        """Executor should resolve observation via direct index and record directive."""
        from openhands.sdk.event.llm_convertible import AgentErrorEvent, MessageEvent, ActionEvent
        from openhands.sdk.llm import Message, TextContent, MessageToolCall
        from openhands.sdk.context.view import View

        state = SimpleNamespace(events=[])

        # Add a user message, then an action + observation pair
        state.events.append(
            MessageEvent(
                source="user",
                llm_message=Message(role="user", content=[TextContent(text="please list files")]),
            )
        )

        tool_call_id = "call_98765"
        response_id = str(uuid4())
        state.events.append(
            ActionEvent(
                thought=[TextContent(text="list files")],
                reasoning_content=None,
                thinking_blocks=[],
                responses_reasoning_item=None,
                action=None,
                tool_name="execute_bash",
                tool_call_id=tool_call_id,
                tool_call=MessageToolCall(
                    id=tool_call_id,
                    name="execute_bash",
                    arguments="{}",
                    origin="completion",
                ),
                llm_response_id=response_id,
            )
        )
        obs = AgentErrorEvent(
            tool_name="execute_bash",
            tool_call_id=tool_call_id,
            error="ls failed: directory not found",
        )
        state.events.append(obs)

        # Build the current view and reconstruct message indices
        view = View.from_events(state.events)
        # Rebuild index using same algorithm: messages = events_to_messages(view.events)
        # Index of the observation message is where role='tool' for 'obs'
        from openhands.sdk.event.base import LLMConvertibleEvent
        msgs = LLMConvertibleEvent.events_to_messages(view.events)
        direct_index = None
        for i, m in enumerate(msgs):
            if m.role == "tool" and m.tool_call_id == tool_call_id:
                direct_index = i
                break
        assert direct_index is not None

        tool = LLMRelevanceCondenserTool.create(state)[0]
        executable = tool.as_executable()

        action = RelevanceCondensationAction(
            tool_call_direct_index=direct_index,
            summary_text="Not useful for next steps.",
        )

        observation = executable(action)

        assert len(state.events) == 4  # user, action, obs + directive
        directive = state.events[-1]
        assert isinstance(directive, RelevanceCondensationDirective)
        # Directive carries the direct index; condenser will resolve at apply time
        assert directive.tool_call_direct_index == direct_index
        assert directive.tool_call_id is None
        assert observation.accepted_event_ids == []
        assert observation.rejected_event_ids == []

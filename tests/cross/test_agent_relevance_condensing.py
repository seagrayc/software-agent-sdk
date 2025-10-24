from types import SimpleNamespace
from uuid import uuid4

from openhands.sdk.context.condenser.llm_relevance_condenser import (
    LLMRelevanceCondenser,
)
from openhands.sdk.context.view import View
from openhands.sdk.event.condenser import Condensation
from openhands.sdk.event.llm_convertible import ActionEvent, AgentErrorEvent, MessageEvent
from openhands.sdk.tool.schema import RelevanceCondensationAction
from openhands.sdk.llm import Message, MessageToolCall, TextContent
from openhands.tools.relevance_condenser.definition import LLMRelevanceCondenserTool


def _message(text: str) -> MessageEvent:
    return MessageEvent(
        source="user",
        llm_message=Message(role="user", content=[TextContent(text=text)]),
    )


def _action(tool_call_id: str, response_id: str) -> ActionEvent:
    return ActionEvent(
        thought=[TextContent(text="considering next command")],
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


def _observation(tool_call_id: str) -> AgentErrorEvent:
    return AgentErrorEvent(
        tool_name="execute_bash",
        tool_call_id=tool_call_id,
        error="bash: cd project: No such file or directory",
    )


def test_relevance_condensing_loop_is_idempotent() -> None:
    """Simulate tool invocation and condenser application across agent steps."""
    state = SimpleNamespace(events=[])

    user_event = _message("List repo files")
    tool_call_id = str(uuid4())
    response_id = str(uuid4())
    action_event = _action(tool_call_id, response_id)
    observation_event = _observation(tool_call_id)

    state.events.extend([user_event, action_event, observation_event])

    tool = LLMRelevanceCondenserTool.create(state)[0]
    executable = tool.as_executable()
    action = RelevanceCondensationAction(
        event_id=observation_event.id,
        summary_text="Directory listing failure no longer relevant.",
    )

    observation = executable(action)
    directive_event = state.events[-1]
    assert observation.accepted_event_ids == [observation_event.id]

    condenser = LLMRelevanceCondenser()
    initial_view = View.from_events(state.events)
    condensation = condenser.condense(initial_view)

    assert isinstance(condensation, Condensation)
    assert observation_event.id in condensation.forgotten_event_ids
    assert directive_event.id in condensation.forgotten_event_ids
    assert condensation.summary == (
        "Response redacted: Directory listing failure no longer relevant."
    )
    assert condensation.summary_offset == 2

    # Agent would append the condensation event before the next step.
    state.events.append(condensation)
    condensed_view = View.from_events(state.events)

    remaining_ids = [event.id for event in condensed_view.events]
    assert observation_event.id not in remaining_ids
    assert directive_event.id not in {
        directive.id for directive in condensed_view.relevance_directives
    }

    # Running the condenser again should be a no-op (idempotent).
    follow_up = condenser.condense(condensed_view)
    assert isinstance(follow_up, View)
    assert follow_up.events == condensed_view.events

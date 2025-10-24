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
        tool_call_id=observation_event.tool_call_id,
        summary_text="Directory listing failure no longer relevant.",
    )

    observation = executable(action)
    directive_event = state.events[-1]
    assert observation.accepted_event_ids == [observation_event.id]

    condenser = LLMRelevanceCondenser()
    initial_view = View.from_events(state.events)
    redacted_view = condenser.condense(initial_view)

    assert isinstance(redacted_view, View)
    # The observation should be redacted inline and tool pairing preserved
    assert len(redacted_view.events) == len(initial_view.events)
    redacted = redacted_view.events[2]
    assert isinstance(redacted, AgentErrorEvent)
    assert redacted.tool_call_id == observation_event.tool_call_id
    assert redacted.error == (
        "Response redacted: Directory listing failure no longer relevant."
    )

    # Directive remains surfaced for idempotence
    assert {d.id for d in redacted_view.relevance_directives} == {directive_event.id}

    # Running the condenser again should be a no-op (idempotent)
    follow_up = condenser.condense(redacted_view)
    assert isinstance(follow_up, View)
    assert [e.id for e in follow_up.events] == [e.id for e in redacted_view.events]

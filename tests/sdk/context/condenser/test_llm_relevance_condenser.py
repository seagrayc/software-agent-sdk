from __future__ import annotations

from uuid import uuid4

from openhands.sdk.context.condenser.llm_relevance_condenser import (
    LLMRelevanceCondenser,
)
from openhands.sdk.context.view import View
from openhands.sdk.event.llm_convertible import (
    ActionEvent,
    AgentErrorEvent,
    MessageEvent,
)
from openhands.sdk.event.relevance_condenser import RelevanceCondensationDirective
from openhands.sdk.llm import Message, MessageToolCall, TextContent


def _message(content: str) -> MessageEvent:
    return MessageEvent(
        source="user",
        llm_message=Message(role="user", content=[TextContent(text=content)]),
    )


def _action(tool_call_id: str, response_id: str) -> ActionEvent:
    return ActionEvent(
        thought=[TextContent(text="requesting tool output")],
        reasoning_content=None,
        thinking_blocks=[],
        responses_reasoning_item=None,
        action=None,
        tool_name="file_browser",
        tool_call_id=tool_call_id,
        tool_call=MessageToolCall(
            id=tool_call_id,
            name="file_browser",
            arguments="{}",
            origin="completion",
        ),
        llm_response_id=response_id,
    )


def _observation(tool_call_id: str) -> AgentErrorEvent:
    return AgentErrorEvent(
        tool_name="file_browser",
        tool_call_id=tool_call_id,
        error="ls failed: directory not found",
    )


def test_condense_returns_view_when_no_directives() -> None:
    events = [
        _message("review repo"),
        _action(tool_call_id=str(uuid4()), response_id=str(uuid4())),
    ]
    view = View.from_events(events)

    condenser = LLMRelevanceCondenser()
    result = condenser.condense(view)

    assert isinstance(result, View)
    assert result.events == view.events
    # Final events forwarded to the LLM are the same as the original view.
    final_events = result.events
    assert len(final_events) == len(view.events)
    assert final_events[0] == view.events[0]



def test_redacts_observation_in_place() -> None:
    tool_call_id = str(uuid4())
    response_id = str(uuid4())
    action_event = _action(tool_call_id=tool_call_id, response_id=response_id)
    observation = _observation(tool_call_id)
    directive = RelevanceCondensationDirective(
        requested_event_id=observation.id,
        summary="no longer actionable",
    )

    events = [
        _message("user requested listing"),
        _action(tool_call_id=tool_call_id, response_id=str(uuid4())),
        action_event,
        observation,
        directive,
    ]

    view = View.from_events(events)
    condenser = LLMRelevanceCondenser()
    redacted_view = condenser.condense(view)

    assert isinstance(redacted_view, View)
    assert len(redacted_view.events) == len(view.events) # replaced in place
    # The observation at the same relative position should now be redacted
    redacted_event = redacted_view.events[3]
    assert isinstance(redacted_event, AgentErrorEvent)
    assert redacted_event.tool_call_id == observation.tool_call_id
    assert redacted_event.tool_name == observation.tool_name
    assert redacted_event.error == "Response redacted: no longer actionable"
    # Directives remain surfaced (idempotent application)
    assert [d.id for d in redacted_view.relevance_directives] == [directive.id]
    # Re-applying is a no-op
    assert condenser.condense(redacted_view).events == redacted_view.events


def test_condense_skips_missing_targets_and_keeps_directive() -> None:
    directive = RelevanceCondensationDirective(
        requested_event_id=str(uuid4()),
        summary="stale directive",
    )
    events = [_message("context"), directive]
    view = View.from_events(events)

    condenser = LLMRelevanceCondenser()
    result = condenser.condense(view)

    # View unchanged since target not found; directive remains active
    assert isinstance(result, View)
    assert [event.id for event in result.events] == [events[0].id]
    assert [d.id for d in result.relevance_directives] == [directive.id]


def test_condense_handles_duplicate_directives_once() -> None:
    tool_call_id = str(uuid4())
    response_id = str(uuid4())
    action = _action(tool_call_id=tool_call_id, response_id=response_id)
    observation = _observation(tool_call_id)
    directive_one = RelevanceCondensationDirective(
        requested_event_id=observation.id,
        summary="first attempt",
    )
    directive_two = RelevanceCondensationDirective(
        requested_event_id=observation.id,
        summary="second attempt",
    )

    events = [
        _message("user request"),
        action,
        observation,
        directive_one,
        directive_two,
    ]

    view = View.from_events(events)
    condenser = LLMRelevanceCondenser()
    redacted_view = condenser.condense(view)

    assert isinstance(redacted_view, View)
    # Original observation should be replaced exactly once
    count_redacted = 0
    for ev in redacted_view.events:
        if isinstance(ev, AgentErrorEvent) and ev.tool_call_id == observation.tool_call_id:
            if ev.error.startswith("Response redacted:"):
                count_redacted += 1
    assert count_redacted == 1
    # Directives remain present for idempotence
    assert {d.id for d in redacted_view.relevance_directives} == {
        directive_one.id,
        directive_two.id,
    }

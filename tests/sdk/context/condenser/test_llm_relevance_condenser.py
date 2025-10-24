from __future__ import annotations

from uuid import uuid4

from openhands.sdk.context.condenser.llm_relevance_condenser import (
    LLMRelevanceCondenser,
)
from openhands.sdk.context.view import View
from openhands.sdk.event.condenser import Condensation, CondensationSummaryEvent
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



def test_reducts_observation() -> None:
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

    condensation = condenser.condense(view)

    assert isinstance(condensation, Condensation)
    assert set(condensation.forgotten_event_ids) == {observation.id, directive.id}
    assert condensation.summary == "Response redacted: no longer actionable"
    assert condensation.summary_offset == 2

    final_view = View.from_events([*events, condensation])
    final_event_ids = [event.id for event in final_view.events]
    assert observation.id not in final_event_ids
    assert directive.id not in {d.id for d in final_view.relevance_directives}
    assert isinstance(final_view.events[-1], CondensationSummaryEvent)


def test_condense_skips_missing_targets_but_drops_directive() -> None:
    directive = RelevanceCondensationDirective(
        requested_event_id=str(uuid4()),
        summary="stale directive",
    )
    events = [_message("context"), directive]
    view = View.from_events(events)

    condenser = LLMRelevanceCondenser()
    condensation = condenser.condense(view)

    assert isinstance(condensation, Condensation)
    assert condensation.forgotten_event_ids == [directive.id]
    assert condensation.summary == "Response redacted: stale directive"
    assert condensation.summary_offset is None

    final_view = View.from_events([*events, condensation])
    assert [event.id for event in final_view.events] == [events[0].id]
    assert final_view.relevance_directives == []


def test_condense_handles_duplicate_directives_once() -> None:
    tool_call_id = str(uuid4())
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
        observation,
        directive_one,
        directive_two,
    ]

    view = View.from_events(events)
    condenser = LLMRelevanceCondenser()
    condensation = condenser.condense(view)

    assert isinstance(condensation, Condensation)
    assert condensation.forgotten_event_ids.count(observation.id) == 1
    assert set(condensation.forgotten_event_ids).issuperset(
        {observation.id, directive_one.id, directive_two.id}
    )

    final_view = View.from_events([*events, condensation])
    final_ids = [event.id for event in final_view.events]
    assert observation.id not in final_ids
    assert all(
        directive.id not in {d.id for d in final_view.relevance_directives}
        for directive in (directive_one, directive_two)
    )
    assert isinstance(final_view.events[-1], CondensationSummaryEvent)

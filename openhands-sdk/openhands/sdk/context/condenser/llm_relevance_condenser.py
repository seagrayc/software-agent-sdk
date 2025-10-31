from logging import getLogger

from openhands.sdk.context.condenser.base import CondenserBase
from openhands.sdk.context.view import View
from openhands.sdk.event.llm_convertible import ObservationBaseEvent
from openhands.sdk.event.llm_convertible.observation import (
    AgentErrorEvent,
    ObservationEvent,
    UserRejectObservation,
)
from openhands.sdk.tool.schema import RelevanceCondensationObservation


logger = getLogger(__name__)


class LLMRelevanceCondenser(CondenserBase):
    """Apply tool-initiated relevance directives in-place and return an updated View.

    Simplified behaviour:
    - Directly returns the modified View
    - Re-apply all relevance directives present in the View by masking each
      targeted observation in place with the redaction summary
    - Leave directives intact; repeated application is a no-op
    """

    def condense(self, view: View) -> View:
        directives = getattr(view, "relevance_directives", [])
        if not directives:
            return view

        # Map target observation id → redaction message
        redactions: dict[str, str] = {}
        # Collect direct indices to resolve within this view
        idx_directives: list[tuple[int, str]] = []  # (index, redaction message)

        for d in directives:
            msg = d.summary.strip()
            if not msg:
                continue
            message = f"Response redacted: {msg}"
            if d.tool_call_id:
                # When tool_call_id is provided on the directive, it contains the target
                # event id to redact.
                redactions[d.tool_call_id] = message
            if d.tool_call_direct_index is not None:
                idx_directives.append((d.tool_call_direct_index, message))

        # Resolve any direct-index directives by mapping message indices to observation events
        if idx_directives:
            # Build mapping: message_index -> observation_event_id for this view
            idx_to_event: dict[int, str] = {}
            idx = 0
            i = 0
            events = list(view.events)
            from openhands.sdk.event.llm_convertible import ActionEvent
            while i < len(events):
                ev = events[i]
                if isinstance(ev, ActionEvent):
                    # Combine adjacent ActionEvents sharing the same llm_response_id
                    response_id = ev.llm_response_id
                    j = i + 1
                    while j < len(events):
                        nxt = events[j]
                        if not isinstance(nxt, ActionEvent) or nxt.llm_response_id != response_id:
                            break
                        j += 1
                    # Assistant message occupies one index
                    idx += 1
                    i = j
                else:
                    # Single non-action event → one message
                    if isinstance(ev, ObservationBaseEvent):
                        idx_to_event[idx] = ev.id
                    idx += 1
                    i += 1

            for index, msg in idx_directives:
                ev_id = idx_to_event.get(index)
                if ev_id:
                    redactions[ev_id] = msg

        if not redactions:
            return view

        # Replace each targeted observation in-place with a minimal redacted observation
        replaced = 0
        new_events = []
        for ev in view.events:
            redaction = redactions.get(ev.id)
            if redaction and isinstance(ev, ObservationBaseEvent):
                if isinstance(ev, ObservationEvent):
                    new_obs = RelevanceCondensationObservation(message=redaction)
                    new_ev = ObservationEvent(
                        tool_name=ev.tool_name,
                        tool_call_id=ev.tool_call_id,
                        action_id=ev.action_id,
                        observation=new_obs,
                    )
                elif isinstance(ev, AgentErrorEvent):
                    new_ev = AgentErrorEvent(
                        tool_name=ev.tool_name,
                        tool_call_id=ev.tool_call_id,
                        error=redaction,
                    )
                elif isinstance(ev, UserRejectObservation):
                    new_ev = UserRejectObservation(
                        tool_name=ev.tool_name,
                        tool_call_id=ev.tool_call_id,
                        action_id=ev.action_id,
                        rejection_reason=redaction,
                    )
                else:
                    # Fallback: ensure we keep tool pairing via AgentErrorEvent
                    new_ev = AgentErrorEvent(
                        tool_name=ev.tool_name,
                        tool_call_id=ev.tool_call_id,
                        error=redaction,
                    )
                new_events.append(new_ev)
                replaced += 1
            else:
                new_events.append(ev)

        logger.debug(
            "Applied %d relevance directives; redacted %d observations in-place",
            len(directives),
            replaced,
        )

        # Return a fresh View; directives remain so reapplying is idempotent
        return View(
            events=new_events,
            unhandled_condensation_request=view.unhandled_condensation_request,
            condensations=view.condensations,
            relevance_directives=view.relevance_directives,
        )

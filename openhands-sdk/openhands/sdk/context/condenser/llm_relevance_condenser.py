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
        for d in directives:
            msg = d.summary.strip()
            if msg:
                redactions[d.requested_event_id] = f"Response redacted: {msg}"

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

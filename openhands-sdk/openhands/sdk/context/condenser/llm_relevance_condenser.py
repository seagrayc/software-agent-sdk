from logging import getLogger

from openhands.sdk.context.condenser.base import CondenserBase
from openhands.sdk.context.view import View
from openhands.sdk.event.condenser import Condensation
from openhands.sdk.event.llm_convertible import ObservationBaseEvent


logger = getLogger(__name__)


class LLMRelevanceCondenser(CondenserBase):
    """Apply tool-initiated relevance directives without losing surrounding context."""

    def condense(self, view: View) -> View | Condensation:
        directives = getattr(view, "relevance_directives", [])
        if not directives:
            return view

        event_lookup = {
            event.id: (index, event) for index, event in enumerate(view.events)
        }

        forgotten_ids: list[str] = []
        seen_ids: set[str] = set()
        summary_lines: list[str] = []
        summary_offsets: list[int] = []

        for directive in directives:
            target_id = directive.requested_event_id
            target_entry = event_lookup.get(target_id)

            if target_entry is None:
                logger.debug(
                    "Relevance directive %s target %s not present in view; skipping",
                    directive.id,
                    target_id,
                )
            else:
                index, target_event = target_entry
                if isinstance(target_event, ObservationBaseEvent):
                    if target_id not in seen_ids:
                        forgotten_ids.append(target_id)
                        seen_ids.add(target_id)
                        summary_offsets.append(index)
                else:
                    logger.debug(
                        "Relevance directive %s target %s is not an observation; "
                        "keeping event in view",
                        directive.id,
                        target_id,
                    )

            if directive.id not in seen_ids:
                forgotten_ids.append(directive.id)
                seen_ids.add(directive.id)

            summary_text = directive.summary.strip()
            if summary_text:
                summary_lines.append(f"Response redacted: {summary_text}")

        if not forgotten_ids:
            return view

        summary = "\n".join(summary_lines) if summary_lines else None
        summary_offset = min(summary_offsets) if summary_offsets else None

        return Condensation(
            forgotten_event_ids=forgotten_ids,
            summary=summary,
            summary_offset=summary_offset,
        )

Temp file generation pattern

        with tempfile.TemporaryDirectory() as temp_dir:


=====

def test_view_drops_forgotten_relevance_directives() -> None:
    """Directives should disappear once a condensation forgets them."""
    message = message_event("Event 0")
    directive = RelevanceCondensationDirective(
        requested_event_id=message.id,
        summary="no longer relevant",
    )
    condensation = Condensation(forgotten_event_ids=[directive.id])

    view = View.from_events([message, directive, condensation])

    assert [event.id for event in view.events] == [message.id]
    assert view.relevance_directives == []

What part of the Relevance Condensation logic is the above test covering?
Relevance Condensation must not drop messages; on reduct them; is there a risk existing Condensation impl logic could clash with this?

===

    condenser = LLMRelevanceCondenser()
    result = condenser.condense(view)
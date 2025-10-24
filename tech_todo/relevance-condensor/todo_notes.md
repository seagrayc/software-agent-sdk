~/repos/agent-sdk$ git diff --name-only main
converstation-persistance-to-network-mapping-gemini.md
examples/01_standalone_sdk/10_persistence.py
openhands-sdk/openhands/sdk/context/condenser/__init__.py
openhands-sdk/openhands/sdk/context/condenser/llm_relevance_condenser.py
openhands-sdk/openhands/sdk/context/view.py
openhands-sdk/openhands/sdk/event/__init__.py
openhands-sdk/openhands/sdk/event/relevance_condenser.py
openhands-sdk/openhands/sdk/tool/__init__.py
openhands-sdk/openhands/sdk/tool/schema.py
openhands-tools/openhands/tools/relevance_condenser/__init__.py
openhands-tools/openhands/tools/relevance_condenser/definition.py
tech_todo/relevance-condensor/design_decisions.md
tech_todo/relevance-condensor/design_decisions_prompt.md
tech_todo/relevance-condensor/todo_notes.md
tests/cross/test_agent_relevance_condensing.py
tests/sdk/context/condenser/test_llm_relevance_condenser.py
tests/sdk/context/test_view.py
tests/sdk/tool/test_llm_relevance_condenser_tool.py

===

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
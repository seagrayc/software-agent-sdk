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

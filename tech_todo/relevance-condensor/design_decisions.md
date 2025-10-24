# Relevance-Oriented Condensation Options

## Requirement Recap
- Minimize stale tool calls and observations in the running conversation while preserving the thread with relevant context for the current task.
- Allow the LLM to keep working with the most relevant subset of actions, observations, and summaries without losing grounding needed for future decisions.

- When a prior tool call is no longer relevant or actionable, mask (redact) it rather than remove it. Keep the original tool invocation event; replace its observation with a concise redaction note that includes the reason. The redaction must be much shorter than the original output to save tokens while preserving continuity and discouraging duplicate re-invocation.


## Existing Integration Points
- Condensers implement `condense(view)` and return either a trimmed `View` or a `Condensation` event (`openhands-sdk/openhands/sdk/context/condenser/base.py`).
- The agent calls the condenser once per `step()` before sampling the next LLM action, and short-circuits to handle emitted `Condensation` events (`openhands-sdk/openhands/sdk/agent/agent.py`).
- `View.from_events(...)` already maps the conversation history to the LLM-facing slice, respecting previous condensations, summaries, and condensation requests. It also filters unmatched tool-call / observation pairs to avoid dangling batches (`openhands-sdk/openhands/sdk/context/view.py`).
- `LLMSummarizingCondenser` shows how a condenser can call an LLM directly to summarize forgotten segments while inserting a `Condensation` event that records masked/forgotten IDs and an optional synopsis (`openhands-sdk/openhands/sdk/context/condenser/llm_summarizing_condenser.py`).

These hooks determine how any relevance-driven strategy must surface its decisions and how the agent reacts to them.

## Option 1 — (Selected) Tool-Driven Relevance Condenser

### Concept
- Introduce `LLMRelevanceCondenserTool` that is always available for the LLM to invoke. Expectation is for the LLM to invoke as a "background" task, using parallel tool calling to perform concurrently with "the next step" in its plan execution. For example, view contents of the next round of files to review while flagging previous tool responses as no longer relevant.
- When the LLM believes some tool calls or observations are no longer needed, it invokes the tool with the event IDs and a redaction explanation to maintain consistency. The condenser must mask the observation(s) of those tool calls instead of deleting the tool invocation(s).
- The companion `LLMRelevanceCondenser`, which is invoked by the Agent on every step, will apply the reductions based on previous tool calls made by the LLM.

### Benefits
- The LLM schedules condensation in parallel with other tool usage; no additional turns are forced solely for summarization.
- Gives the model full situational awareness: it prunes precisely when it deems context pressure high or when it recognizes dead-ends.
- Compatible with existing `handles_condensation_requests()` - if needed, path—tool invocation can emit a `CondensationRequest`. Alternatively the existing condensor invocation in the Agent step will "apply" the tool call.

### Challenges
- Requires a robust API between the tool and condenser (event selection schema, error handling, safeguards against over-forgetting).
- Higher prompt surface: the tool instruction must include guardrails to prevent removal of essential actions, and to ensure the LLM references canonical event IDs.

### Implementation Sketch
- Tool input: `{event_id, summary_text}`. `summary_text` must provide a 1–3 sentence explanation of the redaction to keep continuity.
- Tool output: acknowledgement message; the actual condenser logic runs asynchronously to the tool invocation.

## Option 2 — (Rejected) Autonomous LLM Summarizing Condenser (Pull Model)

### Concept
- Extend the `RollingCondenser` pattern, similar to `LLMSummarizingCondenser`, but change `should_condense(view)` to look for relevance triggers (e.g., number of tool calls exceeding threshold, inactivity on tool outputs).
- When triggered, the condenser calls `self.llm.completion(...)` to classify recent tool calls as stale vs. keep-worthy and to generate a replacement summary.
- Outputs a `Condensation` event that masks stale tool call observations (keeping the original tool invocation events) and optionally injects a synthesized summary at `summary_offset`.

### Challenges
- Condenser runs sequentially in the agent loop, so any condensation requires waiting for its additional LLM request before the main action sample proceeds.
- Harder for the main LLM to express subjective judgments about relevance; it may repeatedly see irrelevant history until the condenser threshold trips.
- Requires careful prompt engineering to ensure the condenser’s LLM understands tool semantics and preserves chains-of-thought needed later.

## Option 1 — Tool-Driven Relevance Condenser - Implementation Details

### Flow Overview
- LLM issues potentially multiple `LLMRelevanceCondenserTool` calls with `{event_id: ..., summary_text: ...}` while it continues other work.
- Tool executor validates the payload, records a structured reduction request event, and responds with an acknowledgement message that the agent relays to the LLM.
- On the following `Agent.step`, the `LLMRelevanceCondenser` inspects the `View`, replays all stored reduction directives against the current event list, and emits a `Condensation` that applies masks to the requested event observations and injects the provided synopsis. The original tool invocation events remain in the history, paired with the redacted observation placeholders.
- The condensed view feeds back into normal action selection, keeping the conversation thread tight without an extra turn.

## Masking vs. Removal (Clarification)

- The flow results masking results, not deleting stale or non-actionable tool calls.
- Keep the original tool invocation event (including parameters) to maintain causal continuity and provide evidence that the call already occurred.
- Replace the associated observation payload with a concise redaction note, e.g., "Observation redacted: superseded by newer results — no longer relevant." Include the reason from `summary_text`.
- Ensure the redaction is significantly shorter than the original observation to reduce context footprint without losing continuity or prompting duplicate tool calls.
- Rationale: retaining the invocation preserves conversation structure and intent, while concise masking recovers tokens and signals completion/irrelevance.

### Implementation Plan
1. **Define tool contract and models**
   - Add a dedicated request/response schema in `openhands-sdk/openhands/sdk/tool/schema.py` for the condenser tool, including strict validation of `event_id` (UUID string belonging to tool call or observation events) and `summary_text` (bounded length).
   - Create a sibling data class in `openhands-sdk/openhands/sdk/event/relevance_condenser.py` to represent a tool-driven reduction directive (`RelevanceCondensationDirective`) capturing the event reference and summary provided by the tool.
   - Update serialization utilities to ensure the new schema is available to OpenAI-style tool specs (`ToolBase.to_openai_tool`, `registry.py`).

2. **Implement `LLMRelevanceCondenserTool`**
   - Create `openhands-tools/openhands/tools/relevance_condenser/definition.py` implementing a `ToolBase` subclass with the above schema.
   - Build a lightweight executor that:
     - Validates that the referenced event ID exists in the persisted conversation (`ConversationState.events`).
     - Emits a new `RelevanceCondensationDirective` event via the provided `on_event` hook, capturing `requested_event_id`, `summary`, and requester metadata (LLM step, timestamp).
     - Returns an `Observation` acknowledging acceptance or listing rejected IDs so the LLM can retry.

3. **Build `LLMRelevanceCondenser`**
   - Create `openhands-sdk/openhands/sdk/context/condenser/llm_relevance_condenser.py` implementing `CondenserBase`.
   - On each `condense(view)`:
     - Collect all `RelevanceCondensationDirective` events surfaced in the view and deterministically reapply them to the underlying event sequence in the order they were issued.
     - Ensure the referenced event IDs are still present; if any are missing, log telemetry and continue so directives can remain idempotent across steps.
     - Aggregate directives while preventing double-removals (e.g., maintain a set of forgotten IDs) so repeated applications remain stable.
     - Record metadata via `self.add_metadata` (e.g., number of events pruned, directive ID) for observability.
   - Optionally emit a short `MessageEvent` summarizing the applied condensation so the LLM receives confirmation without relying on the tool response alone.

4. **Safety, guardrails, and fallback behaviour**
   - Apply guardrails in the tool prompt (tool description + input schema docs) emphasizing that only stale tool outputs should be referenced, and that removing system/user messages is forbidden.
   - Enforce server-side checks: reject directives targeting critical events (e.g., most recent user message, security prompts); respond with a structured error observation so the LLM learns the constraints.

5. **Testing strategy**
   - **Schema and tool validation** (`tests/sdk/tool/test_llm_relevance_condenser_tool.py`): follow the pydantic contract checks used in `tests/sdk/tool/test_tool_definition.py` to assert rejection of malformed payloads (missing summary, invalid UUID, duplicate directives) and to verify acknowledgement/error observations emitted by the executor.
   - **Condenser behaviour** (`tests/sdk/context/condenser/test_llm_relevance_condenser.py`): mirror the `pytest` fixture pattern from `tests/sdk/context/condenser/test_llm_summarizing_condenser.py` to exercise directive replay, overlapping removals, missing events, and metadata/log emission without mutating the original `View`.
   - **View safety checks** (`tests/sdk/context/test_view.py` adjuncts): add targeted cases ensuring condensed views preserve ordering, do not double prune, and remain idempotent when the same directives are re-applied across steps.
   - **Agent loop regression** (`tests/cross/test_agent_relevance_condensing.py`): stage a scripted conversation similar to `tests/cross/test_agent_reconciliation.py` where the tool issues a directive, the subsequent agent step applies the condensation, and repeated steps leave the state stable while surfacing confirmations back to the LLM.

## Appendix — Testing Patterns
- **Pytest unit tests with helper factories** — see [`tests/sdk/context/condenser/test_llm_summarizing_condenser.py`](../../tests/sdk/context/condenser/test_llm_summarizing_condenser.py) for using lightweight event builders and `MagicMock` LLMs; this pattern will validate `LLMRelevanceCondenser` directive handling in isolation.
- **Schema-focused validation suites** — [`tests/sdk/tool/test_tool_definition.py`](../../tests/sdk/tool/test_tool_definition.py) demonstrates asserting pydantic field constraints and executor behaviour; new tool tests will reuse this style to cover directive payload validation and acknowledgement messaging.
- **Stateful conversation regressions** — [`tests/cross/test_agent_reconciliation.py`](../../tests/cross/test_agent_reconciliation.py) shows how to persist and restart conversations to catch regressions; the relevance condenser integration test will adopt this approach to confirm directives survive across agent steps.
- **View invariant checks** — [`tests/sdk/context/test_view.py`](../../tests/sdk/context/test_view.py) captures expectations around how condensations mutate views; additional cases will extend these invariants to the relevance-driven pruning path so the feature remains covered end-to-end.

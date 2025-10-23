Task focus: Create a document in ./tech_todo which provides a detailed background on implementation options for the below requirement.

Requirement: As a coding agent progresses it accumulatets a large number of tool calls; and context. Certain tool calls may long longer be relevant, once subsequent actions are completed. For example - a broard search, followed by viewing multiple files. Once the correct files are found; the search, and any irrelevant files are not longer needed.
Review
[base.py](openhands-sdk/openhands/sdk/context/condenser/base.py) 
[agent.py](openhands-sdk/openhands/sdk/agent/agent.py) - specifically
def step(
        self,
        state: ConversationState,
        on_event: ConversationCallbackType,
    ) -> None:
        # Check for pending actions (implicit confirmation)
        # and execute them before sampling new actions.
        pending_actions = ConversationState.get_unmatched_actions(state.events)
        if pending_actions:
            logger.info(
                "Confirmation mode: Executing %d pending action(s)",
                len(pending_actions),
            )
            self._execute_actions(state, pending_actions, on_event)
            return

        # If a condenser is registered with the agent, we need to give it an
        # opportunity to transform the events. This will either produce a list
        # of events, exactly as expected, or a new condensation that needs to be
        # processed before the agent can sample another action.
        if self.condenser is not None:
            view = View.from_events(state.events)
            condensation_result = self.condenser.condense(view)

            match condensation_result:
                case View():
                    llm_convertible_events = condensation_result.events

                case Condensation():
                    on_event(condensation_result)
                    return None

        else:
            llm_convertible_events = [
                e for e in state.events if isinstance(e, LLMConvertibleEvent)
            ]

One implementation approach would be to create a  `LLMRelevanceCondenserTool` and `LLMRelevanceCondenser` - so both a Tool and a Condensor.
The LLM will invoke the LLMRelevanceCondenserTool at any point that it sees "no longer" relevant tool calls and responses in the chat history.

The Agent would invoke the line `if self.condenser is not None:` and the logic of the LLMRelevanceCondenser would be based on invocations of LLMRelevanceCondenserTool

A potential alternative is an approach similar to [llm_summarizing_condenser.py](openhands-sdk/openhands/sdk/context/condenser/llm_summarizing_condenser.py) - where the condensor directly makes a `self.llm.completion(` call. However, this doesn't feel optimal - as it could create a large amount of additional LLM calls; compared to the tool be called in parallel to other tool calls as needed.

Focus the document on these initial design decisions whilst highlighting other approaches which could be considered.
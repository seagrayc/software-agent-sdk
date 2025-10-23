
# Conversation Persistence and Network Mapping for Gemini

This document explains how the agent-sdk handles conversation persistence and maps conversation events to network requests sent to the Gemini LLM, with a focus on Gemini-specific details.

## Conversation Persistence

The agent-sdk employs a two-tiered approach for persisting conversation state:

1.  **Base State:** The core `ConversationState` object, containing configuration like the agent, workspace, and current status, is serialized to JSON and saved as `base_state.json` within the specified `persistence_dir`. This allows for resuming conversations with their fundamental settings intact.
2.  **Events:** Individual conversation events (user messages, agent actions, LLM responses, tool outputs, etc.) are stored in an `EventLog`. This `EventLog` utilizes a `FileStore` (e.g., `LocalFileStore`) to manage a collection of event files, likely in JSON format, within a dedicated directory (e.g., `EVENTS_DIR`).

When resuming a conversation, the `ConversationState.create` method first attempts to load `base_state.json`. If found, it deserializes the state and then attaches the `EventLog` to load the historical events.

## Mapping Conversation Events to Network Requests (Gemini)

The process of sending conversation history to an LLM involves several layers:

1.  **Event to Message Conversion:** Conversation events stored in the `EventLog` are processed and converted into `Message` objects. The `Message` class defines a standardized structure for representing conversational turns, including roles, content (text, images), tool calls, and reasoning.

2.  **Message to LLM API Format:** The `Message` object's methods, such as `to_llm_dict` and `MessageToolCall`'s serialization methods (`to_chat_dict`, `to_responses_dict`), transform the standardized `Message` structure into formats compatible with different LLM providers.

3.  **LiteLLM Abstraction:** The `litellm` library acts as an abstraction layer. The `LLM` class in the agent-sdk configures `litellm` with provider-specific details (model name, API key, base URL). When `LLM.completion` is called, `litellm` takes the formatted messages and parameters and constructs the actual API request.

4.  **Gemini-Specific Network Request:**
    *   **Transformation:** For Gemini, the `litellm` library uses specific modules (e.g., `litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py`). The `sync_transform_request_body` function is responsible for converting the `Message` objects into Gemini's expected request payload.
    *   **Request Structure:** Gemini's API typically expects a `contents` array, where each element represents a turn in the conversation. This array contains objects with `role` (e.g., "user", "model") and `parts` (containing text, inline data, or function calls).
    *   **Parameters:** Standard LLM parameters like `temperature`, `top_p`, `max_output_tokens`, and `stop_sequences` are mapped to Gemini's specific parameter names. Tool definitions are included in the `tools` and `tool_config` fields of the request.
    *   **Example Snippet (Conceptual):**
        ```json
        {
          "contents": [
            {
              "role": "user",
              "parts": [{"text": "Hello, how are you?"}]
            },
            {
              "role": "model",
              "parts": [{"text": "I'm doing well, thank you!"}]
            },
            // ... more conversation history
          ],
          "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 256,
            "stopSequences": [],
            // ... other Gemini-specific config
          },
          "tools": [
            // ... tool definitions for function calling
          ]
        }
        ```
    *   **API Endpoint:** The `httpx` client in `litellm` makes a POST request to a Vertex AI endpoint, typically structured like: `https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/publishers/google/models/{model}:generateContent` (or similar variations for chat-specific endpoints).

5.  **Response Handling:**
    *   Gemini's responses are parsed by `litellm`'s `transform_response` and `ModelResponseIterator` functions.
    *   The response typically includes `candidates` (containing `content` with `parts`), `usageMetadata`, and potentially other metadata like `grounding_metadata` or `safety_ratings`.
    *   `litellm` maps this Gemini-specific response structure back into its generic `ModelResponse` and `ModelResponseStream` objects, which are then used by the agent-sdk.

## Differences Between Model Providers

While the core workflow (Event -> Message -> LiteLLM -> Provider API) remains consistent, the specific network request and response formats vary significantly between providers. `litellm` abstracts these differences:

*   **OpenAI:** Uses the Chat Completions API format (`messages` array with `role`, `content`, `tool_calls`).
*   **Anthropic:** Uses a different message structure and has specific parameters for "thinking blocks" and extended reasoning.
*   **Gemini:** Uses the `contents` array with `parts`, specific `generationConfig`, and a distinct tool definition/response schema.

The `LLM` class and `litellm`'s provider-specific implementations handle these variations.

## Format of Conversation Events Persisted

*   **Base State:** `ConversationState` is persisted as a JSON string.
*   **Events:** Individual events are stored within an `EventLog` managed by a `FileStore`. While the exact file format isn't detailed here, it's standard practice for such logs to be stored as JSON objects, ensuring structured and parseable data.

## Network Request Format

The network request format is **not** the exact network request previously sent, nor is it a direct Open Hands format or an llmlite format. Instead:

1.  The agent-sdk's `Message` objects are transformed into a provider-agnostic intermediate format.
2.  `litellm` then takes this intermediate format and constructs the **provider-specific API request payload** (e.g., JSON for Gemini's `generateContent` endpoint).
3.  This provider-specific payload is what is actually sent over the network.

Therefore, the network request format is dictated by the LLM provider's API schema, as implemented and translated by `litellm`.

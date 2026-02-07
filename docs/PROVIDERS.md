# Provider Comparison

This document compares the five LLM providers supported by `agent-driver-rs`,
covering authentication, streaming mechanisms, tool formats, configuration, and
known gotchas.

## Comparison Table

| Aspect | Anthropic | OpenAI | Bedrock | OpenRouter | Ollama |
|---|---|---|---|---|---|
| Auth | API key (`ANTHROPIC_API_KEY`) | API key (`OPENAI_API_KEY`) | AWS IAM (auto from env) | API key (`OPENROUTER_API_KEY`) | None (local) |
| Streaming | SSE (HTTP) | SDK stream | SDK `.recv()` | SSE (HTTP) | SDK stream |
| Stream adapter | `buffered_sse_stream` | `buffered_sdk_stream` | Custom `.recv()` loop | `buffered_sse_stream` | `buffered_sdk_stream` |
| Tool format | `input_schema` | `parameters` | `inputSchema` (Converse API) | `parameters` (OpenAI compat) | `parameters` (OpenAI compat) |
| Tool serializer | `ToolFormat::Claude` | `ToolFormat::OpenAi` | `ToolFormat::Claude` | `ToolFormat::OpenAi` | `ToolFormat::OpenAi` |
| Tool stop signal | `stop_reason: "tool_use"` | `finish_reason: "tool_calls"` | `stopReason: "tool_use"` | `finish_reason: "tool_calls"` | `done_reason: "stop"` |
| Model naming | `claude-sonnet-4-20250514` | `gpt-4o` | Inference profile ARN | `provider/model` (must contain `/`) | `llama3.2:3b` |
| Temperature | 0.0-1.0 | 0.0-2.0 (not on reasoning models) | 0.0-1.0 | Varies by model | 0.0-2.0 |
| Parallel tool calls | Yes | Yes | Yes | Yes | Yes |
| Extended thinking | Yes (`budget_tokens`) | o1/o3/GPT-5 reasoning (`reasoning_effort`) | Via Claude models | Depends on underlying model | No |
| Feature flag | `anthropic` | `openai` | `bedrock` | `openrouter` | `ollama` |
| Extra deps | None | `async-openai` | `aws-sdk-bedrockruntime`, `aws-config`, `aws-smithy-types` | None | `ollama-rs` |

## Per-Provider Configuration

### Anthropic

```bash
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-20250514
ANTHROPIC_MAX_TOKENS=4096
ANTHROPIC_TEMPERATURE=0.7          # optional, 0.0-1.0
ANTHROPIC_THINKING_BUDGET=4096     # optional, minimum 1024, must be < max_tokens
```

**Model variants:**
- `claude-opus-4-20250514`
- `claude-sonnet-4-20250514` (default)
- `claude-3-5-haiku-20241022`
- Any custom string

**Extended thinking:** Set `ANTHROPIC_THINKING_BUDGET` to enable. Must be at
least 1024 tokens and strictly less than `ANTHROPIC_MAX_TOKENS`. Supported on
Claude Opus 4 and Claude Sonnet 4 models.

**Gotchas:** None specific.

---

### OpenAI

```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o                # default
OPENAI_MAX_TOKENS=4096
OPENAI_TEMPERATURE=0.7             # optional, not allowed on reasoning models
OPENAI_REASONING_EFFORT=medium     # optional: none/minimal/low/medium/high/xhigh
```

**Model variants:**
- `gpt-4o` (default), `gpt-4o-mini`
- `gpt-5`, `gpt-5.1`, `gpt-5.2`
- `o1`, `o1-mini`, `o3`, `o3-mini`
- Any custom string

**Gotchas:**
- **o3 and o3-mini do not support streaming.** The library checks
  `model.supports_streaming()` and these models return `false`.
- **Temperature is not allowed on reasoning models.** GPT-5.x, o1, o3 series
  models do not support the temperature parameter. Config validation rejects
  a temperature setting on these models.
- **Reasoning configuration** is only available on reasoning models (GPT-5.x,
  o1, o3 series). Set `OPENAI_REASONING_EFFORT` to control the reasoning effort level.

---

### Bedrock

```bash
AWS_REGION=us-east-1               # or via AWS config/credentials chain
BEDROCK_MODEL=claude-sonnet-4.5    # default
BEDROCK_MAX_TOKENS=4096
BEDROCK_TEMPERATURE=0.7            # optional
BEDROCK_INFERENCE_PROFILE=us.anthropic.claude-sonnet-4-5-20250929-v1:0
```

**Model variants:**
- `claude-sonnet-4`, `claude-sonnet-4.5` (default)
- `claude-opus-4`, `claude-opus-4.5`
- `claude-haiku-4.5`, `claude-haiku-3.5` (alias: `claude-3.5-haiku`)
- Any custom model ID string

**Gotchas:**
- **Modern Claude models require inference profiles.** You must set
  `BEDROCK_INFERENCE_PROFILE` to a valid inference profile ARN like
  `us.anthropic.claude-sonnet-4-5-20250929-v1:0`. Without this, the API
  call will fail for recent Claude models.
- **`EventReceiver` uses `.recv()`, not `StreamExt::next()`.** The AWS SDK's
  streaming response does not implement `futures::Stream`, so the Bedrock
  provider has its own polling loop instead of using the shared `buffered_sdk_stream`
  adapter. This means the cancellation pattern is slightly different -- the
  provider manually checks cancellation around each `.recv()` call.
- **`aws_smithy_types::Document::Null`** is a unit variant, not `Null(true)`.
  Tool input JSON conversion must handle this correctly.

---

### OpenRouter

```bash
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=anthropic/claude-sonnet-4    # must contain "/"
OPENROUTER_MAX_TOKENS=4096
OPENROUTER_TEMPERATURE=0.7                    # optional
```

**Model variants:**
- `anthropic/claude-sonnet-4` (default)
- `openai/gpt-4o`
- `google/gemini-2.0-flash`
- `meta-llama/llama-3.3-70b`
- Any custom string in `provider/model` format

**Provider preferences:** The config supports optional `ProviderPreferences`
with `allow`, `deny`, and `require_primary` fields for routing control.

**Gotchas:**
- **Model names must be in `provider/model` format** (i.e., they must contain
  a `/`). Config validation enforces this for custom model strings.
- **May not emit `ContentBlockStop` events.** The library handles this by
  calling `flush_pending()` on `CollectedResponse` when the stream ends, which
  captures any accumulated content that was not explicitly closed. The
  `CollectedResponse::apply_delta(ToolUseStart { .. })` method also auto-finalises
  any previous pending tool use block, handling interleaved tool calls gracefully.
- **Uses the same SSE format as Anthropic** but with OpenAI-compatible tool
  serialization (`ToolFormat::OpenAi`).

---

### Ollama

```bash
OLLAMA_BASE_URL=http://localhost:11434    # optional, this is the default
OLLAMA_MODEL=llama3.2                     # default
OLLAMA_NUM_CTX=8192                       # REQUIRED: context window size
OLLAMA_MAX_TOKENS=4096                    # optional
OLLAMA_TEMPERATURE=0.7                    # optional
OLLAMA_KEEP_ALIVE=5m                      # optional: minutes, "indefinite"/-1, "unload"/0
```

**Well-known model variants:**
- `llama3.2`, `llama3.2:3b`
- `qwen3:30b`, `qwen3:14b`, `qwen3-coder:14b`
- `deepseek-r1`, `mistral`
- Any custom model string (matching what Ollama has pulled locally)

**Context window (`OLLAMA_NUM_CTX`):** This is **required** and must be a
positive integer. Ollama does not auto-detect context window sizes, so the
library requires explicit configuration. Uses the `NumCtx` newtype with
`NonZeroU32` validation.

**Keep-alive options:**
- `indefinite` or `-1`: Keep model loaded in VRAM indefinitely
- A number: Minutes to keep the model loaded (e.g., `5` for 5 minutes)
- `unload` or `0`: Unload model immediately after each request

**Gotchas:**
- **Ollama must be running locally** (or at the configured `OLLAMA_BASE_URL`).
  The library does not start Ollama for you.
- **Some models do not support tool calling.** Not all Ollama models have been
  fine-tuned for function calling. Models like `qwen3` and `llama3.2` generally
  support it; smaller or older models may not.
- **No extended thinking support.** Ollama models do not emit thinking/reasoning
  content blocks.

## Streaming Format Differences

All providers emit the same `StreamEvent` types, but the raw wire formats differ
significantly. Here is how each provider's events map to the standard types:

### Anthropic and OpenRouter (SSE)

These providers use Server-Sent Events over HTTP. The SSE event types map to
`StreamEvent` as follows:

| SSE event type | StreamEvent |
|---|---|
| `message_start` | `Started { metadata }` |
| `content_block_start` | `ContentBlockStart { index, block_type }` |
| `content_block_delta` (text) | `Delta(TextDelta { text })` |
| `content_block_delta` (thinking) | `Delta(ThinkingDelta { thinking })` |
| `content_block_delta` (tool input) | `Delta(ToolInputDelta { id, partial_json })` |
| `content_block_stop` | `ContentBlockStop { index }` |
| `message_stop` / `message_delta` | `Completed { metadata }` |

Both use the `buffered_sse_stream` adapter. OpenRouter additionally checks for a
`[DONE]` marker in the SSE data stream (configured via the `done_message` parameter).

### OpenAI and Ollama (SDK Stream)

These providers use SDK client libraries that return `futures::Stream` items:

| SDK event | StreamEvent |
|---|---|
| First chunk (role present) | `Started { metadata }` |
| `delta.content` | `Delta(TextDelta { text })` |
| `delta.tool_calls[].function.name` | `Delta(ToolUseStart { id, name })` |
| `delta.tool_calls[].function.arguments` | `Delta(ToolInputDelta { id, partial_json })` |
| Stream ends (inner yields `None`) | `Completed { metadata }` (via `on_end` callback) |

Both use the `buffered_sdk_stream` adapter. Block lifecycle events
(`ContentBlockStart`/`ContentBlockStop`) are synthesised by the parse function
based on the presence of tool call start markers and content boundaries.

### Bedrock (AWS SDK `.recv()`)

Bedrock uses the AWS SDK's `ConverseStreamOutput` enum with its own variant names:

| SDK variant | StreamEvent |
|---|---|
| `ContentBlockStart` | `ContentBlockStart { index, block_type }` |
| `ContentBlockDelta` (text) | `Delta(TextDelta { text })` |
| `ContentBlockDelta` (tool input) | `Delta(ToolInputDelta { id, partial_json })` |
| `ContentBlockStop` | `ContentBlockStop { index }` |
| `MessageStart` | `Started { metadata }` |
| `MessageStop` | `Completed { metadata }` |
| `Metadata` | Updates `CompletionMetadata` (stop reason, usage) |

Bedrock does not use either shared stream adapter. Its provider implementation
has a manual `loop` that calls `event_receiver.recv().await` and converts each
variant to `StreamEvent`, checking cancellation on each iteration via
`tokio::select! { biased; }`.

## Tool Serialization Formats

The `ToolFormat` enum in `tool/serializer.rs` handles the two JSON shapes:

### Claude format (`ToolFormat::Claude`)

Used by Anthropic (direct API) and Bedrock (Converse API):

```json
{
  "name": "read_file",
  "description": "Read a file from disk",
  "input_schema": {
    "type": "object",
    "properties": { "path": { "type": "string" } },
    "required": ["path"]
  }
}
```

### OpenAI format (`ToolFormat::OpenAi`)

Used by OpenAI, OpenRouter, and Ollama:

```json
{
  "type": "function",
  "function": {
    "name": "read_file",
    "description": "Read a file from disk",
    "parameters": {
      "type": "object",
      "properties": { "path": { "type": "string" } },
      "required": ["path"]
    },
    "strict": true
  }
}
```

The `strict` field is only included when using `ToolFormat::openai_strict()`.

### Tool result formats

Claude format:
```json
{
  "type": "tool_result",
  "tool_use_id": "call_123",
  "content": "File contents here",
  "is_error": false
}
```

OpenAI format:
```json
{
  "role": "tool",
  "tool_call_id": "call_123",
  "content": "File contents here"
}
```

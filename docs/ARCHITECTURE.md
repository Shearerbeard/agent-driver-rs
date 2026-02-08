# Architecture

This document describes the internal architecture of `agent-driver-rs`: how modules
depend on each other, how data flows through the system, and how concurrency is managed.

## Module Dependency Flow

```
lib.rs
  |
  +-- error.rs                (no deps -- define first!)
  |
  +-- types/                  (depends on: error)
  |   +-- model.rs            (ModelId, MaxTokens, Temperature)
  |   +-- message.rs          (Role, Message, ContentBlock, ToolName, ToolCallId, SystemPrompt)
  |   +-- correlation.rs      (CorrelationId)
  |
  +-- config/                 (depends on: types, error)
  |   +-- common.rs           (ApiKey, AwsRegion -- shared primitives)
  |   +-- provider.rs         (ProviderConfig enum -- the top-level discriminator)
  |   +-- anthropic.rs        (AnthropicConfig, AnthropicModel, ThinkingConfig)
  |   +-- openai.rs           (OpenAiConfig, OpenAiModel, ReasoningConfig)
  |   +-- bedrock.rs          (BedrockConfig, BedrockModel)
  |   +-- openrouter.rs       (OpenRouterConfig, OpenRouterModel, ProviderPreferences)
  |   +-- ollama.rs           (OllamaConfig, OllamaModel, NumCtx, KeepAlive)
  |
  +-- streaming.rs            (depends on: types, error)
  |   StreamEvent, StreamDelta, StreamHandle, CollectedResponse,
  |   CompletionStream, CompletionMetadata, StopReason, TokenUsage
  |
  +-- tool/                   (depends on: types, error)
  |   +-- types.rs            (ToolSchema, ToolSource, McpServerName, PluginId)
  |   +-- definition.rs       (ToolDefinition, ToolAnnotations)
  |   +-- executor.rs         (Tool trait, ToolContext, ToolInput, ToolResult, DynTool, FnTool)
  |   +-- registry.rs         (ToolRegistry -- RwLock<HashMap<ToolName, DynTool>>)
  |   +-- serializer.rs       (ToolFormat -- Claude/OpenAI serialization)
  |   +-- mcp.rs              (McpConnection, McpManager) [feature = "mcp"]
  |
  +-- provider/               (depends on: types, error, streaming, tool)
  |   +-- provider.rs         (Provider trait, ProviderKind, CompletionRequest,
  |   |                        ProviderContext, ProviderInfo, ProviderCapabilities,
  |   |                        CompletionConfig, BoxedProvider, SharedProvider)
  |   +-- stream_adapter.rs   (buffered_sse_stream, buffered_sdk_stream)
  |   +-- retry.rs            (with_retry, RetryConfig)
  |   +-- mock.rs             (MockProvider) [cfg(test) or feature = "test-support"]
  |   +-- anthropic.rs        [feature = "anthropic"]
  |   +-- openai.rs           [feature = "openai"]
  |   +-- bedrock.rs          [feature = "bedrock"]
  |   +-- openrouter.rs       [feature = "openrouter"]
  |   +-- ollama.rs           [feature = "ollama"]
  |
  +-- task/                   (depends on: types, error)
  |   +-- pool.rs             (TaskPool -- registration-before-execution)
  |   +-- handle.rs           (TaskHandle -- JoinHandle wrapper with cancellation)
  |   +-- spawn.rs            (TrackedSpawn -- extension trait for CorrelationId)
  |
  +-- session.rs              (depends on: provider, streaming, tool, types, error)
  |   Session, SessionBuilder, SessionConfig
  |
  +-- agent/                  (depends on: session, streaming, tool, types, error)
      +-- config.rs           (AgentLoopConfig, MaxToolDepth)
      +-- observer.rs         (AgentObserver trait, AgentEvent, LoopStopReason)
      +-- driver.rs           (AgentLoop, AgentOutcome)
```

**Key rule:** `error.rs` has no dependencies and must be defined first. Everything
else depends on it (directly or transitively). The dependency graph is strictly
acyclic -- lower layers never import from higher layers.

## Data Flow: Message to Response

### Non-streaming path (`session.send()`)

```
 User code               Session                   Provider           LLM API
 ---------               -------                   --------           -------
    |                       |                          |                  |
    | send("Hello")         |                          |                  |
    |---------------------->|                          |                  |
    |                       | add Message::user        |                  |
    |                       | to history               |                  |
    |                       |                          |                  |
    |                       | build_completion_request()|                  |
    |                       | (snapshot system_prompt,  |                  |
    |                       |  messages, tools under    |                  |
    |                       |  separate locks)          |                  |
    |                       |                          |                  |
    |                       | complete_stream(req, ctx) |                  |
    |                       |------------------------->|                  |
    |                       |                          | HTTP / SDK call  |
    |                       |                          |----------------->|
    |                       |                          |                  |
    |                       |                          | StreamHandle     |
    |                       |<-------------------------|                  |
    |                       |                          |                  |
    |                       | handle.collect()         |                  |
    |                       | (accumulates deltas      |                  |
    |                       |  into CollectedResponse) |                  |
    |                       |                          |                  |
    |                       | add Message::assistant   |                  |
    |                       | to history               |                  |
    |                       |                          |                  |
    | CollectedResponse     |                          |                  |
    |<----------------------|                          |                  |
```

Step by step:

1. User calls `session.send("Hello")` or `session.send_streaming("Hello")`.
2. Session creates a `Message::user("Hello")` and appends it to the message history
   (acquiring the `messages` write lock, then releasing it).
3. Session snapshots state into a `CompletionRequest` by reading three independent
   locks sequentially: `system_prompt` (RwLock), `messages` (RwLock), and
   `tools` (via `registry.list()`). Each lock is acquired and released individually.
4. Session calls `provider.complete_stream(request, ctx)` where `ctx` carries
   a child `CancellationToken`, the session's `CorrelationId`, and a `TaskTracker`.
5. The provider converts `CompletionRequest` to its API-specific format:
   - Anthropic/OpenRouter: HTTP POST with JSON body, SSE response
   - OpenAI/Ollama: SDK client call returning a `Stream`
   - Bedrock: AWS SDK `converse_stream()` returning an `EventReceiver`
6. The provider wraps the response in a `StreamHandle` containing the
   `CompletionStream` (a `Pin<Box<dyn Stream<Item = Result<StreamEvent, StreamError>>>>`)
   and the child cancellation token.
7. Each provider's stream emits standardised `StreamEvent` variants regardless
   of the underlying wire format (SSE, SDK stream, or `.recv()` loop).
8. `StreamHandle::collect()` races the stream against the cancellation token
   using `tokio::select! { biased; }`. It accumulates `StreamDelta`s into a
   `CollectedResponse` via `apply_delta()` and `finalize_block()`, tracking
   block types per index in a `HashMap<usize, ContentBlockType>`.
9. On the non-streaming path (`send()`), Session adds the assistant response
   as `Message::with_content(Role::Assistant, response.content.clone())` to
   history. On the streaming path (`send_streaming()`), the caller is responsible
   for adding the assistant message (the agent loop handles this).

### Streaming path (`session.send_streaming()`)

The streaming path returns the `StreamHandle` directly to the caller, who can
consume events one at a time via the `Stream` trait or call `collect()`. The
`AgentLoop` uses this path -- it calls `collect_with_observer()` to forward
`TextDelta` and `ThinkingDelta` events to the observer in real time.

## Tool Flow

```
 Registry        Session             Provider          LLM         AgentLoop
 --------        -------             --------          ---         ---------
    |                |                   |               |              |
    | register(tool) |                   |               |              |
    |<---------------|                   |               |              |
    |                |                   |               |              |
    | list() -> Vec<ToolDefinition>      |               |              |
    |--------------->|                   |               |              |
    |                | include in        |               |              |
    |                | CompletionRequest |               |              |
    |                |------------------>|               |              |
    |                |                   | serialize via  |              |
    |                |                   | ToolFormat     |              |
    |                |                   |-------------->|              |
    |                |                   |               |              |
    |                |                   | tool_use in   |              |
    |                |                   | stream events |              |
    |                |                   |<--------------|              |
    |                |                   |               |              |
    |                | CollectedResponse with            |              |
    |                | ContentBlock::ToolUse             |              |
    |                |<------------------|               |              |
    |                |                                   |              |
    |                |                   execute_tool(id, name, input)  |
    |                |<--------------------------------------------|
    |                |                                                  |
    | get(name) -> DynTool                                             |
    |--------------->|                                                  |
    |                | tool.execute(&input, &ctx)                       |
    |                | -> ToolResult                                    |
    |                |                                                  |
    |                | add Message::tool_result(id, content, is_error)  |
    |                | to history                                       |
    |                |                                                  |
    |                | continue_streaming()                             |
    |                | (re-snapshots tools from registry)               |
    |                |------------------------------------------->      |
```

1. Tools are registered in the `ToolRegistry` -- either native `FnTool` closures
   or `McpToolWrapper` instances from MCP servers.
2. When building a `CompletionRequest`, `Session` calls `registry.list()` to
   snapshot all currently registered `ToolDefinition`s. This snapshot happens
   *each turn*, so tools added or removed between turns are picked up.
3. The provider serializes definitions using `ToolFormat::claude()` (Anthropic,
   Bedrock) or `ToolFormat::openai()` (OpenAI, OpenRouter, Ollama). The two
   formats differ in JSON structure:
   - Claude: `{ name, description, input_schema }`
   - OpenAI: `{ type: "function", function: { name, description, parameters } }`
4. When the model decides to use a tool, the stream emits:
   - `StreamDelta::ToolUseStart { id, name }` -- begins a tool call block
   - `StreamDelta::ToolInputDelta { id, partial_json }` -- incremental JSON input
   - `StreamEvent::ContentBlockStop` -- finalises the block
5. `CollectedResponse` accumulates these into `ContentBlock::ToolUse { id, name, input }`.
   If the JSON is malformed or the stream ends early, the input falls back to `{}`.
6. `Session::execute_tool()` looks up the tool by name, calls `tool.execute(&input, &ctx)`
   with a `ToolContext` carrying a child `CancellationToken`, and adds the result to
   history as `Message::tool_result()`.
7. `Session::continue_streaming()` re-snapshots state (including fresh tool list)
   and sends the updated history for the next model turn.

## Agent Loop State Machine

The `AgentLoop` orchestrates multi-turn tool-calling conversations. It borrows
a `&Session` and drives the send/collect/execute/continue cycle.

```
                    +-------------+
                    |   run()     |
                    | send_streaming
                    +------+------+
                           |
                    +------v------+
                    |  collect    |
                    |  response   |
                    +------+------+
                           |
                    +------v------+     No tool_use
                    | has_tool_use| ------------------> complete_loop(EndTurn)
                    |   ?         |
                    +------+------+
                           | Yes
                    +------v------+
                    | depth >=    |     Yes
                    | max_depth?  | ------------------> complete_loop(MaxToolDepthReached)
                    +------+------+
                           | No
                    +------v------+
                    | cancelled?  |     Yes
                    |             | ------------------> complete_loop(Cancelled)
                    +------+------+
                           | No
                    +------v------+
                    | execute     |
                    | tools       |
                    +------+------+
                           |
                    +------v------+
                    | tool_error  |     Yes + !continue_on_error
                    |   ?         | ------------------> complete_loop(ToolError)
                    +------+------+
                           | No / continue
                    +------v------+
                    | continue    |
                    | streaming   | ----> back to "collect response"
                    +-------------+
```

**Key behaviours:**

- **Depth counting:** Counts *tool execution rounds*, not model responses. A
  single round may execute multiple parallel tool calls. The default limit is
  25 rounds (`MaxToolDepth::default()`).
- **Observer forwarding:** During `collect_with_observer()`, `TextDelta` and
  `ThinkingDelta` events are forwarded to the observer in real time. Other stream
  events (block lifecycle, metadata) are consumed silently.
- **Tool error handling:** When `continue_on_tool_error` is `true` (the default),
  tool errors are sent back to the model as error results, letting it recover.
  When `false`, the loop stops on the first tool error.
- **`complete_loop()` deduplication:** All exit paths go through `complete_loop()`,
  which fires the `LoopComplete` observer event and assembles the `AgentOutcome`.

The `AgentOutcome` returned from `run()` contains:
- `final_response` -- the last `CollectedResponse`
- `responses` -- all responses from every turn (including intermediate tool turns)
- `stop_reason` -- a `LoopStopReason` variant
- `iterations` -- number of tool execution rounds completed

## Concurrency Model

### Split Locks in Session

Session uses separate `RwLock`s for each piece of independent state:

```rust
pub struct Session {
    system_prompt: RwLock<SystemPrompt>,  // Rarely changes
    messages: RwLock<Vec<Message>>,        // Frequently appended
    tools: Arc<ToolRegistry>,              // Has its own RwLock internally
    // ...
}
```

This avoids a single lock bottleneck. When building a `CompletionRequest`, each
lock is acquired and released independently -- the system prompt lock is dropped
before the messages lock is acquired, and so on.

### CancellationToken Hierarchy

```
Session root token
  |
  +-- child token (provider call 1)
  |
  +-- child token (provider call 2)
  |
  +-- child token (provider call N)
  |
  +-- child token (tool execution via ToolContext)
```

The Session owns a root `CancellationToken` (from `tokio-util`). Each call to
`new_provider_context()` creates a child token. Cancelling the root (via
`session.cancel()` or `session.shutdown()`) cascades to all child tokens,
causing all in-flight provider streams to terminate.

Providers check cancellation in their stream loops using `tokio::select! { biased; }`,
which checks the cancellation future before polling the network stream. The
`biased` keyword ensures deterministic cancellation -- when both futures are
ready, cancellation always wins.

### TaskPool

The `TaskPool` provides a registration-before-execution guarantee:

1. A oneshot channel gates the task -- it cannot execute until after registration.
2. The task is inserted into a `HashMap<CorrelationId, RegisteredTask>` while
   the gate is still closed.
3. Only then is the gate opened (`start_tx.send(())`).

This eliminates TOCTOU races where a task could complete (and try to deregister
itself) before it was ever registered. A secondary TOCTOU guard re-checks the
`accepting` flag after insertion to handle races with `shutdown()`.

Shutdown is three-phase:
1. `accepting.store(false)` -- stop new tasks
2. `root_token.cancel()` -- cancel all existing tasks
3. `tracker.wait()` -- wait for all tasks to complete

### ToolRegistry

The `ToolRegistry` wraps a `RwLock<HashMap<ToolName, DynTool>>`. Multiple readers
can snapshot tools concurrently (via `list()` and `get()`), while writes
(`register()` and `unregister()`) take an exclusive lock. This is fine because
tool registration/removal is infrequent compared to reads.

### Stream Adapters

Two shared streaming helpers eliminate duplication across providers:

**`buffered_sse_stream`** (used by Anthropic and OpenRouter):
- Wraps a `reqwest_eventsource::EventSource`
- Uses `futures::stream::unfold` with a `VecDeque` buffer for multi-event SSE messages
- Calls provider's parse function for each SSE `data` payload
- Checks a provider-specific "done" marker (e.g., `"[DONE]"` for OpenRouter)
- Cancellation-aware via `tokio::select! { biased; }`
- `on_done` callback fires when the done marker is received, returning an optional
  final `StreamEvent` (typically `Completed` with metadata)

**`buffered_sdk_stream`** (used by OpenAI and Ollama):
- Wraps any `futures::Stream` from an SDK client
- Same `unfold` + `VecDeque` buffer pattern
- `on_end` callback fires when the inner stream yields `None`
- Uses `SdkState` with an `ended: bool` flag to prevent infinite polling after
  the stream terminates (fixes a bug where `on_end` returning `Some` would cause
  the unfold to re-poll the exhausted inner stream indefinitely)

**Bedrock** does not use either adapter. The AWS SDK's `EventReceiver` has a
`.recv()` API that does not implement `futures::Stream`. The Bedrock provider
has its own polling loop that calls `.recv()` in a `loop` with cancellation checks.

## Retry Logic

The `with_retry()` function in `provider/retry.rs` provides exponential backoff
for rate-limited requests:

- Only retries on `ProviderError::RateLimited` -- all other errors are returned
  immediately without retry.
- Respects the `retry_after` duration from the provider if present; otherwise
  uses exponential backoff via the `backoff` crate.
- Default configuration: 3 retries, 500ms initial interval, 30s max interval,
  2x multiplier.

## Feature Flags

Each provider is gated behind a feature flag. Only enabled providers are compiled:

```toml
[features]
default = ["anthropic", "openai", "openrouter"]
anthropic = []
openai = ["dep:async-openai"]
bedrock = ["dep:aws-sdk-bedrockruntime", "dep:aws-config", "dep:aws-smithy-types"]
openrouter = []
ollama = ["dep:ollama-rs"]
mcp = ["dep:rmcp"]
mcp-http = ["mcp", "rmcp/transport-streamable-http-client-reqwest"]
```

The `MockProvider` is available under `cfg(test)` (always in test builds) or
the `test-support` feature flag (for downstream crate integration tests).

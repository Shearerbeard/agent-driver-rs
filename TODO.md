# TODO - agent-driver-rs

## Next Up: Post-Implementation Cleanup & Hardening

The agentic loop + MCP integration was implemented in a single deep session and needs
a thorough review pass for correctness, code quality, and adherence to project principles.

### Known Bugs (from integration testing)

- [ ] **Bedrock "Invalid request: service error"** on final text response after tool iterations.
  Likely a message formatting issue in `continue_streaming()` — Bedrock's converse API
  may reject the message history shape after tool_use + tool_result turns. Reproduce:
  `echo "List files in /tmp" | PROVIDER=bedrock cargo run --features "bedrock mcp" --bin chat -- --mcp 'npx -y @modelcontextprotocol/server-filesystem /tmp'`

- [ ] **`ToolInput::from_value` rejects null/empty args** — MCP tools with no required
  parameters (e.g., `list_allowed_directories`) fail when the model sends `null` or no
  arguments instead of `{}`. The model retries and eventually sends `{}`, but this wastes
  a tool iteration. Fix `ToolInput::from_value` to treat `null`/missing as empty object.

### Code Review & Cleanup (run with reviewer agents)

- [ ] **Review `src/agent/driver.rs`** — The core loop went through multiple borrow-checker
  refactors (methods extracted to free functions). Verify the final structure is clean,
  well-documented, and idiomatic. Check error handling paths.

- [ ] **Review `src/tool/mcp.rs`** — Full rewrite from stub. Verify rmcp API usage is correct,
  error handling covers all failure modes, and the `McpToolWrapper` properly converts
  all rmcp Content types (currently only handles text).

- [ ] **Review `src/provider/openrouter.rs`** — VecDeque buffering was added reactively.
  Verify the `UnfoldState` pattern is clean and consistent with Bedrock's approach.
  Consider extracting a shared buffered-stream helper if patterns are identical.

- [ ] **Review `src/provider/bedrock.rs`** — VecDeque buffering + stop_reason state changes.
  Verify no regressions in non-tool-use streaming paths.

- [ ] **Review `src/streaming.rs`** — `flush_pending()` was added as a safety net. Verify
  it doesn't cause duplicate content blocks when providers DO emit `ContentBlockStop`.
  Add unit tests for the flush path.

- [ ] **Review `src/bin/chat.rs`** — Large rewrite for agent loop + MCP. Check error handling,
  graceful shutdown of MCP connections, and observer output formatting.

- [ ] **Review `src/session.rs`** — `continue_streaming()` duplicates logic from
  `send_streaming()`. Consider extracting shared request-building logic.

- [ ] **Audit for code duplication** — The VecDeque buffering pattern is duplicated between
  Bedrock and OpenRouter providers. Extract a shared `BufferedStreamUnfold` if appropriate.

- [ ] **Verify design principles compliance** — Split locks, cancellation everywhere,
  no free spawning, newtype validation. Check all new code follows CLAUDE.md principles.

### Testing Gaps

- [ ] **Unit tests for `flush_pending()`** — Verify flush works correctly for text, thinking,
  and tool_use pending state. Verify no double-flush issues.

- [ ] **Unit tests for agent loop** — Currently only has `stop_reason_mapping` and
  `agent_outcome_default_iterations`. Needs tests with mock tools for:
  - Single tool call → text response
  - Multi-turn tool calls
  - MaxToolDepth reached
  - Tool error with `continue_on_tool_error: false`
  - Cancellation mid-loop

- [ ] **Unit tests for MCP** — Only has `extract_text_content` tests. Needs tests for:
  - McpToolWrapper content type conversion (text, error, mixed)
  - Tool schema conversion from rmcp format

- [ ] **Integration test** — End-to-end test with mock MCP server (or in-process tool)
  that exercises the full agent loop without requiring live API keys.

### Documentation

- [ ] **Write ADR-0002** — Capture final design decisions for agent loop and dynamic tools
  (tool depth counting, observer pattern, flush_pending safety net, etc.)

- [ ] **Rustdoc for new public API** — `AgentLoop`, `AgentOutcome`, `AgentObserver`,
  `AgentEvent`, `McpConnection`, `McpManager` all need proper doc comments with examples.

## High Priority

- [ ] **Anthropic provider tool support** — Verify Anthropic direct API works with agent loop
  (uses Claude format, not OpenAI format). Test end-to-end with MCP.

## Medium Priority

- [ ] **Vision Support** - Image content blocks for multimodal
- [ ] **Extended Thinking** - Claude's thinking blocks, signature handling
- [ ] **Structured Output** - JSON schema enforcement for OpenAI
- [ ] **Token Counting** - Estimate tokens before sending requests
- [ ] **Conversation Summarization** - Auto-summarize long conversations

## Low Priority / Future

- [ ] **Caching** - Response caching with cache control headers
- [ ] **Prompt Templates** - Reusable prompt components
- [ ] **Cost Tracking** - Track API costs per session
- [ ] **Metrics/Telemetry** - OpenTelemetry integration
- [ ] **WebSocket Transport** - Alternative to SSE for some providers

## Technical Debt

- [ ] Add integration tests with mocked HTTP responses
- [ ] Improve error messages with actionable suggestions
- [ ] Add tracing spans for debugging
- [ ] Document public API with rustdoc examples
- [ ] Benchmark streaming performance

## Completed

- [x] Core types with validation (ModelId, ToolName, etc.)
- [x] Error types (AgentDriverError hierarchy)
- [x] Configuration system with env loading
- [x] Task tracking with cancellation (TaskPool)
- [x] Streaming types (StreamEvent, StreamDelta, StreamHandle)
- [x] Tool system (ToolDefinition, ToolRegistry)
- [x] Provider trait with boxed futures
- [x] Anthropic provider (HTTP + SSE)
- [x] Bedrock provider (AWS SDK converse_stream)
- [x] OpenAI provider (async-openai with streaming)
- [x] Ollama provider (ollama-rs with streaming, num_ctx support)
- [x] OpenRouter provider (reqwest-eventsource SSE, OpenAI-compatible format)
- [x] Session with split locks
- [x] CLI chat client
- [x] Bedrock inference profile support
- [x] **Agent loop** (`src/agent/`) — typed orchestrator with observer pattern
- [x] **MCP integration** (`src/tool/mcp.rs`) — real rmcp wiring, McpConnection/McpManager
- [x] **Session::continue_streaming()** — tool result continuation turns
- [x] **Chat binary agent loop** — `--mcp` CLI flags, AgentLoop integration
- [x] **Bedrock VecDeque buffering** — multi-event stream fix
- [x] **Bedrock duplicate Completed fix** — MessageStop vs Metadata dedup
- [x] **OpenRouter VecDeque buffering** — same multi-event stream fix
- [x] **OpenRouter tool_choice:auto** — enable tool calling
- [x] **flush_pending()** — safety net for providers without ContentBlockStop
- [x] **Serializer strict field** — only include when true

## Notes

### Provider Status
| Provider | Status | Notes |
|----------|--------|-------|
| Anthropic | ✅ Working | Direct HTTP + SSE (untested with agent loop) |
| Bedrock | ✅ Working | Agent loop + MCP tested, final-response bug pending |
| OpenAI | ✅ Working | async-openai with streaming (untested with agent loop) |
| Ollama | ✅ Working | ollama-rs with streaming (untested with agent loop) |
| OpenRouter | ✅ Working | Agent loop + MCP tested, fully working |

### Testing Commands
```bash
# Quick check
cargo check --all-features

# Run all tests
cargo test --all-features

# Test agent loop with MCP (OpenRouter - fully working)
echo "List files in /tmp" | PROVIDER=openrouter cargo run --features "openrouter mcp" --bin chat -- --mcp 'npx -y @modelcontextprotocol/server-filesystem /tmp'

# Test agent loop with MCP (Bedrock - has final-response bug)
echo "List files in /tmp" | PROVIDER=bedrock cargo run --features "bedrock mcp" --bin chat -- --mcp 'npx -y @modelcontextprotocol/server-filesystem /tmp'
```

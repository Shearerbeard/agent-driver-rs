# TODO - agent-driver-rs

## Next Up: Post-Implementation Cleanup & Hardening

The agentic loop + MCP integration was implemented in a single deep session and needs
a thorough review pass for correctness, code quality, and adherence to project principles.

### Known Bugs (from integration testing)

- [ ] **Bedrock "Invalid request: service error"** on final text response after tool iterations.
  Likely a message formatting issue in `continue_streaming()` — Bedrock's converse API
  may reject the message history shape after tool_use + tool_result turns. Reproduce:
  `echo "List files in /tmp" | PROVIDER=bedrock cargo run --features "bedrock mcp" --bin chat -- --mcp 'npx -y @modelcontextprotocol/server-filesystem /tmp'`

- [x] **`ToolInput::from_value` rejects null/empty args** — Fixed: now treats `null` as
  empty object `{}`.

### Code Review & Cleanup (run with reviewer agents)

- [x] **Review `src/agent/driver.rs`** — Audited: HashMap-based block type tracking,
  ContentFilter stop reason handling, expanded doc examples, #[must_use] annotations.

- [x] **Review `src/tool/mcp.rs`** — Audited: doc examples added for McpConnection and
  McpManager, module docs enhanced.

- [x] **Review `src/provider/openrouter.rs`** — Audited: HashMap for parallel tool calls,
  VecDeque buffer bounds documented, ModelId type consistency, tracing for parse errors.

- [x] **Review `src/provider/bedrock.rs`** — Audited: PosInt/NegInt fix, safe integer casts,
  spurious Started event fix, section comments, buffer bounds documented.

- [x] **Review `src/streaming.rs`** — Audited: flush_pending confirmed safe (std::mem::take),
  parallel tool use finalization, HashMap block type tracking in collect(),
  StreamHandle cancellation latency documented.

- [x] **Review `src/bin/chat.rs`** — Audited: UTF-8 safe truncate, section comments added,
  clippy print_with_newline fixed.

- [x] **Review `src/session.rs`** — Audited: O(n²) message trimming replaced with drain(),
  doc example added.

- [x] **Audit for code duplication** — VecDeque buffering now consistent across all 5
  providers (Anthropic, Bedrock, OpenAI, OpenRouter, Ollama) with documented buffer bounds.

- [x] **Verify design principles compliance** — Full audit across 5 dimensions (type design
  95, modern Rust 96, readability 96, parsing 97, threading 96). All at 95%+.

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

- [x] **Rustdoc for new public API** — Doc examples added for `AgentLoop`, `Session`,
  `StreamHandle`, `McpConnection`, `McpManager`, `Provider`. 14 doc-tests passing.

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
- [x] Improve error messages with actionable suggestions
- [ ] Add tracing spans for debugging
- [x] Document public API with rustdoc examples
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
- [x] **ToolInput null/empty fix** — `from_value` accepts `null` as empty object
- [x] **Code review & cleanup** — All 8 review items completed via 5-dimension audit
- [x] **Rustdoc for public API** — Doc examples on 6 key types, 14 doc-tests passing
- [x] **Actionable error messages** — ModelIdError, TaskPoolError messages improved
- [x] **Clippy clean** — All 9 warnings resolved (derivable_impls, unnecessary_cast, etc.)
- [x] **Type safety hardening** — Display impls, Option<ModelId>, #[must_use], expect()
- [x] **Provider event buffering** — VecDeque in all 5 providers (was dropping events)
- [x] **Parsing fixes** — PosInt/NegInt, safe casts, ContentFilter, parallel tool calls
- [x] **Threading fixes** — O(n²) drain, TaskPool TOCTOU, cancellation docs

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

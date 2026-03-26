# TODO - agent-driver-rs

## Milestone 0: Code Quality Review

Review codebase to owner's specifications before feature work begins.

- [ ] Code quality review pass (scope TBD with owner)

## Milestone 1: Fix Bedrock Tool Loop Bug

- [ ] **Bedrock "Invalid request: service error"** on final text response after tool iterations.
  Likely a message formatting issue in `continue_streaming()` — Bedrock's converse API
  may reject the message history shape after tool_use + tool_result turns. Reproduce:
  `echo "List files in /tmp" | PROVIDER=bedrock cargo run --features "bedrock mcp" --bin chat -- --mcp 'npx -y @modelcontextprotocol/server-filesystem /tmp'`

## Milestone 2: Agent Loop Tests & ADR

- [ ] **Write ADR-0002** — Capture design decisions for agent loop and dynamic tools
  (tool depth counting, observer pattern, flush_pending safety net, etc.)

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

## Milestone 3: Unified Claude Provider Tool Support

Anthropic direct API and Bedrock both target the Claude Messages API but currently
build messages independently (Anthropic via custom JSON, Bedrock via AWS SDK types).
OpenRouter uses OpenAI-compatible format — separate concern.

- [ ] **Verify Anthropic provider works with agent loop + MCP** end-to-end
- [ ] **Evaluate shared Claude message formatting** — Anthropic's `serialize_message()` and
  Bedrock's `convert_messages()` both produce Claude-shaped messages. Determine if a shared
  "Claude message builder" layer is worth extracting, or if the AWS SDK type boundary
  makes this impractical.
- [ ] **Extract shared code if viable** — Common Claude content block building (tool_use,
  tool_result, thinking) that both providers adapt to their transport

## Milestone 4: Integration Tests

- [x] Phoenix integration examples with Mock provider (phoenix_integration_mock.rs)
- [x] Phoenix integration examples with Bedrock provider (phoenix_integration_bedrock.rs)
- [x] Phoenix integration examples with Ollama provider (phoenix_integration_ollama.rs)
- [x] OTel Collector docker-compose setup (docker-compose.phoenix.yaml)
- [x] Test runner script (scripts/test-phoenix-integration.sh)
- [x] Documentation updates (PHOENIX_TODO.md, manual-testing.md)
- [ ] **Next Session: Full OTel Integration**
  - [ ] Add `init_tracer()` to Bedrock and Ollama examples
  - [ ] Pass tracer via `.otel_tracer(tracer)` to SessionBuilder
  - [ ] Add assertions to verify reasoning output quality
  - [ ] End-to-end test with mock provider — exercises the full agent loop without
    requiring live API keys (mock HTTP responses or in-process mock provider)
  - [ ] Mock MCP server test — in-process tool that validates the MCP ↔ agent loop wiring
  - [ ] Add integration tests with mocked HTTP responses for individual providers

## Someday

### Features
- [ ] Vision Support — Image content blocks for multimodal
- [ ] Extended Thinking — Claude's thinking blocks, signature handling
- [ ] Structured Output — JSON schema enforcement for OpenAI
- [ ] Token Counting — Estimate tokens before sending requests
- [ ] Conversation Summarization — Auto-summarize long conversations
- [ ] Caching — Response caching with cache control headers
- [ ] Prompt Templates — Reusable prompt components
- [ ] Cost Tracking — Track API costs per session
- [ ] Metrics/Telemetry — OpenTelemetry integration
- [ ] WebSocket Transport — Alternative to SSE for some providers

### Tech Debt
- [ ] Add tracing spans for debugging
- [ ] Benchmark streaming performance

---

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
- [x] Agent loop (`src/agent/`) — typed orchestrator with observer pattern
- [x] MCP integration (`src/tool/mcp.rs`) — real rmcp wiring, McpConnection/McpManager
- [x] Session::continue_streaming() — tool result continuation turns
- [x] Chat binary agent loop — `--mcp` CLI flags, AgentLoop integration
- [x] Bedrock VecDeque buffering — multi-event stream fix
- [x] Bedrock duplicate Completed fix — MessageStop vs Metadata dedup
- [x] OpenRouter VecDeque buffering — same multi-event stream fix
- [x] OpenRouter tool_choice:auto — enable tool calling
- [x] flush_pending() — safety net for providers without ContentBlockStop
- [x] Serializer strict field — only include when true
- [x] ToolInput null/empty fix — `from_value` accepts `null` as empty object
- [x] Code review & cleanup — All 8 review items completed via 5-dimension audit
- [x] Rustdoc for public API — Doc examples on 6 key types, 14 doc-tests passing
- [x] Actionable error messages — ModelIdError, TaskPoolError messages improved
- [x] Clippy clean — All 9 warnings resolved
- [x] Type safety hardening — Display impls, Option<ModelId>, #[must_use], expect()
- [x] Provider event buffering — VecDeque in all 5 providers
- [x] Parsing fixes — PosInt/NegInt, safe casts, ContentFilter, parallel tool calls
- [x] Threading fixes — O(n²) drain, TaskPool TOCTOU, cancellation docs
- [x] Phoenix integration examples with Bedrock provider (phoenix_integration_bedrock.rs)
- [x] Phoenix integration examples with Ollama provider (phoenix_integration_ollama.rs)
- [x] OTel Collector docker-compose setup (docker-compose.phoenix.yaml)
- [x] Test runner script (scripts/test-phoenix-integration.sh)
- [x] Documentation updates (PHOENIX_TODO.md, manual-testing.md)

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

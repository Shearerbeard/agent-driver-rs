# TODO - agent-driver-rs

> Organized by ADR implementation waves. See [docs/adr/README.md](docs/adr/README.md) for ADR index and [docs/internal/agent-driver-roadmap.md](docs/internal/agent-driver-roadmap.md) for phase mapping.

---

## Wave 1: ADR-0002 — Thinking & Reasoning Support (Phase 0)

**Status:** Proposed — correctness bug, multi-turn with thinking is broken.

- [ ] **Fix multi-turn signature loss** — `StreamDelta::SignatureDelta` is explicitly discarded in `CollectedResponse::apply_delta()`. Add `ContentBlock::Signature { signature: String }` and accumulate it. Without this, Anthropic rejects subsequent turns with 400 errors.
- [ ] **Add `ContentBlock::RedactedThinking { data: String }`** — safety-flagged reasoning blocks are silently dropped, breaking multi-turn.
- [ ] **Bedrock thinking support** — `additionalModelRequestFields.thinking` for Claude models. Handle `ContentBlockDelta::ReasoningContent` in streaming (both `.text` and `.signature` members).
- [ ] **Anthropic adaptive mode + effort + display** — current config only covers deprecated manual mode. Add `ThinkingConfig { type: adaptive, display, output_config.effort }`.
- [ ] **OpenAI `ReasoningConfig` wiring** — dead code exists but is never sent in requests. Wire `reasoning_effort` to Chat Completions API.
- [ ] **OpenRouter reasoning pass-through** — parse `reasoning` and `reasoning_details` from responses. Pass `reasoning_details` back in multi-turn.
- [ ] **Ollama streaming verification** — confirm thinking tokens arrive incrementally (two-phase: thinking then content), not all at once.
- [ ] **Budget validation** — `budget_tokens < max_tokens` checked at request-build time.
- [ ] **`ThinkingSupport` capability enum** — replace `extended_thinking: bool` with provider-specific variants (`BudgetTokens`, `ReasoningEffort`, `Toggle`).

**Reproduction:**
```bash
# Multi-turn with thinking (currently broken on Anthropic/Bedrock)
echo "What is 2+2? Think about it first." | PROVIDER=anthropic cargo run --features "anthropic phoenix" --bin chat
```

---

## Wave 2: ADR-0003 — Unit Test Coverage (Phase 2)

**Status:** Proposed — test foundation, must precede live tests.

### P0: Critical gaps (zero coverage)
- [ ] **Bedrock `parse_bedrock_event`** (8+ tests) — MessageStart, ContentBlockStart (text vs tool_use), ContentBlockDelta, ContentBlockStop, MessageStop, Metadata, malformed input, parallel tool calls
- [ ] **`buffered_sse_stream`** (4+ tests) — SSE framing, partial events, reconnection, malformed data

### P1: Format divergence risk
- [ ] **OpenAI `convert_messages`** (4+ tests) — role mapping, tool_use blocks, tool result blocks, empty messages
- [ ] **OpenRouter `serialize_message`** (3+ tests) — shares Anthropic SSE path but own serialization
- [ ] **Anthropic `serialize_message` / `build_request_body`** (3+ tests) — lowest risk, well-exercised via examples

**Target:** 22+ new unit tests. See ADR-0003 for detailed test patterns.

---

## Wave 3: ADR-0004 — Live Provider Integration Tests (Phase 4)

**Status:** Proposed — validates thinking fix end-to-end across providers.

### P0: Ollama + Bedrock (free/local + primary production)
- [ ] **`tests/live_provider.rs` scaffold** — provider-agnostic test structure, env-gated via `LIVE_TEST_PROVIDERS`
- [ ] **Basic text response** — all providers
- [ ] **Streaming deltas** — TextDelta events arrive before Completed
- [ ] **Tool calling round-trip** — model calls tool, we execute, model uses result
- [ ] **Multi-turn conversation** — 2+ user messages, context preserved

### P1: Anthropic + OpenAI + OpenRouter
- [ ] Same test categories as P0

### P2: Extended scenarios
- [ ] **Stop reason mapping** — EndTurn maps correctly from provider response
- [ ] **Token counts** — `CompletionMetadata.usage` populated with non-zero values
- [ ] **Content filter** — provider content filter triggers gracefully (Anthropic, OpenAI, Bedrock)
- [ ] **Extended thinking** — thinking blocks emitted before text (Anthropic, Bedrock, Ollama)

**Canonical test tool:** `sum(a, b)` — deterministic answer, all models can use it.

---

## Wave 4: ADR-0005 — Prompt Caching Support (Phase 5)

**Status:** Proposed — extends `TokenUsage`, needed before multi-agent traces.

- [ ] **`PromptCacheConfig`, `CachePolicy`, `RequestCacheConfig`** — provider-agnostic options
- [ ] **Extend `TokenUsage` with cache fields** — `cached_tokens`, `cache_creation_tokens`, `cache_read_tokens`, `bedrock_cache_read_tokens`, `bedrock_cache_write_tokens`
- [ ] **Add `CacheInfo` to `CompletionMetadata`** — `cache_hit`, `cache_key`, `ttl_secs`
- [ ] **OpenAI automatic caching** — `OpenAI-Behavior-Experimental` header support
- [ ] **Anthropic `cache_control`** — automatic (top-level) + explicit (per-block breakpoints)
- [ ] **Bedrock `CachePointBlock`** — Converse API system, messages, tools fields
- [ ] **OpenRouter header forwarding** — passes underlying provider's caching headers
- [ ] **Cache metrics in OTel spans** — `cache.hit_ratio`, `cache.tokens` attributes
- [ ] **`ProviderCapabilities` update** — `prompt_caching: bool`, `cache_metrics: bool`

---

## Wave 5: ADR-0006 — Multi-Agent Trace Composition (Phase 7)

**Status:** Proposed — depends on OTel infra (Phase 3) + TokenUsage (Wave 4).

- [ ] **`GraphNodeId` newtype** — validated constructor, `#[serde(transparent)]`, `Display`, `TryFrom`
- [ ] **`AgentNodeName` newtype** — validated (non-empty, length limit, charset), dedicated `AgentNodeNameError`
- [ ] **`AgentTopology` enum** — `Coordinator { node_id, node_name }` / `Worker { node_id, node_name, parent }`. Self-parenting rejected at construction.
- [ ] **`TraceContextHeaders`** — private fields, constructible only via `propagation::extract_context()`
- [ ] **`TraceParent`, `TraceState`, `Baggage` newtypes** — W3C validated, `from_header()` only constructors
- [ ] **`AgentLoopSpan::new_coordinator()`** — no parent context required, `graph.node.parent_id = ""`
- [ ] **`AgentLoopSpan::new_worker()`** — `parent_context: &Context` required (not `Option`), prevents orphan spans
- [ ] **`propagation` module** — `TextMapPropagator` integration with `HeaderExtractor`/`HeaderInjector`
- [ ] **`graph.node.*` attributes** — `graph.node.id`, `graph.node.name`, `graph.node.parent_id` on agent spans
- [ ] **MCP request envelope with trace context** — JSON-RPC metadata propagation
- [ ] **HTTP transport integration** — `traceparent`, `tracestate`, `baggage` header injection

---

## Independent / Parallel Work

These can run alongside any wave:

### Phase 1: Aura Compatibility
- [ ] Sequential tool execution config (`sequential_tool_execution: bool` in `AgentLoopConfig`)
- [ ] MCP cancellation propagation (port Aura's `InFlightRequests` + `notifications/cancelled`)
- [ ] MCP progress notifications (`ClientHandler` with request-scoped progress routing)
- [ ] MCP header forwarding (`connect_http()` accepts `HashMap<String, String>`)

### Phase 3.5: Unified Claude Provider
- [ ] Evaluate shared Claude message builder (Anthropic `serialize_message()` vs Bedrock `convert_messages()`)
- [ ] Extract shared code if viable (common content block building)
- [ ] Verify Anthropic + Bedrock parity

### Phase 6: Integration Test Suite
- [ ] Mock HTTP response integration tests (per-provider, no live API keys)
- [ ] Mock MCP server integration test (in-process tool validation)
- [ ] Sequential vs parallel execution tests
- [ ] Cancellation propagation tests (end-to-end)
- [ ] Multi-provider agent loop tests (cross-provider parity)

---

## Completed

### ADRs
- [x] ADR-0001: Tool System & MCP Integration (Accepted)
- [x] ADR-0002: Thinking & Reasoning Support (Proposed)
- [x] ADR-0003: Unit Test Coverage (Proposed)
- [x] ADR-0004: Live Provider Integration Tests (Proposed)
- [x] ADR-0005: Prompt Caching Support (Proposed)
- [x] ADR-0006: Multi-Agent Trace Composition (Proposed)

### Core Infrastructure
- [x] Core types with validation (ModelId, ToolName, etc.)
- [x] Error types (AgentDriverError hierarchy)
- [x] Configuration system with env loading
- [x] Task tracking with cancellation (TaskPool)
- [x] Streaming types (StreamEvent, StreamDelta, StreamHandle)
- [x] Tool system (ToolDefinition, ToolRegistry)
- [x] Provider trait with boxed futures
- [x] Session with split locks

### Providers
- [x] Anthropic provider (HTTP + SSE)
- [x] Bedrock provider (AWS SDK converse_stream)
- [x] OpenAI provider (async-openai with streaming)
- [x] Ollama provider (ollama-rs with streaming, num_ctx support)
- [x] OpenRouter provider (reqwest-eventsource SSE, OpenAI-compatible format)
- [x] Bedrock inference profile support
- [x] Bedrock VecDeque buffering — multi-event stream fix
- [x] Bedrock duplicate Completed fix — MessageStop vs Metadata dedup
- [x] OpenRouter VecDeque buffering — same multi-event stream fix
- [x] OpenRouter tool_choice:auto — enable tool calling
- [x] Provider event buffering — VecDeque in all 5 providers
- [x] Parsing fixes — PosInt/NegInt, safe casts, ContentFilter, parallel tool calls

### Agent Loop & MCP
- [x] Agent loop (`src/agent/`) — typed orchestrator with observer pattern
- [x] MCP integration (`src/tool/mcp.rs`) — real rmcp wiring, McpConnection/McpManager
- [x] Session::continue_streaming() — tool result continuation turns
- [x] Chat binary agent loop — `--mcp` CLI flags, AgentLoop integration
- [x] flush_pending() — safety net for providers without ContentBlockStop
- [x] Serializer strict field — only include when true
- [x] ToolInput null/empty fix — `from_value` accepts `null` as empty object

### OTel / Phoenix
- [x] Phoenix integration examples (Mock, Bedrock, Ollama)
- [x] OTel Collector docker-compose setup (docker-compose.phoenix.yaml)
- [x] Test runner script (scripts/test-phoenix-integration.sh)
- [x] Fix opentelemetry-otlp dependency version mismatch (0.15 → 0.27)
- [x] Real RAII span guards (AgentLoopSpan, ToolSpan, SessionOperationSpan)
- [x] `init_phoenix()` / `shutdown_phoenix()` convenience functions
- [x] OTLP gRPC exporter wired to all Phoenix examples
- [x] OpenInference-compliant spans (AGENT/CHAIN/TOOL)

### Code Quality
- [x] Code review & cleanup — All 8 review items completed via 5-dimension audit
- [x] Rustdoc for public API — Doc examples on 6 key types, 14 doc-tests passing
- [x] Actionable error messages — ModelIdError, TaskPoolError messages improved
- [x] Clippy clean — All 9 warnings resolved
- [x] Type safety hardening — Display impls, Option<ModelId>, #[must_use], expect()
- [x] Threading fixes — O(n²) drain, TaskPool TOCTOU, cancellation docs

---

## Notes

### Provider Status
| Provider | Status | Notes |
|----------|--------|-------|
| Anthropic | ✅ Working | Direct HTTP + SSE (untested with agent loop) |
| Bedrock | ⚠️ Bug | Agent loop + MCP tested, final-response bug after tool iterations |
| OpenAI | ✅ Working | async-openai with streaming (untested with agent loop) |
| Ollama | ✅ Working | ollama-rs with streaming (untested with agent loop) |
| OpenRouter | ✅ Working | Agent loop + MCP tested, fully working |

### Testing Commands
```bash
# Quick check
cargo check --all-features
cargo test --all-features
cargo clippy --features bedrock -- -D warnings

# Live smoke test (provider/agent/tool changes)
echo "What is 2 + 2? Answer in one sentence." | \
    PROVIDER=bedrock cargo run --features bedrock --bin chat

# Live tool calling test (agent/tool/provider changes)
echo "List the contents of the docs/adr directory" | \
    PROVIDER=bedrock cargo run --features "bedrock mcp" --bin chat -- \
    --mcp "npx -y @modelcontextprotocol/server-filesystem $(pwd)"

# Phoenix tracing test (otel changes)
PHOENIX_ENDPOINT=http://your-phoenix-host:4317 \
    cargo run --features phoenix --example phoenix_integration_mock

# Test agent loop with MCP (OpenRouter - fully working)
echo "List files in /tmp" | PROVIDER=openrouter cargo run --features "openrouter mcp" --bin chat -- --mcp 'npx -y @modelcontextprotocol/server-filesystem /tmp'
```

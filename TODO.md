# TODO - agent-driver-rs

> Last reviewed: 2026-06-21. This is the canonical active roadmap. Historical
> handoffs such as `docs/next-session.md` are snapshots, not active planning
> sources. ADR wave details live in [docs/adr/README.md](docs/adr/README.md);
> longer-range phase mapping lives in [docs/internal/agent-driver-roadmap.md](docs/internal/agent-driver-roadmap.md).

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

### P0: Critical gaps
- [ ] **Bedrock `parse_bedrock_event`** (3 of 8+ done; tool-use paths only, via the June quality audit) — still missing MessageStart, ContentBlockStart text, ContentBlockStop, MessageStop, Metadata, malformed input, parallel tool calls
- [ ] **`buffered_sse_stream`** (4+ tests) — SSE framing, partial events, reconnection, malformed data

### P1: Format divergence risk
- [ ] **OpenAI `convert_messages`** (1 of 4+ done: system-dedup) — still missing role mapping, tool_use blocks, tool result blocks, empty messages
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

### Rust Toolchain Modernization (completed)
- [x] **Set current Rust/MSRV policy**: MSRV 1.91.1 declared in `Cargo.toml`, `rust-toolchain.toml` pins stable.
- [x] **Upgrade to Rust edition 2024**: migrated with `make check` as the exit gate. No source changes needed beyond fmt.
- [x] **Revisit ADR-0007 after edition migration**: `thiserror` 2 promoted; Dylint and stricter lint waves remain deferred.
- [x] **Modernize `async-openai`**: migrated from `0.28.3` to `0.41.1`; removed `backoff` direct dep; clears RUSTSEC-2024-0384 and RUSTSEC-2025-0012.
- [x] **Update AWS SDK crates**: `aws-sdk-bedrockruntime` 1.124→1.135, `aws-config` 1.8.13→1.8.18; disabled legacy `rustls` default feature to drop rustls 0.21 stack; clears RUSTSEC-2026-0098/0099/0104.

### Tier 0: Usability and Documentation Hygiene (ADR-0007 audit follow-up)
- [x] ADR-0007: Codex-Style Lint & Tooling Adoption drafted and indexed
- [x] Fix README/CLAUDE/manual-testing provider-default inconsistency (Anthropic default, Bedrock production smoke path)
- [x] Add README run/test cheat sheet
- [x] Mark `TODO.md` as canonical active roadmap and demote `docs/next-session.md` to historical handoff snapshot
- [x] Add `CHANGELOG.md`
- [x] Add minimal CI compile workflow for the public feature set (`schema-sanitize` joined it once its private git dependency was replaced by a local port)
- [x] Add ADR-0007 clippy, cargo-deny, and cargo-shear gates

### Rust Quality Audit (2026-06-21, completed)
Stages 1-5 of the quality audit are done; P0/P1 fixes (HashMap tool-delta
accumulation, `ProviderContext::default()` removal, `#[non_exhaustive]`,
Bedrock parser tests) landed on master. Full logs:
`docs/rust-quality-audit-plan.md` and `docs/rust-quality-audit-checklist.md`.

### Tier 1: ADR-0007 — Codex-Style Lint & Tooling Adoption
- [x] Layer A: `clippy.toml` (`await-holding-invalid-types`, test-only unwrap/expect allowances, anti-slop methods)
- [x] Layer B: promote codex core clippy set to `deny`, one lint per commit
- [x] Layer C: add `deny.toml`; pin `mcp-openai-bridge` to an immutable `rev`
- [x] Layer D: add cargo-shear unused-dependency check

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
- [x] ADR-0007: Codex-Style Lint & Tooling Adoption (Accepted 2026-07-18; Dylint and stricter waves deferred)

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

---

## Benchmark Results & Investigation Items

### Run: 2026-02-09 (gpt-4o-mini, n=30, warmup=3, release build)

```
Scenario         Library                Min   Median     Mean   StdDev      Max       Alloc  vs
────────────────────────────────────────────────────────────────────────────────────────────────────────────
cold_start       agent-driver-rs       59us     89us     87us     16us    110us      4.6 KB
cold_start       rig.rs                60us     76us     81us     25us    169us      8.8 KB  ◀ 1.2x
────────────────────────────────────────────────────────────────────────────────────────────────────────────
ttft             agent-driver-rs    377.8ms  413.7ms  436.0ms   61.4ms  685.7ms     87.9 KB  ◀ 1.2x
ttft             rig.rs             391.5ms  509.2ms  616.7ms  346.8ms    1.93s    120.6 KB
────────────────────────────────────────────────────────────────────────────────────────────────────────────
tool_roundtrip   agent-driver-rs    990.8ms    1.16s    1.27s  383.9ms    2.95s    241.9 KB  ◀ 1.5x
tool_roundtrip   rig.rs               1.43s    1.78s    1.81s  267.0ms    2.55s    383.9 KB
────────────────────────────────────────────────────────────────────────────────────────────────────────────
mcp_discovery    agent-driver-rs      320us    415us    455us    118us    725us    213.2 KB
mcp_discovery    rig.rs               327us    393us    456us    131us    999us    210.3 KB  ◀ 1.1x
────────────────────────────────────────────────────────────────────────────────────────────────────────────
mcp_roundtrip    agent-driver-rs      1.31s    1.76s    1.83s  558.1ms    4.22s    675.5 KB  ◀ 1.3x
mcp_roundtrip    rig.rs               1.76s    2.20s    2.30s  507.2ms    4.25s    622.6 KB
```

Errors during run: 2x "Empty response from agent loop" (ours/mcp_rt), 1x "MaxTurnError" (rig/mcp_rt)

### Analysis

#### Where we win
- **ttft (1.2x)**: Consistent advantage. Our median 414ms vs rig's 509ms. Notably, our stddev is 61ms vs rig's 347ms — we're not just faster, we're dramatically more consistent. Rig had a 1.93s max outlier.
- **tool_roundtrip (1.5x)**: Largest win. Our streaming agent loop completes a full tool call round in 1.16s median vs rig's 1.78s. Both now stream, so this is a genuine throughput advantage.
- **mcp_roundtrip (1.3x)**: Same pattern — our agent loop + MCP tool execution is faster end-to-end.

#### Where rig wins
- **cold_start (1.2x rig)**: Rig's object construction is ~13us faster at median (76us vs 89us). Negligible in absolute terms but we allocate ~half the memory (4.6 KB vs 8.8 KB). Not a concern.
- **mcp_discovery (1.1x rig)**: Virtual tie at ~400us median. Both just call `list_all_tools()` over stdio now (after the fairness fix). The 22us difference is noise.

### Benchmark Investigation Items

#### HIGH PRIORITY

- [ ] **cold_start: 89us vs 76us — profile our SessionBuilder overhead**
  - We lose despite allocating half the memory (4.6 KB vs 8.8 KB). The 13us gap suggests overhead in `SessionBuilder::build()` or `OpenAiProvider::new()` — possibly the split-lock `RwLock` initialization or `ToolRegistry::new()`. Worth profiling with `flamegraph` to identify if there's unnecessary work during construction. Low absolute impact but easy to fix if found.

- [ ] **ttft stddev: investigate rig's 347ms stddev vs our 61ms**
  - Our consistency advantage is striking. This is likely our stream adapter buffering + `biased` select providing more deterministic first-token delivery. Worth understanding *why* to ensure it's architectural and not accidental. Could inform documentation/marketing claims about reliability.

- [ ] **mcp_roundtrip: 2/30 "Empty response" errors from max_tool_depth=2**
  - When the model uses both allowed tool rounds for file listing, it sometimes doesn't produce a final text response before the depth cap. The harness skips these (correct behavior), but 6.7% error rate means we're measuring 28/30 iterations, not 30/30. Two options:
    1. Bump `max_tool_depth` to 3 (risk: reintroduces the asymmetry with rig's `max_turns(2)`)
    2. Use a simpler prompt that needs only 1 tool round (e.g., "What is the current directory?" which just needs `pwd`)
  - Rig also hit 1 MaxTurnError, so this is partly a model behavior issue with the prompt.

#### MEDIUM PRIORITY

- [ ] **tool_roundtrip: our 384ms stddev — investigate outlier sources**
  - Min 991ms, max 2.95s. The 3x spread suggests occasional OpenAI API latency spikes hitting our streaming path harder than rig's. Could be worth adding retry/timeout instrumentation to understand if it's network or stream-processing overhead.

- [ ] **Allocation: our mcp_roundtrip uses 675 KB vs rig's 623 KB**
  - We're 8% higher on allocation despite being 1.3x faster on time. This is likely the `DynTool` wrapping + `ToolRegistry` overhead for MCP tools. Not urgent since we win on latency, but worth tracking — if allocation grows with tool count it could become a concern for large MCP deployments.

#### LOW PRIORITY

- [ ] **Add `--json` output flag for CI integration**
  - Would enable tracking these numbers over time and catching regressions automatically.

- [ ] **Test with gpt-4o and claude-sonnet-4-5 via OpenRouter**
  - Current results are gpt-4o-mini only. Larger models have different streaming characteristics (bigger tokens, longer generation). Results may differ significantly.

- [ ] **StreamHandle drop doesn't cancel — consider adding CancellationToken propagation on drop**
  - The ttft scenario explicitly drops the stream but the underlying connection may linger. This was flagged in the original audit. Low impact for benchmarks (warmup absorbs it), but matters for production use where rapid stream abandonment is common.

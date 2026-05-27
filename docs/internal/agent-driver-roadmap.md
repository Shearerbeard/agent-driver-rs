# agent-driver-rs Roadmap

> See also: [thinking-reasoning-status.md](thinking-reasoning-status.md) — per-provider architecture map and prioritized todo (2026-05-26)

> **Created:** May 2026
> **Context:** Cross-repo evaluation for replacing rig.rs in Aura orchestration mode

## Current Status

**Version:** 0.1.0 (pre-production)
**Fully tested provider:** OpenRouter (agent loop + MCP)
**Partially tested:** Anthropic, OpenAI, Bedrock, Ollama (functional but untested with agent loop)
**Known bug:** Bedrock tool loop — final text response after tool iterations fails with "Invalid request: service error"

## Strategic Context

Aura (aura-orchestration-mode) is evaluating agent-driver-rs as a replacement for rig.rs as its inference layer. Aura is a production-ready multi-agent orchestration system with:

- Coordinator/worker architecture with DAG-based parallel execution
- 5 LLM providers (OpenAI, Anthropic, Bedrock, Gemini, Ollama)
- Full MCP integration (HTTP, SSE, STDIO transports)
- Request-scoped MCP progress and cancellation
- Custom SSE streaming events
- Token usage tracking across multi-turn conversations

The evaluation identified agent-driver-rs as **architecturally sound but early-stage**. The inference/provider layer is mature enough for a coordinator-only spike. Full worker replacement requires gap fixes.

## Roadmap Phases

### Phase 0: Stabilize Foundation (Current — 3-5 days)

**Goal:** Complete existing milestones and fill critical test gaps before any new feature work.

| Priority | Task | Effort | Notes | ADR |
|----------|------|--------|-------|-----|
| P0 | Fix thinking multi-turn bug (signature loss) | 1-2 days | **ADR-0002 Wave 1** — ContentBlock::Signature + RedactedThinking | ADR-0002 |
| P0 | Bedrock thinking support | 1 day | additionalModelRequestFields.thinking, ReasoningContentBlockDelta | ADR-0002 |
| P0 | Agent loop unit tests (Milestone 2) | 1-2 days | See test-plan.md for detailed cases | — |
| P0 | MCP unit tests (Milestone 2) | 1 day | Schema conversion, content type handling | — |
| P0 | `flush_pending()` tests (Milestone 2) | 0.5 days | Text, thinking, tool_use pending state | — |
| P1 | Anthropic adaptive mode + effort + display | 0.5 days | Current config only covers deprecated manual mode | ADR-0002 |
| P1 | OpenAI ReasoningConfig wiring | 0.5 days | Dead code → working code | ADR-0002 |
| P1 | Verify Anthropic with agent loop + MCP | 0.5 days | Smoke test | — |
| P1 | Verify OpenAI with agent loop + MCP | 0.5 days | Smoke test | — |
| P1 | Verify Ollama with agent loop + MCP | 0.5 days | Smoke test | — |

**Completion criteria:**
- All existing TODO.md Milestone 1-2 items checked off
- `cargo test --all-features` passes with >100 tests (currently ~61)
- All 5 providers verified working with agent loop + MCP
- Bedrock tool loop bug fixed and regression-tested
- Multi-turn conversations with extended thinking work (signature round-trip)

### Phase 1: Aura Compatibility Layer (5-7 days)

**Goal:** Fill gaps identified by Aura integration analysis that block production use.

| Priority | Task | Effort | Details |
|----------|------|--------|---------|
| P0 | Sequential tool execution config | 0.5 days | Add `sequential_tool_execution: bool` to `AgentLoopConfig`. Aura's FIFO `tool_event_broker` requires this. |
| P0 | MCP cancellation propagation | 2-3 days | Port Aura's `InFlightRequests` tracker + `notifications/cancelled` to MCP servers. |
| P0 | MCP progress notifications | 2-3 days | Implement `ClientHandler` with request-scoped progress routing. |
| P1 | MCP header forwarding | 0.5 days | Extend `connect_http()` to accept `HashMap<String, String>` of headers. |
| P2 | Gemini provider | 2-3 days | New provider implementation. Blocks Aura configs using Gemini. |
| P2 | Structured MCP content extraction | 1-2 days | Handle `structured_content`, embedded resources, resource links. |

**Completion criteria:**
- Sequential mode tested with FIFO queue simulation
- MCP cancellation verified: server receives `notifications/cancelled` on client disconnect
- Progress notifications routed to correct request context
- Header forwarding tested with authenticated MCP server

### Phase 2: Unit Test Coverage (2-3 days)

**Goal:** Fill critical test gaps per ADR-0003. Unit tests must precede live integration tests.

| Task | Effort | Notes | ADR |
|------|--------|-------|-----|
| Bedrock `parse_bedrock_event` tests (8+) | 1 day | Highest risk, zero coverage. Test all event types. | ADR-0003 |
| `buffered_sse_stream` tests (4+) | 0.5 days | Shared infrastructure, Anthropic + OpenRouter depend on it | ADR-0003 |
| OpenAI `convert_messages` tests (4+) | 0.5 days | Format divergence risk | ADR-0003 |
| OpenRouter `serialize_message` tests (3+) | 0.5 days | Shares Anthropic SSE path but own serialization | ADR-0003 |
| Anthropic `serialize_message` / `build_request_body` (3+) | 0.5 days | Lowest risk, well-exercised via examples | ADR-0003 |

**Completion criteria:**
- 22+ new unit tests across all provider parse/convert functions
- `cargo test --all-features` passes with all new tests
- Bedrock streaming bugs caught before live testing

### Phase 3: OTel Integration (3-5 days)

**Goal:** Complete existing Milestone 4 from TODO.md.

| Task | Effort | Notes | ADR |
|------|--------|-------|-----|
| Bedrock OTel tracer integration | 1 day | Already partially planned in PLAN_NEXT_SESSION.md | — |
| Ollama OTel tracer integration | 1 day | Already partially planned | — |
| Agent loop span creation | 1 day | Iteration + tool execution spans | — |
| End-to-end mock provider test | 1 day | Full agent loop without live API keys | — |
| Mock MCP server test | 0.5 days | Validate MCP ↔ agent loop wiring | — |

**Note:** This work is already planned in `PLAN_NEXT_SESSION.md`. The Aura integration analysis does not change this priority.

### Phase 3.5: Unified Claude Provider (2-3 days)

**Goal:** Milestone 3 from TODO.md — reduce duplication between Anthropic and Bedrock message formatting.

| Task | Effort | Notes | ADR |
|------|--------|-------|-----|
| Evaluate shared Claude message builder | 0.5 days | Assess AWS SDK type boundary | — |
| Extract shared code if viable | 1-2 days | Common content block building | — |
| Verify Anthropic + Bedrock parity | 0.5 days | Same behavior through both providers | — |

### Phase 4: Live Integration Tests (2-3 days)

**Goal:** Comprehensive live provider tests per ADR-0004. Must follow Phase 2 (unit tests).

| Task | Effort | Notes | ADR |
|------|--------|-------|-----|
| `tests/live_provider.rs` scaffold | 0.5 days | Provider-agnostic test structure, env-gated | ADR-0004 |
| Ollama + Bedrock P0 tests | 1 day | Basic text, streaming, tool calling | ADR-0004 |
| Anthropic + OpenAI + OpenRouter P1 tests | 1 day | Same test scenarios, different providers | ADR-0004 |
| Extended thinking tests | 0.5 days | Validates ADR-0002 fixes end-to-end | ADR-0002, ADR-0004 |

**Completion criteria:**
- All P0 test categories pass on Ollama and Bedrock
- Thinking multi-turn validated on Anthropic + Bedrock
- CI configured to skip live tests without credentials

### Phase 5: Prompt Caching (2-3 days)

**Goal:** Provider-agnostic prompt caching per ADR-0005.

| Task | Effort | Notes | ADR |
|------|--------|-------|-----|
| `PromptCacheConfig`, `CachePolicy`, `RequestCacheConfig` | 0.5 days | Provider-agnostic options | ADR-0005 |
| Extend `TokenUsage` with cache fields | 0.5 days | OpenAI, Anthropic, Bedrock metrics | ADR-0005 |
| OpenAI automatic caching header | 0.5 days | `OpenAI-Behavior-Experimental` header | ADR-0005 |
| Anthropic `cache_control` (automatic + explicit) | 0.5 days | Content block cache breakpoints | ADR-0005 |
| Bedrock `CachePointBlock` in Converse API | 0.5 days | System, messages, tools fields | ADR-0005 |
| Cache metrics in OTel spans | 0.5 days | `cache.hit_ratio`, `cache.tokens` attributes | ADR-0005 |

**Completion criteria:**
- Cache config ignored for providers that don't support it
- Cache metrics visible in Phoenix traces
- Live tests validate cache hit reporting

### Phase 6: Integration Test Suite (2-3 days)

**Goal:** Comprehensive integration tests that exercise the full stack.

| Task | Effort | Notes | ADR |
|------|--------|-------|-----|
| Mock HTTP response integration tests | 1 day | Per-provider, no live API keys | — |
| Mock MCP server integration test | 0.5 days | In-process tool validation | — |
| Sequential vs parallel execution tests | 0.5 days | Verify both modes work correctly | — |
| Cancellation propagation tests | 0.5 days | End-to-end cancellation chain | — |
| Multi-provider agent loop tests | 0.5 days | Cross-provider behavior parity | — |

### Phase 7: Multi-Agent Trace Composition (2-3 days)

**Goal:** Multi-agent span composition per ADR-0006. Depends on Phase 3 (OTel) and Phase 5 (caching).

| Task | Effort | Notes | ADR |
|------|--------|-------|-----|
| `GraphNodeId`, `AgentNodeName`, `AgentTopology` types | 0.5 days | Validated newtypes, enum topology | ADR-0006 |
| `TraceContextHeaders`, `TraceParent`, `TraceState`, `Baggage` | 0.5 days | W3C Trace Context newtypes | ADR-0006 |
| `AgentLoopSpan::new_coordinator` / `new_worker` | 0.5 days | Split constructors, mandatory parent context | ADR-0006 |
| `propagation` module (extract/inject) | 0.5 days | `TextMapPropagator` integration | ADR-0006 |
| `graph.node.*` attributes on agent spans | 0.5 days | OpenInference semantic conventions | ADR-0006 |
| MCP request envelope with trace context | 0.5 days | JSON-RPC metadata propagation | ADR-0006 |

**Completion criteria:**
- Multi-agent traces render correctly in Phoenix (coordinate/collaborate/route modes)
- Cross-process propagation preserves trace_id across HTTP and MCP boundaries
- No orphan worker spans possible (type-enforced)

### Phase 8: Aura Integration Spike (2-3 days, aura repo work)

**Goal:** Coordinator-only slot-in to validate the inference layer replacement.

This phase is primarily aura repo work, but agent-driver-rs must provide:

| Requirement | Status | Notes |
|-------------|--------|-------|
| Stable Provider API | Ready | Provider trait is stable |
| Stable Session API | Ready | SessionBuilder is stable |
| Stable AgentLoop API | Ready | AgentLoop::run() is stable |
| MockProvider for testing | Ready | Behind `test-support` feature |
| Sequential execution config | Phase 1 | Needed for worker testing |
| StreamHandle public API | Ready | Can consume events manually |

**Aura-side work:**
- `DriverAgent` enum parallel to `ProviderAgent`
- Streaming adapter: `StreamEvent` → `StreamItem` mapping
- `StreamingAgent` trait implementation
- Coordinator-only integration test (hybrid rig + agent-driver)

### Phase 9: Production Readiness (3-5 days)

**Goal:** Prepare for production use in Aura.

| Task | Effort | Notes |
|------|--------|-------|
| Performance benchmarking | 1 day | Compare vs rig.rs latency/throughput |
| API stability review | 1 day | Identify public API surface for 1.0 |
| Documentation overhaul | 1-2 days | Usage guide, provider guide, MCP guide |
| Publish to crates.io | 0.5 days | First public release |
| Version 1.0.0 planning | 0.5 days | Breaking change audit |

### Someday / Future Features

From TODO.md, plus additions from Aura integration analysis:

| Feature | Priority | Notes | ADR |
|---------|----------|-------|-----|
| Vision support | P2 | Image content blocks for multimodal | — |
| Extended thinking | **P0** | Claude thinking blocks, signature handling | ADR-0002 |
| Structured output | P2 | JSON schema enforcement for OpenAI | — |
| Token counting | P3 | Estimate tokens before sending | — |
| Conversation summarization | P3 | Auto-summarize long conversations | — |
| Response caching | **P1** | Cache control headers | ADR-0005 |
| Prompt templates | P3 | Reusable components | — |
| Cost tracking | P3 | Per-session API costs | — |
| WebSocket transport | P3 | Alternative to SSE | — |
| OpenRouter as Aura provider | P2 | Aura could add OpenRouter support | — |
| Multi-agent composition | **P1** | Coordinator/worker trace composition | ADR-0006 |

## Dependency Graph

```
Phase 0 (Stabilize + ADR-0002: Thinking fix)
  ├── Thinking signature/multi-turn fix ──────────────┐
  ├── Bedrock thinking support ───────────────────────┤
  ├── Agent loop tests ───────────────────────────────┤
  └── MCP tests ──────────────────────────────────────┤
                                                      │
Phase 1 (Aura Compatibility) ◄───────────────────────┘
  ├── Sequential execution ◄── tests from Phase 0
  ├── MCP cancellation
  ├── MCP progress
  └── Header forwarding

Phase 2 (Unit Tests + ADR-0003) ◄── Phase 0
  ├── Bedrock parse tests (P0)
  ├── buffered_sse_stream tests
  ├── OpenAI convert_messages tests
  └── OpenRouter/Anthropic serialization tests

Phase 3 (OTel) — independent, can run in parallel with Phase 1/2

Phase 3.5 (Unified Claude) — independent, can run in parallel

Phase 4 (Live Tests + ADR-0004) ◄── Phase 2 + Phase 3
  ├── Ollama + Bedrock P0 tests
  ├── Anthropic + OpenAI + OpenRouter P1 tests
  └── Extended thinking validation ◄── Phase 0

Phase 5 (Caching + ADR-0005) ◄── Phase 4
  ├── PromptCacheConfig, CachePolicy
  ├── TokenUsage cache fields
  ├── Provider-specific caching (OpenAI, Anthropic, Bedrock)
  └── Cache metrics in OTel spans ◄── Phase 3

Phase 6 (Integration Tests) ◄── Phase 0 + Phase 1 + Phase 3

Phase 7 (Multi-Agent Traces + ADR-0006) ◄── Phase 3 + Phase 5
  ├── GraphNodeId, AgentTopology types
  ├── TraceContextHeaders, W3C propagation
  ├── new_coordinator / new_worker constructors
  └── graph.node.* attributes on spans

Phase 8 (Aura Spike) ◄── Phase 0 (for MockProvider stability)

Phase 9 (Production) ◄── All previous phases
```

## Milestone Tracking

| Milestone | Status | Target | ADR |
|-----------|--------|--------|-----|
| M0: Code quality review | Not started | TBD | — |
| M1: Bedrock tool loop bug | In progress | Phase 0 | ADR-0002 |
| M2: Agent loop tests + ADR | In progress | Phase 0 | — |
| M3: Unified Claude provider | Not started | Phase 3.5 | — |
| M4: Integration tests (OTel) | In progress | Phase 3 | — |
| M5: Aura compatibility | Not started | Phase 1 | — |
| M6: Production readiness | Not started | Phase 9 | — |
| M7: Unit test coverage | Not started | Phase 2 | ADR-0003 |
| M8: Live provider tests | Not started | Phase 4 | ADR-0004 |
| M9: Prompt caching | Not started | Phase 5 | ADR-0005 |
| M10: Multi-agent traces | Not started | Phase 7 | ADR-0006 |

## Key Decisions Pending

1. **Sequential vs parallel default:** Should sequential execution be the default, or opt-in? Recommendation: opt-in (preserve current performance for non-Aura users).

2. **Gemini provider priority:** Is Gemini support required for the initial Aura integration spike? Recommendation: no — use OpenRouter as fallback for the spike.

3. **MCP rmcp version:** agent-driver-rs uses rmcp 0.14, Aura uses 0.12. Version drift may cause compatibility issues. Recommendation: align on latest (0.14) and update Aura.

4. **Public API surface for 1.0:** What constitutes the stable public API? Recommendation: Provider trait, Session, SessionBuilder, AgentLoop, AgentObserver, Tool trait, ToolRegistry, StreamHandle, CollectedResponse.

5. **OpenAI Responses API vs Chat Completions:** Should we support the OpenAI Responses API as a separate provider, or extend the existing OpenAI provider with a mode flag? (ADR-0002)

6. **Ollama GPT-OSS string think values:** ollama-rs only supports `Option<bool>` — should we fork/patch ollama-rs or use raw HTTP for string think values? (ADR-0002)

7. **Nested worker depth:** Should multi-agent worker nesting be type-bounded or left unbounded? (ADR-0006)

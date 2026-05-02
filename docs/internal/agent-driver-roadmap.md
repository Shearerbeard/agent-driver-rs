# agent-driver-rs Roadmap

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

| Priority | Task | Effort | Notes |
|----------|------|--------|-------|
| P0 | Fix Bedrock tool loop bug (Milestone 1) | 1-2 days | Blocks Bedrock testing with agent loop |
| P0 | Agent loop unit tests (Milestone 2) | 1-2 days | See test-plan.md for detailed cases |
| P0 | MCP unit tests (Milestone 2) | 1 day | Schema conversion, content type handling |
| P0 | `flush_pending()` tests (Milestone 2) | 0.5 days | Text, thinking, tool_use pending state |
| P0 | Write ADR-0002 (Milestone 2) | 0.5 days | Agent loop + dynamic tool design decisions |
| P1 | Verify Anthropic with agent loop + MCP | 0.5 days | Smoke test |
| P1 | Verify OpenAI with agent loop + MCP | 0.5 days | Smoke test |
| P1 | Verify Ollama with agent loop + MCP | 0.5 days | Smoke test |

**Completion criteria:**
- All existing TODO.md Milestone 1-2 items checked off
- `cargo test --all-features` passes with >100 tests (currently ~61)
- All 5 providers verified working with agent loop + MCP
- Bedrock tool loop bug fixed and regression-tested

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

### Phase 2: OTel Integration (Current Milestone 4 — 3-5 days)

**Goal:** Complete existing Milestone 4 from TODO.md.

| Task | Effort | Notes |
|------|--------|-------|
| Bedrock OTel tracer integration | 1 day | Already partially planned in PLAN_NEXT_SESSION.md |
| Ollama OTel tracer integration | 1 day | Already partially planned |
| Agent loop span creation | 1 day | Iteration + tool execution spans |
| End-to-end mock provider test | 1 day | Full agent loop without live API keys |
| Mock MCP server test | 0.5 days | Validate MCP ↔ agent loop wiring |

**Note:** This work is already planned in `PLAN_NEXT_SESSION.md`. The Aura integration analysis does not change this priority.

### Phase 3: Unified Claude Provider (2-3 days)

**Goal:** Milestone 3 from TODO.md — reduce duplication between Anthropic and Bedrock message formatting.

| Task | Effort | Notes |
|------|--------|-------|
| Evaluate shared Claude message builder | 0.5 days | Assess AWS SDK type boundary |
| Extract shared code if viable | 1-2 days | Common content block building |
| Verify Anthropic + Bedrock parity | 0.5 days | Same behavior through both providers |

### Phase 4: Integration Test Suite (2-3 days)

**Goal:** Comprehensive integration tests that exercise the full stack.

| Task | Effort | Notes |
|------|--------|-------|
| Mock HTTP response integration tests | 1 day | Per-provider, no live API keys |
| Mock MCP server integration test | 0.5 days | In-process tool validation |
| Sequential vs parallel execution tests | 0.5 days | Verify both modes work correctly |
| Cancellation propagation tests | 0.5 days | End-to-end cancellation chain |
| Multi-provider agent loop tests | 0.5 days | Cross-provider behavior parity |

### Phase 5: Aura Integration Spike (2-3 days, aura repo work)

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

### Phase 6: Production Readiness (3-5 days)

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

| Feature | Priority | Notes |
|---------|----------|-------|
| Vision support | P2 | Image content blocks for multimodal |
| Extended thinking | P2 | Claude thinking blocks, signature handling |
| Structured output | P2 | JSON schema enforcement for OpenAI |
| Token counting | P3 | Estimate tokens before sending |
| Conversation summarization | P3 | Auto-summarize long conversations |
| Response caching | P3 | Cache control headers |
| Prompt templates | P3 | Reusable components |
| Cost tracking | P3 | Per-session API costs |
| WebSocket transport | P3 | Alternative to SSE |
| OpenRouter as Aura provider | P2 | Aura could add OpenRouter support |

## Dependency Graph

```
Phase 0 (Stabilize)
  ├── Bedrock bug fix ──────────────────────┐
  ├── Agent loop tests ─────────────────────┤
  └── MCP tests ────────────────────────────┤
                                            │
Phase 1 (Aura Compatibility) ◄─────────────┘
  ├── Sequential execution ◄── tests from Phase 0
  ├── MCP cancellation
  ├── MCP progress
  └── Header forwarding

Phase 2 (OTel) — independent, can run in parallel with Phase 1

Phase 3 (Unified Claude) — independent, can run in parallel

Phase 4 (Integration Tests) ◄── Phase 0 + Phase 1 + Phase 2

Phase 5 (Aura Spike) ◄── Phase 0 (for MockProvider stability)

Phase 6 (Production) ◄── All previous phases
```

## Milestone Tracking

| Milestone | Status | Target |
|-----------|--------|--------|
| M0: Code quality review | Not started | TBD |
| M1: Bedrock tool loop bug | In progress | Phase 0 |
| M2: Agent loop tests + ADR | In progress | Phase 0 |
| M3: Unified Claude provider | Not started | Phase 3 |
| M4: Integration tests (OTel) | In progress | Phase 2 |
| M5: Aura compatibility | Not started | Phase 1 |
| M6: Production readiness | Not started | Phase 6 |

## Key Decisions Pending

1. **Sequential vs parallel default:** Should sequential execution be the default, or opt-in? Recommendation: opt-in (preserve current performance for non-Aura users).

2. **Gemini provider priority:** Is Gemini support required for the initial Aura integration spike? Recommendation: no — use OpenRouter as fallback for the spike.

3. **MCP rmcp version:** agent-driver-rs uses rmcp 0.14, Aura uses 0.12. Version drift may cause compatibility issues. Recommendation: align on latest (0.14) and update Aura.

4. **Public API surface for 1.0:** What constitutes the stable public API? Recommendation: Provider trait, Session, SessionBuilder, AgentLoop, AgentObserver, Tool trait, ToolRegistry, StreamHandle, CollectedResponse.

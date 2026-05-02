# agent-driver-rs Test Plan

> **Created:** May 2026
> **Context:** Critical test gaps identified before further core development and Aura integration spike

## Current Test Coverage

**Total tests:** ~61
**Test files:**
- `tests/agent_loop.rs` — 12 tests (comprehensive agent loop coverage)
- `tests/streaming.rs` — 7 tests (streaming infrastructure)
- `tests/session.rs` — 8 tests (session management)
- `tests/cluster_guardian.rs` — (unknown, not reviewed)

**Coverage gaps:**
- No MCP unit tests beyond `extract_text_content`
- No `flush_pending()` tests
- No per-provider agent loop tests (only OpenRouter verified)
- No sequential execution tests (parallel is default)
- No cancellation propagation tests
- No MCP cancellation/progress tests
- No error classification tests

## Priority 1: Existing TODO.md Milestone 2 Tests

These are already identified in TODO.md and must be completed before any new feature work.

### 1.1 `flush_pending()` Tests

**File:** `tests/streaming.rs` (new section)

| # | Test Name | Description |
|---|-----------|-------------|
| 1 | `flush_pending_captures_text_without_stop` | Text delta without ContentBlockStop → flush_pending captures it |
| 2 | `flush_pending_captures_thinking_without_stop` | Thinking delta without ContentBlockStop → flush_pending captures it |
| 3 | `flush_pending_captures_tool_input_without_stop` | ToolInputDelta without ContentBlockStop → flush_pending captures it |
| 4 | `flush_pending_no_double_flush` | ContentBlockStop followed by Completed → flush_pending does not double-flush |
| 5 | `flush_pending_empty_stream` | Empty stream → flush_pending returns empty content |
| 6 | `flush_pending_mixed_content_blocks` | Multiple content blocks (text + thinking + tool_use) → all flushed correctly |

**Effort:** 0.5 days

### 1.2 Agent Loop Unit Tests

**File:** `tests/agent_loop.rs` (new section)

| # | Test Name | Description |
|---|-----------|-------------|
| 1 | `single_tool_call_then_text` | One tool call → model responds with text → loop ends |
| 2 | `multi_turn_tool_chain` | Tool call → result → tool call → result → text (3+ turns) |
| 3 | `max_tool_depth_reached` | `max_tool_depth: 2` with 4 tool rounds → stops at 2 |
| 4 | `tool_error_stops_loop` | `continue_on_tool_error: false` → tool error → loop stops |
| 5 | `tool_error_recovers` | `continue_on_tool_error: true` → tool error → model recovers with text |
| 6 | `cancellation_mid_loop` | Cancel token fired during tool execution → loop returns Cancelled |
| 7 | `cancellation_before_loop` | Cancel token fired before run() → immediate Cancelled error |
| 8 | `fallback_tool_parsing_xml` | Model emits tool call as XML → fallback parsing extracts it |
| 9 | `fallback_tool_parsing_json` | Model emits tool call as fenced JSON → fallback parsing extracts it |
| 10 | `observer_receives_all_event_types` | RecordingObserver captures TextDelta, ThinkingDelta, ToolCallStart, ToolCallComplete, IterationStart, IterationComplete, LoopComplete |

**Note:** Many of these already exist! Review current `tests/agent_loop.rs` — it already has:
- `multi_round_tool_chain_depth_3` ✅
- `parallel_tool_calls_single_response` ✅
- `tool_error_stops_loop` ✅
- `tool_error_continues_when_configured` ✅
- `cancellation_stops_loop` ✅
- `observer_event_sequence_full_round` ✅
- `observer_receives_thinking_deltas` ✅
- `parallel_tool_preserves_history_order` ✅
- `parallel_tool_observer_batching` ✅
- `parallel_tool_mixed_success_failure` ✅
- `parallel_tool_error_stops_loop` ✅
- `max_tool_depth_1` ✅
- `content_filter_stop_reason` ✅

**Remaining from this list:** `fallback_tool_parsing_xml`, `fallback_tool_parsing_json`

**Effort:** 0.5 days (only fallback parsing tests needed)

### 1.3 MCP Unit Tests

**File:** `tests/mcp.rs` (new file)

| # | Test Name | Description |
|---|-----------|-------------|
| 1 | `mcp_tool_wrapper_text_content` | MCP tool returns text content → McpToolWrapper converts to ToolResult |
| 2 | `mcp_tool_wrapper_error_content` | MCP tool returns error → McpToolWrapper converts to error ToolResult |
| 3 | `mcp_tool_wrapper_mixed_content` | MCP tool returns text + error → McpToolWrapper handles mixed |
| 4 | `mcp_tool_wrapper_resource_content` | MCP tool returns resource (blob/base64) → McpToolWrapper extracts text |
| 5 | `mcp_tool_schema_conversion` | rmcp Tool → ToolDefinition conversion (name, description, input_schema) |
| 6 | `mcp_tool_schema_sanitization` | MCP tool with invalid schema characters → sanitized name |
| 7 | `mcp_manager_connect_stdio` | McpManager connects to stdio MCP server → tools discovered |
| 8 | `mcp_manager_connect_http` | McpManager connects to HTTP MCP server → tools discovered |
| 9 | `mcp_manager_disconnect` | McpManager disconnects → connections cleaned up |
| 10 | `mcp_manager_sync_tools` | McpManager syncs tools after connection → registry updated |
| 11 | `mcp_tool_cancellation` | MCP tool execution cancelled via CancellationToken → returns error |
| 12 | `mcp_manager_parallel_connect` | McpManager connects multiple servers in parallel → all succeed |

**Effort:** 1-2 days

### 1.4 ADR-0002: Agent Loop + Dynamic Tools

**File:** `docs/adr/ADR-0002-agent-loop-dynamic-tools.md`

Document the following design decisions:

1. **Tool depth counting** — counts tool execution rounds, not model responses
2. **Observer pattern** — why AgentObserver instead of Stream hooks
3. **`flush_pending()` safety net** — handles providers without ContentBlockStop
4. **Parallel tool execution** — why `join_all` over sequential (and the tradeoffs)
5. **Tool registry re-read each turn** — dynamic tool add/remove between turns
6. **3-phase tool execution** — emit starts, execute, emit completes (order preservation)
7. **`continue_on_tool_error` semantics** — error as model feedback vs hard stop
8. **AgentLoop consumes self** — why `run(self)` not `run(&self)`

**Effort:** 0.5 days

## Priority 2: Aura Integration Test Requirements

These tests are required to validate agent-driver-rs as a rig.rs replacement.

### 2.1 Sequential Tool Execution Tests

**File:** `tests/agent_loop.rs` (new section) or `tests/sequential_execution.rs`

| # | Test Name | Description |
|---|-----------|-------------|
| 1 | `sequential_execution_single_tool` | `sequential_tool_execution: true` with 1 tool → executes sequentially (baseline) |
| 2 | `sequential_execution_multiple_tools` | `sequential_tool_execution: true` with 3 tools → executes one at a time |
| 3 | `sequential_execution_order_preserved` | Tools A, B, C in response → execute in A, B, C order (not parallel race) |
| 4 | `sequential_execution_observer_order` | RecordingObserver: ToolCallStart(A) → ToolCallComplete(A) → ToolCallStart(B) → ToolCallComplete(B) |
| 5 | `sequential_execution_fifo_queue_simulation` | Simulate Aura's FIFO tool_event_broker: push all IDs, then peek/pop sequentially → correct mapping |
| 6 | `parallel_execution_still_works` | `sequential_tool_execution: false` (default) → parallel execution unchanged |
| 7 | `sequential_execution_slower_but_correct` | Measure timing: 3 tools at 50ms each → sequential takes ~150ms, parallel takes ~50ms |

**Critical test:** #5 — this directly validates that Aura's `tool_event_broker` would work correctly with sequential mode.

**Effort:** 1 day

### 2.2 MCP Cancellation Tests

**File:** `tests/mcp.rs` (new section)

| # | Test Name | Description |
|---|-----------|-------------|
| 1 | `mcp_cancelled_notification_sent` | Client disconnects → MCP server receives `notifications/cancelled` |
| 2 | `mcp_in_flight_tracking` | Multiple MCP calls for same request → all tracked by request_id |
| 3 | `mcp_cancel_all_for_request` | Cancel all in-flight MCP calls for a request → all receive cancellation |
| 4 | `mcp_cancel_does_not_affect_other_requests` | Cancel request A → request B's MCP calls continue |
| 5 | `mcp_cancel_during_long_operation` | MCP tool takes 5s → cancelled at 1s → server receives notification |
| 6 | `mcp_cancel_and_close` | Cancel + close connection → all in-flight cancelled, connection closed |

**Requires:** Mock MCP server that tracks received notifications.

**Effort:** 1-2 days

### 2.3 MCP Progress Tests

**File:** `tests/mcp.rs` (new section)

| # | Test Name | Description |
|---|-----------|-------------|
| 1 | `mcp_progress_received` | MCP server sends progress → client receives it |
| 2 | `mcp_progress_routed_to_correct_request` | Progress for request A → delivered to request A's channel, not B's |
| 3 | `mcp_progress_orphaned_warning` | Progress arrives after request cancelled → warning logged |
| 4 | `mcp_progress_subscription` | Client subscribes to progress → receives notifications during tool execution |
| 5 | `mcp_progress_with_cancellation` | Progress arrives during cancellation → handled gracefully |

**Requires:** Mock MCP server that sends progress notifications.

**Effort:** 1 day

### 2.4 MCP Header Forwarding Tests

**File:** `tests/mcp.rs` (new section)

| # | Test Name | Description |
|---|-----------|-------------|
| 1 | `mcp_http_static_headers` | HTTP MCP connection with static headers → headers sent on all requests |
| 2 | `mcp_http_forwarded_headers` | HTTP MCP connection with forwarded headers → headers from request passed through |
| 3 | `mcp_http_bearer_auth` | HTTP MCP connection with bearer token → Authorization header sent |
| 4 | `mcp_stdio_no_headers` | STDIO MCP connection → header forwarding is no-op (expected) |

**Requires:** Mock HTTP MCP server that echoes received headers.

**Effort:** 0.5 days

### 2.5 Per-Provider Agent Loop Tests

**File:** `tests/provider_agent_loop.rs` (new file)

| # | Test Name | Provider | Description |
|---|-----------|----------|-------------|
| 1 | `anthropic_agent_loop_basic` | Anthropic | Simple text response via agent loop |
| 2 | `anthropic_agent_loop_tool` | Anthropic | Tool call + result + text response |
| 3 | `anthropic_agent_loop_mcp` | Anthropic | MCP tool via agent loop |
| 4 | `openai_agent_loop_basic` | OpenAI | Simple text response via agent loop |
| 5 | `openai_agent_loop_tool` | OpenAI | Tool call + result + text response |
| 6 | `openai_agent_loop_mcp` | OpenAI | MCP tool via agent loop |
| 7 | `ollama_agent_loop_basic` | Ollama | Simple text response via agent loop |
| 8 | `ollama_agent_loop_tool` | Ollama | Tool call + result + text response |
| 9 | `ollama_agent_loop_mcp` | Ollama | MCP tool via agent loop |
| 10 | `bedrock_agent_loop_tool` | Bedrock | Tool call + result + text (regression test for bug fix) |
| 11 | `bedrock_agent_loop_mcp` | Bedrock | MCP tool via agent loop (regression test for bug fix) |

**Note:** These are smoke tests — they verify the full path works, not edge cases. Use live API keys (feature-gated behind `live-tests` or similar).

**Effort:** 2-3 days (mostly manual setup + verification)

### 2.6 Error Classification Tests

**File:** `tests/error_classification.rs` (new file)

| # | Test Name | Description |
|---|-----------|-------------|
| 1 | `is_retriable_rate_limited` | RateLimited error → is_retriable() = true |
| 2 | `is_retriable_timeout` | Timeout error → is_retriable() = true |
| 3 | `is_retriable_5xx` | HTTP 5xx error → is_retriable() = true |
| 4 | `is_retriable_4xx` | HTTP 4xx error → is_retriable() = false |
| 5 | `is_recoverable_context_window` | ContextWindowExceeded → is_recoverable() = true |
| 6 | `is_recoverable_content_policy` | ContentPolicyViolation → is_recoverable() = true |
| 7 | `is_recoverable_rate_limited` | RateLimited → is_recoverable() = true |
| 8 | `context_overflow_detection_sse` | SSE provider returns 400 → AgentLoopError::is_context_overflow() = true |
| 9 | `context_overflow_detection_sdk` | SDK provider returns error → AgentLoopError::is_context_overflow() = true |
| 10 | `content_policy_detection_sse` | SSE provider returns 400 → AgentLoopError::is_content_policy_violation() = true |

**Effort:** 0.5 days

## Priority 3: Integration Test Suite

### 3.1 End-to-End Mock Provider Test

**File:** `examples/e2e_mock_provider.rs` (new file)

Full agent loop exercise without live API keys:

1. MockProvider with predefined response sequence
2. Session with tools (native + MCP mock)
3. AgentLoop with observer
4. Verify: correct event sequence, tool execution, final response, token usage

**Effort:** 0.5 days

### 3.2 Mock MCP Server Integration Test

**File:** `examples/mcp_mock_server.rs` + `tests/mcp_integration.rs`

In-process MCP server that:
- Registers tools
- Validates tool call format
- Returns predefined responses
- Sends progress notifications
- Receives cancellation notifications

**Effort:** 1 day

### 3.3 Cancellation Chain Test

**File:** `tests/cancellation.rs` (new file)

| # | Test Name | Description |
|---|-----------|-------------|
| 1 | `session_cancel_cascades_to_provider` | Session.cancel() → provider call cancelled |
| 2 | `session_cancel_cascades_to_tool` | Session.cancel() → tool execution cancelled |
| 3 | `agent_loop_cancel_stops_collection` | AgentLoop cancellation → stream collection stops |
| 4 | `tool_cancel_does_not_affect_session` | ToolContext cancel → session still operational |
| 5 | `shutdown_waits_for_tasks` | Session.shutdown() → waits for TaskPool tasks to complete |

**Effort:** 0.5 days

## Test Execution Strategy

### CI Integration

```bash
# Fast tests (every PR)
cargo test --all-features

# MCP tests (every PR, requires no external deps)
cargo test --features "mcp test-support"

# Live provider tests (nightly or manual)
cargo test --features "anthropic openai bedrock ollama live-tests" -- test_provider_

# Sequential execution tests (every PR)
cargo test --features "test-support" -- test_sequential_
```

### Test Feature Flags

Proposed additions to `Cargo.toml`:

```toml
[features]
test-support = []          # MockProvider for integration tests
live-tests = []            # Live API key tests (anthropic, openai, etc.)
mcp-http = ["mcp", ...]    # HTTP MCP transport
```

## Test Count Projections

| Category | Current | After P1 | After P2 | After P3 |
|----------|---------|----------|----------|----------|
| Agent loop | 13 | 15 | 22 | 22 |
| Streaming | 7 | 13 | 13 | 13 |
| Session | 8 | 8 | 8 | 8 |
| MCP | ~2 | 14 | 26 | 28 |
| Error classification | 0 | 10 | 10 | 10 |
| Sequential execution | 0 | 0 | 7 | 7 |
| Cancellation | 1 | 1 | 6 | 6 |
| Per-provider (live) | 0 | 0 | 11 | 11 |
| Integration (e2e) | 0 | 0 | 0 | 3 |
| **Total** | **~61** | **~61** | **~103** | **~108** |

## Effort Summary

| Priority | Tasks | Effort |
|----------|-------|--------|
| P1: TODO.md Milestone 2 | flush_pending, fallback parsing, MCP unit tests, ADR-0002 | 3-4 days |
| P2: Aura integration | Sequential execution, MCP cancellation, MCP progress, header forwarding, per-provider smoke tests, error classification | 6-8 days |
| P3: Integration suite | E2E mock, mock MCP server, cancellation chain | 2-3 days |
| **Total** | | **11-15 days** |

## Dependencies

```
P1 (Milestone 2)
  ├── flush_pending tests
  ├── Agent loop tests (mostly done)
  ├── MCP unit tests
  └── ADR-0002

P2 (Aura Integration)
  ├── Sequential execution tests ◄── requires P1 MCP tests (for McpToolWrapper)
  ├── MCP cancellation tests ◄── requires P1 MCP tests
  ├── MCP progress tests ◄── requires P1 MCP tests
  ├── Header forwarding tests ◄── requires P1 MCP tests
  ├── Per-provider smoke tests ◄── requires Bedrock bug fix (Milestone 1)
  └── Error classification tests (independent)

P3 (Integration Suite)
  ├── E2E mock provider ◄── requires P1 + P2
  ├── Mock MCP server ◄── requires P1 MCP tests
  └── Cancellation chain ◄── independent
```

## Success Criteria

Before starting the Aura integration spike (Phase 5 of roadmap):

- [ ] All P1 tests passing (>90 tests total)
- [ ] Bedrock tool loop bug fixed and regression-tested
- [ ] Sequential execution config implemented and tested
- [ ] All 5 providers verified with agent loop (smoke tests)
- [ ] MCP cancellation propagation working
- [ ] MCP progress notifications working
- [ ] MCP header forwarding working
- [ ] ADR-0002 written and reviewed

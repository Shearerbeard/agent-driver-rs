# Thinking/Reasoning Status Map

> Generated 2026-05-26 from a full code audit. Living doc — update as work progresses.

## Architecture Map: Thinking/Reasoning Flow

```
Request Config          Parsing (streaming)       Collection (shared)      Serialization (multi-turn)
─────────────          ────────────────────       ───────────────────      ─────────────────────────
[per-provider]    →    [per-provider]        →    streaming.rs         →   [per-provider]
                                                  CollectedResponse
                                                  apply_delta()
                                                  finalize_block()
```

### Shared modules (touch once, all providers benefit)

| Module | File | What it does |
|--------|------|-------------|
| **ContentBlock enum** | `src/types/message.rs:192-217` | `Text`, `Thinking`, `ToolUse`, `ToolResult` — **missing `Signature`, `RedactedThinking`** |
| **StreamDelta enum** | `src/streaming.rs:28-46` | `ThinkingDelta`, `SignatureDelta` exist — SignatureDelta **dropped in apply_delta (line 199)** |
| **CollectedResponse** | `src/streaming.rs:147-293` | `pending_thinking` accumulator works. `flush_pending` handles thinking. **Signatures ignored.** |
| **buffered_sse_stream** | `src/provider/stream_adapter.rs:36-125` | Shared SSE adapter — used by **Anthropic + OpenRouter** |
| **buffered_sdk_stream** | `src/provider/stream_adapter.rs:140-213` | Shared SDK adapter — used by **OpenAI + Ollama** |

### Per-provider status

| Provider | Config | Parsing | Multi-turn serialize | Shared infra |
|----------|--------|---------|---------------------|-------------|
| **Anthropic** | `budget_tokens` wired (line 112) | ThinkingDelta + SignatureDelta parsed (lines 360, 384) | `type: "thinking"` preserved (line 139) | `buffered_sse_stream` |
| **Ollama** | `.think(true)` wired (line 264) | ThinkingDelta parsed (line 430) | Merged into text via `extract_text()` (line 156) | `buffered_sdk_stream` |
| **Bedrock** | Hardcoded `false` (line 65) | **Nothing** — no ReasoningContent handling | Lossy `<thinking>` XML (line 154) | Custom AWS SDK stream |
| **OpenAI** | Dead code — `ReasoningConfig` exists (config/openai.rs:106) but never sent | **Nothing** — no `reasoning_content` parsing | Merged into text (line 138) | `buffered_sdk_stream` |
| **OpenRouter** | **Nothing** | **Nothing** — no `reasoning_details` parsing | **Nothing** — thinking not handled | `buffered_sse_stream` |

---

## P0 — Trust the foundation

### 1. Core types fix (shared — all providers benefit)

**Files:** `src/types/message.rs`, `src/streaming.rs`

- [ ] Add `ContentBlock::Signature { signature: String }` to ContentBlock enum
- [ ] Add `ContentBlock::RedactedThinking { data: String }` to ContentBlock enum
- [ ] Fix `CollectedResponse::apply_delta` (line 199) — accumulate `SignatureDelta` into new `pending_signature: Option<String>` instead of ignoring it
- [ ] Add `finalize_block` handling for signature (flush pending_signature → `ContentBlock::Signature`)
- [ ] Update `flush_pending` to handle pending_signature

**Why first:** Every provider that parses signatures (currently only Anthropic, but Bedrock will too) depends on this fix. One change, multiple beneficiaries.

### 2. Anthropic thinking — complete the last mile

**File:** `src/provider/anthropic.rs`

**Already working:** ThinkingDelta parsing (line 360), SignatureDelta parsing (line 384), multi-turn serialization of `ContentBlock::Thinking` (line 139), `budget_tokens` config (line 112).

- [ ] Serialize `ContentBlock::Signature` back in multi-turn requests (add case in `serialize_message` around line 139)
- [ ] Serialize `ContentBlock::RedactedThinking` back in multi-turn requests
- [ ] Add adaptive mode config: `{ "type": "adaptive", "display": ..., "output_config": { "effort": ... } }` — currently only manual `{ "type": "enabled", "budget_tokens": N }` (line 112-117)
- [ ] Add `ThinkingConfig` variants in `src/config/anthropic.rs` for adaptive + display + effort

**Effort:** ~0.5 days. Most of the pipeline works — just signature round-trip and config expansion.

### 3. Bedrock thinking — build from scratch

**File:** `src/provider/bedrock.rs`

**Nothing works today.** `extended_thinking` is hardcoded `false` (line 65). No parsing, no config, lossy XML serialization.

- [ ] Enable `extended_thinking` capability based on config (line 65)
- [ ] Add `BedrockThinkingConfig` to `src/config/bedrock.rs` — `additionalModelRequestFields.thinking` with `type: "enabled"` + `budget_tokens`
- [ ] Wire thinking config into request body (Converse API `additional_model_request_fields`)
- [ ] Parse `ContentBlockDelta::ReasoningContent` in `parse_bedrock_event` (line 538 area) — has `.text` and `.signature` members
- [ ] Emit `StreamDelta::ThinkingDelta` for reasoning text
- [ ] Emit `StreamDelta::SignatureDelta` for reasoning signature
- [ ] Fix multi-turn: serialize `ContentBlock::Thinking` as proper Bedrock reasoning block, not `<thinking>` XML (line 154)
- [ ] Serialize `ContentBlock::Signature` in multi-turn

**Effort:** ~1.5 days. AWS SDK types are `#[non_exhaustive]` so reasoning variants may need runtime detection.

**Shares with Anthropic:** Same `ContentBlock::Signature` type (from #1), same `CollectedResponse` accumulation. Different wire format.

### 4. OpenAI reasoning — wire the dead code

**File:** `src/provider/openai.rs`

Config exists (`src/config/openai.rs:106-137` — `ReasoningConfig` with `effort` + `summary`) but is never applied to requests.

- [ ] Wire `reasoning_effort` into `CreateChatCompletionRequestArgs` builder (around line 299-330)
- [ ] Parse `reasoning_content` from response delta in `parse_openai_chunk` (line 455 area) — emit as `StreamDelta::ThinkingDelta`
- [ ] Handle `reasoning_content` in multi-turn: include as separate content block, not merged into text (line 138)

**Effort:** ~0.5 days. Config + types exist, just needs wiring.

**Uses:** `buffered_sdk_stream` (shared with Ollama).

### 5. OpenRouter reasoning — variant of OpenAI

**File:** `src/provider/openrouter.rs`

OpenRouter uses OpenAI-compatible format but different reasoning keys: `reasoning_details` in responses vs OpenAI's `reasoning_content`.

- [ ] Parse `reasoning` and `reasoning_details` from SSE response in `parse_openrouter_event` (line 394 area) — emit as `StreamDelta::ThinkingDelta`
- [ ] Add `reasoning_effort` config to `OpenRouterConfig`
- [ ] Wire reasoning config into request body
- [ ] Handle reasoning in multi-turn serialization (pass `reasoning_details` back)

**Effort:** ~0.5 days. Very similar to OpenAI (#4) but uses `buffered_sse_stream` (shared with Anthropic) and different response keys.

**Pattern:** OpenAI and OpenRouter are both "OpenAI-compatible" but diverge on: (a) reasoning key names, (b) transport (SDK vs SSE), (c) model-specific quirks. Not worth a shared abstraction — the duplication is small and the divergence is meaningful.

### 6. Parallel tool execution with proper IDs

**Files:** `src/agent/driver.rs`, `src/tool/executor.rs`

- [ ] Audit tool_call_id assignment across all providers — ensure IDs are unique and stable
- [ ] Verify parallel tool calls return results correlated to correct IDs
- [ ] Test: 3 parallel tool calls → results arrive in any order → correct IDs matched

**Why this framing:** The goal isn't "add sequential mode" — it's "make parallel correct so consumers can trust it."

### 7. Unit test coverage (ADR-0003)

**Files:** `tests/` directory

- [ ] Bedrock `parse_bedrock_event` — 8+ tests (zero coverage today). All event types: MessageStart, ContentBlockStart, ContentBlockDelta (text, tool, reasoning), ContentBlockStop, MessageStop, Metadata, malformed input
- [ ] `buffered_sse_stream` — 4+ tests. SSE framing, partial events, reconnection, malformed data
- [ ] OpenAI `convert_messages` — 4+ tests. Role mapping, tool_use blocks, tool results, empty messages
- [ ] OpenRouter `serialize_message` — 3+ tests
- [ ] Anthropic `serialize_message` — 3+ tests

**Target:** 22+ new unit tests.

### 8. Live integration tests with provider argument

- [ ] `tests/live_provider.rs` scaffold — provider-agnostic test structure, env-gated via `LIVE_TEST_PROVIDERS`
- [ ] Canonical test tool: `sum(a, b)` — deterministic, all models can use it
- [ ] Test categories: basic text, streaming deltas, tool calling round-trip, multi-turn, thinking (where supported)
- [ ] CLI-driven: `LIVE_TEST_PROVIDERS=bedrock,openai cargo test --features "bedrock openai" live_`

---

## P1 — Blockers (need before production)

### 9. MCP progress notifications

**Current state:** Zero support. No `ClientHandler`, no `ProgressToken`, no request-scoped routing.

- [ ] Design: how does progress route from MCP server → agent loop → observer?
- [ ] Implement `ClientHandler` for receiving server notifications
- [ ] Add `ProgressToken` support in tool call requests
- [ ] Route progress to `AgentObserver` (new event type)

### 10. MCP server-side cancellation notification

**Current state:** Client-side cancellation fully works (CancellationToken → tokio::select! in mcp.rs:233-240). Missing: telling the server to stop.

- [ ] Track outbound MCP request IDs in `McpToolWrapper`
- [ ] On cancellation, send `notifications/cancelled` with the request ID
- [ ] Needs rmcp notification handler setup (currently `().serve(transport)` — empty handler, mcp.rs:75-76)

### 11. Gemini provider

Production configs depend on Gemini — this is a real blocker.

- [ ] New provider implementation
- [ ] Streaming, tool calling, multi-turn
- [ ] Live integration tests

### 12. OpenRouter provider completeness

Beyond reasoning (#5 above):
- [ ] User-testable CLI smoke test mode
- [ ] Model listing / discovery

---

## P2 — Important, not blocking

### 13. Prompt caching (ADR-0005)

Huge value but needs architectural dividing lines between providers settled first. Don't scatter cache config across 5 providers before the composition patterns are right.

- [ ] `PromptCacheConfig`, `CachePolicy`, `RequestCacheConfig` — provider-agnostic
- [ ] `TokenUsage` cache fields
- [ ] Per-provider wiring: OpenAI (header), Anthropic (cache_control), Bedrock (CachePointBlock)

### 14. Unified Claude provider

Dedupe Anthropic/Bedrock message formatting. Lower priority — the duplication is manageable.

### 15. Ollama thinking verification

Confirm streaming is truly incremental vs all-at-once.

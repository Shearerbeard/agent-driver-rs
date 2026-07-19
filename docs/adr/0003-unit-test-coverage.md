# ADR-0003: Unit Test Coverage Across Provider Branches

**Status:** Proposed
**Date:** 2026-05-03
**Context tags:** [testing] [provider] [streaming]

---

## Context

A test audit (May 2026) revealed uneven unit test coverage across provider implementations. Parsing and message conversion functions are the critical path for every provider — a bug here silently corrupts streaming data, tool calls, or stop reasons. Some providers have excellent coverage (Anthropic: 10 tests, Ollama: 10 tests), while others have critical gaps (Bedrock parse: 0 tests, SSE stream adapter: 0 tests).

### Current Coverage

Counts updated 2026-07-18. The May 2026 audit found Bedrock parse and OpenAI
`convert_messages` at zero; the June quality-audit added 3 Bedrock parser tests
(tool-use paths only) and 1 OpenAI system-dedup test. Both remain below the
targets in the Decision.

| Provider | Parse Function | Tests | Convert Messages | Tests | Build Request | Tests |
|----------|---------------|-------|-----------------|-------|---------------|-------|
| Anthropic | `parse_anthropic_event` | **10** | `serialize_message` | 0 | `build_request_body` | 0 |
| Bedrock | `parse_bedrock_event` | **3** | `convert_messages` | 4 | N/A (SDK) | — |
| OpenAI | `parse_openai_chunk` | **4** | `convert_messages` | 1 | N/A (SDK) | — |
| Ollama | `parse_ollama_response` | **10** | `convert_messages` | 2 | N/A (SDK) | — |
| OpenRouter | `parse_openrouter_event` | **6** | `serialize_message` | 0 | `build_request_body` | 0 |

| Shared | Function | Tests |
|--------|----------|-------|
| stream_adapter | `buffered_sse_stream` | **0** |
| stream_adapter | `buffered_sdk_stream` | **5** |
| retry | `with_retry` | **4** |
| mock | helpers + provider | **8** |

### Risk Assessment

- **Bedrock parse_bedrock_event: HIGH** — Zero coverage for the entire streaming pipeline. This is the provider most likely to be used in production (AWS). Event types include MessageStart, ContentBlockStart (text vs tool_use), ContentBlockDelta, ContentBlockStop, MessageStop, and Metadata — none are tested.
- **buffered_sse_stream: MEDIUM** — Used by Anthropic and OpenRouter. The SDK variant has 5 tests; the SSE variant has 0. SSE parsing has different failure modes (reconnection, partial events, malformed data).
- **OpenAI convert_messages: MEDIUM** — Message format conversion untested. OpenAI's format diverges from Claude (roles, tool_call structure, function calling).
- **Serialization functions: LOW** — `build_request_body` and `serialize_message` for Anthropic/OpenRouter are exercised indirectly via live examples. Bugs here surface quickly.

---

## Decision

### 1. Coverage Requirements

Every provider parse function must have unit tests covering:
- **Happy path per event type** — one test per distinct event variant the provider can emit (e.g., MessageStart, TextDelta, ToolUseStart, ContentBlockStop, Completed)
- **Stop reason mapping** — every stop reason variant the provider returns (end_turn, max_tokens, tool_use, content_filter)
- **Malformed input** — at least one test for graceful handling of unparseable data
- **Tool call parsing** — single and parallel tool calls, including argument accumulation across deltas

Every provider message conversion function must have unit tests covering:
- **Role mapping** — user, assistant, system, tool roles convert correctly
- **Tool use blocks** — tool_use content blocks serialize to the provider's expected format
- **Tool result blocks** — tool results map to the correct provider format
- **Edge cases** — empty messages, messages with mixed content types

### 2. Priority Order

1. **Bedrock `parse_bedrock_event`** — highest risk, zero coverage
2. **`buffered_sse_stream`** — shared infrastructure, two providers depend on it
3. **OpenAI `convert_messages`** — format divergence risk
4. **OpenRouter `serialize_message`** — shares Anthropic's SSE path but has its own serialization
5. **Anthropic `serialize_message` / `build_request_body`** — lowest risk (well-exercised via examples)

### 3. Test Patterns

Tests follow the existing project pattern: construct input fixture data, call the function, assert on the returned `StreamEvent` or provider-specific type. No network calls, no mocking of external services.

```rust
#[test]
fn parse_message_start() {
    let json: JsonValue = serde_json::from_str(r#"{"type":"message_start","message":{...}}"#).unwrap();
    let events = parse_provider_event(&json);
    assert!(matches!(events[0], StreamEvent::Started { .. }));
}
```

Bedrock tests require constructing AWS SDK types (`ConverseStreamOutput` variants) rather than JSON, following the pattern in existing `convert_messages` tests. Since AWS SDK types are verbose to construct, add fixture factory functions (e.g., `make_text_delta(text)`, `make_tool_use_start(name)`) in the test module to minimize boilerplate.

### 4. Test Count Targets

| Provider | Current (at decision, May 2026) | Target | Delta |
|----------|---------------------------------|--------|-------|
| Bedrock parse | 0 | 8+ | +8 |
| buffered_sse_stream | 0 | 4+ | +4 |
| OpenAI convert_messages | 0 | 4+ | +4 |
| OpenRouter serialize | 0 | 3+ | +3 |
| Anthropic serialize | 0 | 3+ | +3 |
| **Total** | **0** | **22+** | **+22** |

---

## Consequences

### Positive
- Bedrock streaming bugs caught before live testing (the current Bedrock tool-loop bug in TODO.md may be a parse issue)
- SSE adapter tested at the same level as the SDK adapter
- Confidence in message format conversion before adding live provider tests (ADR-0004)
- Parsing regressions caught immediately by CI

### Negative
- ~22 new tests to write and maintain
- Bedrock tests require constructing AWS SDK types, which can be verbose
- Some conversion tests may be brittle if provider SDKs change type signatures on upgrade

### Neutral
- Does not replace live integration tests (ADR-0004) — unit tests verify parsing logic, live tests verify the full request→response→parse pipeline
- No changes to production code required

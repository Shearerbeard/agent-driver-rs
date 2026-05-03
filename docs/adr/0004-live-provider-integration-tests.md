# ADR-0004: Live Provider Integration Tests

**Status:** Proposed
**Date:** 2026-05-03
**Context tags:** [testing] [provider] [integration] [ci]

---

## Context

All existing integration tests use `MockProvider` with deterministic responses. This gives strong coverage of the agent loop, session, and streaming infrastructure — but misses the entire class of bugs that live between our code and the provider's API:

- Message serialization accepted by the provider
- Streaming format actually parseable (SSE framing, chunking, backpressure)
- Tool calling round-trips (model decides to call a tool, we execute, model incorporates result)
- Token counts in `CompletionMetadata` populated from real responses
- Provider-specific error responses (rate limits, context window, content policy)
- Stop reason mapping from actual model behavior (not synthetic fixtures)
- OTel span attributes from real responses (for future LLM span work)

The existing examples (`phoenix_integration_bedrock.rs`, `phoenix_integration_ollama.rs`) exercise live providers but don't assert behavior — they print output and exit. They also require manual invocation.

### Provider Landscape

| Provider | Auth | Cost | Latency | Availability |
|----------|------|------|---------|-------------|
| Ollama | None (local) | Free | ~1-5s | Requires local/Docker instance |
| Bedrock | AWS SSO/IAM | Per-token | ~2-8s | Requires AWS account |
| Anthropic | API key | Per-token | ~2-6s | Requires API key |
| OpenAI | API key | Per-token | ~1-5s | Requires API key |
| OpenRouter | API key | Per-token | ~2-10s | Requires API key, routes to upstream |

---

## Decision

### 1. Test File Structure

A single test file `tests/live_provider.rs` that takes provider and model as configuration, not as code paths. Each test is a scenario (basic text, tool calling, streaming, etc.) that runs identically against any provider.

```
tests/live_provider.rs
├── helper: create_session(provider, model, tracer?) → Session
├── test: basic_text_response
├── test: tool_calling_round_trip
├── test: streaming_text_deltas
├── test: multi_turn_conversation
├── test: stop_reason_end_turn
└── test: token_count_populated
```

### 2. Provider Configuration via Environment

Tests are gated by environment variables. If the variable is not set, the test is skipped (not failed). This makes them safe to run in CI without credentials.

```
# Provider selection (one or more, comma-separated)
LIVE_TEST_PROVIDERS=ollama,bedrock,anthropic

# Per-provider config (existing from_env patterns)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3:8b

BEDROCK_INFERENCE_PROFILE=us.anthropic.claude-sonnet-4-5-20250929-v1:0
# (uses AWS default credential chain)

ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-20250514

OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o

OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=anthropic/claude-sonnet-4
```

### 3. Test Categories

Each category tests a specific capability. Not all providers support all categories.

| Category | What it verifies | Providers |
|----------|-----------------|-----------|
| **Basic text** | Send prompt, get non-empty text response | All 5 |
| **Streaming deltas** | TextDelta events arrive before Completed | All 5 |
| **Tool calling** | Model calls a tool, we execute, model uses result | All 5 |
| **Multi-turn** | 2+ user messages, conversation context preserved | All 5 |
| **Stop reasons** | EndTurn maps correctly from provider response | All 5 |
| **Token counts** | `CompletionMetadata.usage` is `Some` with non-zero values | All 5 (provider-dependent) |
| **Content filter** | Provider content filter triggers gracefully | Anthropic, OpenAI, Bedrock |
| **Extended thinking** | Thinking blocks emitted before text | Anthropic, Bedrock, Ollama |

### 4. Test Execution Pattern

```rust
/// Skip test if provider not configured
fn providers() -> Vec<(Box<dyn Provider>, ModelId, &'static str)> {
    let mut providers = vec![];
    let requested = std::env::var("LIVE_TEST_PROVIDERS").unwrap_or_default();

    if requested.contains("ollama") {
        if let Ok(config) = OllamaConfig::from_env() {
            let model = config.model.as_str().to_string();
            providers.push((
                Box::new(OllamaProvider::new(config).unwrap()) as Box<dyn Provider>,
                ModelId::new(&model).unwrap(),
                "ollama",
            ));
        }
    }
    // ... same for each provider
    providers
}

#[tokio::test]
async fn basic_text_response() {
    for (provider, model, name) in providers() {
        let session = SessionBuilder::new()
            .with_provider(provider)
            .model(model)
            .build().await.unwrap();

        let outcome = AgentLoop::new(&session)
            .run("What is 2 + 2? Answer in one sentence.")
            .await
            .unwrap();

        assert!(!outcome.final_response.text().is_empty(),
            "[{}] empty response", name);
        assert!(outcome.final_response.text().contains('4'),
            "[{}] expected '4' in response: {}", name, outcome.final_response.text());

        session.shutdown().await;
    }
}
```

### 5. Provider × Category Matrix

This is the target — not all cells need to be filled at once.

| | Ollama | Bedrock | Anthropic | OpenAI | OpenRouter |
|---|---|---|---|---|---|
| Basic text | P0 | P0 | P1 | P1 | P1 |
| Streaming deltas | P0 | P0 | P1 | P1 | P1 |
| Tool calling | P0 | P0 | P1 | P1 | P1 |
| Multi-turn | P1 | P1 | P2 | P2 | P2 |
| Stop reasons | P1 | P1 | P2 | P2 | P2 |
| Token counts | P2 | P2 | P2 | P2 | P2 |
| Content filter | — | P2 | P2 | P2 | — |
| Extended thinking | P2 | P2 | P2 | — | — |

P0 = implement first (Ollama is free/local, Bedrock is primary production target).

### 6. Cargo Configuration

```toml
[[test]]
name = "live_provider"
required-features = ["test-support"]
# Note: no feature gate per provider — env vars control which providers run.
# The binary links all provider code; only the env check skips execution.
```

Alternatively, feature-gate at compile time if linking all providers is too heavy:
```toml
required-features = ["test-support", "ollama", "bedrock", "anthropic", "openai", "openrouter"]
```

### 7. CI Strategy

- **Local dev:** Run with `LIVE_TEST_PROVIDERS=ollama` (free, fast)
- **PR checks:** Skip live tests (no env vars set = all tests skipped, zero failures)
- **Nightly/manual:** Run with all providers configured via CI secrets
- **Cost control:** Prompts designed to produce short responses ("Answer in one sentence")

---

## Consequences

### Positive
- Real provider bugs caught before release (serialization, parsing, stop reason mapping)
- Confidence that provider SDK upgrades don't break streaming
- OTel span attributes from real responses available for future LLM span work
- Same test code validates all providers — parity bugs surface immediately
- Ollama tests are free and fast, suitable for local development loops

### Negative
- Live tests are slow (~2-10s per provider per test) and non-deterministic
- API-key providers incur cost per test run (mitigated by short prompts)
- Flaky network conditions can cause false failures
- Provider API changes can break tests (but that's the point — we want to know)

### Neutral
- Does not replace unit tests (ADR-0003) — unit tests verify parsing logic in isolation, live tests verify the full pipeline
- Does not replace MockProvider integration tests — those test agent loop logic, these test provider compatibility
- Test output is non-deterministic (model responses vary) — assertions must be fuzzy ("response contains '4'", not exact string match)

### Relationship to ADR-0003

ADR-0003 (unit test coverage) should be completed first. The unit parse tests catch the obvious bugs cheaply. Live tests then validate that the serialization → API → streaming → parsing pipeline works end-to-end, which unit tests cannot cover.

# CLAUDE.md - Project Guide for Claude Code

## Project Overview

`agent-driver-rs` is a Rust library providing a unified abstraction over multiple LLM providers (Anthropic, OpenAI, Bedrock, OpenRouter, Ollama) with streaming completions, dynamic tool registration, mutable system prompts, and robust task tracking with cascading cancellation.

## Quick Start

Use `README.md` as the canonical source for first-run setup, feature flags,
environment variables, and the run/test cheat sheet. Use
`docs/manual-testing.md` for the full provider smoke-test checklist.

## Architecture

Use `docs/ARCHITECTURE.md` for the canonical architecture overview and module
map. This file records agent-facing implementation rules and current priorities
only.

## Key Design Principles

### 1. Error Types First

Always define errors in `src/error.rs` before other modules. Other code depends on these.

### 2. Newtype Pattern with Validation

All domain types use newtypes with protected constructors:

```rust
// Good: Validation on construction
let model = ModelId::new("claude-sonnet-4.5")?;

// Types derive Eq + Hash when used as HashMap keys
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ToolName(String);
```

### 3. Split Locks

Separate RwLocks for independent state to avoid contention:

```rust
pub struct Session {
    system_prompt: RwLock<SystemPrompt>,  // Independent
    messages: RwLock<Vec<Message>>,        // Independent
    tools: Arc<ToolRegistry>,              // Has own lock
}
```

### 4. Streaming First

All providers implement streaming. Non-streaming via `collect()`:

```rust
let handle = provider.complete_stream(request, ctx).await?;
let response = handle.collect().await?;  // Accumulates stream
```

### 5. Cancellation Everywhere

Use `CancellationToken` from tokio-util. Providers MUST check cancellation in stream loops:

```rust
tokio::select! {
    biased;
    _ = ctx.cancellation.cancelled() => return None,
    event = stream.next() => { /* handle */ }
}
```

### 6. No Free Spawning

All tasks go through `TaskPool` with registration-before-execution guarantee.

### 7. Agent Loop = Typed Infrastructure

The agent loop (`src/agent/`) is lower-level infrastructure, not a prompting strategy:

```rust
// The loop drives: send → detect tool_use → execute → continue
let outcome = AgentLoop::new(&session)
    .with_observer(MyObserver)
    .run("Use tools to answer this")
    .await?;
```

- **Observer-driven streaming**: Loop owns the stream, forwards `TextDelta`/`ThinkingDelta` to observer
- **Tool depth counting**: Counts tool execution rounds, not model responses. Safety limit via `MaxToolDepth`
- **Re-reads registry each turn**: `continue_streaming()` snapshots tools, so tools added/removed between turns are picked up
- ReAct and other strategies compose _on top_ of this loop

## Provider Implementation Checklist

When implementing a new provider:

1. [ ] Add config in `src/config/{provider}.rs`
2. [ ] Add variant to `ProviderConfig` enum
3. [ ] Implement `Provider` trait with `complete_stream`
4. [ ] Handle rate limiting with `RetryConfig` in `src/provider/retry.rs`
5. [ ] Check `ctx.cancellation` in stream poll loop
6. [ ] Map provider errors to `ProviderError` variants
7. [ ] Add feature flag to `Cargo.toml`
8. [ ] Update `src/bin/chat.rs` to handle new provider
9. [ ] Test with live API

## Common Patterns

### Adding a new model to existing provider

```rust
// In config/{provider}.rs
pub enum ProviderModel {
    ExistingModel,
    NewModel,  // Add variant
    Custom(String),
}

impl ProviderModel {
    pub fn model_id(&self) -> &str {
        match self {
            Self::NewModel => "actual-api-model-id",
            // ...
        }
    }
}
```

### Handling provider-specific streaming formats

Each provider has different SSE formats. Parse in provider, emit standard `StreamEvent`:

```rust
// Provider receives: {"type": "content_block_delta", "delta": {"text": "Hi"}}
// Emit: StreamEvent::Delta(StreamDelta::TextDelta { text: "Hi".into() })
```

## Testing

Before committing a feature, run through `docs/manual-testing.md`. For ordinary
docs-only changes, run the relevant link/coherence checks instead of provider
smoke tests. For source changes, `make check` is the default gate.

## Contributing

### Commit conventions

Use [Conventional Commits](https://www.conventionalcommits.org/). First line lowercase, no trailing period, under 72 characters.

Format: `<type>(<optional scope>): <description>`

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`

### Branch naming

`<type>/<short-description>` — e.g. `feat/gemini-provider`, `fix/bedrock-tool-loop`

### Before submitting

```bash
make check  # runs fmt, clippy, test
```

All clippy warnings must be resolved. See `LINT_WAVES.md` for the lint configuration.

### PR process

Open a PR against `master`. Include what changed and why. Link relevant ADRs if applicable.

## Architecture Decision Records

Design decisions are documented in `docs/adr/`. Read these before making architectural changes.

- **ADR-0001:** Tool System & MCP Integration — how tools flow through prompts, MCP negotiation (Accepted)
- **ADR-0002:** Thinking & Reasoning Support Across Providers (Proposed)
- **ADR-0003:** Unit Test Coverage Across Provider Branches (Proposed)
- **ADR-0004:** Live Provider Integration Tests (Proposed)
- **ADR-0005:** Prompt Caching Support Across Providers (Proposed)
- **ADR-0006:** Multi-Agent Trace Composition with OpenInference (Proposed)
- **ADR-0007:** Codex-Style Lint & Tooling Adoption (Accepted)
- See `docs/adr/README.md` for the full index and ADR format

When proposing a significant architectural change (new subsystem, protocol integration, cross-cutting concern), write an ADR first. ADRs focus on _context and consequences_, not implementation details.

## Task Tracking

- **TODO.md**: Wave-by-wave work items for the library
- **docs/internal/agent-driver-roadmap.md**: Longer-range phase mapping
- **docs/adr/README.md**: ADR index with implementation order

**Current Priorities (in order):**

1. **ADR-0002** (Wave 1): Fix thinking/reasoning — signature loss (multi-turn broken), Bedrock thinking, Anthropic adaptive mode, OpenAI dead config
2. **ADR-0003** (Wave 2): Unit test coverage — Bedrock parse (3 tests, tool-use paths only), SSE adapter (0 tests), OpenAI convert_messages (1 test)
3. **ADR-0004** (Wave 3): Live integration tests — parameterized `tests/live_provider.rs`, Ollama + Bedrock P0
4. **ADR-0005** (Wave 4): Prompt caching support — `PromptCacheConfig`, `TokenUsage` cache fields, provider headers
5. **ADR-0006** (Wave 5): Multi-agent trace composition — `AgentTopology` enum, W3C context propagation, `graph.node.*` spans
6. **ADR-0007** (parallel): Codex-style lint/tooling — continue deferred items (`thiserror` 2 done, Dylint and stricter lint waves remain)

**Completed:**

- Toolchain modernization — MSRV 1.91.1, edition 2024, `rust-toolchain.toml` pinning stable
- Dependency modernization — `async-openai` 0.41, AWS SDK 1.135, `thiserror` 2, `backoff` removed, OTel 0.32; cargo-deny exceptions 6 → 0
- OTel/Phoenix integration — OpenInference-compliant spans (AGENT/CHAIN/TOOL), 7 conformance tests
- Phoenix: self-hosted, set via `PHOENIX_ENDPOINT` (port 4317 OTLP, port 6006 UI)
- `PHOENIX_ENDPOINT=http://your-phoenix-host:4317`
- ADR-0001: Tool System & MCP Integration (Accepted)
- ADR-0002 through ADR-0007: All written and reviewed

## Gotchas

1. **Bedrock models need inference profiles** - Modern Claude models require `BEDROCK_INFERENCE_PROFILE` env var
2. **aws_smithy_types::Document::Null** - Unit variant, not `Null(true)`
3. **OpenAI o3/o3-mini don't support streaming** - Check `model.supports_streaming()`
4. **Temperature not allowed on reasoning models** - GPT-5, o1, o3 series
5. **Edition 2024** - MSRV 1.91.1, `rust-toolchain.toml` pins stable

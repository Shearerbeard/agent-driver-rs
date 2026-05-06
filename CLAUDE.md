# CLAUDE.md - Project Guide for Claude Code

## Project Overview

`agent-driver-rs` is a Rust library providing a unified abstraction over multiple LLM providers (Anthropic, OpenAI, Bedrock, OpenRouter, Ollama) with streaming completions, dynamic tool registration, mutable system prompts, and robust task tracking with cascading cancellation.

## Quick Start

```bash
# Run with default provider (Bedrock Sonnet 4.5)
cargo run --features bedrock --bin chat

# Run tests
cargo test --features bedrock

# Check all features compile
cargo check --all-features
```

## Architecture

```
src/
├── lib.rs                 # Crate root, re-exports
├── error.rs               # All error types (define first!)
├── types/                 # Core newtypes with validation
│   ├── model.rs           # ModelId, MaxTokens, Temperature
│   ├── message.rs         # Role, Message, ContentBlock, ToolName
│   └── correlation.rs     # CorrelationId for request tracking
├── config/                # Provider-specific configurations
│   ├── provider.rs        # ProviderConfig enum (discriminator-based)
│   ├── anthropic.rs, openai.rs, bedrock.rs, etc.
├── provider/              # Provider implementations
│   ├── mod.rs             # Provider trait definition
│   ├── anthropic.rs       # Direct HTTP + SSE
│   ├── bedrock.rs         # AWS SDK converse_stream
│   └── (openai.rs, etc.)  # Other providers
├── streaming.rs           # StreamEvent, StreamDelta, StreamHandle
├── agent/                 # Agentic tool loop
│   ├── mod.rs             # Re-exports
│   ├── config.rs          # AgentLoopConfig, MaxToolDepth
│   ├── observer.rs        # AgentObserver trait, AgentEvent, LoopStopReason
│   └── driver.rs          # AgentLoop struct, AgentOutcome, run()
├── tool/                  # Tool system
│   ├── types.rs           # ToolSchema, ToolDefinition
│   ├── executor.rs        # Tool trait
│   ├── registry.rs        # Dynamic tool registration
│   └── mcp.rs             # MCP integration (McpConnection, McpManager) [mcp feature]
├── otel/                  # OpenTelemetry / Phoenix tracing [phoenix feature]
│   ├── mod.rs             # Provider init, init_phoenix(), shutdown
│   ├── types.rs           # Attribute types, enums, newtypes
│   └── instrumentation.rs # RAII span guards (AgentLoopSpan, ToolSpan, etc.)
├── task/                  # Task tracking with cancellation
│   ├── pool.rs            # TaskPool (registration-before-execution)
│   └── handle.rs          # TaskHandle wrapper
├── session.rs             # Session with split locks
└── bin/chat.rs            # CLI chat client
```

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
- ReAct and other strategies compose *on top* of this loop

## Feature Flags

```toml
[features]
default = ["anthropic", "openai", "openrouter"]
anthropic = []
openai = ["dep:async-openai"]
bedrock = ["dep:aws-sdk-bedrockruntime", "dep:aws-config", "dep:aws-smithy-types"]
openrouter = []
ollama = ["dep:ollama-rs"]
mcp = ["dep:rmcp"]
phoenix = ["dep:opentelemetry", "dep:opentelemetry_sdk", "dep:opentelemetry-otlp", "dep:opentelemetry-semantic-conventions"]
```

## Provider Implementation Checklist

When implementing a new provider:

1. [ ] Add config in `src/config/{provider}.rs`
2. [ ] Add variant to `ProviderConfig` enum
3. [ ] Implement `Provider` trait with `complete_stream`
4. [ ] Handle rate limiting with `backoff` crate
5. [ ] Check `ctx.cancellation` in stream poll loop
6. [ ] Map provider errors to `ProviderError` variants
7. [ ] Add feature flag to `Cargo.toml`
8. [ ] Update `src/bin/chat.rs` to handle new provider
9. [ ] Test with live API

## Environment Configuration

Copy `.env.example` or create `.env`:
```bash
PROVIDER=bedrock  # anthropic, openai, bedrock, openrouter, ollama

# Bedrock (current default)
BEDROCK_MODEL=claude-sonnet-4.5
BEDROCK_MAX_TOKENS=4096
BEDROCK_INFERENCE_PROFILE=us.anthropic.claude-sonnet-4-5-20250929-v1:0

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-20250514

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
```

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

Before committing a feature, run through the full checklist in [`docs/manual-testing.md`](docs/manual-testing.md). At minimum:

```bash
# Compile + test + lint (every change)
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
```

See `docs/manual-testing.md` for the full checklist, multi-tool testing, per-provider commands, and known gotchas.

## Architecture Decision Records

Design decisions are documented in `docs/adr/`. Read these before making architectural changes.

- **ADR-0001:** Tool System & MCP Integration — how tools flow through prompts, MCP negotiation (Accepted)
- **ADR-0002:** Thinking & Reasoning Support Across Providers (Proposed)
- **ADR-0003:** Unit Test Coverage Across Provider Branches (Proposed)
- **ADR-0004:** Live Provider Integration Tests (Proposed)
- **ADR-0005:** Prompt Caching Support Across Providers (Proposed)
- **ADR-0006:** Multi-Agent Trace Composition with OpenInference (Proposed)
- See `docs/adr/README.md` for the full index and ADR format

When proposing a significant architectural change (new subsystem, protocol integration, cross-cutting concern), write an ADR first. ADRs focus on *context and consequences*, not implementation details.

## Task Tracking

- **TODO.md**: Active development tasks and milestones (ADR-driven waves)
- **docs/internal/agent-driver-roadmap.md**: Strategic roadmap (Phases 0-9)
- **docs/adr/README.md**: ADR index with implementation order

**Current Priorities (in order):**
1. **ADR-0002** (Wave 1): Fix thinking/reasoning — signature loss (multi-turn broken), Bedrock thinking, Anthropic adaptive mode, OpenAI dead config
2. **ADR-0003** (Wave 2): Unit test coverage — Bedrock parse (0 tests), SSE adapter (0 tests), OpenAI convert_messages
3. **ADR-0004** (Wave 3): Live integration tests — parameterized `tests/live_provider.rs`, Ollama + Bedrock P0
4. **ADR-0005** (Wave 4): Prompt caching support — `PromptCacheConfig`, `TokenUsage` cache fields, provider headers
5. **ADR-0006** (Wave 5): Multi-agent trace composition — `AgentTopology` enum, W3C context propagation, `graph.node.*` spans

**Completed:**
- OTel/Phoenix integration — OpenInference-compliant spans (AGENT/CHAIN/TOOL), 7 conformance tests
- Phoenix: `your-phoenix-host` (port 4317 OTLP, port 6006 UI)
- `PHOENIX_ENDPOINT=http://your-phoenix-host:4317`
- ADR-0001: Tool System & MCP Integration (Accepted)
- ADR-0002 through ADR-0006: All written and reviewed

## Gotchas

1. **Bedrock models need inference profiles** - Modern Claude models require `BEDROCK_INFERENCE_PROFILE` env var
2. **aws_smithy_types::Document::Null** - Unit variant, not `Null(true)`
3. **OpenAI o3/o3-mini don't support streaming** - Check `model.supports_streaming()`
4. **Temperature not allowed on reasoning models** - GPT-5, o1, o3 series
5. **Edition 2021** - Rust 2024 edition doesn't exist yet

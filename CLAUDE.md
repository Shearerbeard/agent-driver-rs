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
├── tool/                  # Tool system
│   ├── types.rs           # ToolSchema, ToolDefinition
│   ├── executor.rs        # Tool trait
│   └── registry.rs        # Dynamic tool registration
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

```bash
# Unit tests (no API calls)
cargo test --features bedrock

# Integration test with live API
PROVIDER=bedrock cargo run --bin chat

# Test specific provider
cargo test --features openai openai_
```

## Gotchas

1. **Bedrock models need inference profiles** - Modern Claude models require `BEDROCK_INFERENCE_PROFILE` env var
2. **aws_smithy_types::Document::Null** - Unit variant, not `Null(true)`
3. **OpenAI o3/o3-mini don't support streaming** - Check `model.supports_streaming()`
4. **Temperature not allowed on reasoning models** - GPT-5, o1, o3 series
5. **Edition 2021** - Rust 2024 edition doesn't exist yet

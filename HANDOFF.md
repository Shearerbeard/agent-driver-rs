# HANDOFF.md - Session Continuation Guide

## Current State (Last Updated: 2026-02-04)

### What Works
- **Bedrock Provider**: Fully functional with Claude Sonnet 4.5
- **Anthropic Provider**: Fully functional with SSE streaming
- **OpenAI Provider**: Fully functional with async-openai streaming (GPT-4o, GPT-4o-mini)
- **Ollama Provider**: Fully functional with ollama-rs streaming (local models)
- **OpenRouter Provider**: Fully functional with SSE streaming (any OpenRouter model)
- **Core Infrastructure**: Types, config, streaming, tools, session, task pool
- **CLI Chat**: Interactive chat with streaming output

### Active Configuration
```bash
# .env
PROVIDER=bedrock
BEDROCK_MODEL=claude-sonnet-4.5
BEDROCK_MAX_TOKENS=4096
BEDROCK_INFERENCE_PROFILE=us.anthropic.claude-sonnet-4-5-20250929-v1:0
```

### Test Commands
```bash
# Verify everything compiles
cargo check --features bedrock

# Run tests (61 passing)
cargo test --features bedrock

# Interactive test
echo "What is 2+2?" | cargo run --features bedrock --bin chat
```

## Next Task: MCP Tool Integration

### Overview
Connect the rmcp crate for Model Context Protocol server tools.
The `mcp` feature flag is already defined in Cargo.toml.

### Files to Modify
1. `src/tool/mcp.rs` - Implement MCP tool discovery and execution
2. `src/session.rs` - Add MCP tool registration to session

### Implementation Notes
- Use rmcp crate for MCP client functionality
- Tool discovery from MCP servers
- Execution via MCP protocol

## Recently Completed: OpenRouter Provider

The OpenRouter provider is fully functional with:
- SSE streaming via reqwest-eventsource
- OpenAI-compatible request format
- Tool/function calling support
- Provider preferences routing

### Testing OpenRouter
```bash
PROVIDER=openrouter OPENROUTER_API_KEY=sk-or-... cargo run --features openrouter --bin chat
```

## Recently Completed: OpenAI Provider

The OpenAI provider is fully functional with:
- Streaming via async-openai
- Tool/function calling support
- Temperature handling (disabled for reasoning models)
- Support for GPT-4o, GPT-4o-mini, o1, o1-mini

### Testing OpenAI
```bash
PROVIDER=openai OPENAI_API_KEY=sk-... cargo run --features openai --bin chat
```

## Architecture Reference

### Provider Trait
```rust
pub trait Provider: Send + Sync {
    fn info(&self) -> &ProviderInfo;

    fn complete_stream(
        &self,
        request: CompletionRequest,
        ctx: ProviderContext,
    ) -> Pin<Box<dyn Future<Output = Result<StreamHandle, ProviderError>> + Send + '_>>;

    fn list_models(&self, ctx: ProviderContext)
        -> Pin<Box<dyn Future<Output = Result<Vec<ModelInfo>, ProviderError>> + Send + '_>>;
}
```

### StreamEvent Flow
```
Started → ContentBlockStart → Delta* → ContentBlockStop → Completed
```

### Key Types
- `CompletionRequest`: model, system, messages, tools, config
- `ProviderContext`: correlation_id, cancellation, task_tracker
- `StreamHandle`: stream + cancellation + correlation_id
- `StreamDelta`: TextDelta, ThinkingDelta, ToolUseStart, ToolInputDelta

## Session Handoff Protocol

When continuing work:

1. Read this file and CLAUDE.md
2. Check TODO.md for current task
3. Run `cargo check --features <provider>` to verify state
4. Run `cargo test --features <provider>` to confirm tests pass
5. Continue with next task

## Recent Changes

### 2026-02-04 (Session 4)
- Implemented OpenRouter provider with reqwest-eventsource SSE
- OpenAI-compatible request/response format
- Tool calling support
- Provider preferences routing support
- Set up GitHub repo (Shearerbeard/agent-driver-rs)
- Established feature branch workflow

### 2026-02-04 (Session 3)
- Implemented Ollama provider with ollama-rs
- Full streaming support with num_ctx configuration
- Token usage reporting from final_data
- Tested with qwen3:14b locally

### 2026-02-04 (Session 2)
- Implemented OpenAI provider with async-openai
- Full streaming support with cancellation
- Tool calling support
- Temperature restrictions for reasoning models

### 2026-02-04 (Session 1)
- Fixed Bedrock `Document::Null` syntax (unit variant, not function)
- Added Claude 4.5 models (Sonnet, Opus, Haiku) to BedrockModel enum
- Added inference profile support for Bedrock
- Verified streaming works with Sonnet 4.5
- Created CLAUDE.md, TODO.md, HANDOFF.md

## Known Issues

1. **Bedrock models need inference profiles** - Set `BEDROCK_INFERENCE_PROFILE` env var
2. **Some OpenAI models don't stream** - o3, o3-mini require non-streaming path
3. **Temperature restrictions** - Reasoning models (GPT-5, o-series) don't accept temperature

## Contact

Project: agent-driver-rs
Primary Language: Rust 2021 Edition
Key Dependencies: tokio, aws-sdk-bedrockruntime, async-openai, reqwest-eventsource

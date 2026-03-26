# agent-driver-rs

A Rust library providing a unified abstraction over multiple LLM providers with streaming completions, tool calling, and agentic loops.

## Features

- **5-provider support**: Anthropic Claude, OpenAI (GPT-4o, o3), AWS Bedrock, OpenRouter, Ollama
- **Streaming-first**: All providers emit standard `StreamEvent` types
- **Tool calling**: Dynamic tool registration, JSON Schema input validation, native + MCP tools
- **Agent loop**: Typed infrastructure for multi-turn tool-calling conversations with observer pattern
- **Session management**: Mutable system prompts, message history with trimming, split-lock concurrency
- **Task tracking**: Cascading cancellation via `CancellationToken`, `TaskPool` with registration-before-execution
- **Feature-flagged providers**: Only compile what you need

## Quick Start

```bash
# Run with default provider (Bedrock Sonnet 4.5)
cargo run --features bedrock --bin chat

# Run tests
cargo test --all-features

# Check all features compile
cargo check --all-features
```

## Feature Flags

| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `anthropic` | Anthropic API (direct HTTP + SSE) | — |
| `openai` | OpenAI API | `async-openai` |
| `bedrock` | AWS Bedrock (converse_stream) | `aws-sdk-bedrockruntime`, `aws-config`, `aws-smithy-types` |
| `openrouter` | OpenRouter API | — |
| `ollama` | Ollama local models | `ollama-rs` |
| `mcp` | MCP tool integration | `rmcp` |
| `test-support` | Expose `MockProvider` for integration tests | — |

Default features: `anthropic`, `openai`, `openrouter`

## Phoenix/OpenTelemetry Integration

This library supports comprehensive OpenTelemetry integration for tracing provider calls, agent loops, and tool executions.

### Bedrock Provider Tests

Prerequisites:
- AWS credentials configured (via `aws sso login` or environment variables)
- Docker installed for OTel collector

```bash
# Start OTel Collector
docker compose -f docker-compose.phoenix.yaml up -d

# Run Bedrock test (requires AWS credentials)
cargo run --example phoenix_integration_bedrock --features "phoenix bedrock"
```

### Ollama Provider Tests

Prerequisites:
- Ollama server running (`ollama serve`)
- Docker for OTel collector

```bash
# Start OTel Collector
docker compose -f docker-compose.phoenix.yaml up -d

# Run Ollama test (requires Ollama server running)
cargo run --example phoenix_integration_ollama --features "phoenix ollama"
```

### All Tests

```bash
# Run all tests with OTel collector
./scripts/test-phoenix-integration.sh all
```

### Viewing OTel Spans

```bash
# View OTel Collector logs
docker compose -f docker-compose.phoenix.yaml logs -f otel-collector
```

## Architecture Overview

- **Provider layer** abstracts 5 backends behind a single `Provider` trait
- **Session** manages conversation state with split locks (system prompt, messages, tools)
- **Agent loop** drives the send -> tool_use -> execute -> continue cycle
- **Tool system** supports both native Rust tools and MCP server connections
- **Streaming infrastructure** converts provider-specific formats into standard `StreamEvent`s

See `docs/ARCHITECTURE.md` for details.

## Environment Configuration

| Variable | Provider | Description |
|----------|----------|-------------|
| `PROVIDER` | All | Provider to use: `anthropic`, `openai`, `bedrock`, `openrouter`, `ollama` |
| `ANTHROPIC_API_KEY` | Anthropic | API key |
| `OPENAI_API_KEY` | OpenAI | API key |
| `BEDROCK_INFERENCE_PROFILE` | Bedrock | Inference profile ARN |
| `OLLAMA_HOST` | Ollama | Host URL (default: `http://localhost:11434`) |

## Documentation

- `docs/ARCHITECTURE.md` — detailed architecture and data flow
- `docs/PROVIDERS.md` — provider comparison and configuration
- `docs/adr/` — architecture decision records
- `docs/manual-testing.md` — testing checklist

## License

See LICENSE

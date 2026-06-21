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

## Prerequisites

- **Rust 1.91.1+** (MSRV declared in `Cargo.toml`, pinned via `rust-toolchain.toml`)
- Edition 2024
- `make` (for the `make check` / `make deny` / `make unused-deps` gates)

## Quick Start

The default feature set compiles Anthropic, OpenAI, and OpenRouter. The
checked-in `.env.example` defaults to Anthropic, so the first live chat path is:

```bash
# 1. Configure a provider
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY

# 2. Run the chat client
cargo run --bin chat

# 3. Run tests
cargo test --all-features

# 4. Check all features compile
cargo check --all-features
```

For a local no-cloud-provider smoke test, use Ollama instead:

```bash
ollama serve
PROVIDER=ollama cargo run --no-default-features --features ollama --bin chat
```

Bedrock is supported but is not the default feature set. Modern Claude models
on Bedrock require AWS credentials and `BEDROCK_INFERENCE_PROFILE`; see
`.env.example` and `docs/manual-testing.md`.

## Run / Test Cheat Sheet

| Goal | Command | Requirements |
|------|---------|--------------|
| Format, lint, and test | `make check` | Rust toolchain |
| Compile all feature-gated code | `cargo check --all-features` | Rust toolchain |
| Run all tests | `cargo test --all-features` | Rust toolchain |
| Run Anthropic chat | `cargo run --bin chat` | `.env` with `PROVIDER=anthropic`, `ANTHROPIC_API_KEY` |
| Run local Ollama chat | `PROVIDER=ollama cargo run --no-default-features --features ollama --bin chat` | `ollama serve` |
| Run Bedrock chat | `PROVIDER=bedrock cargo run --no-default-features --features bedrock --bin chat` | AWS credentials, `BEDROCK_INFERENCE_PROFILE` |
| Live MCP tool smoke | `echo "List docs/adr" \| PROVIDER=bedrock cargo run --features "bedrock mcp" --bin chat -- --mcp "npx -y @modelcontextprotocol/server-filesystem $(pwd)"` | AWS credentials, Node/npm |
| Phoenix mock tracing | `cargo run --example phoenix_integration_mock --features phoenix` | No provider key |
| Benchmark smoke | `cd bench && OPENAI_API_KEY=test cargo run -- -n 1 -w 0 -s cold_start --json` | No live API call |

For the full checklist, including provider-specific live tests and known MCP
gotchas, see `docs/manual-testing.md`.

## Feature Flags

| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `anthropic` | Anthropic API (direct HTTP + SSE) | — |
| `openai` | OpenAI API | `async-openai` |
| `bedrock` | AWS Bedrock (converse_stream) | `aws-sdk-bedrockruntime`, `aws-config`, `aws-smithy-types` |
| `openrouter` | OpenRouter API | — |
| `ollama` | Ollama local models | `ollama-rs` |
| `mcp` | MCP tool integration | `rmcp` |
| `mcp-http` | MCP streamable HTTP transport | `mcp`, `rmcp/transport-streamable-http-client-reqwest` |
| `schema-sanitize` | OpenAI strict schema sanitization | `mcp-openai-bridge` |
| `phoenix` | OpenTelemetry/Phoenix tracing | `opentelemetry`, `opentelemetry_sdk`, `opentelemetry-otlp` |
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

Copy `.env.example` to `.env`. `.env.example` is the canonical config
reference and lists provider-specific models, token limits, temperature knobs,
reasoning settings, Bedrock inference profiles, and Ollama options.

At minimum set `PROVIDER` to one of `anthropic`, `openai`, `bedrock`,
`openrouter`, or `ollama`, then set the matching API key or local endpoint.

## Documentation

- `docs/ARCHITECTURE.md` — detailed architecture and data flow
- `docs/PROVIDERS.md` — provider comparison and configuration
- `docs/adr/` — architecture decision records
- `docs/manual-testing.md` — testing checklist

## Benchmarks

A standalone benchmark harness lives in `bench/`. It compares agent-driver-rs against rig-core:

```bash
cd bench && cargo run --release -- -n 20 -w 3 --json
```

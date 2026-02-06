# Manual Testing Guide

Run through this checklist before committing a feature. Every step must pass.

## 1. Automated Checks (required, every change)

```bash
# Compile check — the active provider feature
cargo check --features bedrock

# Compile check — all features together (catches conditional compilation issues)
cargo check --all-features

# Unit tests — active provider
cargo test --features bedrock

# Unit tests — all features
cargo test --all-features

# Clippy — zero warnings policy
cargo clippy --features bedrock -- -D warnings
```

If your change touches provider-agnostic code (`types/`, `agent/`, `tool/`,
`session.rs`, `streaming.rs`), also run:

```bash
cargo clippy --all-features -- -D warnings
```

## 2. Live Smoke Test — Basic Chat (required for provider changes)

Verify the default provider still works end-to-end with streaming.

```bash
echo "What is 2 + 2? Answer in one sentence." | \
    PROVIDER=bedrock cargo run --features bedrock --bin chat
```

Expected: Streamed text response, token usage line, clean exit. No errors.

If your change touches a specific provider, test that provider too:

```bash
# Anthropic
echo "What is 2 + 2?" | PROVIDER=anthropic cargo run --features anthropic --bin chat

# OpenRouter
echo "What is 2 + 2?" | PROVIDER=openrouter cargo run --features openrouter --bin chat

# Ollama (requires local server)
echo "What is 2 + 2?" | PROVIDER=ollama cargo run --features ollama --bin chat
```

## 3. Live Smoke Test — Tool Calling (required for agent/tool/provider changes)

Verify the agent loop can call tools and send results back to the provider.

### Single tool call

```bash
echo "List the contents of the docs/adr directory" | \
    PROVIDER=bedrock cargo run --features "bedrock mcp" --bin chat -- \
    --mcp "npx -y @modelcontextprotocol/server-filesystem $(pwd)"
```

Expected:
- `[calling tool: list_directory]` appears
- `[tool list_directory done: ...]` appears (green)
- Model responds with file listing text
- `[loop done: end_turn, 1 tool iteration(s)]`

### Multi-tool call (exercises message merging)

```bash
echo "Get the file info for $(pwd)/docs/adr/README.md and also list the $(pwd)/docs/adr directory. Do both at once." | \
    PROVIDER=bedrock cargo run --features "bedrock mcp" --bin chat -- \
    --mcp "npx -y @modelcontextprotocol/server-filesystem $(pwd)"
```

Expected:
- Two `[calling tool: ...]` lines in the same iteration
- Both tools complete successfully
- Model summarizes both results
- No Bedrock `ValidationException` errors

## 4. Debug Logging

When something fails and the error is unclear:

```bash
RUST_LOG=debug PROVIDER=bedrock cargo run --features bedrock --bin chat
```

## Summary Checklist

Copy this into your PR or commit notes:

```
- [ ] `cargo check --features bedrock`
- [ ] `cargo check --all-features`
- [ ] `cargo test --features bedrock`
- [ ] `cargo test --all-features`
- [ ] `cargo clippy --features bedrock -- -D warnings`
- [ ] Live: basic chat smoke test
- [ ] Live: tool calling smoke test (if agent/tool/provider changed)
```

## Known Gotchas

- **MCP package name**: Use `@modelcontextprotocol/server-filesystem`, not the
  old `@anthropic/mcp-filesystem-server` (removed from npm).
- **macOS `/tmp`**: Symlinks to `/private/tmp`. MCP filesystem server rejects
  `/tmp` if only `/private/tmp` is allowed. Use `$(pwd)` or absolute real paths.
- **MCP no-argument tools**: Tools that take no arguments (e.g.
  `list_allowed_directories`) may fail with `expected object` validation error.
  Pre-existing MCP/model issue — not a sign your change broke something.
- **Bedrock "service error" after MCP error results**: When a tool returns an
  MCP error containing JSON/special characters, the follow-up Bedrock request
  sometimes fails with a generic "service error". Pre-existing issue — test with
  tools that succeed to get a clean signal.

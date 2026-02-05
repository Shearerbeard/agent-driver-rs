# TODO - agent-driver-rs

## Next Up

- [ ] **OpenRouter Provider** - SSE streaming via reqwest-eventsource
  - API: `https://openrouter.ai/api/v1/chat/completions`
  - Auth: `Authorization: Bearer $OPENROUTER_API_KEY`
  - Format: OpenAI-compatible with SSE streaming
  - Config exists: `src/config/openrouter.rs`
  - Stub exists: `src/provider/openrouter.rs`

## High Priority

- [ ] **MCP Tool Integration** - Connect rmcp for MCP server tools

## Medium Priority

- [ ] **Vision Support** - Image content blocks for multimodal
- [ ] **Extended Thinking** - Claude's thinking blocks, signature handling
- [ ] **Structured Output** - JSON schema enforcement for OpenAI
- [ ] **Token Counting** - Estimate tokens before sending requests
- [ ] **Conversation Summarization** - Auto-summarize long conversations

## Low Priority / Future

- [ ] **Caching** - Response caching with cache control headers
- [ ] **Prompt Templates** - Reusable prompt components
- [ ] **Cost Tracking** - Track API costs per session
- [ ] **Metrics/Telemetry** - OpenTelemetry integration
- [ ] **WebSocket Transport** - Alternative to SSE for some providers

## Technical Debt

- [ ] Add integration tests with mocked HTTP responses
- [ ] Improve error messages with actionable suggestions
- [ ] Add tracing spans for debugging
- [ ] Document public API with rustdoc examples
- [ ] Benchmark streaming performance

## Completed

- [x] Core types with validation (ModelId, ToolName, etc.)
- [x] Error types (AgentDriverError hierarchy)
- [x] Configuration system with env loading
- [x] Task tracking with cancellation (TaskPool)
- [x] Streaming types (StreamEvent, StreamDelta, StreamHandle)
- [x] Tool system (ToolDefinition, ToolRegistry)
- [x] Provider trait with boxed futures
- [x] Anthropic provider (HTTP + SSE)
- [x] Bedrock provider (AWS SDK converse_stream)
- [x] **OpenAI provider** (async-openai with streaming)
- [x] **Ollama provider** (ollama-rs with streaming, num_ctx support)
- [x] Session with split locks
- [x] CLI chat client
- [x] Bedrock inference profile support

## Notes

### Provider Status
| Provider | Status | Notes |
|----------|--------|-------|
| Anthropic | ✅ Working | Direct HTTP + SSE |
| Bedrock | ✅ Working | Requires inference profile for new models |
| OpenAI | ✅ Working | async-openai with streaming |
| Ollama | ✅ Working | ollama-rs with streaming, num_ctx support |
| OpenRouter | 🚧 In Progress | SSE similar to Anthropic |

### Testing Commands
```bash
# Quick check
cargo check --all-features

# Run all tests
cargo test --features bedrock

# Test with live API
echo "Hello" | cargo run --features bedrock --bin chat
```

# Phoenix/OpenTelemetry Integration - Remaining Tasks

## Current State ✅

All examples compile successfully:
- `phoenix_integration_mock.rs` - Mock provider tests (passes ✅)
- `phoenix_integration_bedrock.rs` - Bedrock provider (compiles ✅, requires AWS credentials)
- `phoenix_integration_ollama.rs` - Ollama provider (compiles ✅, requires Ollama running)

## Completed ✅

- [x] Add opentelemetry_sdk dependency to Cargo.toml
- [x] Fix src/otel/mod.rs imports for concrete types
- [x] Fix src/session.rs imports and add otel_tracer getter
- [x] Fix src/agent/driver.rs type mismatches
- [x] Fix compilation warnings
- [x] Create docker-compose.phoenix.yaml
- [x] Create otel-collector-config.yaml  
- [x] Create phoenix_integration_mock.rs with 7 test scenarios (mock passes ✅)
- [x] Create test runner script

## In Progress 🚧

### Priority 2: Full End-to-End OTLP Export

Currently using `TracerProvider::default()` which doesn't export anywhere.

**Need to implement:**
1. Real OTLP exporter in examples
2. Connect to OTel Collector running in Docker
3. Verify spans appear in collector logs

### Priority 3: Span Attribute Population

The instrumentation.rs stubs don't actually create spans with attributes. Currently just stubs.

**Need to implement in src/otel/instrumentation.rs:**
1. `AgentLoopSpan` - creates proper spans with attributes:
   - session.id
   - model
   - provider  
   - iteration count
   - stop_reason

2. `ToolSpan` - creates tool execution spans with:
   - tool_name
   - success/failure

3. `CompletionSpan` - wraps provider calls with:
   - llm.model
   - llm.provider
   - token counts (if available)

### Priority 4: Session/Provider Integration

Need to ensure spans are actually created during execution:
- `session.send_streaming()` should create session.operation span
- `AgentLoop::run()` should create agent_loop span with children
- Provider calls should have completion spans

### Priority 5: Documentation

- [ ] Update docs/manual-testing.md with Phoenix testing
- [ ] Add PHOENIX.md integration guide

### Priority 2: Full End-to-End OTLP Export

Currently using `TracerProvider::default()` which doesn't export anywhere.

**Need to implement:**
1. Real OTLP exporter in examples
2. Connect to OTel Collector running in Docker
3. Verify spans appear in collector logs

### Priority 3: Span Attribute Population

The instrumentation.rs stubs don't actually create spans with attributes. Currently just stubs.

**Need to implement in src/otel/instrumentation.rs:**
1. `AgentLoopSpan` - creates proper spans with attributes:
   - session.id
   - model
   - provider  
   - iteration count
   - stop_reason

2. `ToolSpan` - creates tool execution spans with:
   - tool_name
   - success/failure

3. `CompletionSpan` - wraps provider calls with:
   - llm.model
   - llm.provider
   - token counts (if available)

### Priority 4: Session/Provider Integration

Need to ensure spans are actually created during execution:
- `session.send_streaming()` should create session.operation span
- `AgentLoop::run()` should create agent_loop span with children
- Provider calls should have completion spans

### Priority 5: Documentation

- [ ] Update docs/manual-testing.md with Phoenix testing
- [ ] Add PHOENIX.md integration guide

---

## Quick Start (Current State)

```bash
# Run mock tests (works now)
cargo run --example phoenix_integration_mock --features phoenix

# Run Bedrock tests (requires AWS credentials)
cargo run --example phoenix_integration_bedrock --features "phoenix bedrock"

# Run Ollama tests (requires Ollama running)
cargo run --example phoenix_integration_ollama --features "phoenix ollama"

# Start OTel Collector
docker compose -f docker-compose.phoenix.yaml up -d

# View logs
docker compose -f docker-compose.phoenix.yaml logs -f otel-collector

# Check spans arrive
docker compose -f docker-compose.phoenix.yaml logs otel-collector | grep -i span
```

---

## Dependencies

The LSP shows phantom errors in src/otel/instrumentation.rs that don't affect compilation:
- These appear to be stale LSP diagnostics
- `cargo check --features phoenix` passes clean
- Can ignore these LSP errors

---

## Notes

- Bedrock: Need to use `aws sso login` for credentials
- Ollama: Need qwen3:30b-a3b-thinking-2507-q4_K_M model installed
- Current tests use MockProvider which validates the tracing infrastructure works
- Real provider tests validate the full integration with actual spans

## ⚠️ LSP Errors Are Phantom

The LSP shows errors in src/otel/instrumentation.rs but these are STALE - compilation passes clean:
```
$ cargo check --features phoenix
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.32s
```

Ignore the LSP diagnostics - they're from a broken static analysis pass and don't affect the build.

# Research: Multi-Agent Trace Composition

**Date:** 2026-05-05
**Related ADR:** [ADR-0006](../adr/0006-multi-agent-trace-composition.md)
**Status:** Research complete

## Sources

### OpenInference Specification

- **Semantic Conventions**: https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md
  - Defines `openinference.span.kind` values: `LLM`, `EMBEDDING`, `CHAIN`, `RETRIEVER`, `RERANKER`, `TOOL`, `AGENT`, `GUARDRAIL`, `EVALUATOR`, `PROMPT`
  - Defines `agent.name` attribute
  - Defines `graph.node.id`, `graph.node.name`, `graph.node.parent_id` for execution graph visualization
  - `graph.node.parent_id` empty string = root node

- **Traces Concept**: https://mintlify.com/Arize-ai/openinference/concepts/traces
  - Traces are trees of spans connected by parent-child relationships
  - All spans share the same `trace_id`
  - Root span has no `parent_id`
  - Context propagation: `trace_id` propagated to children, parent's `span_id` becomes child's `parent_id`

- **Spans Concept**: https://www.mintlify.com/Arize-ai/openinference/concepts/spans
  - Span fields: name, parent_span_id, start/end timestamps, span context, attributes, events, status, kind
  - `openinference.span.kind` is REQUIRED for all spans

- **Data Model**: https://mintlify.com/Arize-ai/openinference/spec/data-model
  - Hierarchical data model for AI application execution as distributed traces

### OpenInference GitHub Issues (Known Bugs & Patterns)

- **Issue #1748**: feat: Add semantic conventions for agent and graph attributes
  - Added `agent.name`, `graph.node.id`, `graph.node.name`, `graph.node.parent_id`
  - Changed from `next_node`/`next_agent` to `parent_id` pattern (more aligned with OTEL convention)
  - PR merged: https://github.com/Arize-ai/openinference/commit/5f90a8014c216a821c299d0c7ea7aa79a4fc738d

- **Issue #2917**: Teams inside Workflows create separate root traces
  - Bug: `_get_team_span_context` checks for `_AGNO_PARENT_NODE_CONTEXT_KEY` to decide if Team is top-level
  - When Team runs inside Workflow, Workflow doesn't set the key → Team forces new root trace
  - Workaround: check `trace_api.get_current_span()` before forcing `INVALID_SPAN`
  - Root cause: incorrect attach/detach ordering in async generator boundaries
  - PR #2681 addresses this

- **Issue #5573**: Agno OpenInference creates nested traces instead of separate traces for subsequent Team runs
  - `parent_run_id` field used internally by Agno for session tracking
  - OpenInference interprets these as span parent-child relationships
  - Fixed in `openinference-instrumentation-agno: v0.1.25` (PR #2533)

- **Issue #2640**: Remote team spans no longer being nested properly
  - Version 0.1.25 accidentally broke remote agent nesting
  - Remote agents show as independent spans, not linked to original team
  - Fix requires explicit parent-child relationship model (not relying on context propagation alone)
  - PR #2935 addresses this

- **Issue #2748**: SmolAgents not trace correlation
  - Python generators don't preserve OpenTelemetry context across `yield` statements
  - Fixed in v0.1.17 with `wrapped_generator()` that reattaches tracing context
  - ThreadPoolExecutor in v1.24.0 broke trace context for parallel tool calls

- **Issue #2275**: LangChain instrumentation not correctly nesting httpx spans
  - `with_structured_output` loses `parent_run_id`
  - Known limitation: manual parent spans don't compose properly with LangChain's internal context management

- **Issue #2190**: Langchain/LangGraph async tool calls using context argument
  - `AsyncioInstrumentor` required for context propagation in async Python code
  - Manual parent spans often show empty because LangChain's auto-instrumentation doesn't reliably nest under them

- **Issue #1604**: Identify top-level agent spans
  - No standard way to identify the start of spans issued by the same agent
  - Suggestion: add `agent_id` metadata attribute for filtering

### OpenInference PRs

- **PR #2053**: feat(Agno): Capture graph attributes for agent visual
  - Captures `graph.node.id`, `graph.node.name`, `graph.node.parent_id` for Agno agent team visualization
  - Supports coordinate, collaborate, and route modes

- **PR #2090**: feat(Agno): Capture agent graph attributes (merged)
  - Redo of PR #2053, merged Aug 2025

- **Issue #2564**: feat: OpenAI agents sdk instrumentation
  - Maps OpenAI Agents SDK traces/spans to OpenTelemetry using OpenInference semantics
  - Includes handoff graph with `graph.node.id/parent_id`

### OpenTelemetry Rust

- **Context**: https://docs.rs/opentelemetry/latest/opentelemetry/context/struct.Context.html
  - Execution-scoped collection of values
  - Immutable, write operations create new context
  - `attach()` / `detach()` for thread-local management
  - `with_value()` for storing application-specific types

- **Tracer**: https://docs.rs/opentelemetry/latest/opentelemetry/trace/trait.Tracer.html
  - `start_with_context()` for creating child spans
  - `FutureExt::with_context()` for async context propagation
  - Active span becomes parent of new spans

- **SpanKind**: https://docs.rs/opentelemetry/latest/opentelemetry/trace/enum.SpanKind.html
  - `Client`, `Server`, `Producer`, `Consumer`, `Internal`
  - OpenInference uses `Internal` for most AI spans

- **Async context propagation**: https://docs.rs/otel/latest/otel/
  - OpenTelemetry context stored in thread-local storage
  - Async tasks migrate between threads at `.await` points
  - Must use `FutureExt::with_context` to propagate across async boundaries

- **PR #2378**: Allow overlapping context scopes
  - Changes single `_current` Context to a stack resilient to out-of-order deactivation
  - Needed for interop with tokio-rs/tracing

### Blog Posts

- **Contextful Context**: https://blog.scottgerring.com/posts/contextful-context/
  - Explains how OpenTelemetry context works in Rust async
  - Context stored in thread-local storage, cloned via Arc
  - Async code yields and may resume on different thread
  - `FutureExt` wrapper re-attaches correct context when future is polled

## Key Findings

### 1. Graph Attributes Are the Standard for Multi-Agent

OpenInference has standardized on `graph.node.*` attributes for multi-agent visualization:
- `graph.node.id` — unique identifier for the node
- `graph.node.name` — human-readable name
- `graph.node.parent_id` — references parent node ID (empty = root)

This was added in mid-2025 and is actively used by Agno, OpenAI Agents SDK, and other frameworks.

### 2. Context Propagation Is the #1 Source of Bugs

Every major Python framework has hit trace-splitting bugs:
- **Agno**: Teams inside Workflows create separate root traces (#2917)
- **SmolAgents**: Generator yields lose context (#2748)
- **LangGraph**: Manual parent spans don't nest with auto-instrumentation (#2190)
- **LangChain**: `with_structured_output` loses `parent_run_id` (#2275)

The common pattern: **implicit context propagation is fragile**. Explicit context extraction/injection is more reliable.

### 3. Explicit Parent-Child > Implicit Context

Multiple issues point to the same conclusion: relying solely on OpenTelemetry's thread-local context is insufficient for multi-agent compositions. The fix is always to use an explicit parent-child relationship model (propagating parent IDs) in addition to context propagation.

This directly informs our decision to use `AgentRole::Worker { parent: GraphNodeId }` — the parent is encoded in the type, not inferred from runtime context.

### 4. Rust Has Advantages Over Python for This

Python's issues stem from:
- Generators don't preserve context across `yield`
- ThreadPoolExecutor creates new threads without context
- AsyncioInstrumentor required but often forgotten

Rust avoids these because:
- `FutureExt::with_context` explicitly attaches context to futures
- Tokio's task spawning can carry context explicitly
- Type system enforces parent-child relationships at compile time

### 5. W3C Trace Context Is the Cross-Process Standard

OpenTelemetry natively supports W3C Trace Context propagation:
- `traceparent` header: `version-traceid-parentid-flags`
- `tracestate` header: vendor-specific key-value pairs
- Works across HTTP, gRPC, and any transport that can carry headers

This is the mechanism for propagating trace context to remote workers.

## Design Implications

1. **Type-level topology**: `AgentRole::Worker { parent }` prevents orphan spans at compile time
2. **Explicit context extraction**: `AgentLoopSpan::extract_trace_context()` for cross-process delegation
3. **Graph attributes on every agent span**: `graph.node.*` for Phoenix visualization
4. **W3C Trace Context for transport**: Standard headers for HTTP/MCP propagation
5. **RAII span guards**: Existing pattern extends naturally to multi-agent

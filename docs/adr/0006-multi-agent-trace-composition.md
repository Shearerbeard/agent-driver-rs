# ADR-0006: Multi-Agent Trace Composition with OpenInference

**Status:** Proposed
**Date:** 2026-05-05
**Context tags:** [otel] [phoenix] [multi-agent] [tracing] [architecture]

## Context

agent-driver-rs currently supports single-agent tracing via `AgentLoopSpan`, `ToolSpan`, and `SessionOperationSpan` in `src/otel/instrumentation.rs`. These span types work well for a single `AgentLoop` driving tool calls against one `Session`.

However, multi-agent compositions are a natural extension: a **coordinator** agent may delegate sub-tasks to multiple **worker** agents, each with their own `AgentLoop`, `Session`, and tool set. This raises several tracing questions:

1. **Parent-child span relationships**: How do worker agent spans nest under a coordinator span within a single trace?
2. **Cross-process propagation**: When a worker agent runs in a separate process (e.g., a remote MCP server or a microservice), how does the trace context propagate so that all spans appear in the same trace in Phoenix?
3. **Graph-level attribution**: OpenInference defines `graph.node.id`, `graph.node.name`, and `graph.node.parent_id` attributes for visualizing agent execution graphs. How do we map our Rust types to these?
4. **Making illegal states unrepresentable**: How do we use Rust's type system to prevent invalid trace compositions (e.g., a worker span without a parent, a coordinator without a trace root, mismatched trace IDs)?

### Forces

- **OpenTelemetry context propagation** relies on thread-local state in sync code and `FutureExt::with_context` in async code. Across process boundaries, context must be serialized into transport headers (W3C Trace Context).
- **OpenInference semantic conventions** define `agent.name`, `graph.node.id`, `graph.node.name`, and `graph.node.parent_id` for multi-agent graph visualization in Phoenix.
- **Real-world pain points**: Python frameworks (Agno, LangGraph, SmolAgents) have all hit bugs where multi-agent traces split into separate root traces due to context propagation failures across async boundaries or process boundaries. See [research doc](../internal/multi-agent-trace-research.md) for detailed issue analysis.
- **Our design principles**: newtypes with validated constructors, enums over magic strings, `#[non_exhaustive]` for forward compatibility, RAII guards for span lifecycle.

## Decision

### 1. New Types for Multi-Agent Topology

Introduce newtypes and an enum that encode the agent topology at the type level:

```rust
/// Error type for `GraphNodeId` validation failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphNodeIdError {
    Empty,
    InvalidCharacters,
}

/// Identifies a node in the agent execution graph.
/// Backed by a validated string (alphanumeric + underscore + hyphen).
/// Inner field is private — no arbitrary construction.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct GraphNodeId(String);

impl GraphNodeId {
    /// Create a validated node ID. Rejects empty strings and invalid characters.
    #[must_use = "this returns a Result that should be checked"]
    pub fn new(id: impl Into<String>) -> Result<Self, GraphNodeIdError>;

    pub fn as_str(&self) -> &str;
}

impl TryFrom<String> for GraphNodeId {
    type Error = GraphNodeIdError;
    // ...
}

impl std::fmt::Display for GraphNodeId {
    // ...
}
```

```rust
/// Error type for `AgentNodeName` validation failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentNodeNameError {
    Empty,
    TooLong,
    InvalidCharacters,
}

/// A validated, human-readable name for an agent node.
/// Used for Phoenix UI display (graph.node.name).
/// Inner field is private — no arbitrary construction.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct AgentNodeName(String);

impl AgentNodeName {
    /// Create a validated name. Rejects empty, overly long, or invalid strings.
    #[must_use = "this returns a Result that should be checked"]
    pub fn new(name: impl Into<String>) -> Result<Self, AgentNodeNameError>;

    pub fn as_str(&self) -> &str;
}

impl TryFrom<String> for AgentNodeName {
    type Error = AgentNodeNameError;
    // ...
}

impl std::fmt::Display for AgentNodeName {
    // ...
}
```

```rust
/// Error type for illegal agent topology configurations.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TopologyError {
    /// Worker cannot be its own parent.
    SelfParenting { node_id: GraphNodeId },
    /// Parent node ID does not exist in the graph.
    ParentNotFound { parent_id: GraphNodeId },
    /// Node name validation failed.
    InvalidName(AgentNodeNameError),
    /// Node ID validation failed.
    InvalidId(GraphNodeIdError),
    /// Expected a Coordinator topology but received a Worker.
    ExpectedCoordinator,
    /// Expected a Worker topology but received a Coordinator.
    ExpectedWorker,
}

/// Encodes the full trace topology for a multi-agent composition.
///
/// This is an enum, not a struct, so the relationship between role and
/// parent is structurally enforced. A Coordinator has no parent; a Worker
/// must have one. The fields are private — topology is immutable after
/// construction.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum AgentTopology {
    /// Top-level coordinator — no parent agent.
    Coordinator {
        node_id: GraphNodeId,
        node_name: AgentNodeName,
    },
    /// Worker agent delegated by a coordinator or another worker.
    Worker {
        node_id: GraphNodeId,
        node_name: AgentNodeName,
        parent: GraphNodeId,
    },
}

impl AgentTopology {
    /// Create a coordinator topology.
    #[must_use = "this returns a Result that should be checked"]
    pub fn coordinator(
        node_id: GraphNodeId,
        node_name: AgentNodeName,
    ) -> Result<Self, TopologyError>;

    /// Create a worker topology.
    /// Fails if `parent == node_id` (self-parenting).
    #[must_use = "this returns a Result that should be checked"]
    pub fn worker(
        node_id: GraphNodeId,
        node_name: AgentNodeName,
        parent: GraphNodeId,
    ) -> Result<Self, TopologyError>;

    pub fn node_id(&self) -> &GraphNodeId;
    pub fn node_name(&self) -> &AgentNodeName;
    pub fn parent_id(&self) -> Option<&GraphNodeId>;
    pub fn is_coordinator(&self) -> bool;
}
```

#### Why this makes illegal states unrepresentable

- **`AgentTopology` is an enum**, not a struct — you cannot have a `Worker` without a `parent` field, and you cannot have a `Coordinator` with a parent. The role and parent are structurally coupled.
- **`GraphNodeId::new()` validates** the string format at construction time — no invalid IDs can exist in the type.
- **`AgentNodeName::new()` validates** at construction time — no empty or invalid names.
- **`AgentTopology::worker()` rejects self-parenting** at construction time (`node_id == parent` → `TopologyError::SelfParenting`).
- **All inner fields are private** — callers cannot mutate a `Coordinator` into a `Worker` post-construction, or swap a valid `node_id` for an invalid string.
- **`SpanKind::Agent` is already an enum**; `AgentTopology` adds the structural relationship layer on top.

### 2. Span Hierarchy for Multi-Agent Composition

The span tree for a coordinator delegating to two workers:

```
Trace ID: abc-123 (shared across all spans)
│
├─ [CHAIN] "session.send"                          (SessionOperationSpan)
│  └─ [LLM] "completion"                           (implicit, via provider)
│
├─ [AGENT] "agent_loop:coordinator"                (AgentLoopSpan::new_coordinator)
│  ├─ graph.node.id: "coordinator"
│  ├─ graph.node.name: "Research Coordinator"
│  ├─ graph.node.parent_id: ""  (empty = root)
│  │
│  ├─ [TOOL] "tool.delegate_to_worker_a"           (ToolSpan)
│  │  └─ [AGENT] "agent_loop:worker_a"             (AgentLoopSpan::new_worker)
│  │     ├─ graph.node.id: "worker_a"
│  │     ├─ graph.node.name: "Web Search Agent"
│  │     ├─ graph.node.parent_id: "coordinator"
│  │     ├─ agent.name: "worker_a"
│  │     │
│  │     ├─ [LLM] "completion"                     (worker's LLM call)
│  │     └─ [TOOL] "tool.web_search"               (worker's tool call)
│  │
│  └─ [TOOL] "tool.delegate_to_worker_b"           (ToolSpan)
│     └─ [AGENT] "agent_loop:worker_b"             (AgentLoopSpan::new_worker)
│        ├─ graph.node.id: "worker_b"
│        ├─ graph.node.name: "Summarization Agent"
│        ├─ graph.node.parent_id: "coordinator"
│        ├─ agent.name: "worker_b"
│        │
│        ├─ [LLM] "completion"                     (worker's LLM call)
│        └─ [TOOL] "tool.summarize"                (worker's tool call)
│
└─ [CHAIN] "session.continue"                      (SessionOperationSpan)
```

Key rules:
- All spans share the **same `trace_id`** (OpenTelemetry context propagation).
- Each agent span carries `graph.node.id`, `graph.node.name`, and `graph.node.parent_id`.
- The coordinator's `graph.node.parent_id` is empty string (root node per OpenInference spec).
- Worker agent spans are children of the **delegating tool span**, which is itself a child of the coordinator agent span.

### 3. Cross-Process Propagation Strategy

When a worker agent runs in a separate process (remote MCP server, microservice, or spawned subprocess), the trace context must be serialized and transmitted. We use the **W3C Trace Context** standard, which OpenTelemetry natively supports.

#### 3.1 Context Serialization Types

```rust
/// Error type for traceparent validation failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceParentError {
    InvalidFormat,
    InvalidVersion,
    InvalidTraceId,
    InvalidParentId,
}

/// W3C traceparent: version-traceid-parentid-flags
/// Inner field is private — only constructible via `from_header`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct TraceParent(String);

impl TraceParent {
    /// Parse from a W3C traceparent header string.
    /// This is the ONLY constructor.
    #[must_use = "this returns a Result that should be checked"]
    pub fn from_header(value: &str) -> Result<Self, TraceParentError>;

    pub fn to_header(&self) -> &str;
}

impl std::fmt::Display for TraceParent {
    // ...
}
```

```rust
/// W3C tracestate header value.
/// Inner field is private.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct TraceState(String);

impl TraceState {
    #[must_use = "this returns a Result that should be checked"]
    pub fn from_header(value: &str) -> Result<Self, TraceStateError>;
    pub fn to_header(&self) -> &str;
}
```

```rust
/// W3C baggage header value.
/// Inner field is private.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct Baggage(String);

impl Baggage {
    #[must_use = "this returns a Result that should be checked"]
    pub fn from_header(value: &str) -> Result<Self, BaggageError>;
    pub fn to_header(&self) -> &str;
}
```

```rust
/// Serialized trace context for cross-process propagation.
///
/// Fields are private — this struct can only be constructed via the
/// `propagation` module using `TextMapPropagator`.
#[derive(Debug, Clone)]
pub struct TraceContextHeaders {
    traceparent: TraceParent,
    tracestate: Option<TraceState>,
    baggage: Option<Baggage>,
}

impl TraceContextHeaders {
    pub fn traceparent(&self) -> &TraceParent;
    pub fn tracestate(&self) -> Option<&TraceState>;
    pub fn baggage(&self) -> Option<&Baggage>;
}
```

**Implementation note**: We leverage OpenTelemetry's `TextMapPropagator` trait (with `HeaderExtractor` and `HeaderInjector`) to populate and extract `TraceContextHeaders`. This avoids manual parsing of W3C headers and ensures compatibility with any future changes to the OpenTelemetry propagation format.

```rust
pub mod propagation {
    use opentelemetry::propagation::{Injector, TextMapPropagator};

    /// Extract trace context from a carrier (e.g., HTTP headers, MCP metadata).
    /// This is the only way to construct `TraceContextHeaders`.
    pub fn extract_context(carrier: &impl HeaderExtractor) -> TraceContextHeaders;

    /// Inject trace context into a carrier.
    pub fn inject_context(context: &TraceContextHeaders, carrier: &mut impl Injector);
}
```

#### 3.2 Propagation Flow

```
Coordinator Process                          Worker Process
┌──────────────────────┐                    ┌──────────────────────┐
│ AgentLoopSpan        │                    │                      │
│   (active span)      │                    │                      │
│        │             │                    │                      │
│        ▼             │                    │                      │
│ Extract context      │  HTTP/MCP headers  │  Inject context      │
│ → TraceContextHeaders│ ──────────────────>│ → restore to Context │
│                      │   traceparent      │                      │
│                      │   tracestate       │                      │
│                      │                    │        │             │
│                      │                    │        ▼             │
│                      │                    │ Create worker span   │
│                      │                    │ with parent context  │
│                      │                    │ → same trace_id      │
└──────────────────────┘                    └──────────────────────┘
```

#### 3.3 API for Cross-Process Delegation

```rust
impl AgentLoopSpan {
    /// Extract the current trace context for cross-process propagation.
    /// Returns headers that should be sent to the worker process.
    pub fn extract_trace_context(&self) -> TraceContextHeaders;
}
```

On the **worker side**, the remote process receives `TraceContextHeaders`, restores the OTel `Context` via `propagation::extract_context`, and then creates a worker span with the **mandatory** parent context:

```rust
// Worker process: restore context and create span
let parent_cx = propagation::extract_context(&incoming_headers);
let topology = AgentTopology::worker(
    GraphNodeId::new("worker_a")?,
    AgentNodeName::new("Web Search Agent")?,
    GraphNodeId::new("coordinator")?,
)?;

let worker_span = AgentLoopSpan::new_worker(
    &tracer,
    "agent_loop:worker_a",
    &topology,
    &parent_cx,  // REQUIRED — not Option
)?;
```

#### 3.4 MCP Transport Integration

For MCP servers, trace context propagates via JSON-RPC metadata:

```rust
/// MCP request envelope with trace context injection.
/// Fields are private — construct via `McpRequestWithTrace::new(...)`.
pub struct McpRequestWithTrace<T> {
    jsonrpc: McpJsonRpcVersion,
    id: RequestId,
    method: McpMethod,
    params: T,
    /// W3C trace context — injected by the coordinator, extracted by the worker.
    #[serde(skip_serializing_if = "Option::is_none")]
    trace_context: Option<TraceContextHeaders>,
}

impl<T> McpRequestWithTrace<T> {
    pub fn new(
        id: RequestId,
        method: McpMethod,
        params: T,
        trace_context: Option<TraceContextHeaders>,
    ) -> Self;

    pub fn trace_context(&self) -> Option<&TraceContextHeaders>;
}
```

#### 3.5 HTTP Transport Integration

For HTTP-based worker delegation:

```rust
/// Standard W3C Trace Context header names.
pub mod trace_headers {
    pub const TRACEPARENT: &str = "traceparent";
    pub const TRACESTATE: &str = "tracestate";
    pub const BAGGAGE: &str = "baggage";
}

/// Inject trace context into an HTTP request builder.
/// Uses TextMapPropagator for serialization.
pub fn inject_trace_context<T>(
    builder: http::request::Builder,
    context: &TraceContextHeaders,
) -> http::request::Builder;
```

### 4. OpenInference Attribute Constants

Add the graph attributes to the existing `attr` module:

```rust
pub mod attr {
    // ... existing constants ...

    // Agent graph attributes (OpenInference semantic conventions)
    pub const AGENT_NAME: &str = "agent.name";
    pub const GRAPH_NODE_ID: &str = "graph.node.id";
    pub const GRAPH_NODE_NAME: &str = "graph.node.name";
    pub const GRAPH_NODE_PARENT_ID: &str = "graph.node.parent_id";
}
```

### 5. Span Creation API

Split constructors by role so that a `Worker` **cannot** be created without a parent OTel context. This prevents the trace-splitting bugs seen in Python frameworks.

```rust
/// Error type for span creation failures.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SpanCreationError {
    /// The topology was invalid (e.g., self-parenting).
    InvalidTopology(TopologyError),
    /// The OTel tracer was not available.
    TracerNotFound,
    /// Failed to build the span.
    BuildError(String),
}

impl AgentLoopSpan {
    /// Create a coordinator (root) agent span.
    ///
    /// No parent context is required — this span becomes the trace root.
    /// `graph.node.parent_id` is set to `""` (root sentinel per OpenInference).
    #[must_use = "this returns a Result that should be checked"]
    pub fn new_coordinator(
        tracer: &Tracer,
        name: &str,
        topology: &AgentTopology,
    ) -> Result<Self, SpanCreationError> {
        // Runtime assert: topology must be Coordinator
        let (node_id, node_name) = match topology {
            AgentTopology::Coordinator { node_id, node_name } => (node_id, node_name),
            AgentTopology::Worker { .. } => {
                return Err(SpanCreationError::InvalidTopology(
                    TopologyError::ExpectedCoordinator,
                ));
            }
        };

        let span = tracer
            .span_builder(name.to_string())
            .with_kind(OtelSpanKind::Internal)
            .with_attributes(vec![
                KeyValue::new(attr::SPAN_KIND, SpanKind::Agent.as_str()),
                KeyValue::new(attr::AGENT_NAME, node_name.as_str()),
                KeyValue::new(attr::GRAPH_NODE_ID, node_id.as_str()),
                KeyValue::new(attr::GRAPH_NODE_NAME, node_name.as_str()),
                KeyValue::new(attr::GRAPH_NODE_PARENT_ID, ""),
            ])
            .start(tracer);

        Ok(Self {
            context: Context::current_with_span(span),
            tracer: tracer.clone(),
        })
    }

    /// Create a worker agent span with a mandatory parent context.
    ///
    /// `parent_context` is **required** (not `Option`) — a worker without a parent
    /// context would become an orphaned root span, which is the exact bug this
    /// design prevents.
    ///
    /// `graph.node.parent_id` is set from `topology.parent()`.
    #[must_use = "this returns a Result that should be checked"]
    pub fn new_worker(
        tracer: &Tracer,
        name: &str,
        topology: &AgentTopology,
        parent_context: &Context,
    ) -> Result<Self, SpanCreationError> {
        let (node_id, node_name, parent) = match topology {
            AgentTopology::Worker {
                node_id,
                node_name,
                parent,
            } => (node_id, node_name, parent),
            AgentTopology::Coordinator { .. } => {
                return Err(SpanCreationError::InvalidTopology(
                    TopologyError::ExpectedWorker,
                ));
            }
        };

        let span = tracer
            .span_builder(name.to_string())
            .with_kind(OtelSpanKind::Internal)
            .with_attributes(vec![
                KeyValue::new(attr::SPAN_KIND, SpanKind::Agent.as_str()),
                KeyValue::new(attr::AGENT_NAME, node_name.as_str()),
                KeyValue::new(attr::GRAPH_NODE_ID, node_id.as_str()),
                KeyValue::new(attr::GRAPH_NODE_NAME, node_name.as_str()),
                KeyValue::new(attr::GRAPH_NODE_PARENT_ID, parent.as_str().to_string()),
            ])
            .start_with_context(tracer, parent_context);

        Ok(Self {
            context: parent_context.with_span(span),
            tracer: tracer.clone(),
        })
    }
}
```

## Consequences

### What becomes easier

- **Phoenix visualization**: With `graph.node.*` attributes, Phoenix can render the agent execution graph (coordinate, collaborate, route modes) correctly.
- **Debugging multi-agent flows**: A single trace shows the full delegation chain — coordinator → worker A → worker B — with clear parent-child relationships.
- **Cross-process debugging**: W3C Trace Context propagation (via `TextMapPropagator`) ensures that remote workers appear in the same trace, not as orphaned root spans (a common bug in Python frameworks).
- **Type safety**: The type system prevents constructing orphan worker spans, invalid node IDs, or topology configurations without required fields.
- **Forward compatibility**: `#[non_exhaustive]` on `AgentTopology` allows adding new variants (e.g., `Supervisor`, `Reviewer`) without breaking existing code.
- **Metadata propagation**: Inclusion of W3C `baggage` headers allows propagating application-specific metadata (session IDs, user IDs) across process boundaries alongside trace context.

### What becomes harder

- **API surface grows**: Users must now understand `AgentTopology` (enum), `GraphNodeId`, `AgentNodeName`, and `TraceContextHeaders` when doing multi-agent work. The single-agent path (`AgentLoopSpan::new`) remains unchanged.
- **Cross-process requires explicit opt-in**: Users must manually extract/inject trace context when delegating across processes. This is intentional (avoids magic) but adds boilerplate.
- **MCP server support**: MCP servers must be updated to extract trace context from JSON-RPC metadata and restore it before creating spans.

### Tradeoffs

| Approach | Pros | Cons |
|----------|------|------|
| **Type-level topology** (chosen) | Illegal states unrepresentable; compile-time guarantees | More types to learn |
| Builder pattern with runtime validation | Simpler API | Errors at runtime, not compile time |
| Magic strings for node IDs | Zero new types | No validation, easy to typo, no IDE support |
| Implicit context propagation | Less boilerplate | Fragile, hard to debug when it breaks (see Python ecosystem bugs) |
| Manual W3C header parsing | Full control | Error-prone, breaks with OTEL spec updates |
| **`TextMapPropagator`** (chosen) | Idiomatic OTEL, future-proof | Requires `HeaderExtractor`/`HeaderInjector` adapters |

### Migration Path

- Existing single-agent code using `AgentLoopSpan::new(tracer, name, agent_name)` continues to work unchanged.
- Multi-agent support is additive: `AgentLoopSpan::new_coordinator()` and `AgentLoopSpan::new_worker()` are new constructors alongside the existing `new()`.
- `AgentLoopSpan::new` is **not** deprecated; the new constructors are the extended versions for multi-agent scenarios.

### Open Questions

1. **Nested worker depth**: Should we limit worker nesting depth (e.g., coordinator → worker → sub-worker) with a type-level bound, or leave it unbounded? Proposed: leave unbounded but document best practices.
2. **Batch delegation**: When a coordinator delegates to N workers in parallel, should we create a `graph.node.group_id` attribute to link sibling workers? Proposed: defer until we have a concrete use case.

//! OpenInference-compliant instrumentation for Arize Phoenix.
//!
//! Spec: https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md
//!
//! This module is the single source of truth for OpenInference attribute keys
//! and span construction. Attribute key strings are defined as constants in
//! [`attr`]. Span kinds are baked into each span struct's constructor —
//! you cannot create a ToolSpan with kind AGENT.

/// OpenInference semantic convention attribute keys.
///
/// Every `KeyValue::new()` in this module uses these constants as the key.
/// Grep `KeyValue::new("` in `src/otel/` — there should be zero hits.
#[cfg(feature = "phoenix")]
pub mod attr {
    pub const SPAN_KIND: &str = "openinference.span.kind";

    pub const LLM_MODEL_NAME: &str = "llm.model_name";
    pub const LLM_SYSTEM: &str = "llm.system";
    pub const LLM_TOKEN_COUNT_PROMPT: &str = "llm.token_count.prompt";
    pub const LLM_TOKEN_COUNT_COMPLETION: &str = "llm.token_count.completion";
    pub const LLM_TOKEN_COUNT_TOTAL: &str = "llm.token_count.total";
    pub const LLM_INVOCATION_PARAMETERS: &str = "llm.invocation_parameters";

    pub const INPUT_VALUE: &str = "input.value";
    pub const OUTPUT_VALUE: &str = "output.value";

    pub const TOOL_NAME: &str = "tool.name";
    pub const TOOL_DESCRIPTION: &str = "tool.description";

    pub const AGENT_NAME: &str = "agent.name";
    pub const AGENT_ITERATIONS: &str = "agent.iterations";
    pub const AGENT_STOP_REASON: &str = "agent.stop_reason";
}

#[cfg(feature = "phoenix")]
use crate::agent::LoopStopReason;
#[cfg(feature = "phoenix")]
use crate::types::ToolName;
#[cfg(feature = "phoenix")]
use opentelemetry::trace::{SpanKind as OtelSpanKind, TraceContextExt, Tracer};
#[cfg(feature = "phoenix")]
use opentelemetry::{Context, KeyValue};

/// OpenInference span kinds recognized by Phoenix.
#[cfg(feature = "phoenix")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SpanKind {
    Agent,
    Llm,
    Tool,
    Chain,
}

#[cfg(feature = "phoenix")]
impl SpanKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Agent => "AGENT",
            Self::Llm => "LLM",
            Self::Tool => "TOOL",
            Self::Chain => "CHAIN",
        }
    }
}

/// RAII span guard for the agent loop. Kind is always AGENT.
#[cfg(feature = "phoenix")]
pub struct AgentLoopSpan {
    context: Context,
    tracer: opentelemetry_sdk::trace::Tracer,
}

#[cfg(feature = "phoenix")]
impl AgentLoopSpan {
    pub fn new(
        tracer: &opentelemetry_sdk::trace::Tracer,
        name: &str,
        agent_name: &str,
    ) -> Result<Self, String> {
        let span = tracer
            .span_builder(name.to_string())
            .with_kind(OtelSpanKind::Server)
            .with_attributes(vec![
                KeyValue::new(attr::SPAN_KIND, SpanKind::Agent.as_str()),
                KeyValue::new(attr::AGENT_NAME, agent_name.to_string()),
            ])
            .start(tracer);
        let context = Context::current_with_span(span);
        Ok(Self {
            context,
            tracer: tracer.clone(),
        })
    }

    pub fn record_outcome(&self, stop_reason: &LoopStopReason, iterations: u32) {
        let span = self.context.span();
        span.set_attribute(KeyValue::new(attr::AGENT_ITERATIONS, iterations as i64));
        span.set_attribute(KeyValue::new(
            attr::AGENT_STOP_REASON,
            stop_reason.to_string(),
        ));
        if matches!(stop_reason, LoopStopReason::ToolError { .. }) {
            span.set_status(opentelemetry::trace::Status::error(
                stop_reason.to_string(),
            ));
        } else {
            span.set_status(opentelemetry::trace::Status::Ok);
        }
    }

    pub fn create_tool_span(&self, tool_name: &ToolName) -> Option<ToolSpan> {
        let span = self
            .tracer
            .span_builder(format!("tool.{}", tool_name.as_str()))
            .with_kind(OtelSpanKind::Internal)
            .with_attributes(vec![
                KeyValue::new(attr::SPAN_KIND, SpanKind::Tool.as_str()),
                KeyValue::new(attr::TOOL_NAME, tool_name.as_str().to_string()),
            ])
            .start_with_context(&self.tracer, &self.context);
        let context = self.context.with_span(span);
        Some(ToolSpan { context })
    }

    pub fn create_session_span(
        &self,
        tracer: &opentelemetry_sdk::trace::Tracer,
        operation: &str,
    ) -> Option<SessionOperationSpan> {
        SessionOperationSpan::with_parent(tracer, operation, &self.context).ok()
    }

    pub fn context(&self) -> &Context {
        &self.context
    }
}

#[cfg(feature = "phoenix")]
impl Drop for AgentLoopSpan {
    fn drop(&mut self) {
        self.context.span().end();
    }
}

/// RAII span guard for tool execution. Kind is always TOOL.
#[cfg(feature = "phoenix")]
pub struct ToolSpan {
    context: Context,
}

#[cfg(feature = "phoenix")]
impl ToolSpan {
    pub fn mark_success(&self) {
        self.context
            .span()
            .set_status(opentelemetry::trace::Status::Ok);
    }

    pub fn mark_failure(&self, message: impl Into<String>) {
        self.context
            .span()
            .set_status(opentelemetry::trace::Status::error(message.into()));
    }
}

#[cfg(feature = "phoenix")]
impl Drop for ToolSpan {
    fn drop(&mut self) {
        self.context.span().end();
    }
}

/// RAII span guard for session operations (send/continue). Kind is always CHAIN.
#[cfg(feature = "phoenix")]
pub struct SessionOperationSpan {
    context: Context,
}

#[cfg(feature = "phoenix")]
impl SessionOperationSpan {
    pub fn new(
        tracer: &opentelemetry_sdk::trace::Tracer,
        operation: &str,
    ) -> Result<Self, String> {
        let span = tracer
            .span_builder(operation.to_string())
            .with_kind(OtelSpanKind::Internal)
            .with_attributes(vec![KeyValue::new(
                attr::SPAN_KIND,
                SpanKind::Chain.as_str(),
            )])
            .start(tracer);
        let context = Context::current_with_span(span);
        Ok(Self { context })
    }

    pub fn with_parent(
        tracer: &opentelemetry_sdk::trace::Tracer,
        operation: &str,
        parent: &Context,
    ) -> Result<Self, String> {
        let span = tracer
            .span_builder(operation.to_string())
            .with_kind(OtelSpanKind::Internal)
            .with_attributes(vec![KeyValue::new(
                attr::SPAN_KIND,
                SpanKind::Chain.as_str(),
            )])
            .start_with_context(tracer, parent);
        let context = parent.with_span(span);
        Ok(Self { context })
    }
}

#[cfg(feature = "phoenix")]
impl Drop for SessionOperationSpan {
    fn drop(&mut self) {
        self.context.span().end();
    }
}

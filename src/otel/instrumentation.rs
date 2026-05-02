//! OpenTelemetry instrumentation helpers
//!
//! Provides RAII guards for automatic span lifecycle management.
//! Each span struct holds an `opentelemetry::Context` that owns the span
//! via interior mutability. Spans are automatically ended on Drop.

#[cfg(feature = "phoenix")]
use crate::otel::types::{AgentStopReason, CompletionAttributes, ToolExecutionAttributes};
#[cfg(feature = "phoenix")]
use opentelemetry::trace::{SpanKind, TraceContextExt, Tracer};
#[cfg(feature = "phoenix")]
use opentelemetry::{Context, KeyValue};

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
            .with_kind(SpanKind::Server)
            .with_attributes(vec![KeyValue::new(
                "agent.name",
                agent_name.to_string(),
            )])
            .start(tracer);
        let context = Context::current_with_span(span);
        Ok(Self {
            context,
            tracer: tracer.clone(),
        })
    }

    pub fn record_outcome(&self, stop_reason: AgentStopReason, iterations: u32) {
        let span = self.context.span();
        span.set_attribute(KeyValue::new("agent.iterations", iterations as i64));
        let reason_str = match &stop_reason {
            AgentStopReason::Normal => "normal",
            AgentStopReason::ToolExecutionCompleted => "tool_execution_completed",
            AgentStopReason::MaxToolDepthReached => "max_tool_depth_reached",
            AgentStopReason::Cancelled => "cancelled",
            AgentStopReason::ToolError { .. } => "tool_error",
        };
        span.set_attribute(KeyValue::new("agent.stop_reason", reason_str));
        if let AgentStopReason::ToolError {
            tool_name, message, ..
        } = &stop_reason
        {
            span.set_attribute(KeyValue::new("agent.error.tool", tool_name.clone()));
            span.set_status(opentelemetry::trace::Status::error(message.clone()));
        } else {
            span.set_status(opentelemetry::trace::Status::Ok);
        }
    }

    pub fn create_tool_span(&self, tool_name: &str) -> Option<ToolSpan> {
        let span = self
            .tracer
            .span_builder(format!("tool.{}", tool_name))
            .with_kind(SpanKind::Internal)
            .with_attributes(vec![KeyValue::new("tool.name", tool_name.to_string())])
            .start_with_context(&self.tracer, &self.context);
        let context = self.context.with_span(span);
        Some(ToolSpan { context })
    }

    pub fn context(&self) -> &Context {
        &self.context
    }

    pub fn create_session_span(
        &self,
        tracer: &opentelemetry_sdk::trace::Tracer,
        operation: &str,
    ) -> Option<SessionOperationSpan> {
        SessionOperationSpan::with_parent(tracer, operation, &self.context).ok()
    }

    pub fn create_iteration_span(&self, iteration: u32) -> Option<IterationSpan> {
        let span = self
            .tracer
            .span_builder(format!("iteration.{}", iteration))
            .with_kind(SpanKind::Internal)
            .with_attributes(vec![KeyValue::new("agent.iteration", iteration as i64)])
            .start_with_context(&self.tracer, &self.context);
        let context = self.context.with_span(span);
        Some(IterationSpan { context })
    }
}

#[cfg(feature = "phoenix")]
impl Drop for AgentLoopSpan {
    fn drop(&mut self) {
        self.context.span().end();
    }
}

#[cfg(feature = "phoenix")]
pub struct ToolSpan {
    context: Context,
}

#[cfg(feature = "phoenix")]
impl ToolSpan {
    pub fn new(tracer: &opentelemetry_sdk::trace::Tracer, name: &str) -> Result<Self, String> {
        let span = tracer
            .span_builder(name.to_string())
            .with_kind(SpanKind::Internal)
            .with_attributes(vec![KeyValue::new("tool.name", name.to_string())])
            .start(tracer);
        let context = Context::current_with_span(span);
        Ok(Self { context })
    }

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

#[cfg(feature = "phoenix")]
pub struct IterationSpan {
    context: Context,
}

#[cfg(feature = "phoenix")]
impl IterationSpan {
    pub fn new(tracer: &opentelemetry_sdk::trace::Tracer, iteration: u32) -> Result<Self, String> {
        let span = tracer
            .span_builder(format!("iteration.{}", iteration))
            .with_kind(SpanKind::Internal)
            .with_attributes(vec![KeyValue::new("agent.iteration", iteration as i64)])
            .start(tracer);
        let context = Context::current_with_span(span);
        Ok(Self { context })
    }
}

#[cfg(feature = "phoenix")]
impl Drop for IterationSpan {
    fn drop(&mut self) {
        self.context.span().end();
    }
}

#[cfg(feature = "phoenix")]
pub struct CompletionSpan {
    context: Context,
}

#[cfg(feature = "phoenix")]
impl CompletionSpan {
    pub fn new(
        tracer: &opentelemetry_sdk::trace::Tracer,
        name: &str,
        attributes: CompletionAttributes,
    ) -> Result<Self, String> {
        let span = tracer
            .span_builder(name.to_string())
            .with_kind(SpanKind::Client)
            .with_attributes(create_completion_attributes(&attributes))
            .start(tracer);
        let context = Context::current_with_span(span);
        Ok(Self { context })
    }

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
impl Drop for CompletionSpan {
    fn drop(&mut self) {
        self.context.span().end();
    }
}

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
            .with_kind(SpanKind::Internal)
            .with_attributes(vec![KeyValue::new(
                "session.operation",
                operation.to_string(),
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
            .with_kind(SpanKind::Internal)
            .with_attributes(vec![KeyValue::new(
                "session.operation",
                operation.to_string(),
            )])
            .start_with_context(tracer, parent);
        let context = parent.with_span(span);
        Ok(Self { context })
    }

    pub fn inner(&self) -> Option<&()> {
        Some(&())
    }
}

#[cfg(feature = "phoenix")]
impl Drop for SessionOperationSpan {
    fn drop(&mut self) {
        self.context.span().end();
    }
}

#[cfg(feature = "phoenix")]
pub fn create_completion_attributes(attributes: &CompletionAttributes) -> Vec<KeyValue> {
    let mut attrs = vec![
        KeyValue::new("llm.model", attributes.model.clone()),
        KeyValue::new("llm.provider", attributes.provider.clone()),
    ];
    if let Some(t) = attributes.prompt_tokens {
        attrs.push(KeyValue::new("llm.prompt_tokens", t as i64));
    }
    if let Some(t) = attributes.completion_tokens {
        attrs.push(KeyValue::new("llm.completion_tokens", t as i64));
    }
    if let Some(t) = attributes.total_tokens {
        attrs.push(KeyValue::new("llm.total_tokens", t as i64));
    }
    if let Some(t) = attributes.temperature {
        attrs.push(KeyValue::new("llm.temperature", t as f64));
    }
    attrs
}

#[cfg(feature = "phoenix")]
pub fn create_agent_loop_attributes(
    attributes: &crate::otel::types::AgentLoopAttributes,
) -> Vec<KeyValue> {
    let mut attrs = vec![KeyValue::new("agent.name", attributes.name.clone())];
    if let Some(iteration) = attributes.iteration {
        attrs.push(KeyValue::new("agent.iteration", iteration as i64));
    }
    if let Some(tool_count) = attributes.tool_count {
        attrs.push(KeyValue::new("agent.tool_count", tool_count as i64));
    }
    attrs
}

#[cfg(feature = "phoenix")]
pub fn create_tool_attributes(attributes: &ToolExecutionAttributes) -> Vec<KeyValue> {
    let mut attrs = vec![
        KeyValue::new("tool.name", attributes.tool_name.clone()),
        KeyValue::new("tool.success", attributes.success),
    ];
    if let Some(msg) = &attributes.error_message {
        attrs.push(KeyValue::new("tool.error", msg.clone()));
    }
    attrs
}

//! OpenTelemetry instrumentation helpers
//!
//! Provides RAII guards for automatic span lifecycle management.

#[cfg(feature = "phoenix")]
use crate::otel::types::{
    AgentLoopAttributes, AgentStopReason, CompletionAttributes, ToolExecutionAttributes,
};

#[cfg(feature = "phoenix")]
#[allow(dead_code)]
pub struct CompletionSpan {
    attributes: CompletionAttributes,
}

#[cfg(feature = "phoenix")]
impl CompletionSpan {
    pub fn new(
        _tracer: &opentelemetry_sdk::trace::Tracer,
        name: &str,
        attributes: CompletionAttributes,
    ) -> Result<Self, String> {
        let _ = name;
        Ok(Self { attributes })
    }

    pub fn mark_success(&mut self) {}
    pub fn mark_failure(&mut self, _message: impl Into<String>) {}
}

#[cfg(feature = "phoenix")]
pub struct AgentLoopSpan {
    _name: String,
}

#[cfg(feature = "phoenix")]
impl AgentLoopSpan {
    pub fn new(
        _tracer: &opentelemetry_sdk::trace::Tracer,
        name: &str,
        agent_name: &str,
    ) -> Result<Self, String> {
        let _ = (name, agent_name);
        Ok(Self {
            _name: name.to_string(),
        })
    }

    pub fn record_outcome(&mut self, _stop_reason: AgentStopReason, _iterations: u32) {}

    pub fn create_tool_span(&self, _tool_name: &str) -> Option<ToolSpan> {
        None
    }

    pub fn create_iteration_span(&self, _iteration: u32) -> Option<IterationSpan> {
        None
    }
}

#[cfg(feature = "phoenix")]
pub struct ToolSpan {
    _name: String,
}

#[cfg(feature = "phoenix")]
impl ToolSpan {
    pub fn new(_tracer: &opentelemetry_sdk::trace::Tracer, name: &str) -> Result<Self, String> {
        Ok(Self {
            _name: name.to_string(),
        })
    }

    pub fn mark_success(&mut self) {}
    pub fn mark_failure(&mut self, _message: impl Into<String>) {}
}

#[cfg(feature = "phoenix")]
pub struct IterationSpan {
    _iteration: u32,
}

#[cfg(feature = "phoenix")]
impl IterationSpan {
    pub fn new(_tracer: &opentelemetry_sdk::trace::Tracer, iteration: u32) -> Result<Self, String> {
        Ok(Self {
            _iteration: iteration,
        })
    }
}

#[cfg(feature = "phoenix")]
pub struct SessionOperationSpan {
    _operation: String,
}

#[cfg(feature = "phoenix")]
impl SessionOperationSpan {
    pub fn new(
        _tracer: &opentelemetry_sdk::trace::Tracer,
        operation: &str,
    ) -> Result<Self, String> {
        Ok(Self {
            _operation: operation.to_string(),
        })
    }

    pub fn inner(&self) -> Option<&()> {
        None
    }
}

#[cfg(feature = "phoenix")]
pub fn create_completion_attributes(
    attributes: &CompletionAttributes,
) -> Vec<(String, opentelemetry::Value)> {
    vec![
        (
            "llm.model".to_string(),
            attributes.model.as_str().to_string().into(),
        ),
        (
            "llm.provider".to_string(),
            attributes.provider.as_str().to_string().into(),
        ),
    ]
}

#[cfg(feature = "phoenix")]
pub fn create_agent_loop_attributes(
    attributes: &AgentLoopAttributes,
) -> Vec<(String, opentelemetry::Value)> {
    vec![(
        "agent.name".to_string(),
        attributes.name.as_str().to_string().into(),
    )]
}

#[cfg(feature = "phoenix")]
pub fn create_tool_attributes(
    attributes: &ToolExecutionAttributes,
) -> Vec<(String, opentelemetry::Value)> {
    vec![(
        "tool.name".to_string(),
        attributes.tool_name.as_str().to_string().into(),
    )]
}

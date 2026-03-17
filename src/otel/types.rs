//! OpenTelemetry types and configurations for Arize Phoenix tracing
//!
//! This module provides finite types following the project's coding principles:
//! - Newtypes for endpoint configuration
//! - Enums for span kinds and completion status
//! - Type-safe attribute representations

#[cfg(feature = "phoenix")]
use crate::error::OtelError;
#[cfg(feature = "phoenix")]
use std::fmt;

#[cfg(feature = "phoenix")]
/// Phoenix collector endpoint
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OtelEndpoint(String);

#[cfg(feature = "phoenix")]
impl OtelEndpoint {
    /// Create a new endpoint from a string
    pub fn new(endpoint: impl Into<String>) -> Result<Self, OtelError> {
        let endpoint = endpoint.into();
        if endpoint.is_empty() {
            return Err(OtelError::TracerInit(
                "OTEL endpoint cannot be empty".to_string(),
            ));
        }
        Ok(Self(endpoint))
    }

    /// Get the endpoint as a string slice
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[cfg(feature = "phoenix")]
impl fmt::Display for OtelEndpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(feature = "phoenix")]
/// OTLP exporter configuration
#[derive(Debug, Clone)]
pub struct OtlpExporterConfig {
    endpoint: OtelEndpoint,
    timeout: std::time::Duration,
    headers: Option<std::collections::HashMap<String, String>>,
}

#[cfg(feature = "phoenix")]
impl Default for OtlpExporterConfig {
    fn default() -> Self {
        Self {
            endpoint: OtelEndpoint::new("http://localhost:4317").unwrap(),
            timeout: std::time::Duration::from_secs(10),
            headers: None,
        }
    }
}

#[cfg(feature = "phoenix")]
impl OtlpExporterConfig {
    /// Create a new exporter configuration
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: OtelEndpoint::new(endpoint.into()).unwrap(),
            ..Default::default()
        }
    }

    /// Set the timeout for OTLP exports
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set custom headers for the OTLP request
    pub fn with_headers(mut self, headers: std::collections::HashMap<String, String>) -> Self {
        self.headers = Some(headers);
        self
    }

    /// Get the endpoint
    pub fn endpoint(&self) -> &OtelEndpoint {
        &self.endpoint
    }

    /// Get the timeout
    pub fn timeout(&self) -> std::time::Duration {
        self.timeout
    }

    /// Get headers if set
    pub fn headers(&self) -> Option<&std::collections::HashMap<String, String>> {
        self.headers.as_ref()
    }
}

#[cfg(feature = "phoenix")]
/// OpenInference span kind
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpanKind {
    /// Internal operation
    Internal,
    /// Server-side operation
    Server,
    /// Client-side operation
    Client,
    /// Producer operation
    Producer,
    /// Consumer operation
    Consumer,
}

#[cfg(feature = "phoenix")]
impl SpanKind {
    /// Convert to OpenTelemetry span kind
    pub fn to_otlp_span_kind(&self) -> opentelemetry::trace::SpanKind {
        match self {
            Self::Internal => opentelemetry::trace::SpanKind::Internal,
            Self::Server => opentelemetry::trace::SpanKind::Server,
            Self::Client => opentelemetry::trace::SpanKind::Client,
            Self::Producer => opentelemetry::trace::SpanKind::Producer,
            Self::Consumer => opentelemetry::trace::SpanKind::Consumer,
        }
    }
}

#[cfg(feature = "phoenix")]
/// Completion status for LLM completions
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CompletionStatus {
    /// Successful completion
    Success,
    /// Partial completion (streaming)
    Partial,
    /// Failed completion
    Error { message: String },
}

#[cfg(feature = "phoenix")]
impl CompletionStatus {
    /// Check if the completion was successful
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success)
    }

    /// Check if the completion is partial
    pub fn is_partial(&self) -> bool {
        matches!(self, Self::Partial)
    }

    /// Check if the completion failed
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error { .. })
    }

    /// Get the error message if present
    pub fn error_message(&self) -> Option<&str> {
        match self {
            Self::Error { message } => Some(message),
            _ => None,
        }
    }
}

#[cfg(feature = "phoenix")]
/// Agent loop stop reason for tracing
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AgentStopReason {
    /// Normal completion without tool use
    Normal,
    /// Tool execution completed all requested tools
    ToolExecutionCompleted,
    /// Maximum tool depth reached
    MaxToolDepthReached,
    /// Agent loop cancelled
    Cancelled,
    /// Tool execution error
    ToolError { tool_name: String, message: String },
}

#[cfg(feature = "phoenix")]
/// Provider kind for span attributes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProviderKind {
    /// Anthropic provider
    Anthropic,
    /// OpenAI provider
    OpenAi,
    /// AWS Bedrock provider
    Bedrock,
    /// OpenRouter provider
    OpenRouter,
    /// Ollama provider
    Ollama,
}

#[cfg(feature = "phoenix")]
impl ProviderKind {
    /// Get the OpenInference provider string
    pub fn as_inference_provider(&self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::OpenAi => "openai",
            Self::Bedrock => "amazon_bedrock",
            Self::OpenRouter => "openrouter",
            Self::Ollama => "ollama",
        }
    }

    /// Get the OpenTelemetry provider string
    pub fn as_otlp_provider(&self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::OpenAi => "openai",
            Self::Bedrock => "bedrock",
            Self::OpenRouter => "openrouter",
            Self::Ollama => "ollama",
        }
    }
}

#[cfg(feature = "phoenix")]
/// Tool execution result for tracing
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ToolExecutionResult {
    /// Tool name
    pub tool_name: String,
    /// Whether the execution succeeded
    pub success: bool,
    /// Error message if failed
    pub error_message: Option<String>,
}

#[cfg(feature = "phoenix")]
impl ToolExecutionResult {
    /// Create a new successful tool execution result
    pub fn success(tool_name: impl Into<String>) -> Self {
        Self {
            tool_name: tool_name.into(),
            success: true,
            error_message: None,
        }
    }

    /// Create a new failed tool execution result
    pub fn failure(tool_name: impl Into<String>, error_message: impl Into<String>) -> Self {
        Self {
            tool_name: tool_name.into(),
            success: false,
            error_message: Some(error_message.into()),
        }
    }

    /// Check if the execution succeeded
    pub fn is_success(&self) -> bool {
        self.success
    }
}

#[cfg(feature = "phoenix")]
/// LLM finish reason for span attributes
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LlmFinishReason {
    /// Normal stop
    Stop,
    /// Maximum length reached
    Length,
    /// Tool use required
    ToolUse,
    /// Content filter triggered
    ContentFilter,
    /// API error occurred
    Error { message: String },
}

#[cfg(feature = "phoenix")]
impl LlmFinishReason {
    /// Get the OpenInference finish reason string
    pub fn as_inference_reason(&self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::ToolUse => "tool_use",
            Self::ContentFilter => "content_filter",
            Self::Error { .. } => "error",
        }
    }
}

#[cfg(feature = "phoenix")]
/// Span name for LLM operations
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SpanName(String);

#[cfg(feature = "phoenix")]
impl SpanName {
    /// Create a new span name
    pub fn new(name: impl Into<String>) -> Result<Self, OtelError> {
        let name = name.into();
        if name.is_empty() {
            return Err(OtelError::SpanCreation(
                "Span name cannot be empty".to_string(),
            ));
        }
        Ok(Self(name))
    }

    /// Get the span name as a string slice
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[cfg(feature = "phoenix")]
/// Completion span attributes
#[derive(Debug, Clone)]
pub struct CompletionAttributes {
    pub model: String,
    pub provider: String,
    pub completion_status: CompletionStatus,
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

#[cfg(feature = "phoenix")]
/// Agent loop span attributes
#[derive(Debug, Clone)]
pub struct AgentLoopAttributes {
    pub name: String,
    pub iteration: Option<u32>,
    pub stop_reason: AgentStopReason,
    pub tool_count: Option<usize>,
    pub response_text: Option<String>,
}

#[cfg(feature = "phoenix")]
/// Tool execution span attributes
#[derive(Debug, Clone)]
pub struct ToolExecutionAttributes {
    pub tool_name: String,
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
    pub success: bool,
    pub error_message: Option<String>,
}

#[cfg(feature = "phoenix")]
/// Session span attributes
#[derive(Debug, Clone)]
pub struct SessionAttributes {
    pub message_count: usize,
    pub session_id: String,
    pub operation: String,
}

#[cfg(feature = "phoenix")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_otel_endpoint_creation() {
        let endpoint = OtelEndpoint::new("http://localhost:4317").unwrap();
        assert_eq!(endpoint.as_str(), "http://localhost:4317");

        let empty = OtelEndpoint::new("").unwrap_err();
        assert!(matches!(empty, OtelError::TracerInit(_)));
    }

    #[test]
    fn test_span_name_creation() {
        let name = SpanName::new("llm.completion").unwrap();
        assert_eq!(name.as_str(), "llm.completion");

        let empty = SpanName::new("").unwrap_err();
        assert!(matches!(empty, OtelError::SpanCreation(_)));
    }

    #[test]
    fn test_completion_status() {
        assert!(CompletionStatus::Success.is_success());
        assert!(!CompletionStatus::Success.is_partial());
        assert!(!CompletionStatus::Success.is_error());

        assert!(CompletionStatus::Partial.is_partial());
        assert!(!CompletionStatus::Partial.is_success());
        assert!(!CompletionStatus::Partial.is_error());

        assert!(CompletionStatus::Error {
            message: "failed".to_string()
        }
        .is_error());
        assert!(!CompletionStatus::Error {
            message: "failed".to_string()
        }
        .is_success());
    }

    #[test]
    fn test_agent_stop_reason() {
        assert!(matches!(AgentStopReason::Normal, AgentStopReason::Normal));
        assert!(matches!(
            AgentStopReason::MaxToolDepthReached,
            AgentStopReason::MaxToolDepthReached
        ));
        assert!(matches!(
            AgentStopReason::Cancelled,
            AgentStopReason::Cancelled
        ));
    }

    #[test]
    fn test_tool_execution_result() {
        let success = ToolExecutionResult::success("test_tool");
        assert_eq!(success.tool_name, "test_tool");
        assert!(success.is_success());

        let failure = ToolExecutionResult::failure("test_tool", "error");
        assert_eq!(failure.tool_name, "test_tool");
        assert!(!failure.is_success());
        assert_eq!(failure.error_message, Some("error".to_string()));
    }

    #[test]
    fn test_provider_kind() {
        assert_eq!(ProviderKind::Anthropic.as_inference_provider(), "anthropic");
        assert_eq!(ProviderKind::OpenAi.as_inference_provider(), "openai");
        assert_eq!(
            ProviderKind::Bedrock.as_inference_provider(),
            "amazon_bedrock"
        );
        assert_eq!(ProviderKind::Ollama.as_inference_provider(), "ollama");
    }
}

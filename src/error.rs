//! Error types for agent-driver-rs.
//!
//! All error types are defined in this module so that other modules can depend on them
//! without circular imports. The hierarchy is:
//!
//! - [`AgentDriverError`] -- top-level enum that wraps all other error types
//!   - [`ConfigError`] -- configuration loading and validation failures
//!   - [`ProviderError`] -- LLM provider communication errors
//!   - [`ToolError`] -- tool lookup and execution failures
//!   - [`TaskPoolError`] -- task spawning and lifecycle errors
//!   - [`SessionError`] -- session-level operation errors
//!   - [`AgentLoopError`] -- agent loop orchestration errors
//!
//! Newtype validation errors ([`ModelIdError`], [`ToolNameError`], [`TemperatureError`])
//! are separate because they occur at construction time, not during operations.
//!
//! ## When to use which error
//!
//! | Error | When to use | Recovery |
//! |-------|-------------|----------|
//! | [`ConfigError`] | Startup-time: missing env vars, invalid config values | Fix configuration and restart |
//! | [`ProviderError`] | Network failures, auth errors, rate limits, context overflow, content policy | Retry (with backoff for rate limits), trim history (context overflow), modify content (policy) |
//! | [`StreamError`] | Mid-stream failures: connection lost, parse errors | Retry the request; cancelled streams are intentional |
//! | [`ToolError`] | Tool not found, invalid input, execution failure | Check tool name, validate input schema, handle gracefully |
//! | [`SessionError`] | Wraps provider/tool/stream errors at session level | Match inner error and handle accordingly |
//! | [`AgentLoopError`] | Loop-level: cancelled, max depth, session errors | Check stop reason; Cancelled is usually intentional |
//! | [`TaskPoolError`] | Pool shutdown or task not found | Create new pool, or check task ID |

use std::fmt;
use std::time::Duration;

use thiserror::Error;

use crate::provider::ProviderKind;
use crate::types::ToolName;

/// Why authentication failed — enables programmatic retry/reporting decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthErrorKind {
    /// Required API key is missing from configuration/environment.
    MissingApiKey,
    /// API key contains invalid characters (e.g., non-ASCII in HTTP header).
    InvalidApiKey,
    /// Provider returned 401/Unauthorized.
    Rejected,
}

impl fmt::Display for AuthErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingApiKey => f.write_str("missing API key"),
            Self::InvalidApiKey => f.write_str("invalid API key"),
            Self::Rejected => f.write_str("rejected by provider"),
        }
    }
}

/// What kind of stream connection failure occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamErrorKind {
    /// SSE/HTTP connection dropped.
    ConnectionDropped,
    /// Provider returned an error event (e.g., Anthropic "overloaded").
    ProviderError,
    /// Underlying SDK or transport error.
    TransportError,
}

impl fmt::Display for StreamErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConnectionDropped => f.write_str("connection dropped"),
            Self::ProviderError => f.write_str("provider error"),
            Self::TransportError => f.write_str("transport error"),
        }
    }
}

/// Top-level crate error
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AgentDriverError {
    #[error(transparent)]
    Config(#[from] ConfigError),
    #[error(transparent)]
    Provider(#[from] ProviderError),
    #[error(transparent)]
    Tool(#[from] ToolError),
    #[error(transparent)]
    Task(#[from] TaskPoolError),
    #[error(transparent)]
    Session(#[from] SessionError),
    #[error(transparent)]
    AgentLoop(#[from] AgentLoopError),
}

/// Configuration-related errors
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ConfigError {
    #[error("Missing required field: {field}")]
    MissingField { field: &'static str },
    #[error("Invalid value for {field}: {reason}")]
    InvalidValue { field: String, reason: String },
    #[error("Environment variable error: {0}")]
    Env(#[from] std::env::VarError),
    #[error("Unknown provider: {0}")]
    UnknownProvider(String),
}

impl From<TemperatureError> for ConfigError {
    fn from(err: TemperatureError) -> Self {
        Self::InvalidValue {
            field: "temperature".to_owned(),
            reason: err.to_string(),
        }
    }
}

impl From<std::convert::Infallible> for ConfigError {
    fn from(never: std::convert::Infallible) -> Self {
        match never {}
    }
}

/// Provider-related errors
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ProviderError {
    #[error("[{provider}] authentication failed ({kind}): {message}")]
    Auth {
        provider: ProviderKind,
        kind: AuthErrorKind,
        message: String,
    },
    #[error("[{provider}] rate limited, retry after {retry_after:?}")]
    RateLimited {
        provider: ProviderKind,
        retry_after: Option<Duration>,
    },
    #[error("[{provider}] model not found: {model}")]
    ModelNotFound {
        provider: ProviderKind,
        model: String,
    },
    #[error("Request cancelled")]
    Cancelled,
    #[error("Request timeout after {0:?}")]
    Timeout(Duration),
    #[error("Stream error: {0}")]
    Stream(#[from] StreamError),
    #[error("[{provider}] HTTP error (status {status:?}): {message}")]
    HttpError {
        provider: ProviderKind,
        status: Option<u16>,
        message: String,
    },
    #[error("[{provider}] streaming not supported for model {model}")]
    StreamingNotSupported {
        provider: ProviderKind,
        model: String,
    },
    #[error("[{provider}] invalid request: {message}")]
    InvalidRequest {
        provider: ProviderKind,
        message: String,
    },
    #[error("[{provider}] context window exceeded: {message}")]
    ContextWindowExceeded {
        provider: ProviderKind,
        message: String,
        /// The provider's maximum context window size in tokens, if known.
        context_window: Option<u32>,
        /// How many tokens the request used, if known.
        tokens_used: Option<u32>,
    },
    #[error("[{provider}] content policy violation: {message}")]
    ContentPolicyViolation {
        provider: ProviderKind,
        message: String,
    },
}

impl ProviderError {
    /// Returns `true` if the error is retriable (rate limited, timeout, or 5xx HTTP).
    ///
    /// Retriable means the same request can be sent again without modification.
    pub fn is_retriable(&self) -> bool {
        match self {
            Self::RateLimited { .. } | Self::Timeout(_) => true,
            Self::HttpError {
                status: Some(s), ..
            } => *s >= 500,
            Self::Auth { .. }
            | Self::ModelNotFound { .. }
            | Self::Cancelled
            | Self::Stream(_)
            | Self::HttpError { status: None, .. }
            | Self::StreamingNotSupported { .. }
            | Self::InvalidRequest { .. }
            | Self::ContextWindowExceeded { .. }
            | Self::ContentPolicyViolation { .. } => false,
        }
    }

    /// Returns `true` if the error is recoverable by modifying the request.
    ///
    /// Recoverable errors include context window exceeded (trim history),
    /// content policy violations (modify content), and rate limits (wait and retry).
    /// Unlike [`is_retriable`](Self::is_retriable), recoverable errors generally
    /// require changing the request before retrying.
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::ContextWindowExceeded { .. }
                | Self::ContentPolicyViolation { .. }
                | Self::RateLimited { .. }
        )
    }

    /// Returns the retry-after duration hint, if available.
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::RateLimited { retry_after, .. } => *retry_after,
            Self::Auth { .. }
            | Self::ModelNotFound { .. }
            | Self::Cancelled
            | Self::Timeout(_)
            | Self::Stream(_)
            | Self::HttpError { .. }
            | Self::StreamingNotSupported { .. }
            | Self::InvalidRequest { .. }
            | Self::ContextWindowExceeded { .. }
            | Self::ContentPolicyViolation { .. } => None,
        }
    }

    /// Returns the provider kind, if the error is provider-specific.
    pub fn provider(&self) -> Option<ProviderKind> {
        match self {
            Self::Auth { provider, .. }
            | Self::RateLimited { provider, .. }
            | Self::ModelNotFound { provider, .. }
            | Self::HttpError { provider, .. }
            | Self::StreamingNotSupported { provider, .. }
            | Self::InvalidRequest { provider, .. }
            | Self::ContextWindowExceeded { provider, .. }
            | Self::ContentPolicyViolation { provider, .. } => Some(*provider),
            Self::Cancelled | Self::Timeout(_) | Self::Stream(_) => None,
        }
    }
}

/// Stream-related errors
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum StreamError {
    #[error("Stream cancelled")]
    Cancelled,
    #[error("Connection lost ({kind}): {message}")]
    ConnectionLost {
        kind: StreamErrorKind,
        message: String,
    },
    #[error("Deserialization failed: {message}")]
    Deserialize {
        message: String,
        raw_data: Option<String>,
    },
}

/// Tool-related errors
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ToolError {
    #[error("Tool not found: {0}")]
    NotFound(ToolName),
    #[error("Invalid input for {}: {message}", tool_name.as_ref().map(ToolName::as_str).unwrap_or("<unknown>"))]
    InvalidInput {
        tool_name: Option<ToolName>,
        message: String,
    },
    #[error("Execution of tool '{}' failed: {message}", tool_name.as_str())]
    ExecutionFailed {
        tool_name: ToolName,
        message: String,
    },
    #[error("MCP error: {0}")]
    Mcp(#[from] McpToolError),
}

/// MCP-specific tool errors
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum McpToolError {
    #[error("[{server_name}] connection failed: {message}")]
    ConnectionFailed {
        server_name: String,
        message: String,
    },
    #[error("[{server_name}] server error: {message}")]
    ServerError {
        server_name: String,
        message: String,
    },
    #[error("[{server_name}] tool discovery failed: {message}")]
    ToolDiscoveryFailed {
        server_name: String,
        message: String,
    },
    #[error("[{server_name}] disconnected from MCP server")]
    Disconnected { server_name: String },
    #[error("[{server_name}] protocol error: {message}")]
    ProtocolError {
        server_name: String,
        message: String,
    },
}

/// Task pool errors.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TaskPoolError {
    #[error(
        "Task pool has been shut down and is no longer accepting new tasks. Create a new TaskPool to spawn more tasks"
    )]
    Shutdown,
    #[error(
        "Task not found with correlation ID: {0}. The task may have already completed or been removed"
    )]
    TaskNotFound(String),
}

/// Session-related errors
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SessionError {
    #[error(transparent)]
    Provider(#[from] ProviderError),
    #[error(transparent)]
    Tool(#[from] ToolError),
    #[error(transparent)]
    Stream(#[from] StreamError),
    #[error("MCP error: {0}")]
    Mcp(#[from] McpToolError),
}

/// Agent loop errors
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AgentLoopError {
    #[error(transparent)]
    Session(#[from] SessionError),
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("Agent loop cancelled")]
    Cancelled,
    #[error("Max tool depth reached ({0} iterations)")]
    MaxToolDepthReached(u32),
}

impl AgentLoopError {
    /// Returns `true` if the error is a context window overflow.
    ///
    /// Checks both the `ProviderError::ContextWindowExceeded` path (SDK providers)
    /// and the `StreamError::ConnectionLost` path (SSE providers where HTTP 400
    /// errors arrive mid-stream).
    pub fn is_context_overflow(&self) -> bool {
        match self {
            Self::Session(SessionError::Provider(ProviderError::ContextWindowExceeded {
                ..
            })) => true,
            Self::Session(SessionError::Stream(StreamError::ConnectionLost {
                message, ..
            })) => is_context_window_message(message),
            Self::Session(_)
            | Self::InvalidConfig(_)
            | Self::Cancelled
            | Self::MaxToolDepthReached(_) => false,
        }
    }

    /// Returns `true` if the error is a content policy violation.
    ///
    /// Checks both the `ProviderError::ContentPolicyViolation` path (SDK providers)
    /// and the `StreamError::ConnectionLost` path (SSE providers).
    pub fn is_content_policy_violation(&self) -> bool {
        match self {
            Self::Session(SessionError::Provider(ProviderError::ContentPolicyViolation {
                ..
            })) => true,
            Self::Session(SessionError::Stream(StreamError::ConnectionLost {
                message, ..
            })) => is_content_policy_message(message),
            Self::Session(_)
            | Self::InvalidConfig(_)
            | Self::Cancelled
            | Self::MaxToolDepthReached(_) => false,
        }
    }

    /// Extracts the inner [`ProviderError`], if present.
    pub fn as_provider_error(&self) -> Option<&ProviderError> {
        match self {
            Self::Session(SessionError::Provider(e)) => Some(e),
            Self::Session(_)
            | Self::InvalidConfig(_)
            | Self::Cancelled
            | Self::MaxToolDepthReached(_) => None,
        }
    }
}

/// OpenTelemetry errors
#[cfg(feature = "phoenix")]
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum OtelError {
    #[error("Failed to initialize OTEL tracer: {0}")]
    TracerInit(String),
    #[error("Failed to create span: {0}")]
    SpanCreation(String),
    #[error("Failed to export span: {0}")]
    Export(String),
}

// ── Centralized error message detection ──────────────────────────────────

/// Returns `true` if the message indicates a context window overflow.
///
/// This is the single source of truth for context window detection patterns.
/// All providers should use this instead of scattering detection logic.
pub(crate) fn is_context_window_message(msg: &str) -> bool {
    let lower = msg.to_lowercase();
    lower.contains("context_length_exceeded")
        || lower.contains("prompt is too long")
        || (lower.contains("exceed") && lower.contains("context"))
        || lower.contains("too many tokens")
        || lower.contains("maximum context length")
        || lower.contains("input is too long")
}

/// Returns `true` if the message indicates a content policy violation.
///
/// This is the single source of truth for content policy detection patterns.
pub(crate) fn is_content_policy_message(msg: &str) -> bool {
    let lower = msg.to_lowercase();
    lower.contains("content_policy_violation")
        || lower.contains("content policy")
        || lower.contains("usage policy")
        || lower.contains("content management policy")
        || lower.contains("guardrail")
}

// Validation errors for newtypes

/// Error when constructing a [`ModelId`](crate::types::ModelId).
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum ModelIdError {
    #[error(
        "Model ID cannot be empty. Provide a model identifier like \"claude-sonnet-4\" or \"gpt-4o\""
    )]
    Empty,
    #[error(
        "Model ID contains invalid characters. Only alphanumeric, hyphens (-), underscores (_), forward slashes (/), colons (:), and dots (.) are allowed"
    )]
    InvalidCharacters,
}

/// Error when constructing a ToolName
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum ToolNameError {
    #[error("Tool name cannot be empty")]
    Empty,
    #[error("Tool name has invalid format (must be alphanumeric with _, -, or .)")]
    InvalidFormat,
    #[error("Tool name is too long")]
    TooLong,
}

/// Error when constructing a Temperature
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum TemperatureError {
    #[error("Temperature {0} out of range [0.0, 2.0]")]
    OutOfRange(f32),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_error_is_retriable() {
        assert!(
            ProviderError::RateLimited {
                provider: ProviderKind::Anthropic,
                retry_after: None,
            }
            .is_retriable()
        );

        assert!(ProviderError::Timeout(Duration::from_secs(30)).is_retriable());

        assert!(
            ProviderError::HttpError {
                provider: ProviderKind::OpenAi,
                status: Some(503),
                message: "Service unavailable".into(),
            }
            .is_retriable()
        );

        assert!(
            !ProviderError::HttpError {
                provider: ProviderKind::OpenAi,
                status: Some(400),
                message: "Bad request".into(),
            }
            .is_retriable()
        );

        assert!(
            !ProviderError::Auth {
                provider: ProviderKind::Anthropic,
                kind: AuthErrorKind::Rejected,
                message: "unauthorized".into(),
            }
            .is_retriable()
        );
    }

    #[test]
    fn provider_error_retry_after() {
        let dur = Duration::from_secs(5);
        let err = ProviderError::RateLimited {
            provider: ProviderKind::Bedrock,
            retry_after: Some(dur),
        };
        assert_eq!(err.retry_after(), Some(dur));

        let err = ProviderError::Auth {
            provider: ProviderKind::Anthropic,
            kind: AuthErrorKind::Rejected,
            message: "nope".into(),
        };
        assert_eq!(err.retry_after(), None);
    }

    #[test]
    fn provider_error_provider_accessor() {
        let err = ProviderError::InvalidRequest {
            provider: ProviderKind::Ollama,
            message: "bad".into(),
        };
        assert_eq!(err.provider(), Some(ProviderKind::Ollama));

        assert_eq!(ProviderError::Cancelled.provider(), None);
        assert_eq!(
            ProviderError::Timeout(Duration::from_secs(1)).provider(),
            None
        );
    }

    #[test]
    fn provider_error_provider_accessor_new_variants() {
        let err = ProviderError::ContextWindowExceeded {
            provider: ProviderKind::OpenAi,
            message: "too long".into(),
            context_window: Some(128_000),
            tokens_used: Some(130_000),
        };
        assert_eq!(err.provider(), Some(ProviderKind::OpenAi));

        let err = ProviderError::ContentPolicyViolation {
            provider: ProviderKind::Anthropic,
            message: "blocked".into(),
        };
        assert_eq!(err.provider(), Some(ProviderKind::Anthropic));
    }

    #[test]
    fn provider_error_is_recoverable() {
        assert!(
            ProviderError::ContextWindowExceeded {
                provider: ProviderKind::OpenAi,
                message: "too long".into(),
                context_window: None,
                tokens_used: None,
            }
            .is_recoverable()
        );

        assert!(
            ProviderError::ContentPolicyViolation {
                provider: ProviderKind::Anthropic,
                message: "blocked".into(),
            }
            .is_recoverable()
        );

        assert!(
            ProviderError::RateLimited {
                provider: ProviderKind::Bedrock,
                retry_after: None,
            }
            .is_recoverable()
        );

        // Not recoverable:
        assert!(
            !ProviderError::Auth {
                provider: ProviderKind::Anthropic,
                kind: AuthErrorKind::Rejected,
                message: "nope".into(),
            }
            .is_recoverable()
        );

        assert!(
            !ProviderError::InvalidRequest {
                provider: ProviderKind::OpenAi,
                message: "bad request".into(),
            }
            .is_recoverable()
        );

        assert!(!ProviderError::Cancelled.is_recoverable());
    }

    #[test]
    fn context_window_not_retriable() {
        assert!(
            !ProviderError::ContextWindowExceeded {
                provider: ProviderKind::OpenAi,
                message: "too long".into(),
                context_window: None,
                tokens_used: None,
            }
            .is_retriable()
        );

        assert!(
            !ProviderError::ContentPolicyViolation {
                provider: ProviderKind::Anthropic,
                message: "blocked".into(),
            }
            .is_retriable()
        );
    }

    #[test]
    fn is_context_window_message_patterns() {
        assert!(is_context_window_message(
            "This model's maximum context length is 8192 tokens"
        ));
        assert!(is_context_window_message("context_length_exceeded"));
        assert!(is_context_window_message(
            "The prompt is too long for this model"
        ));
        assert!(is_context_window_message(
            "Request exceeds the context window limit"
        ));
        assert!(is_context_window_message("Too many tokens in the request"));
        assert!(is_context_window_message("input is too long"));
        assert!(!is_context_window_message("invalid API key"));
        assert!(!is_context_window_message("rate limited"));
    }

    #[test]
    fn is_content_policy_message_patterns() {
        assert!(is_content_policy_message("content_policy_violation"));
        assert!(is_content_policy_message("Violates our content policy"));
        assert!(is_content_policy_message("Violates our usage policy"));
        assert!(is_content_policy_message(
            "Blocked by content management policy"
        ));
        assert!(is_content_policy_message("Blocked by guardrail"));
        assert!(!is_content_policy_message("invalid request"));
        assert!(!is_content_policy_message("rate limited"));
    }

    #[test]
    fn agent_loop_error_is_context_overflow() {
        // Direct ProviderError path
        let err = AgentLoopError::Session(SessionError::Provider(
            ProviderError::ContextWindowExceeded {
                provider: ProviderKind::OpenAi,
                message: "too long".into(),
                context_window: None,
                tokens_used: None,
            },
        ));
        assert!(err.is_context_overflow());

        // StreamError path (SSE providers)
        let err = AgentLoopError::Session(SessionError::Stream(StreamError::ConnectionLost {
            kind: StreamErrorKind::ProviderError,
            message: "context_length_exceeded: max 128000 tokens".into(),
        }));
        assert!(err.is_context_overflow());

        // Non-matching
        let err = AgentLoopError::Cancelled;
        assert!(!err.is_context_overflow());
    }

    #[test]
    fn agent_loop_error_is_content_policy_violation() {
        let err = AgentLoopError::Session(SessionError::Provider(
            ProviderError::ContentPolicyViolation {
                provider: ProviderKind::Anthropic,
                message: "blocked".into(),
            },
        ));
        assert!(err.is_content_policy_violation());

        let err = AgentLoopError::Session(SessionError::Stream(StreamError::ConnectionLost {
            kind: StreamErrorKind::ProviderError,
            message: "content_policy_violation".into(),
        }));
        assert!(err.is_content_policy_violation());

        let err = AgentLoopError::Cancelled;
        assert!(!err.is_content_policy_violation());
    }

    #[test]
    fn agent_loop_error_as_provider_error() {
        let err = AgentLoopError::Session(SessionError::Provider(
            ProviderError::ContextWindowExceeded {
                provider: ProviderKind::OpenAi,
                message: "too long".into(),
                context_window: Some(128_000),
                tokens_used: None,
            },
        ));
        let pe = err.as_provider_error().unwrap();
        assert!(matches!(pe, ProviderError::ContextWindowExceeded { .. }));
        assert_eq!(pe.provider(), Some(ProviderKind::OpenAi));

        let err = AgentLoopError::Cancelled;
        assert!(err.as_provider_error().is_none());
    }
}

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
//! | [`ProviderError`] | Network failures, auth errors, rate limits | Retry (with backoff for rate limits), check credentials |
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
pub enum ConfigError {
    #[error("Missing required field: {field}")]
    MissingField { field: &'static str },
    #[error("Invalid value for {field}: {reason}")]
    InvalidValue { field: &'static str, reason: String },
    #[error("Environment variable error: {0}")]
    Env(#[from] std::env::VarError),
    #[error("Unknown provider: {0}")]
    UnknownProvider(String),
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
}

impl ProviderError {
    /// Returns `true` if the error is retriable (rate limited, timeout, or 5xx HTTP).
    pub fn is_retriable(&self) -> bool {
        match self {
            Self::RateLimited { .. } | Self::Timeout(_) => true,
            Self::HttpError { status: Some(s), .. } => *s >= 500,
            _ => false,
        }
    }

    /// Returns the retry-after duration hint, if available.
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::RateLimited { retry_after, .. } => *retry_after,
            _ => None,
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
            | Self::InvalidRequest { provider, .. } => Some(*provider),
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
    #[error("Invalid input for {}: {message}", tool_name.as_ref().map(|n| n.as_str()).unwrap_or("<unknown>"))]
    InvalidInput {
        tool_name: Option<ToolName>,
        message: String,
    },
    #[error("Execution of tool '{}' failed: {message}", tool_name.as_str())]
    ExecutionFailed { tool_name: ToolName, message: String },
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
pub enum TaskPoolError {
    #[error("Task pool has been shut down and is no longer accepting new tasks. Create a new TaskPool to spawn more tasks")]
    Shutdown,
    #[error("Task not found with correlation ID: {0}. The task may have already completed or been removed")]
    TaskNotFound(String),
}

/// Session-related errors
#[derive(Debug, Error)]
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

// Validation errors for newtypes

/// Error when constructing a [`ModelId`](crate::types::ModelId).
#[derive(Debug, Clone, Error)]
pub enum ModelIdError {
    #[error("Model ID cannot be empty. Provide a model identifier like \"claude-sonnet-4\" or \"gpt-4o\"")]
    Empty,
    #[error("Model ID contains invalid characters. Only alphanumeric, hyphens (-), underscores (_), forward slashes (/), colons (:), and dots (.) are allowed")]
    InvalidCharacters,
}

/// Error when constructing a ToolName
#[derive(Debug, Clone, Error)]
pub enum ToolNameError {
    #[error("Tool name cannot be empty")]
    Empty,
    #[error("Tool name has invalid format (must be alphanumeric with _, -, or .)")]
    InvalidFormat,
}

/// Error when constructing a Temperature
#[derive(Debug, Clone, Error)]
pub enum TemperatureError {
    #[error("Temperature {0} out of range [0.0, 2.0]")]
    OutOfRange(f32),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_error_is_retriable() {
        assert!(ProviderError::RateLimited {
            provider: ProviderKind::Anthropic,
            retry_after: None,
        }
        .is_retriable());

        assert!(ProviderError::Timeout(Duration::from_secs(30)).is_retriable());

        assert!(ProviderError::HttpError {
            provider: ProviderKind::OpenAi,
            status: Some(503),
            message: "Service unavailable".into(),
        }
        .is_retriable());

        assert!(!ProviderError::HttpError {
            provider: ProviderKind::OpenAi,
            status: Some(400),
            message: "Bad request".into(),
        }
        .is_retriable());

        assert!(!ProviderError::Auth {
            provider: ProviderKind::Anthropic,
            kind: AuthErrorKind::Rejected,
            message: "unauthorized".into(),
        }
        .is_retriable());
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
}

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

use thiserror::Error;

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
    #[error("Authentication failed: {0}")]
    Auth(String),
    #[error("Rate limited, retry after {retry_after:?}")]
    RateLimited {
        retry_after: Option<std::time::Duration>,
    },
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    #[error("Request cancelled")]
    Cancelled,
    #[error("Request timeout after {0:?}")]
    Timeout(std::time::Duration),
    #[error("Stream error: {0}")]
    Stream(#[from] StreamError),
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("Streaming not supported for model {0}")]
    StreamingNotSupported(String),
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
}

/// Stream-related errors
#[derive(Debug, Clone, Error)]
pub enum StreamError {
    #[error("Stream cancelled")]
    Cancelled,
    #[error("Connection lost: {0}")]
    ConnectionLost(String),
    #[error("Deserialization failed: {0}")]
    Deserialize(String),
}

/// Tool-related errors
#[derive(Debug, Error)]
pub enum ToolError {
    #[error("Tool not found: {0}")]
    NotFound(String),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),
    #[error("MCP error: {0}")]
    Mcp(#[from] McpToolError),
}

/// MCP-specific tool errors
#[derive(Debug, Error)]
pub enum McpToolError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    #[error("Server error: {0}")]
    ServerError(String),
    #[error("Tool discovery failed: {0}")]
    ToolDiscoveryFailed(String),
    #[error("Disconnected from MCP server")]
    Disconnected,
    #[error("Protocol error: {0}")]
    ProtocolError(String),
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

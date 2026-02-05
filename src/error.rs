//! Error types for agent-driver-rs
//!
//! This module defines all error types used throughout the crate.
//! Error types are defined upfront as other modules depend on them.

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
}

/// Task pool errors
#[derive(Debug, Error)]
pub enum TaskPoolError {
    #[error("Pool has been shut down")]
    Shutdown,
    #[error("Task not found: {0}")]
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
}

// Validation errors for newtypes

/// Error when constructing a ModelId
#[derive(Debug, Clone, Error)]
pub enum ModelIdError {
    #[error("Model ID cannot be empty")]
    Empty,
    #[error("Model ID contains invalid characters")]
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

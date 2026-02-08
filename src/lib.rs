//! agent-driver-rs: A unified abstraction over multiple LLM providers
//!
//! This library provides:
//! - Streaming completions with multiple LLM providers (Claude, OpenAI, Bedrock, OpenRouter, Ollama)
//! - Dynamic tool registration and execution
//! - Mutable system prompts
//! - Robust task tracking with cascading cancellation
//!
//! # Example
//!
//! ```no_run
//! use agent_driver_rs::{Session, SessionBuilder, SystemPrompt};
//! use agent_driver_rs::config::ProviderConfig;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = ProviderConfig::from_env()?;
//!     // ... build session and use
//!     Ok(())
//! }
//! ```

// Core modules
pub mod error;
pub mod types;
pub mod config;
pub mod streaming;
pub mod tool;
pub mod task;
pub mod provider;
pub mod session;
pub mod agent;

// Re-exports for convenience
pub use error::{
    AgentDriverError, AgentLoopError, AuthErrorKind, ConfigError, McpToolError, ModelIdError,
    ProviderError, SessionError, StreamError, StreamErrorKind, TaskPoolError, TemperatureError,
    ToolError, ToolNameError,
};

pub use types::{
    ContentBlock, CorrelationId, MaxTokens, Message, ModelId, Role, SystemPrompt, Temperature,
    ToolCallId, ToolName, ToolResultContent,
};

pub use config::ProviderConfig;

pub use streaming::{
    CollectedResponse, CompletionMetadata, CompletionStream, ContentBlockType, StopReason,
    StreamDelta, StreamEvent, StreamHandle, TokenUsage,
};

pub use tool::{DynTool, McpServerName, PluginId, Tool, ToolContext, ToolDefinition, ToolInput, ToolRegistry, ToolResult, ToolSchema, ToolSource};

pub use task::{TaskHandle, TaskPool, TrackedSpawn};

pub use provider::{
    BoxedProvider, CompletionConfig, CompletionRequest, Provider, ProviderCapabilities,
    ProviderContext, ProviderInfo, ProviderKind, SharedProvider,
};

#[cfg(any(test, feature = "test-support"))]
pub use provider::mock;

pub use session::{Session, SessionBuilder, SessionConfig};

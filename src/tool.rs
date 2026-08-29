//! Tool system for LLM function calling
//!
//! This module provides:
//! - Tool definitions with JSON Schema input specifications
//! - A registry for dynamic tool management
//! - Tool execution traits
//! - Serialization for different provider formats (Claude, OpenAI)

mod definition;
mod executor;
mod registry;
#[cfg(feature = "schema-sanitize")]
mod schema_sanitize;
mod serializer;
mod types;

// Re-export main types
pub use definition::{ToolAnnotations, ToolDefinition};
pub use executor::{DynTool, FnTool, Tool, ToolContext, ToolInput, ToolResult};
pub use registry::ToolRegistry;
pub use serializer::ToolFormat;
pub use types::{McpServerName, PluginId, ToolSchema, ToolSource};

// MCP support is optional
#[cfg(feature = "mcp")]
pub mod mcp;

#[cfg(feature = "mcp")]
pub use mcp::{McpConnection, McpManager, McpServerSpec};

#[cfg(feature = "mcp-http")]
pub use mcp::McpHttpSpec;

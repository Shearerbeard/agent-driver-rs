//! Tool execution types and traits

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value as JsonValue};
use std::sync::Arc;

use crate::error::ToolError;

use super::definition::ToolDefinition;

/// Input for tool execution
#[derive(Debug, Clone)]
pub struct ToolInput(Map<String, JsonValue>);

impl ToolInput {
    /// Create from a JSON object map
    pub fn new(map: Map<String, JsonValue>) -> Self {
        Self(map)
    }

    /// Create from a JSON value (must be an object)
    pub fn from_value(value: JsonValue) -> Result<Self, ToolError> {
        match value {
            JsonValue::Object(map) => Ok(Self(map)),
            _ => Err(ToolError::InvalidInput(
                "Tool input must be a JSON object".into(),
            )),
        }
    }

    /// Get the inner map
    pub fn inner(&self) -> &Map<String, JsonValue> {
        &self.0
    }

    /// Parse the input into a typed struct
    pub fn parse<T: DeserializeOwned>(&self) -> Result<T, serde_json::Error> {
        serde_json::from_value(JsonValue::Object(self.0.clone()))
    }

    /// Get a specific field
    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        self.0.get(key)
    }

    /// Get a string field
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(|v| v.as_str())
    }

    /// Get an integer field
    pub fn get_i64(&self, key: &str) -> Option<i64> {
        self.0.get(key).and_then(|v| v.as_i64())
    }

    /// Get a boolean field
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.0.get(key).and_then(|v| v.as_bool())
    }
}

impl Default for ToolInput {
    fn default() -> Self {
        Self(Map::new())
    }
}

/// Result of tool execution
#[derive(Debug, Clone)]
pub enum ToolResult {
    /// Successful execution
    Success {
        content: String,
        structured: Option<JsonValue>,
    },
    /// Execution failed
    Error {
        message: String,
        code: Option<String>,
    },
}

impl ToolResult {
    /// Create a success result with text content
    pub fn text(s: impl Into<String>) -> Self {
        Self::Success {
            content: s.into(),
            structured: None,
        }
    }

    /// Create a success result with structured data
    pub fn json(content: impl Into<String>, data: JsonValue) -> Self {
        Self::Success {
            content: content.into(),
            structured: Some(data),
        }
    }

    /// Create an error result
    pub fn error(msg: impl Into<String>) -> Self {
        Self::Error {
            message: msg.into(),
            code: None,
        }
    }

    /// Create an error result with a code
    pub fn error_with_code(msg: impl Into<String>, code: impl Into<String>) -> Self {
        Self::Error {
            message: msg.into(),
            code: Some(code.into()),
        }
    }

    /// Check if this is an error result
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error { .. })
    }

    /// Check if this is a success result
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success { .. })
    }

    /// Get the content (text or error message)
    pub fn content(&self) -> &str {
        match self {
            Self::Success { content, .. } => content,
            Self::Error { message, .. } => message,
        }
    }
}

/// Core tool trait - object-safe for dynamic dispatch
#[async_trait]
pub trait Tool: Send + Sync {
    /// Get the tool definition
    fn definition(&self) -> &ToolDefinition;

    /// Execute the tool with the given input
    async fn execute(&self, input: &ToolInput) -> Result<ToolResult, ToolError>;
}

/// Type-erased tool
pub type DynTool = Arc<dyn Tool>;

/// Wrapper for function-based tools
pub struct FnTool<F>
where
    F: Fn(&ToolInput) -> futures::future::BoxFuture<'static, Result<ToolResult, ToolError>>
        + Send
        + Sync,
{
    definition: ToolDefinition,
    func: F,
}

impl<F> FnTool<F>
where
    F: Fn(&ToolInput) -> futures::future::BoxFuture<'static, Result<ToolResult, ToolError>>
        + Send
        + Sync,
{
    /// Create a new function-based tool
    pub fn new(definition: ToolDefinition, func: F) -> Self {
        Self { definition, func }
    }
}

#[async_trait]
impl<F> Tool for FnTool<F>
where
    F: Fn(&ToolInput) -> futures::future::BoxFuture<'static, Result<ToolResult, ToolError>>
        + Send
        + Sync,
{
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn execute(&self, input: &ToolInput) -> Result<ToolResult, ToolError> {
        (self.func)(input).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::types::ToolSchema;
    use crate::types::ToolName;

    #[test]
    fn tool_input_get_fields() {
        let input = ToolInput::new(
            serde_json::json!({
                "path": "/test/file.txt",
                "count": 42,
                "enabled": true
            })
            .as_object()
            .unwrap()
            .clone(),
        );

        assert_eq!(input.get_str("path"), Some("/test/file.txt"));
        assert_eq!(input.get_i64("count"), Some(42));
        assert_eq!(input.get_bool("enabled"), Some(true));
        assert!(input.get_str("missing").is_none());
    }

    #[test]
    fn tool_result_success() {
        let result = ToolResult::text("Success!");
        assert!(result.is_success());
        assert!(!result.is_error());
        assert_eq!(result.content(), "Success!");
    }

    #[test]
    fn tool_result_error() {
        let result = ToolResult::error("Something went wrong");
        assert!(result.is_error());
        assert!(!result.is_success());
        assert_eq!(result.content(), "Something went wrong");
    }

    #[tokio::test]
    async fn fn_tool_execution() {
        use futures::FutureExt;

        let definition = ToolDefinition::new(
            ToolName::new("test_tool").unwrap(),
            "A test tool",
            ToolSchema::empty(),
        );

        let tool = FnTool::new(definition, |_input| {
            async { Ok(ToolResult::text("Hello from tool!")) }.boxed()
        });

        let result = tool.execute(&ToolInput::default()).await.unwrap();
        assert_eq!(result.content(), "Hello from tool!");
    }
}

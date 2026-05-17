//! Tool execution: the [`Tool`] trait, input/output types, and dynamic dispatch.
//!
//! This module defines how tools are executed:
//! - [`ToolInput`] -- validated JSON object wrapper for tool parameters
//! - [`ToolResult`] -- success or error outcome of tool execution
//! - [`Tool`] trait -- the async interface all tools implement
//! - [`DynTool`] -- type-erased `Arc<dyn Tool>` for registry storage
//! - [`FnTool`] -- convenience wrapper for closure-based tools

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value as JsonValue};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

use crate::error::ToolError;

use super::definition::ToolDefinition;

/// Context passed to tool execution, providing cancellation and future extensibility.
///
/// Wraps a [`CancellationToken`] so tools can check for cancellation during long-running
/// operations. The `#[non_exhaustive]` attribute allows adding fields (e.g., correlation ID,
/// timeout) in future versions without breaking existing implementations.
///
/// # Example
///
/// ```
/// use agent_driver_rs::tool::ToolContext;
/// use tokio_util::sync::CancellationToken;
///
/// let ctx = ToolContext::new(CancellationToken::new());
/// assert!(!ctx.cancellation.is_cancelled());
/// ```
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct ToolContext {
    /// Token for cooperative cancellation of tool execution.
    pub cancellation: CancellationToken,
}

impl ToolContext {
    /// Create a new context with the given cancellation token.
    pub fn new(cancellation: CancellationToken) -> Self {
        Self { cancellation }
    }
}

impl Default for ToolContext {
    fn default() -> Self {
        Self {
            cancellation: CancellationToken::new(),
        }
    }
}

/// Input for tool execution, wrapping a validated JSON object.
///
/// # Example
///
/// ```
/// use agent_driver_rs::tool::ToolInput;
///
/// let input = ToolInput::from_value(serde_json::json!({"path": "/tmp/test.txt"})).unwrap();
/// assert_eq!(input.get_str("path"), Some("/tmp/test.txt"));
/// ```
#[derive(Debug, Clone)]
pub struct ToolInput(Map<String, JsonValue>);

impl ToolInput {
    /// Create from a JSON object map
    pub fn new(map: Map<String, JsonValue>) -> Self {
        Self(map)
    }

    /// Create from a JSON value
    ///
    /// Accepts a JSON object directly, or treats `null` as an empty object.
    /// Some LLMs send `null` or omit arguments entirely for tools with no
    /// required parameters — this avoids wasting a tool iteration on a retry.
    pub fn from_value(value: JsonValue) -> Result<Self, ToolError> {
        // serde_json::Value is from an external crate
        #[allow(clippy::wildcard_enum_match_arm, reason = "serde_json::Value is an external enum that may add variants")]
        match value {
            JsonValue::Object(map) => Ok(Self(map)),
            JsonValue::Null => Ok(Self(Map::new())),
            other => Err(ToolError::InvalidInput {
                tool_name: None,
                message: format!(
                    "Tool input must be a JSON object, got {}",
                    json_type_name(&other),
                ),
            }),
        }
    }

    /// Get the inner map
    pub fn inner(&self) -> &Map<String, JsonValue> {
        &self.0
    }

    /// Parse the input into a typed struct.
    ///
    /// Clones the inner map before deserializing. This is intentional: typical
    /// tool inputs are small JSON objects (a handful of string/number fields),
    /// so the clone cost is negligible. Borrowing would require a `Deserialize<'de>`
    /// bound tied to `&'a self`, propagating lifetime parameters through every
    /// `Tool::execute` implementation — complexity not worth the savings.
    pub fn parse<T: DeserializeOwned>(&self) -> Result<T, serde_json::Error> {
        serde_json::from_value(JsonValue::Object(self.0.clone()))
    }

    /// Get a specific field
    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        self.0.get(key)
    }

    /// Get a string field
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.0.get(key)?.as_str()
    }

    /// Get an integer field
    pub fn get_i64(&self, key: &str) -> Option<i64> {
        self.0.get(key)?.as_i64()
    }

    /// Get a boolean field
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.0.get(key)?.as_bool()
    }
}

impl Default for ToolInput {
    fn default() -> Self {
        Self(Map::new())
    }
}

/// Result of tool execution -- either success with content or an error message.
///
/// # Example
///
/// ```
/// use agent_driver_rs::tool::ToolResult;
///
/// let ok = ToolResult::text("File contents here");
/// assert!(ok.is_success());
///
/// let err = ToolResult::error("File not found");
/// assert!(err.is_error());
/// ```
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
    #[must_use]
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error { .. })
    }

    /// Check if this is a success result
    #[must_use]
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success { .. })
    }

    /// Get the content (text or error message)
    #[must_use]
    pub fn content(&self) -> &str {
        match self {
            Self::Success { content, .. } => content,
            Self::Error { message, .. } => message,
        }
    }
}

/// Human-readable name for a JSON value type (used in error messages)
fn json_type_name(value: &JsonValue) -> &'static str {
    match value {
        JsonValue::Array(_) => "array",
        JsonValue::Bool(_) => "bool",
        JsonValue::Number(_) => "number",
        JsonValue::String(_) => "string",
        JsonValue::Null => "null",
        JsonValue::Object(_) => "object",
    }
}

/// Core tool trait - object-safe for dynamic dispatch
#[async_trait]
pub trait Tool: Send + Sync {
    /// Get the tool definition
    fn definition(&self) -> &ToolDefinition;

    /// Execute the tool with the given input and context
    async fn execute(&self, input: &ToolInput, ctx: &ToolContext) -> Result<ToolResult, ToolError>;
}

/// Type-erased tool
pub type DynTool = Arc<dyn Tool>;

/// Wrapper for function-based tools
pub struct FnTool<F>
where
    F: Fn(
            &ToolInput,
            &ToolContext,
        ) -> futures::future::BoxFuture<'static, Result<ToolResult, ToolError>>
        + Send
        + Sync,
{
    definition: ToolDefinition,
    func: F,
}

impl<F> FnTool<F>
where
    F: Fn(
            &ToolInput,
            &ToolContext,
        ) -> futures::future::BoxFuture<'static, Result<ToolResult, ToolError>>
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
    F: Fn(
            &ToolInput,
            &ToolContext,
        ) -> futures::future::BoxFuture<'static, Result<ToolResult, ToolError>>
        + Send
        + Sync,
{
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn execute(&self, input: &ToolInput, ctx: &ToolContext) -> Result<ToolResult, ToolError> {
        (self.func)(input, ctx).await
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
    fn tool_input_from_null_is_empty_object() {
        let input = ToolInput::from_value(JsonValue::Null).unwrap();
        assert!(input.inner().is_empty());
    }

    #[test]
    fn tool_input_from_object_preserves_fields() {
        let input = ToolInput::from_value(serde_json::json!({"key": "val"})).unwrap();
        assert_eq!(input.get_str("key"), Some("val"));
    }

    #[test]
    fn tool_input_rejects_non_object_types() {
        assert!(ToolInput::from_value(serde_json::json!("string")).is_err());
        assert!(ToolInput::from_value(serde_json::json!(42)).is_err());
        assert!(ToolInput::from_value(serde_json::json!(true)).is_err());
        assert!(ToolInput::from_value(serde_json::json!([1, 2])).is_err());
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

        let tool = FnTool::new(definition, |_input, _ctx| {
            async { Ok(ToolResult::text("Hello from tool!")) }.boxed()
        });

        let result = tool
            .execute(&ToolInput::default(), &ToolContext::default())
            .await
            .unwrap();
        assert_eq!(result.content(), "Hello from tool!");
    }
}

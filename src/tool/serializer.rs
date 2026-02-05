//! Tool serialization for different provider API formats.
//!
//! Each LLM provider expects tools and tool results in a different JSON shape.
//! [`ToolFormat`] abstracts over these differences, supporting Claude (Anthropic/Bedrock)
//! and OpenAI (OpenAI/OpenRouter) formats.

use serde_json::Value as JsonValue;

use crate::types::ToolCallId;

use super::definition::ToolDefinition;
use super::executor::ToolResult;

/// Tool format for different providers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolFormat {
    /// Claude/Anthropic format
    Claude,
    /// OpenAI format
    OpenAi { strict: bool },
}

impl ToolFormat {
    /// Create Claude format
    pub fn claude() -> Self {
        Self::Claude
    }

    /// Create OpenAI format with strict mode
    pub fn openai_strict() -> Self {
        Self::OpenAi { strict: true }
    }

    /// Create OpenAI format without strict mode
    pub fn openai() -> Self {
        Self::OpenAi { strict: false }
    }

    /// Serialize a list of tools for the provider API
    pub fn serialize_tools(&self, tools: &[ToolDefinition]) -> JsonValue {
        let tool_array: Vec<JsonValue> = tools.iter().map(|t| self.serialize_tool(t)).collect();
        JsonValue::Array(tool_array)
    }

    /// Serialize a single tool for the provider API
    pub fn serialize_tool(&self, tool: &ToolDefinition) -> JsonValue {
        match self {
            Self::Claude => {
                // Claude: { name, description, input_schema }
                serde_json::json!({
                    "name": tool.name.as_str(),
                    "description": &tool.description,
                    "input_schema": tool.input_schema.to_value()
                })
            }
            Self::OpenAi { strict } => {
                // OpenAI: { type: "function", function: { name, description, parameters } }
                let mut function = serde_json::json!({
                    "name": tool.name.as_str(),
                    "description": &tool.description,
                    "parameters": tool.input_schema.to_value()
                });
                // Only include strict when true (OpenAI-specific extension)
                if *strict {
                    function["strict"] = serde_json::json!(true);
                }
                serde_json::json!({
                    "type": "function",
                    "function": function
                })
            }
        }
    }

    /// Serialize a tool result for the provider API
    pub fn serialize_result(&self, id: &ToolCallId, result: &ToolResult) -> JsonValue {
        let (content, is_error) = match result {
            ToolResult::Success { content, .. } => (content.clone(), false),
            ToolResult::Error { message, .. } => (message.clone(), true),
        };

        match self {
            Self::Claude => serde_json::json!({
                "type": "tool_result",
                "tool_use_id": id.as_str(),
                "content": content,
                "is_error": is_error
            }),
            Self::OpenAi { .. } => serde_json::json!({
                "role": "tool",
                "tool_call_id": id.as_str(),
                "content": content
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::types::ToolSchema;
    use crate::types::ToolName;

    fn make_test_tool() -> ToolDefinition {
        ToolDefinition::new(
            ToolName::new("read_file").unwrap(),
            "Read a file from disk",
            ToolSchema::from_value(serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"]
            }))
            .unwrap(),
        )
    }

    #[test]
    fn serialize_claude_tool() {
        let tool = make_test_tool();
        let json = ToolFormat::Claude.serialize_tool(&tool);

        assert_eq!(json["name"], "read_file");
        assert_eq!(json["description"], "Read a file from disk");
        assert!(json["input_schema"]["properties"]["path"].is_object());
    }

    #[test]
    fn serialize_openai_tool() {
        let tool = make_test_tool();
        let json = ToolFormat::openai_strict().serialize_tool(&tool);

        assert_eq!(json["type"], "function");
        assert_eq!(json["function"]["name"], "read_file");
        assert_eq!(json["function"]["strict"], true);
    }

    #[test]
    fn serialize_claude_result_success() {
        let id = ToolCallId::new("call_123");
        let result = ToolResult::text("File contents here");
        let json = ToolFormat::Claude.serialize_result(&id, &result);

        assert_eq!(json["type"], "tool_result");
        assert_eq!(json["tool_use_id"], "call_123");
        assert_eq!(json["content"], "File contents here");
        assert_eq!(json["is_error"], false);
    }

    #[test]
    fn serialize_claude_result_error() {
        let id = ToolCallId::new("call_123");
        let result = ToolResult::error("File not found");
        let json = ToolFormat::Claude.serialize_result(&id, &result);

        assert_eq!(json["is_error"], true);
        assert_eq!(json["content"], "File not found");
    }

    #[test]
    fn serialize_openai_result() {
        let id = ToolCallId::new("call_abc");
        let result = ToolResult::text("Success!");
        let json = ToolFormat::openai().serialize_result(&id, &result);

        assert_eq!(json["role"], "tool");
        assert_eq!(json["tool_call_id"], "call_abc");
        assert_eq!(json["content"], "Success!");
    }
}

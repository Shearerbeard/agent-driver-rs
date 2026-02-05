//! Message types for LLM conversations.
//!
//! This module defines the core message primitives used throughout the library:
//! - [`Role`] -- participant role in a conversation (user, assistant, tool, system)
//! - [`Message`] -- a single message with role and content blocks
//! - [`ContentBlock`] -- typed content within a message (text, thinking, tool use, tool result)
//! - [`ToolName`] -- validated tool name with format constraints
//! - [`ToolCallId`] -- opaque provider-assigned identifier for tool calls
//! - [`SystemPrompt`] -- cheap-to-clone system prompt wrapper using `Arc<str>`

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::sync::Arc;

use crate::error::ToolNameError;

/// Message role in conversation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// Validated tool name
///
/// Tool names must be non-empty and contain only alphanumeric characters,
/// underscores, hyphens, and dots (for MCP namespaced tools).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct ToolName(String);

impl ToolName {
    /// Create a new ToolName with validation
    #[must_use = "this returns a Result that should be checked"]
    pub fn new(name: impl Into<String>) -> Result<Self, ToolNameError> {
        let name = name.into();
        if name.is_empty() {
            return Err(ToolNameError::Empty);
        }
        // Allow alphanumeric, underscore, hyphen, and dot (for MCP namespaced tools like mcp__filesystem__read)
        if !name
            .chars()
            .all(|c| c.is_alphanumeric() || matches!(c, '_' | '-' | '.'))
        {
            return Err(ToolNameError::InvalidFormat);
        }
        Ok(Self(name))
    }

    /// Get the tool name as a string slice
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for ToolName {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Self::new(s).map_err(serde::de::Error::custom)
    }
}

impl std::fmt::Display for ToolName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Tool call identifier (provider-assigned, opaque)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ToolCallId(String);

impl ToolCallId {
    /// Create a new ToolCallId
    ///
    /// No validation - this is assigned by providers.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Get the ID as a string slice
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ToolCallId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// System prompt wrapper
///
/// Uses Arc<str> for cheap cloning across async boundaries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SystemPrompt(Arc<str>);

impl SystemPrompt {
    /// Create a new system prompt
    pub fn new(prompt: impl AsRef<str>) -> Self {
        Self(Arc::from(prompt.as_ref()))
    }

    /// Create an empty system prompt
    pub fn empty() -> Self {
        Self(Arc::from(""))
    }

    /// Get the prompt as a string slice
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Check if the prompt is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl std::fmt::Display for SystemPrompt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for SystemPrompt {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for SystemPrompt {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl Serialize for SystemPrompt {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for SystemPrompt {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(Self::new(s))
    }
}

/// Content returned from tool execution
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    // Future: Image support (see TODO.md)
    // Image { data: String, media_type: String },
}

impl From<String> for ToolResultContent {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl From<&str> for ToolResultContent {
    fn from(s: &str) -> Self {
        Self::Text(s.to_string())
    }
}

/// Content block within a message
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Standard text content
    Text { text: String },

    /// Reasoning/thinking content (from extended thinking, o1, GPT-5.x)
    Thinking { text: String },

    /// Tool invocation request from model
    ToolUse {
        id: ToolCallId,
        name: ToolName,
        input: JsonValue,
    },

    /// Result of tool execution
    ToolResult {
        tool_use_id: ToolCallId,
        content: ToolResultContent,
        is_error: bool,
    },
    // Future: Image support (see TODO.md)
    // Image { source: ImageSource },
}

impl ContentBlock {
    /// Create a text content block
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Create a thinking content block
    pub fn thinking(text: impl Into<String>) -> Self {
        Self::Thinking { text: text.into() }
    }

    /// Check if this is a text block and get the text
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text { text } => Some(text),
            _ => None,
        }
    }

    /// Check if this is a tool use block
    #[must_use]
    pub fn as_tool_use(&self) -> Option<(&ToolCallId, &ToolName, &JsonValue)> {
        match self {
            Self::ToolUse { id, name, input } => Some((id, name, input)),
            _ => None,
        }
    }
}

/// A message in the conversation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

impl Message {
    /// Create a new message with the given role and content blocks
    pub fn new(role: Role, content: Vec<ContentBlock>) -> Self {
        Self { role, content }
    }

    /// Create a user message with text content
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    /// Create an assistant message with text content
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    /// Create a message with arbitrary content blocks
    pub fn with_content(role: Role, content: Vec<ContentBlock>) -> Self {
        Self { role, content }
    }

    /// Create a tool result message
    pub fn tool_result(tool_use_id: ToolCallId, content: impl Into<String>, is_error: bool) -> Self {
        Self {
            role: Role::Tool,
            content: vec![ContentBlock::ToolResult {
                tool_use_id,
                content: ToolResultContent::Text(content.into()),
                is_error,
            }],
        }
    }

    /// Get all text content concatenated
    #[must_use]
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|b| b.as_text())
            .collect::<Vec<_>>()
            .join("")
    }

    /// Get all tool use blocks
    #[must_use]
    pub fn tool_uses(&self) -> Vec<(&ToolCallId, &ToolName, &JsonValue)> {
        self.content
            .iter()
            .filter_map(|b| b.as_tool_use())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_name_valid() {
        assert!(ToolName::new("read_file").is_ok());
        assert!(ToolName::new("mcp__filesystem__read").is_ok());
        assert!(ToolName::new("tool-name").is_ok());
        assert!(ToolName::new("tool.name").is_ok());
    }

    #[test]
    fn tool_name_invalid() {
        assert!(matches!(ToolName::new(""), Err(ToolNameError::Empty)));
        assert!(matches!(
            ToolName::new("tool name"),
            Err(ToolNameError::InvalidFormat)
        ));
        assert!(matches!(
            ToolName::new("tool@name"),
            Err(ToolNameError::InvalidFormat)
        ));
    }

    #[test]
    fn message_user() {
        let msg = Message::user("Hello");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.text(), "Hello");
    }

    #[test]
    fn message_assistant() {
        let msg = Message::assistant("Hi there");
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.text(), "Hi there");
    }

    #[test]
    fn system_prompt_clone_is_cheap() {
        let prompt = SystemPrompt::new("A long system prompt that would be expensive to clone");
        let cloned = prompt.clone();
        // Arc makes cloning cheap - just incrementing ref count
        assert_eq!(prompt.as_str(), cloned.as_str());
    }
}

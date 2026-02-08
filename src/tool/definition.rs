//! Tool definitions and metadata annotations.
//!
//! A [`ToolDefinition`] describes a tool's name, purpose, and input schema so that
//! LLM providers can present it in their function-calling APIs. [`ToolAnnotations`]
//! provide optional UI hints (destructive, idempotent, requires confirmation).

use super::types::{ToolSchema, ToolSource};
use crate::types::ToolName;

/// Definition of a tool
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    /// Unique name of the tool
    pub name: ToolName,
    /// Human-readable description
    pub description: String,
    /// JSON Schema for input parameters
    pub input_schema: ToolSchema,
    /// Source of the tool
    pub source: ToolSource,
}

impl ToolDefinition {
    /// Create a new tool definition
    pub fn new(name: ToolName, description: impl Into<String>, input_schema: ToolSchema) -> Self {
        Self {
            name,
            description: description.into(),
            input_schema,
            source: ToolSource::Native,
        }
    }

    /// Set the tool source
    #[must_use]
    pub fn with_source(mut self, source: ToolSource) -> Self {
        self.source = source;
        self
    }

    /// Create a simple tool definition with no parameters
    pub fn simple(name: ToolName, description: impl Into<String>) -> Self {
        Self {
            name,
            description: description.into(),
            input_schema: ToolSchema::empty(),
            source: ToolSource::Native,
        }
    }
}

/// Optional annotations for tools (e.g., for UI hints)
#[derive(Debug, Clone, Default)]
pub struct ToolAnnotations {
    /// Whether the tool is destructive
    pub destructive: bool,
    /// Whether the tool is idempotent
    pub idempotent: bool,
    /// Whether the tool requires confirmation
    pub requires_confirmation: bool,
    /// Tags for categorization
    pub tags: Vec<String>,
}

impl ToolAnnotations {
    /// Create empty annotations
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark as destructive
    #[must_use]
    pub fn destructive(mut self) -> Self {
        self.destructive = true;
        self
    }

    /// Mark as idempotent
    #[must_use]
    pub fn idempotent(mut self) -> Self {
        self.idempotent = true;
        self
    }

    /// Mark as requiring confirmation
    #[must_use]
    pub fn requires_confirmation(mut self) -> Self {
        self.requires_confirmation = true;
        self
    }

    /// Add a tag
    #[must_use]
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::types::McpServerName;

    #[test]
    fn tool_definition_new() {
        let def = ToolDefinition::new(
            ToolName::new("read_file").unwrap(),
            "Read a file",
            ToolSchema::empty(),
        );
        assert_eq!(def.name.as_str(), "read_file");
        assert_eq!(def.description, "Read a file");
        assert!(def.source.is_native());
    }

    #[test]
    fn tool_definition_with_source() {
        let def = ToolDefinition::simple(ToolName::new("mcp_tool").unwrap(), "MCP tool")
            .with_source(ToolSource::Mcp {
                server_name: McpServerName::new("test-server").unwrap(),
            });
        assert!(def.source.is_mcp());
    }

    #[test]
    fn tool_annotations() {
        let annotations = ToolAnnotations::new()
            .destructive()
            .requires_confirmation()
            .with_tag("file-system");

        assert!(annotations.destructive);
        assert!(annotations.requires_confirmation);
        assert!(!annotations.idempotent);
        assert_eq!(annotations.tags, vec!["file-system"]);
    }
}

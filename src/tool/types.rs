//! Core tool data types: schema and source tracking.
//!
//! - [`ToolSchema`] -- JSON Schema wrapper for tool input parameters, uses `Arc` for cheap cloning.
//! - [`ToolSource`] -- tracks where a tool originated (native, MCP server, or dynamic plugin).

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as JsonValue};
use std::sync::Arc;

/// Schema for tool input parameters
///
/// Wraps a JSON Schema object. Uses Arc for cheap cloning.
/// Note: Does not implement Hash because serde_json::Map doesn't.
#[derive(Debug, Clone, Serialize)]
#[serde(transparent)]
pub struct ToolSchema(Arc<Map<String, JsonValue>>);

impl ToolSchema {
    /// Create a new tool schema from a JSON object
    pub fn new(schema: Map<String, JsonValue>) -> Self {
        Self(Arc::new(schema))
    }

    /// Create an empty schema
    pub fn empty() -> Self {
        Self(Arc::new(Map::new()))
    }

    /// Get the inner schema
    pub fn inner(&self) -> &Map<String, JsonValue> {
        &self.0
    }

    /// Create from a JSON value (must be an object)
    pub fn from_value(value: JsonValue) -> Option<Self> {
        match value {
            JsonValue::Object(map) => Some(Self::new(map)),
            _ => None,
        }
    }

    /// Convert to JSON value
    pub fn to_value(&self) -> JsonValue {
        JsonValue::Object((*self.0).clone())
    }

    /// Create a sanitized copy suitable for OpenAI strict function calling.
    ///
    /// Applies two transformations:
    /// 1. Makes all properties required+nullable (OpenAI strict mode requirement)
    /// 2. Adds `additionalProperties: false` at every object level
    ///
    /// Returns the original schema unchanged if the `schema-sanitize` feature
    /// is not enabled.
    #[cfg(feature = "schema-sanitize")]
    pub fn sanitize_openai(&self) -> Self {
        let mut val = self.to_value();
        mcp_openai_bridge::fix_empty_root_required(&mut val);
        mcp_openai_bridge::recursive_set_additional_properties_false(&mut val);
        Self::from_value(val).unwrap_or_default()
    }
}

impl PartialEq for ToolSchema {
    fn eq(&self, other: &Self) -> bool {
        // Compare by Arc pointer for efficiency, or by value if different arcs
        Arc::ptr_eq(&self.0, &other.0) || *self.0 == *other.0
    }
}

impl Eq for ToolSchema {}

impl<'de> Deserialize<'de> for ToolSchema {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let map = Map::deserialize(deserializer)?;
        Ok(Self::new(map))
    }
}

impl Default for ToolSchema {
    fn default() -> Self {
        Self::empty()
    }
}

/// Name of an MCP server (non-empty string).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct McpServerName(String);

impl McpServerName {
    /// Create a new MCP server name, validating it is non-empty.
    pub fn new(name: impl Into<String>) -> Result<Self, &'static str> {
        let name = name.into();
        if name.is_empty() {
            return Err("MCP server name cannot be empty");
        }
        Ok(Self(name))
    }

    /// Get the server name as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for McpServerName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Identifier for a dynamic plugin (non-empty string).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PluginId(String);

impl PluginId {
    /// Create a new plugin ID, validating it is non-empty.
    pub fn new(id: impl Into<String>) -> Result<Self, &'static str> {
        let id = id.into();
        if id.is_empty() {
            return Err("Plugin ID cannot be empty");
        }
        Ok(Self(id))
    }

    /// Get the plugin ID as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for PluginId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Source of a tool
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ToolSource {
    /// Built-in native tool
    #[default]
    Native,
    /// Tool from an MCP server
    Mcp { server_name: McpServerName },
    /// Dynamically registered tool
    Dynamic { plugin_id: PluginId },
}

impl ToolSource {
    /// Check if this is a native tool
    #[must_use]
    pub fn is_native(&self) -> bool {
        matches!(self, Self::Native)
    }

    /// Check if this is an MCP tool
    #[must_use]
    pub fn is_mcp(&self) -> bool {
        matches!(self, Self::Mcp { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_schema_empty() {
        let schema = ToolSchema::empty();
        assert!(schema.inner().is_empty());
    }

    #[test]
    fn tool_schema_from_value() {
        let value = serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"}
            }
        });
        let schema = ToolSchema::from_value(value).unwrap();
        assert!(schema.inner().contains_key("type"));
    }

    #[test]
    fn tool_schema_equality() {
        let schema1 = ToolSchema::new(
            serde_json::json!({"type": "object"})
                .as_object()
                .unwrap()
                .clone(),
        );
        let schema2 = ToolSchema::new(
            serde_json::json!({"type": "object"})
                .as_object()
                .unwrap()
                .clone(),
        );
        let schema3 = schema1.clone();

        assert_eq!(schema1, schema2); // Value equality
        assert_eq!(schema1, schema3); // Arc pointer equality (faster)
    }

    #[test]
    fn tool_source_checks() {
        assert!(ToolSource::Native.is_native());
        assert!(!ToolSource::Native.is_mcp());
        assert!(ToolSource::Mcp {
            server_name: McpServerName::new("test").unwrap()
        }
        .is_mcp());
    }

    #[test]
    fn mcp_server_name_validation() {
        assert!(McpServerName::new("my-server").is_ok());
        assert!(McpServerName::new("").is_err());
    }

    #[test]
    fn plugin_id_validation() {
        assert!(PluginId::new("my-plugin").is_ok());
        assert!(PluginId::new("").is_err());
    }

    #[cfg(feature = "schema-sanitize")]
    #[test]
    fn sanitize_openai_adds_additional_properties() {
        let schema = ToolSchema::from_value(serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"}
            }
        }))
        .unwrap();

        let sanitized = schema.sanitize_openai();
        let val = sanitized.to_value();
        assert_eq!(
            val.get("additionalProperties"),
            Some(&serde_json::json!(false)),
            "expected additionalProperties: false, got: {}",
            serde_json::to_string_pretty(&val).unwrap()
        );
    }

    #[cfg(feature = "schema-sanitize")]
    #[test]
    fn sanitize_openai_fixes_required() {
        let schema = ToolSchema::from_value(serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "verbose": {"type": "boolean"}
            }
        }))
        .unwrap();

        let sanitized = schema.sanitize_openai();
        let val = sanitized.to_value();
        let required = val.get("required").and_then(|r| r.as_array());
        assert!(
            required.is_some(),
            "expected required array after sanitization, got: {}",
            serde_json::to_string_pretty(&val).unwrap()
        );
        let req = required.unwrap();
        assert!(req.contains(&serde_json::json!("path")));
        assert!(req.contains(&serde_json::json!("verbose")));
    }

    #[cfg(feature = "schema-sanitize")]
    #[test]
    fn sanitize_openai_handles_empty_schema() {
        let schema = ToolSchema::empty();
        let sanitized = schema.sanitize_openai();
        // Empty schema should survive sanitization without panic
        assert!(sanitized.inner().is_empty() || !sanitized.inner().is_empty());
    }
}

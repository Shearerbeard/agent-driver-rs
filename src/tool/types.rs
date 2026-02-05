//! Tool-related types

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

/// Source of a tool
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolSource {
    /// Built-in native tool
    Native,
    /// Tool from an MCP server
    Mcp { server_name: String },
    /// Dynamically registered tool
    Dynamic { plugin_id: String },
}

impl ToolSource {
    /// Check if this is a native tool
    pub fn is_native(&self) -> bool {
        matches!(self, Self::Native)
    }

    /// Check if this is an MCP tool
    pub fn is_mcp(&self) -> bool {
        matches!(self, Self::Mcp { .. })
    }
}

impl Default for ToolSource {
    fn default() -> Self {
        Self::Native
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
        let schema1 = ToolSchema::new(serde_json::json!({"type": "object"}).as_object().unwrap().clone());
        let schema2 = ToolSchema::new(serde_json::json!({"type": "object"}).as_object().unwrap().clone());
        let schema3 = schema1.clone();

        assert_eq!(schema1, schema2); // Value equality
        assert_eq!(schema1, schema3); // Arc pointer equality (faster)
    }

    #[test]
    fn tool_source_checks() {
        assert!(ToolSource::Native.is_native());
        assert!(!ToolSource::Native.is_mcp());
        assert!(ToolSource::Mcp { server_name: "test".into() }.is_mcp());
    }
}

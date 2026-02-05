//! Tool registry for dynamic tool management

use std::collections::HashMap;

use tokio::sync::RwLock;

use crate::types::ToolName;

use super::definition::ToolDefinition;
use super::executor::DynTool;

/// Registry for dynamic tool management.
///
/// Tools can be added and removed at runtime. Uses tokio's `RwLock`
/// for async-safe concurrent access. The agent loop re-reads the registry
/// each turn, so tools added/removed between turns are picked up automatically.
///
/// # Example
///
/// ```no_run
/// use agent_driver_rs::tool::{ToolRegistry, ToolDefinition, ToolSchema};
/// use agent_driver_rs::ToolName;
///
/// # async fn example() {
/// let registry = ToolRegistry::new();
/// // Register, list, unregister tools at runtime
/// let tools = registry.list().await;
/// # }
/// ```
pub struct ToolRegistry {
    tools: RwLock<HashMap<ToolName, DynTool>>,
}

impl ToolRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            tools: RwLock::new(HashMap::new()),
        }
    }

    /// Register a tool
    ///
    /// Returns the previously registered tool with the same name, if any.
    pub async fn register(&self, tool: DynTool) -> Option<DynTool> {
        let name = tool.definition().name.clone();
        self.tools.write().await.insert(name, tool)
    }

    /// Unregister a tool by name
    ///
    /// Returns the tool if it was registered.
    pub async fn unregister(&self, name: &ToolName) -> Option<DynTool> {
        self.tools.write().await.remove(name)
    }

    /// Get a tool by name
    pub async fn get(&self, name: &ToolName) -> Option<DynTool> {
        self.tools.read().await.get(name).cloned()
    }

    /// Check if a tool is registered
    pub async fn contains(&self, name: &ToolName) -> bool {
        self.tools.read().await.contains_key(name)
    }

    /// List all registered tool definitions
    pub async fn list(&self) -> Vec<ToolDefinition> {
        self.tools
            .read()
            .await
            .values()
            .map(|t| t.definition().clone())
            .collect()
    }

    /// List all registered tool names
    pub async fn names(&self) -> Vec<ToolName> {
        self.tools.read().await.keys().cloned().collect()
    }

    /// Get the number of registered tools
    pub async fn len(&self) -> usize {
        self.tools.read().await.len()
    }

    /// Check if the registry is empty
    pub async fn is_empty(&self) -> bool {
        self.tools.read().await.is_empty()
    }

    /// Clear all registered tools
    pub async fn clear(&self) {
        self.tools.write().await.clear();
    }

    /// Register multiple tools at once
    pub async fn register_all(&self, tools: impl IntoIterator<Item = DynTool>) {
        let mut guard = self.tools.write().await;
        for tool in tools {
            let name = tool.definition().name.clone();
            guard.insert(name, tool);
        }
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::types::ToolSchema;
    use async_trait::async_trait;
    use std::sync::Arc;

    struct TestTool {
        definition: ToolDefinition,
    }

    #[async_trait]
    impl super::super::executor::Tool for TestTool {
        fn definition(&self) -> &ToolDefinition {
            &self.definition
        }

        async fn execute(
            &self,
            _input: &super::super::executor::ToolInput,
        ) -> Result<super::super::executor::ToolResult, crate::error::ToolError> {
            Ok(super::super::executor::ToolResult::text("test"))
        }
    }

    fn make_test_tool(name: &str) -> DynTool {
        Arc::new(TestTool {
            definition: ToolDefinition::new(
                ToolName::new(name).unwrap(),
                format!("Test tool: {}", name),
                ToolSchema::empty(),
            ),
        })
    }

    #[tokio::test]
    async fn register_and_get() {
        let registry = ToolRegistry::new();
        let tool = make_test_tool("test_tool");

        registry.register(tool.clone()).await;

        let name = ToolName::new("test_tool").unwrap();
        let retrieved = registry.get(&name).await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().definition().name.as_str(), "test_tool");
    }

    #[tokio::test]
    async fn unregister() {
        let registry = ToolRegistry::new();
        let tool = make_test_tool("test_tool");
        registry.register(tool).await;

        let name = ToolName::new("test_tool").unwrap();
        let removed = registry.unregister(&name).await;
        assert!(removed.is_some());
        assert!(registry.get(&name).await.is_none());
    }

    #[tokio::test]
    async fn list_tools() {
        let registry = ToolRegistry::new();
        registry.register(make_test_tool("tool1")).await;
        registry.register(make_test_tool("tool2")).await;

        let definitions = registry.list().await;
        assert_eq!(definitions.len(), 2);
    }

    #[tokio::test]
    async fn replace_existing() {
        let registry = ToolRegistry::new();
        let tool1 = make_test_tool("test_tool");
        let tool2 = make_test_tool("test_tool");

        let old = registry.register(tool1).await;
        assert!(old.is_none());

        let old = registry.register(tool2).await;
        assert!(old.is_some());
    }
}

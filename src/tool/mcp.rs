//! MCP (Model Context Protocol) tool provider
//!
//! This module provides integration with MCP servers for dynamic tool loading.
//! All types are behind `#[cfg(feature = "mcp")]`.

use std::borrow::Cow;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::{mpsc, RwLock};
use tracing;

use crate::error::{McpToolError, ToolError};
use crate::types::ToolName;

use super::definition::ToolDefinition;
use super::executor::{DynTool, Tool, ToolInput, ToolResult};
use super::registry::ToolRegistry;
use super::types::{McpServerName, ToolSchema, ToolSource};

/// Handler for MCP client notifications
///
/// Stores the peer reference and notifies via channel when tool list changes.
pub struct McpClientHandler {
    peer: Option<rmcp::Peer<rmcp::RoleClient>>,
    tool_list_changed_tx: mpsc::Sender<()>,
}

impl McpClientHandler {
    fn new(tool_list_changed_tx: mpsc::Sender<()>) -> Self {
        Self {
            peer: None,
            tool_list_changed_tx,
        }
    }
}

impl rmcp::ClientHandler for McpClientHandler {
    fn get_peer(&self) -> Option<rmcp::Peer<rmcp::RoleClient>> {
        self.peer.clone()
    }

    fn set_peer(&mut self, peer: rmcp::Peer<rmcp::RoleClient>) {
        self.peer = Some(peer);
    }

    fn get_info(&self) -> rmcp::model::ClientInfo {
        rmcp::model::ClientInfo {
            protocol_version: Default::default(),
            capabilities: Default::default(),
            client_info: rmcp::model::Implementation {
                name: "agent-driver-rs".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
        }
    }

    fn on_tool_list_changed(&self) -> impl std::future::Future<Output = ()> + Send + '_ {
        let tx = self.tool_list_changed_tx.clone();
        async move {
            let _ = tx.send(()).await;
        }
    }
}

/// A live connection to one MCP server.
///
/// Connects to an MCP-compatible tool server via stdio transport, discovers
/// its tools, and bridges them into the library's [`ToolRegistry`] so the
/// agent loop can call them automatically.
///
/// # Example
///
/// ```no_run
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// use agent_driver_rs::tool::{McpConnection, ToolRegistry};
///
/// let registry = ToolRegistry::new();
///
/// // Connect to an MCP server and sync its tools into the registry
/// let conn = McpConnection::connect_stdio(
///     "time-server",
///     "npx",
///     &["-y", "@anthropic/mcp-server-time"],
/// ).await?;
///
/// let tool_count = conn.sync_tools(&registry).await?;
/// println!("Loaded {} tools from MCP server", tool_count);
///
/// // Keep `conn` alive for the duration of tool usage; dropping it
/// // terminates the child process.
/// conn.disconnect().await;
/// # Ok(())
/// # }
/// ```
pub struct McpConnection {
    name: String,
    service: rmcp::service::RunningService<rmcp::RoleClient, McpClientHandler>,
    peer: rmcp::Peer<rmcp::RoleClient>,
    // Retained for future reactive tool list updates (on_tool_list_changed notification)
    #[allow(dead_code)]
    tool_list_changed_rx: RwLock<mpsc::Receiver<()>>,
}

impl McpConnection {
    /// Connect to an MCP server via stdio transport
    pub async fn connect_stdio(
        name: impl Into<String>,
        command: impl AsRef<str>,
        args: &[&str],
    ) -> Result<Self, McpToolError> {
        let name = name.into();
        let mut cmd = tokio::process::Command::new(command.as_ref());
        cmd.args(args);

        let transport = rmcp::transport::TokioChildProcess::new(&mut cmd)
            .map_err(|e| McpToolError::ConnectionFailed(format!("{}: {}", name, e)))?;

        let (tx, rx) = mpsc::channel(16);
        let handler = McpClientHandler::new(tx);

        let service: rmcp::service::RunningService<rmcp::RoleClient, McpClientHandler> =
            rmcp::ServiceExt::serve(handler, transport)
                .await
                .map_err(|e: std::io::Error| {
                    McpToolError::ConnectionFailed(format!("{}: {}", name, e))
                })?;

        let peer = service.peer().clone();

        tracing::info!(server = %name, "Connected to MCP server");

        Ok(Self {
            name,
            service,
            peer,
            tool_list_changed_rx: RwLock::new(rx),
        })
    }

    /// Discover available tools from the MCP server
    pub async fn discover_tools(&self) -> Result<Vec<DynTool>, McpToolError> {
        let mcp_tools = self
            .peer
            .list_all_tools()
            .await
            .map_err(|e| McpToolError::ToolDiscoveryFailed(format!("{}: {}", self.name, e)))?;

        let mut tools = Vec::with_capacity(mcp_tools.len());
        for mcp_tool in mcp_tools {
            let tool_name = match ToolName::new(mcp_tool.name.as_ref()) {
                Ok(name) => name,
                Err(e) => {
                    tracing::warn!(
                        server = %self.name,
                        tool = %mcp_tool.name,
                        "Skipping tool with invalid name: {}",
                        e
                    );
                    continue;
                }
            };

            let schema = ToolSchema::from_value(mcp_tool.schema_as_json_value())
                .unwrap_or_default();

            let definition = ToolDefinition::new(
                tool_name,
                mcp_tool.description.as_ref().to_string(),
                schema,
            )
            .with_source(ToolSource::Mcp {
                server_name: McpServerName::new(self.name.clone())
                    .expect("MCP connection name is always non-empty"),
            });

            let wrapper = McpToolWrapper {
                definition,
                peer: self.peer.clone(),
                mcp_tool_name: mcp_tool.name.to_string(),
            };

            tools.push(Arc::new(wrapper) as DynTool);
        }

        tracing::info!(
            server = %self.name,
            count = tools.len(),
            "Discovered MCP tools"
        );

        Ok(tools)
    }

    /// Discover tools and sync them into a registry
    ///
    /// Returns the number of tools synced.
    pub async fn sync_tools(&self, registry: &ToolRegistry) -> Result<usize, McpToolError> {
        let tools = self.discover_tools().await?;
        let count = tools.len();
        registry.register_all(tools).await;
        Ok(count)
    }

    /// Get the server name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Disconnect from the MCP server
    pub async fn disconnect(self) {
        tracing::info!(server = %self.name, "Disconnecting from MCP server");
        let _ = self.service.cancel().await;
    }
}

/// Wrapper that bridges an MCP tool to our Tool trait
struct McpToolWrapper {
    definition: ToolDefinition,
    peer: rmcp::Peer<rmcp::RoleClient>,
    mcp_tool_name: String,
}

#[async_trait]
impl Tool for McpToolWrapper {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn execute(&self, input: &ToolInput) -> Result<ToolResult, ToolError> {
        let arguments = if input.inner().is_empty() {
            None
        } else {
            Some(input.inner().clone())
        };

        let params = rmcp::model::CallToolRequestParam {
            name: Cow::Owned(self.mcp_tool_name.clone()),
            arguments,
        };

        let result = self
            .peer
            .call_tool(params)
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("MCP call_tool failed: {}", e)))?;

        // Check if the result is an error
        if result.is_error.unwrap_or(false) {
            let message = extract_text_content(&result.content);
            return Ok(ToolResult::error(message));
        }

        let content = extract_text_content(&result.content);
        Ok(ToolResult::text(content))
    }
}

/// Extract text content from MCP Content blocks
fn extract_text_content(content: &[rmcp::model::Content]) -> String {
    content
        .iter()
        .filter_map(|c| {
            if let rmcp::model::RawContent::Text(text_content) = &c.raw {
                Some(text_content.text.as_str())
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Manager for multiple MCP server connections.
///
/// A convenience wrapper that owns several [`McpConnection`]s and provides
/// batch operations (connect, sync tools, disconnect). Useful when your
/// application talks to more than one MCP server.
///
/// # Example
///
/// ```no_run
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// use agent_driver_rs::tool::{McpManager, ToolRegistry};
///
/// let registry = ToolRegistry::new();
/// let mut manager = McpManager::new();
///
/// // Connect to multiple servers
/// manager.connect_stdio("time", "npx", &["-y", "@anthropic/mcp-server-time"]).await?;
/// manager.connect_stdio("fs", "npx", &["-y", "@anthropic/mcp-server-fs"]).await?;
///
/// // Sync all tools from all servers in one call
/// let total = manager.sync_all_tools(&registry).await?;
/// println!("{} tools from {} servers", total, manager.server_count());
///
/// // Clean up
/// manager.disconnect_all().await;
/// # Ok(())
/// # }
/// ```
pub struct McpManager {
    connections: Vec<McpConnection>,
}

impl McpManager {
    /// Create a new empty manager
    pub fn new() -> Self {
        Self {
            connections: Vec::new(),
        }
    }

    /// Connect to an MCP server via stdio and add it to the manager
    pub async fn connect_stdio(
        &mut self,
        name: impl Into<String>,
        command: impl AsRef<str>,
        args: &[&str],
    ) -> Result<(), McpToolError> {
        let conn = McpConnection::connect_stdio(name, command, args).await?;
        self.connections.push(conn);
        Ok(())
    }

    /// Sync all tools from all connections into a registry
    ///
    /// Returns the total number of tools synced.
    pub async fn sync_all_tools(&self, registry: &ToolRegistry) -> Result<usize, McpToolError> {
        let mut total = 0;
        for conn in &self.connections {
            total += conn.sync_tools(registry).await?;
        }
        Ok(total)
    }

    /// Get the number of connected servers
    pub fn server_count(&self) -> usize {
        self.connections.len()
    }

    /// Disconnect from all servers
    pub async fn disconnect_all(self) {
        for conn in self.connections {
            conn.disconnect().await;
        }
    }
}

impl Default for McpManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_text_from_content() {
        let content = vec![rmcp::model::Content::text("Hello world")];
        assert_eq!(extract_text_content(&content), "Hello world");
    }

    #[test]
    fn extract_text_multiple_blocks() {
        let content = vec![
            rmcp::model::Content::text("Line 1"),
            rmcp::model::Content::text("Line 2"),
        ];
        assert_eq!(extract_text_content(&content), "Line 1\nLine 2");
    }

    #[test]
    fn mcp_manager_default() {
        let manager = McpManager::new();
        assert_eq!(manager.server_count(), 0);
    }
}

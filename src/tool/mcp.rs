//! MCP (Model Context Protocol) tool provider
//!
//! This module provides integration with MCP servers for dynamic tool loading.
//! All types are behind `#[cfg(feature = "mcp")]`.

use std::sync::Arc;

use async_trait::async_trait;
use tracing;

use crate::error::{McpToolError, ToolError};
use crate::types::ToolName;

use super::definition::ToolDefinition;
use super::executor::{DynTool, Tool, ToolContext, ToolInput, ToolResult};
use super::registry::ToolRegistry;
use super::types::{McpServerName, ToolSchema, ToolSource};

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
    name: McpServerName,
    service: rmcp::service::RunningService<rmcp::RoleClient, ()>,
}

impl McpConnection {
    /// Connect to an MCP server via stdio transport
    pub async fn connect_stdio(
        name: impl AsRef<str>,
        command: impl AsRef<str>,
        args: &[&str],
    ) -> Result<Self, McpToolError> {
        let name =
            McpServerName::new(name.as_ref()).map_err(|reason| McpToolError::ConnectionFailed {
                server_name: name.as_ref().to_owned(),
                message: reason.to_owned(),
            })?;
        let mut cmd = tokio::process::Command::new(command.as_ref());
        cmd.args(args);

        let transport = rmcp::transport::TokioChildProcess::new(cmd).map_err(|e| {
            McpToolError::ConnectionFailed {
                server_name: name.as_str().to_owned(),
                message: e.to_string(),
            }
        })?;

        use rmcp::ServiceExt as _;
        let service =
            ().serve(transport)
                .await
                .map_err(|e| McpToolError::ConnectionFailed {
                    server_name: name.as_str().to_owned(),
                    message: e.to_string(),
                })?;

        tracing::info!(server = %name, "Connected to MCP server");

        Ok(Self { name, service })
    }

    /// Connect to an MCP server via Streamable HTTP transport
    #[cfg(feature = "mcp-http")]
    pub async fn connect_http(
        name: impl AsRef<str>,
        uri: impl AsRef<str>,
    ) -> Result<Self, McpToolError> {
        let name =
            McpServerName::new(name.as_ref()).map_err(|reason| McpToolError::ConnectionFailed {
                server_name: name.as_ref().to_owned(),
                message: reason.to_owned(),
            })?;
        let transport = rmcp::transport::StreamableHttpClientTransport::from_uri(uri.as_ref());

        use rmcp::ServiceExt as _;
        let service =
            ().serve(transport)
                .await
                .map_err(|e| McpToolError::ConnectionFailed {
                    server_name: name.as_str().to_owned(),
                    message: e.to_string(),
                })?;

        tracing::info!(server = %name, "Connected to MCP server via HTTP");

        Ok(Self { name, service })
    }

    /// Discover available tools from the MCP server
    pub async fn discover_tools(&self) -> Result<Vec<DynTool>, McpToolError> {
        let mcp_tools =
            self.service
                .list_all_tools()
                .await
                .map_err(|e| McpToolError::ToolDiscoveryFailed {
                    server_name: self.name.as_str().to_owned(),
                    message: e.to_string(),
                })?;

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

            let schema =
                ToolSchema::from_value(mcp_tool.schema_as_json_value()).unwrap_or_default();

            let definition = ToolDefinition::new(
                tool_name,
                mcp_tool.description.as_deref().unwrap_or("").to_owned(),
                schema,
            )
            .with_source(ToolSource::Mcp {
                server_name: self.name.clone(),
            });

            let wrapper = McpToolWrapper {
                definition,
                peer: self.service.peer().clone(),
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

    /// List raw MCP tool definitions without wrapping them into `DynTool`.
    ///
    /// Unlike [`discover_tools`](Self::discover_tools), this returns the raw
    /// protocol-level tool list with no schema parsing or wrapper construction.
    /// Useful for benchmarking the MCP transport layer in isolation.
    pub async fn list_raw_tools(&self) -> Result<Vec<rmcp::model::Tool>, McpToolError> {
        self.service
            .list_all_tools()
            .await
            .map_err(|e| McpToolError::ToolDiscoveryFailed {
                server_name: self.name.as_str().to_owned(),
                message: e.to_string(),
            })
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
        self.name.as_str()
    }

    /// Disconnect from the MCP server
    pub async fn disconnect(mut self) {
        tracing::info!(server = %self.name, "Disconnecting from MCP server");
        drop(
            self.service
                .close_with_timeout(std::time::Duration::from_secs(5))
                .await,
        );
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

    async fn execute(&self, input: &ToolInput, ctx: &ToolContext) -> Result<ToolResult, ToolError> {
        let arguments = input.inner().clone();

        let params = rmcp::model::CallToolRequestParams::new(self.mcp_tool_name.clone())
            .with_arguments(arguments);

        // Race MCP call against cancellation (matches codebase pattern in
        // collect_with_observer and all stream adapters)
        let result = tokio::select! {
            biased;
            _ = ctx.cancellation.cancelled() => {
                return Err(ToolError::ExecutionFailed {
                    tool_name: self.definition.name.clone(),
                    message: "Tool execution cancelled".to_owned(),
                });
            }
            result = self.peer.call_tool(params) => {
                result.map_err(|e| ToolError::ExecutionFailed {
                    tool_name: self.definition.name.clone(),
                    message: format!("MCP call_tool failed: {e}"),
                })?
            }
        };

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
    let mut result = String::new();
    for c in content {
        if let rmcp::model::RawContent::Text(text_content) = &c.raw {
            if !result.is_empty() {
                result.push('\n');
            }
            result.push_str(&text_content.text);
        }
    }
    result
}

/// Specification for connecting to an MCP server via stdio transport.
///
/// Used with [`McpManager::connect_all_stdio`] for parallel connection setup.
#[derive(Debug, Clone)]
pub struct McpServerSpec {
    /// Human-readable name for this server
    pub name: String,
    /// Command to execute (e.g., "npx")
    pub command: String,
    /// Arguments to the command
    pub args: Vec<String>,
}

/// Specification for connecting to an MCP server via Streamable HTTP transport.
///
/// Used with [`McpManager::connect_all_http`] for parallel connection setup.
#[cfg(feature = "mcp-http")]
#[derive(Debug, Clone)]
pub struct McpHttpSpec {
    /// Human-readable name for this server
    pub name: String,
    /// URI to connect to (e.g., "http://localhost:8000/mcp")
    pub uri: String,
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
/// use agent_driver_rs::tool::{McpManager, McpServerSpec, ToolRegistry};
///
/// let registry = ToolRegistry::new();
/// let mut manager = McpManager::new();
///
/// // Connect to multiple servers in parallel
/// let specs = vec![
///     McpServerSpec {
///         name: "time".into(),
///         command: "npx".into(),
///         args: vec!["-y".into(), "@anthropic/mcp-server-time".into()],
///     },
///     McpServerSpec {
///         name: "fs".into(),
///         command: "npx".into(),
///         args: vec!["-y".into(), "@anthropic/mcp-server-fs".into()],
///     },
/// ];
/// let errors = manager.connect_all_stdio(specs).await;
/// for (name, err) in &errors {
///     eprintln!("Failed to connect to '{}': {}", name, err);
/// }
///
/// // Sync all tools from all servers concurrently
/// let (total, sync_errors) = manager.sync_all_tools_concurrent(&registry).await;
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
        name: impl AsRef<str>,
        command: impl AsRef<str>,
        args: &[&str],
    ) -> Result<(), McpToolError> {
        let conn = McpConnection::connect_stdio(name, command, args).await?;
        self.connections.push(conn);
        Ok(())
    }

    /// Connect to an MCP server via Streamable HTTP and add it to the manager
    #[cfg(feature = "mcp-http")]
    pub async fn connect_http(
        &mut self,
        name: impl AsRef<str>,
        uri: impl AsRef<str>,
    ) -> Result<(), McpToolError> {
        let conn = McpConnection::connect_http(name, uri).await?;
        self.connections.push(conn);
        Ok(())
    }

    /// Connect to multiple MCP servers via stdio in parallel.
    ///
    /// Attempts all connections concurrently via `join_all`. Successful connections
    /// are added to the manager; failures are returned as `(name, error)` pairs.
    pub async fn connect_all_stdio(
        &mut self,
        specs: Vec<McpServerSpec>,
    ) -> Vec<(String, McpToolError)> {
        let futures: Vec<_> = specs
            .into_iter()
            .map(|spec| async move {
                let args: Vec<&str> = spec.args.iter().map(String::as_str).collect();
                let result = McpConnection::connect_stdio(&spec.name, &spec.command, &args).await;
                (spec.name, result)
            })
            .collect();

        let results = futures::future::join_all(futures).await;
        let mut errors = Vec::new();

        for (name, result) in results {
            match result {
                Ok(conn) => self.connections.push(conn),
                Err(e) => errors.push((name, e)),
            }
        }

        errors
    }

    /// Connect to multiple MCP servers via Streamable HTTP in parallel.
    ///
    /// Attempts all connections concurrently via `join_all`. Successful connections
    /// are added to the manager; failures are returned as `(name, error)` pairs.
    #[cfg(feature = "mcp-http")]
    pub async fn connect_all_http(
        &mut self,
        specs: Vec<McpHttpSpec>,
    ) -> Vec<(String, McpToolError)> {
        let futures: Vec<_> = specs
            .into_iter()
            .map(|spec| async move {
                let result = McpConnection::connect_http(&spec.name, spec.uri.as_str()).await;
                (spec.name, result)
            })
            .collect();

        let results = futures::future::join_all(futures).await;
        let mut errors = Vec::new();

        for (name, result) in results {
            match result {
                Ok(conn) => self.connections.push(conn),
                Err(e) => errors.push((name, e)),
            }
        }

        errors
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

    /// Sync tools from all connections concurrently.
    ///
    /// Returns `(total_tools_synced, errors)`. Unlike [`sync_all_tools`](Self::sync_all_tools),
    /// this method continues past individual server failures and reports them all.
    pub async fn sync_all_tools_concurrent(
        &self,
        registry: &ToolRegistry,
    ) -> (usize, Vec<McpToolError>) {
        let futures: Vec<_> = self
            .connections
            .iter()
            .map(|conn| conn.sync_tools(registry))
            .collect();

        let results = futures::future::join_all(futures).await;
        let mut total = 0;
        let mut errors = Vec::new();

        for result in results {
            match result {
                Ok(count) => total += count,
                Err(e) => errors.push(e),
            }
        }

        (total, errors)
    }

    /// Get the number of connected servers
    pub fn server_count(&self) -> usize {
        self.connections.len()
    }

    /// Extract all connections for keepalive purposes.
    ///
    /// Consumes the manager and returns the underlying connections.
    #[must_use = "dropping the connections will disconnect from MCP servers"]
    pub fn into_connections(self) -> Vec<McpConnection> {
        self.connections
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

    /// Empty MCP server names must be rejected at the API boundary rather than
    /// reaching the `.expect()` in `discover_tools`.
    #[tokio::test]
    async fn connect_stdio_rejects_empty_name() {
        let result = McpConnection::connect_stdio("", "nonexistent-command", &[]).await;
        assert!(
            matches!(
                result,
                Err(McpToolError::ConnectionFailed { server_name, message })
                    if server_name.is_empty() && message == "MCP server name cannot be empty"
            ),
            "expected empty name error"
        );
    }

    /// Verify that empty ToolInput produces `Some({})`, not `None`.
    ///
    /// This reproduces the bug where `McpToolWrapper::execute()` converted
    /// empty maps to `None`, causing MCP servers to reject the call with
    /// "Invalid arguments" because the `arguments` field was omitted entirely.
    #[test]
    fn empty_tool_input_produces_some_empty_map() {
        use crate::tool::executor::ToolInput;
        use serde_json::Value as JsonValue;

        let input = ToolInput::from_value(JsonValue::Object(serde_json::Map::new())).unwrap();

        // This is the logic from McpToolWrapper::execute() after the fix.
        // Before the fix, is_empty() caused this to be None.
        let arguments = Some(input.inner().clone());

        assert!(arguments.is_some(), "arguments must be Some, not None");
        assert!(
            arguments.as_ref().unwrap().is_empty(),
            "arguments should be an empty map"
        );
    }
}

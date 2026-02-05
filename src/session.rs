//! Session management for LLM conversations
//!
//! A session manages:
//! - Mutable system prompt
//! - Conversation message history
//! - Tool registry
//! - Cancellation and lifecycle

use std::sync::Arc;

use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

use crate::error::{ConfigError, SessionError, ToolError};
use crate::provider::{CompletionConfig, CompletionRequest, Provider, ProviderContext};
use crate::streaming::{CollectedResponse, StreamHandle};
use crate::tool::{DynTool, ToolDefinition, ToolInput, ToolRegistry, ToolResult};
use crate::types::{
    ContentBlock, CorrelationId, MaxTokens, Message, ModelId, Role, SystemPrompt, ToolCallId,
    ToolName,
};

/// Default max history to prevent unbounded memory growth
pub const DEFAULT_MAX_HISTORY_MESSAGES: usize = 1000;

/// Configuration for a session
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Model to use for completions
    pub model: ModelId,
    /// Default completion configuration
    pub completion_config: CompletionConfig,
    /// Maximum messages to retain in history (None = unlimited)
    pub max_history_messages: Option<usize>,
    /// Timeout for provider requests
    pub request_timeout: Option<std::time::Duration>,
}

/// A conversation session with an LLM
///
/// Sessions manage mutable state with split locks to avoid contention:
/// - System prompt (rarely changes)
/// - Message history (frequently appended)
/// - Tool registry (dynamically updated)
pub struct Session {
    provider: Arc<dyn Provider>,
    config: SessionConfig,

    // Split locks - independent mutation
    system_prompt: RwLock<SystemPrompt>,
    messages: RwLock<Vec<Message>>,
    tools: Arc<ToolRegistry>,

    cancellation: CancellationToken,
    task_tracker: TaskTracker,
    session_id: CorrelationId,
}

impl Session {
    /// Get the session ID
    pub fn session_id(&self) -> CorrelationId {
        self.session_id
    }

    /// Get the model ID
    pub fn model(&self) -> &ModelId {
        &self.config.model
    }

    // System prompt management

    /// Get the current system prompt
    pub async fn system_prompt(&self) -> SystemPrompt {
        self.system_prompt.read().await.clone()
    }

    /// Set the system prompt
    pub async fn set_system_prompt(&self, prompt: SystemPrompt) {
        *self.system_prompt.write().await = prompt;
    }

    // Tool management (delegated to ToolRegistry)

    /// Register a tool
    pub async fn register_tool(&self, tool: DynTool) {
        self.tools.register(tool).await;
    }

    /// Remove a tool by name
    pub async fn remove_tool(&self, name: &ToolName) -> Option<DynTool> {
        self.tools.unregister(name).await
    }

    /// Get a tool by name
    pub async fn get_tool(&self, name: &ToolName) -> Option<DynTool> {
        self.tools.get(name).await
    }

    /// List all registered tools
    pub async fn list_tools(&self) -> Vec<ToolDefinition> {
        self.tools.list().await
    }

    // Message management

    /// Add a message to the history
    pub async fn add_message(&self, message: Message) {
        let mut msgs = self.messages.write().await;
        msgs.push(message);

        // Trim if configured
        if let Some(max) = self.config.max_history_messages {
            while msgs.len() > max {
                msgs.remove(0);
            }
        }
    }

    /// Get all messages in the history
    pub async fn messages(&self) -> Vec<Message> {
        self.messages.read().await.clone()
    }

    /// Clear the message history
    pub async fn clear_messages(&self) {
        self.messages.write().await.clear();
    }

    /// Get the number of messages in history
    pub async fn message_count(&self) -> usize {
        self.messages.read().await.len()
    }

    // Completions

    /// Send a message and get a streaming response
    pub async fn send_streaming(
        &self,
        msg: impl Into<String>,
    ) -> Result<StreamHandle, SessionError> {
        let user_msg = Message::user(msg);
        self.add_message(user_msg).await;

        // Snapshot state (release locks before provider call)
        let system = self.system_prompt.read().await.clone();
        let messages = self.messages.read().await.clone();
        let tools = self.tools.list().await;

        let request = CompletionRequest {
            model: self.config.model.clone(),
            system: Some(system),
            messages,
            tools,
            config: self.config.completion_config.clone(),
        };

        let ctx = ProviderContext::new(
            self.session_id,
            self.cancellation.child_token(),
            self.task_tracker.clone(),
        );

        Ok(self.provider.complete_stream(request, ctx).await?)
    }

    /// Send a message and collect the full response
    pub async fn send(&self, msg: impl Into<String>) -> Result<CollectedResponse, SessionError> {
        let handle = self.send_streaming(msg).await?;
        let response = handle.collect().await?;

        // Add assistant response to history
        if !response.content.is_empty() {
            self.add_message(Message::with_content(
                Role::Assistant,
                response.content.clone(),
            ))
            .await;
        }

        Ok(response)
    }

    /// Send a completion request with explicit messages (doesn't add to history)
    pub async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CollectedResponse, SessionError> {
        let ctx = ProviderContext::new(
            self.session_id,
            self.cancellation.child_token(),
            self.task_tracker.clone(),
        );

        let handle = self.provider.complete_stream(request, ctx).await?;
        Ok(handle.collect().await?)
    }

    // Tool execution

    /// Execute a tool and add the result to history
    pub async fn execute_tool(
        &self,
        id: ToolCallId,
        name: &ToolName,
        input: ToolInput,
    ) -> Result<ToolResult, SessionError> {
        let tool = self
            .tools
            .get(name)
            .await
            .ok_or_else(|| ToolError::NotFound(name.as_str().to_string()))?;

        let result = tool.execute(&input).await?;

        // Add tool result to history
        let content = match &result {
            ToolResult::Success { content, .. } => content.clone(),
            ToolResult::Error { message, .. } => message.clone(),
        };
        self.add_message(Message::tool_result(id, content, result.is_error()))
            .await;

        Ok(result)
    }

    /// Process tool calls from a response
    ///
    /// Executes each tool call and adds results to history.
    /// Returns the tool results.
    pub async fn process_tool_calls(
        &self,
        response: &CollectedResponse,
    ) -> Result<Vec<ToolResult>, SessionError> {
        let mut results = Vec::new();

        for block in &response.content {
            if let ContentBlock::ToolUse { id, name, input } = block {
                let tool_input = ToolInput::from_value(input.clone())
                    .map_err(|e| ToolError::InvalidInput(e.to_string()))?;
                let result = self.execute_tool(id.clone(), name, tool_input).await?;
                results.push(result);
            }
        }

        Ok(results)
    }

    // Lifecycle

    /// Request cancellation of all operations
    pub fn cancel(&self) {
        self.cancellation.cancel();
    }

    /// Check if cancellation has been requested
    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }

    /// Get a child cancellation token
    pub fn child_token(&self) -> CancellationToken {
        self.cancellation.child_token()
    }

    /// Shut down the session gracefully
    pub async fn shutdown(&self) {
        self.cancellation.cancel();
        self.task_tracker.close();
        self.task_tracker.wait().await;
    }
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("session_id", &self.session_id)
            .field("model", &self.config.model)
            .field("is_cancelled", &self.is_cancelled())
            .finish_non_exhaustive()
    }
}

/// Builder for creating sessions
pub struct SessionBuilder {
    provider: Option<Arc<dyn Provider>>,
    model: Option<ModelId>,
    completion_config: Option<CompletionConfig>,
    system_prompt: Option<SystemPrompt>,
    max_history_messages: Option<usize>,
    request_timeout: Option<std::time::Duration>,
    tools: Vec<DynTool>,
}

impl SessionBuilder {
    /// Create a new session builder
    pub fn new() -> Self {
        Self {
            provider: None,
            model: None,
            completion_config: None,
            system_prompt: None,
            max_history_messages: None,
            request_timeout: None,
            tools: Vec::new(),
        }
    }

    /// Set the provider (accepts Arc<dyn Provider>)
    pub fn provider(mut self, p: Arc<dyn Provider>) -> Self {
        self.provider = Some(p);
        self
    }

    /// Convenience: wrap a concrete provider in Arc
    pub fn with_provider(mut self, p: impl Provider + 'static) -> Self {
        self.provider = Some(Arc::new(p));
        self
    }

    /// Set the model to use
    pub fn model(mut self, m: ModelId) -> Self {
        self.model = Some(m);
        self
    }

    /// Set the completion configuration
    pub fn completion_config(mut self, c: CompletionConfig) -> Self {
        self.completion_config = Some(c);
        self
    }

    /// Set max tokens (convenience method)
    pub fn max_tokens(mut self, tokens: MaxTokens) -> Self {
        let config = self
            .completion_config
            .get_or_insert_with(CompletionConfig::default);
        config.max_tokens = tokens;
        self
    }

    /// Set the system prompt
    pub fn system_prompt(mut self, p: SystemPrompt) -> Self {
        self.system_prompt = Some(p);
        self
    }

    /// Add a tool
    pub fn tool(mut self, t: DynTool) -> Self {
        self.tools.push(t);
        self
    }

    /// Add multiple tools
    pub fn tools(mut self, tools: impl IntoIterator<Item = DynTool>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Set the maximum message history size
    pub fn max_history(mut self, max: usize) -> Self {
        self.max_history_messages = Some(max);
        self
    }

    /// Set the request timeout
    pub fn timeout(mut self, t: std::time::Duration) -> Self {
        self.request_timeout = Some(t);
        self
    }

    /// Build the session
    ///
    /// This is async to properly register tools.
    pub async fn build(self) -> Result<Session, ConfigError> {
        let provider = self
            .provider
            .ok_or(ConfigError::MissingField { field: "provider" })?;
        let model = self
            .model
            .ok_or(ConfigError::MissingField { field: "model" })?;
        let completion_config = self.completion_config.unwrap_or_default();

        let registry = Arc::new(ToolRegistry::new());

        let config = SessionConfig {
            model,
            completion_config,
            // Use provided value, or default to prevent unbounded memory growth
            max_history_messages: self
                .max_history_messages
                .or(Some(DEFAULT_MAX_HISTORY_MESSAGES)),
            request_timeout: self.request_timeout,
        };

        // Register initial tools
        for tool in self.tools {
            registry.register(tool).await;
        }

        let session = Session {
            provider,
            config,
            system_prompt: RwLock::new(self.system_prompt.unwrap_or_else(SystemPrompt::empty)),
            messages: RwLock::new(Vec::new()),
            tools: registry,
            cancellation: CancellationToken::new(),
            task_tracker: TaskTracker::new(),
            session_id: CorrelationId::generate(),
        };

        Ok(session)
    }
}

impl Default for SessionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

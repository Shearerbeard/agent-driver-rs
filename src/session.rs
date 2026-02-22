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
use crate::tool::{DynTool, ToolContext, ToolDefinition, ToolInput, ToolRegistry, ToolResult};
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
    /// Sanitize tool schemas for OpenAI strict function calling (default: false)
    ///
    /// When true, tool schemas in outbound requests have all properties made
    /// required+nullable and `additionalProperties: false` added at every
    /// object level. This is needed for models/providers that require strict
    /// JSON Schema compliance.
    #[cfg(feature = "schema-sanitize")]
    pub sanitize_schemas: bool,
}

/// A conversation session with an LLM.
///
/// Sessions manage mutable state with split locks to avoid contention:
/// - System prompt (rarely changes)
/// - Message history (frequently appended)
/// - Tool registry (dynamically updated)
///
/// Use [`SessionBuilder`] to construct a session, then call [`send`](Self::send)
/// for one-shot completions or [`send_streaming`](Self::send_streaming) for
/// real-time streaming.
///
/// # Example
///
/// ```no_run
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// use agent_driver_rs::{SessionBuilder, SystemPrompt, ModelId};
///
/// // Build a session (provider must be set via SessionBuilder::provider)
/// let session = SessionBuilder::new()
///     .model(ModelId::new("claude-sonnet-4")?)
///     // .provider(my_provider)
///     // .system_prompt(SystemPrompt::new("You are helpful."))
///     .build()
///     .await?;
///
/// // Send a message and collect the response
/// let response = session.send("Hello!").await?;
/// println!("{}", response.text());
///
/// // Update the system prompt mid-conversation
/// session.set_system_prompt(SystemPrompt::new("Be concise.")).await;
///
/// // Shut down gracefully
/// session.shutdown().await;
/// # Ok(())
/// # }
/// ```
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

    /// Get a reference to the tool registry
    pub fn tool_registry(&self) -> &Arc<ToolRegistry> {
        &self.tools
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

        // Trim if configured — use drain to avoid O(n^2) repeated remove(0)
        if let Some(max) = self.config.max_history_messages {
            if msgs.len() > max {
                let excess = msgs.len() - max;
                msgs.drain(..excess);
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

        let request = self.build_completion_request().await;
        let ctx = self.new_provider_context();
        Ok(self.provider.complete_stream(request, ctx).await?)
    }

    /// Continue a streaming conversation without adding a new user message
    ///
    /// Use this after processing tool calls: the assistant's tool_use message and
    /// the tool result messages are already in history, so we just need to send
    /// the current history back to the provider for the next turn.
    pub async fn continue_streaming(&self) -> Result<StreamHandle, SessionError> {
        let request = self.build_completion_request().await;
        let ctx = self.new_provider_context();
        Ok(self.provider.complete_stream(request, ctx).await?)
    }

    /// Snapshot session state into a completion request.
    ///
    /// Reads system prompt, messages, and tools under their respective locks,
    /// releasing each lock before moving to the next.
    async fn build_completion_request(&self) -> CompletionRequest {
        let system = self.system_prompt.read().await.clone();
        let messages = self.messages.read().await.clone();
        #[allow(unused_mut)]
        let mut tools = self.tools.list().await;

        // Apply schema sanitization if configured
        #[cfg(feature = "schema-sanitize")]
        if self.config.sanitize_schemas {
            for tool in &mut tools {
                tool.input_schema = tool.input_schema.sanitize_openai();
            }
        }

        CompletionRequest {
            model: self.config.model.clone(),
            system: Some(system),
            messages,
            tools,
            config: self.config.completion_config.clone(),
        }
    }

    /// Create a fresh provider context with a child cancellation token.
    fn new_provider_context(&self) -> ProviderContext {
        ProviderContext::new(
            self.session_id,
            self.cancellation.child_token(),
            self.task_tracker.clone(),
        )
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
        let ctx = self.new_provider_context();
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
            .ok_or_else(|| ToolError::NotFound(name.clone()))?;

        let result = tool
            .execute(&input, &ToolContext::new(self.cancellation.child_token()))
            .await?;

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
                let tool_input =
                    ToolInput::from_value(input.clone()).map_err(|e| ToolError::InvalidInput {
                        tool_name: Some(name.clone()),
                        message: e.to_string(),
                    })?;
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
    #[must_use]
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

    /// Set the provider (accepts `Arc<dyn Provider>`)
    #[must_use]
    pub fn provider(mut self, p: Arc<dyn Provider>) -> Self {
        self.provider = Some(p);
        self
    }

    /// Convenience: wrap a concrete provider in Arc
    #[must_use]
    pub fn with_provider(mut self, p: impl Provider + 'static) -> Self {
        self.provider = Some(Arc::new(p));
        self
    }

    /// Set the model to use
    #[must_use]
    pub fn model(mut self, m: ModelId) -> Self {
        self.model = Some(m);
        self
    }

    /// Set the completion configuration
    #[must_use]
    pub fn completion_config(mut self, c: CompletionConfig) -> Self {
        self.completion_config = Some(c);
        self
    }

    /// Set max tokens (convenience method)
    #[must_use]
    pub fn max_tokens(mut self, tokens: MaxTokens) -> Self {
        let config = self
            .completion_config
            .get_or_insert_with(CompletionConfig::default);
        config.max_tokens = tokens;
        self
    }

    /// Set the system prompt
    #[must_use]
    pub fn system_prompt(mut self, p: SystemPrompt) -> Self {
        self.system_prompt = Some(p);
        self
    }

    /// Add a tool
    #[must_use]
    pub fn tool(mut self, t: DynTool) -> Self {
        self.tools.push(t);
        self
    }

    /// Add multiple tools
    #[must_use]
    pub fn tools(mut self, tools: impl IntoIterator<Item = DynTool>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Set the maximum message history size
    #[must_use]
    pub fn max_history(mut self, max: usize) -> Self {
        self.max_history_messages = Some(max);
        self
    }

    /// Set the request timeout
    #[must_use]
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
            #[cfg(feature = "schema-sanitize")]
            sanitize_schemas: false,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{mock_text_response, MockProvider};
    use crate::tool::{FnTool, ToolDefinition, ToolInput, ToolResult, ToolSchema};
    use futures::FutureExt;
    use std::sync::Arc;

    fn model() -> ModelId {
        ModelId::new("mock-model").unwrap()
    }

    fn mock_session(responses: Vec<Vec<crate::streaming::StreamEvent>>) -> SessionBuilder {
        SessionBuilder::new()
            .with_provider(MockProvider::new(responses))
            .model(model())
    }

    #[tokio::test]
    async fn send_adds_user_and_assistant_messages() {
        let session = mock_session(vec![mock_text_response("Hello back!")])
            .build()
            .await
            .unwrap();

        let response = session.send("hi").await.unwrap();
        assert_eq!(response.text(), "Hello back!");

        let msgs = session.messages().await;
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, Role::User);
        assert_eq!(msgs[0].text(), "hi");
        assert_eq!(msgs[1].role, Role::Assistant);
        assert_eq!(msgs[1].text(), "Hello back!");
    }

    #[tokio::test]
    async fn send_streaming_adds_user_message() {
        let session = mock_session(vec![mock_text_response("streamed")])
            .build()
            .await
            .unwrap();

        let handle = session.send_streaming("hello").await.unwrap();
        let response = handle.collect().await.unwrap();
        assert_eq!(response.text(), "streamed");

        let msgs = session.messages().await;
        // send_streaming only adds the user message; assistant is NOT auto-added
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, Role::User);
        assert_eq!(msgs[0].text(), "hello");
    }

    #[tokio::test]
    async fn history_trimming() {
        let session = mock_session(vec![
            mock_text_response("r1"),
            mock_text_response("r2"),
            mock_text_response("r3"),
        ])
        .max_history(4)
        .build()
        .await
        .unwrap();

        // Each send() adds 2 messages (user + assistant)
        session.send("m1").await.unwrap(); // history: [user, assistant] = 2
        session.send("m2").await.unwrap(); // history: [user, assistant, user, assistant] = 4
        session.send("m3").await.unwrap(); // history would be 6, trimmed to 4

        let msgs = session.messages().await;
        assert_eq!(msgs.len(), 4);
        // Oldest 2 messages were trimmed (user m1, assistant r1)
        assert_eq!(msgs[0].role, Role::User);
        assert_eq!(msgs[0].text(), "m2");
        assert_eq!(msgs[1].role, Role::Assistant);
        assert_eq!(msgs[1].text(), "r2");
    }

    #[tokio::test]
    async fn execute_tool_adds_result_to_history() {
        let session = mock_session(vec![]).build().await.unwrap();

        let definition = ToolDefinition::new(
            ToolName::new("echo").unwrap(),
            "Echoes input",
            ToolSchema::empty(),
        );
        let tool: crate::tool::DynTool = Arc::new(FnTool::new(definition, |_input, _ctx| {
            async { Ok(ToolResult::text("echoed!")) }.boxed()
        }));
        session.register_tool(tool).await;

        let result = session
            .execute_tool(
                ToolCallId::new("call_1"),
                &ToolName::new("echo").unwrap(),
                ToolInput::default(),
            )
            .await
            .unwrap();
        assert!(result.is_success());
        assert_eq!(result.content(), "echoed!");

        let msgs = session.messages().await;
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, Role::Tool);
    }

    #[tokio::test]
    async fn execute_tool_not_found() {
        let session = mock_session(vec![]).build().await.unwrap();

        let result = session
            .execute_tool(
                ToolCallId::new("call_1"),
                &ToolName::new("nonexistent").unwrap(),
                ToolInput::default(),
            )
            .await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(&err, SessionError::Tool(ToolError::NotFound(_))),
            "expected NotFound, got: {:?}",
            err
        );
    }

    #[tokio::test]
    async fn system_prompt_update() {
        let session = mock_session(vec![])
            .system_prompt(SystemPrompt::new("initial"))
            .build()
            .await
            .unwrap();

        assert_eq!(session.system_prompt().await.as_str(), "initial");

        session
            .set_system_prompt(SystemPrompt::new("updated"))
            .await;
        assert_eq!(session.system_prompt().await.as_str(), "updated");
    }

    #[tokio::test]
    async fn clear_messages() {
        let session = mock_session(vec![mock_text_response("hi")])
            .build()
            .await
            .unwrap();

        session.send("hello").await.unwrap();
        assert_eq!(session.message_count().await, 2);

        session.clear_messages().await;
        assert_eq!(session.message_count().await, 0);
        assert!(session.messages().await.is_empty());
    }

    #[tokio::test]
    async fn cancellation() {
        let session = mock_session(vec![]).build().await.unwrap();

        assert!(!session.is_cancelled());
        session.cancel();
        assert!(session.is_cancelled());
    }
}

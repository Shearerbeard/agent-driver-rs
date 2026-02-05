//! Provider trait and related types
//!
//! This module defines the core `Provider` trait that all LLM providers implement.

mod retry;

pub use retry::{with_retry, RetryConfig};

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

use crate::error::ProviderError;
use crate::streaming::{CollectedResponse, StreamHandle};
use crate::tool::ToolDefinition;
use crate::types::{CorrelationId, MaxTokens, Message, ModelId, SystemPrompt, Temperature};

/// Request for an LLM completion.
///
/// # Example
///
/// ```
/// use agent_driver_rs::provider::CompletionRequest;
/// use agent_driver_rs::{ModelId, Message, SystemPrompt};
///
/// let request = CompletionRequest::new(
///     ModelId::new("claude-sonnet-4").unwrap(),
///     vec![Message::user("Hello!")],
/// ).with_system(SystemPrompt::new("You are helpful."));
/// ```
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    /// Model to use
    pub model: ModelId,
    /// System prompt
    pub system: Option<SystemPrompt>,
    /// Conversation messages
    pub messages: Vec<Message>,
    /// Available tools
    pub tools: Vec<ToolDefinition>,
    /// Completion configuration
    pub config: CompletionConfig,
}

impl CompletionRequest {
    /// Create a new completion request
    pub fn new(model: ModelId, messages: Vec<Message>) -> Self {
        Self {
            model,
            system: None,
            messages,
            tools: Vec::new(),
            config: CompletionConfig::default(),
        }
    }

    /// Set the system prompt
    #[must_use]
    pub fn with_system(mut self, system: SystemPrompt) -> Self {
        self.system = Some(system);
        self
    }

    /// Set the tools
    #[must_use]
    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    /// Set the completion config
    #[must_use]
    pub fn with_config(mut self, config: CompletionConfig) -> Self {
        self.config = config;
        self
    }
}

/// Configuration for completion requests
#[derive(Debug, Clone)]
pub struct CompletionConfig {
    /// Maximum tokens in response
    pub max_tokens: MaxTokens,
    /// Sampling temperature
    pub temperature: Option<Temperature>,
    /// Stop sequences
    pub stop_sequences: Vec<String>,
}

impl CompletionConfig {
    /// Create with builder-style validation
    pub fn new(max_tokens: MaxTokens) -> Self {
        Self {
            max_tokens,
            temperature: None,
            stop_sequences: Vec::new(),
        }
    }

    /// Set the temperature
    #[must_use]
    pub fn with_temperature(mut self, temp: Temperature) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Set stop sequences
    #[must_use]
    pub fn with_stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.stop_sequences = sequences;
        self
    }
}

impl Default for CompletionConfig {
    fn default() -> Self {
        Self {
            max_tokens: MaxTokens::new(4096).expect("4096 is valid"),
            temperature: None,
            stop_sequences: Vec::new(),
        }
    }
}

/// Context for provider operations
///
/// Carries cancellation token and correlation ID for request tracking.
#[derive(Debug, Clone)]
pub struct ProviderContext {
    /// Correlation ID for this request
    pub correlation_id: CorrelationId,
    /// Cancellation token
    pub cancellation: CancellationToken,
    /// Task tracker for spawned tasks
    pub task_tracker: TaskTracker,
}

impl ProviderContext {
    /// Create a new provider context
    pub fn new(
        correlation_id: CorrelationId,
        cancellation: CancellationToken,
        task_tracker: TaskTracker,
    ) -> Self {
        Self {
            correlation_id,
            cancellation,
            task_tracker,
        }
    }

    /// Create a child context for sub-operations
    pub fn child(&self) -> Self {
        Self {
            correlation_id: CorrelationId::generate(),
            cancellation: self.cancellation.child_token(),
            task_tracker: self.task_tracker.clone(),
        }
    }

    /// Check if the context is cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }
}

impl Default for ProviderContext {
    fn default() -> Self {
        Self {
            correlation_id: CorrelationId::generate(),
            cancellation: CancellationToken::new(),
            task_tracker: TaskTracker::new(),
        }
    }
}

/// Provider capabilities for feature detection
#[derive(Debug, Clone, Default)]
pub struct ProviderCapabilities {
    /// Supports streaming responses
    pub streaming: bool,
    /// Supports tool/function calling
    pub tools: bool,
    /// Supports vision/images
    pub vision: bool,
    /// Supports extended thinking
    pub extended_thinking: bool,
    /// Maximum context tokens (if known)
    pub max_context_tokens: Option<u32>,
}

/// Information about a provider
#[derive(Debug, Clone)]
pub struct ProviderInfo {
    /// Provider identifier
    pub id: &'static str,
    /// Human-readable name
    pub name: &'static str,
    /// Provider capabilities
    pub capabilities: ProviderCapabilities,
}

/// Information about a model
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model identifier
    pub id: ModelId,
    /// Human-readable name
    pub name: String,
    /// Context window size (if known)
    pub context_window: Option<u32>,
}

/// Core provider trait
///
/// All LLM providers implement this trait. Uses boxed futures for object safety.
///
/// # Cancellation contract
///
/// Provider implementations **must** respect the `CancellationToken` in
/// `ctx.cancellation`. Specifically, the stream returned inside the
/// `StreamHandle` must check for cancellation on every iteration of its
/// poll loop. The canonical pattern uses `tokio::select!` with the `biased`
/// hint so cancellation is checked before the next network read:
///
/// ```ignore
/// loop {
///     tokio::select! {
///         biased;
///         _ = ctx.cancellation.cancelled() => return None,
///         event = inner_stream.next() => {
///             // ... parse and yield StreamEvent ...
///         }
///     }
/// }
/// ```
///
/// This is a **convention-only** requirement -- there is no compile-time
/// mechanism to enforce it. The `StreamHandle` wrapper adds a synchronous
/// `is_cancelled()` check in its own `poll_next`, but that only takes
/// effect *between* polls.  The provider's stream loop is the first line
/// of defence for prompt cancellation during a long-running network read.
///
/// See [`StreamHandle`] for additional notes on cancellation latency.
pub trait Provider: Send + Sync {
    /// Get provider information
    fn info(&self) -> &ProviderInfo;

    /// Start a streaming completion.
    ///
    /// Returns a [`StreamHandle`] that yields [`StreamEvent`]s. The returned
    /// stream **must** honour `ctx.cancellation` -- see the
    /// [cancellation contract](Provider#cancellation-contract) on the trait
    /// for the expected pattern and rationale.
    ///
    /// [`StreamEvent`]: crate::streaming::StreamEvent
    fn complete_stream(
        &self,
        request: CompletionRequest,
        ctx: ProviderContext,
    ) -> Pin<Box<dyn Future<Output = Result<StreamHandle, ProviderError>> + Send + '_>>;

    /// List available models
    fn list_models(
        &self,
        ctx: ProviderContext,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<ModelInfo>, ProviderError>> + Send + '_>>;
}

/// Extension trait for non-streaming completion
#[async_trait]
pub trait ProviderExt: Provider {
    /// Collect a streaming completion into a single response
    async fn complete(
        &self,
        request: CompletionRequest,
        ctx: ProviderContext,
    ) -> Result<CollectedResponse, ProviderError> {
        let handle = self.complete_stream(request, ctx).await?;
        handle.collect().await.map_err(ProviderError::from)
    }
}

// Blanket implementation
impl<T: Provider> ProviderExt for T {}

/// Type alias for a boxed provider
pub type BoxedProvider = Box<dyn Provider>;

/// Type alias for a shared provider
pub type SharedProvider = Arc<dyn Provider>;

// Provider implementations are in separate modules
#[cfg(feature = "anthropic")]
pub mod anthropic;

#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "bedrock")]
pub mod bedrock;

#[cfg(feature = "openrouter")]
pub mod openrouter;

#[cfg(feature = "ollama")]
pub mod ollama;

// Re-export provider implementations when available
#[cfg(feature = "anthropic")]
pub use anthropic::AnthropicProvider;

#[cfg(feature = "openai")]
pub use openai::OpenAiProvider;

#[cfg(feature = "bedrock")]
pub use bedrock::BedrockProvider;

#[cfg(feature = "openrouter")]
pub use openrouter::OpenRouterProvider;

#[cfg(feature = "ollama")]
pub use ollama::OllamaProvider;

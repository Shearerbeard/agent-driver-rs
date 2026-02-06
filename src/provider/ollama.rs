//! Ollama provider implementation using the `ollama-rs` crate.
//!
//! Connects to a local Ollama instance for running open-source models like Llama,
//! Qwen, Mistral, and DeepSeek. Supports streaming chat completions with configurable
//! context window size (`num_ctx`) and model keep-alive settings.

use std::future::Future;
use std::pin::Pin;

use ollama_rs::generation::chat::request::ChatMessageRequest;
use ollama_rs::generation::chat::{ChatMessage, ChatMessageResponseStream, MessageRole};
use ollama_rs::models::ModelOptions;
use ollama_rs::Ollama;

use crate::config::OllamaConfig;
use crate::error::{ProviderError, StreamError};
use crate::streaming::{
    CompletionMetadata, ContentBlockType, StopReason, StreamDelta, StreamEvent, StreamHandle,
    TokenUsage,
};
use crate::types::{ContentBlock, Message, ModelId, Role};

use super::{
    CompletionRequest, ModelInfo, Provider, ProviderCapabilities, ProviderContext, ProviderInfo,
};

/// Ollama provider for local LLM inference
pub struct OllamaProvider {
    config: OllamaConfig,
    client: Ollama,
    info: ProviderInfo,
}

impl OllamaProvider {
    /// Create a new Ollama provider
    pub fn new(config: OllamaConfig) -> Result<Self, ProviderError> {
        // Parse the base URL to extract host and port
        let host = config
            .base_url
            .host_str()
            .unwrap_or("localhost")
            .to_string();
        let port = config.base_url.port().unwrap_or(11434);

        let client = Ollama::new(format!("http://{}", host), port);

        let info = ProviderInfo {
            kind: super::ProviderKind::Ollama,
            name: "Ollama",
            capabilities: ProviderCapabilities {
                streaming: true,
                tools: true,
                vision: false,
                extended_thinking: false,
                max_context_tokens: Some(config.num_ctx.get()),
            },
        };

        Ok(Self {
            config,
            client,
            info,
        })
    }

    /// Get the configured context window size
    pub fn num_ctx(&self) -> u32 {
        self.config.num_ctx.get()
    }

    /// Convert our messages to Ollama format
    fn convert_messages(&self, messages: &[Message]) -> Vec<ChatMessage> {
        messages
            .iter()
            .filter_map(|msg| {
                let role = match msg.role {
                    Role::User => MessageRole::User,
                    Role::Assistant => MessageRole::Assistant,
                    Role::System => MessageRole::System,
                    Role::Tool => MessageRole::User, // Tool results as user messages
                };

                // Extract text content
                let text: String = msg
                    .content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::Text { text } => Some(text.as_str()),
                        ContentBlock::Thinking { text } => Some(text.as_str()),
                        ContentBlock::ToolResult { content, .. } => match content {
                            crate::types::ToolResultContent::Text(s) => Some(s.as_str()),
                        },
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                if text.is_empty() {
                    None
                } else {
                    Some(ChatMessage::new(role, text))
                }
            })
            .collect()
    }
}

impl Provider for OllamaProvider {
    fn info(&self) -> &ProviderInfo {
        &self.info
    }

    fn complete_stream(
        &self,
        request: CompletionRequest,
        ctx: ProviderContext,
    ) -> Pin<Box<dyn Future<Output = Result<StreamHandle, ProviderError>> + Send + '_>> {
        Box::pin(async move {
            let model = self.config.model.as_str().to_string();

            // Convert messages
            let mut messages = Vec::new();

            // Add system prompt if present
            if let Some(ref system) = request.system {
                if !system.as_str().is_empty() {
                    messages.push(ChatMessage::new(
                        MessageRole::System,
                        system.as_str().to_string(),
                    ));
                }
            }

            // Add conversation messages
            messages.extend(self.convert_messages(&request.messages));

            // Build model options with num_ctx (critical for Ollama!)
            let mut options = ModelOptions::default().num_ctx(self.config.num_ctx.get() as u64);

            // Add temperature if specified
            if let Some(temp) = request.config.temperature {
                options = options.temperature(temp.get());
            }

            // Add max tokens if specified
            if let Some(max_tokens) = self.config.max_tokens {
                options = options.num_predict(max_tokens.get() as i32);
            }

            // Build request
            let chat_request =
                ChatMessageRequest::new(model.clone(), messages).options(options);

            // Create streaming request
            let stream: ChatMessageResponseStream = self
                .client
                .send_chat_messages_stream(chat_request)
                .await
                .map_err(|e| ProviderError::InvalidRequest(format!("Ollama error: {}", e)))?;

            let event_stream = super::stream_adapter::buffered_sdk_stream(
                stream,
                ctx.cancellation.clone(),
                StreamState::new(model.clone()),
                |item, state| match item {
                    Ok(response) => parse_ollama_response(response, state),
                    Err(()) => vec![Err(StreamError::ConnectionLost("Stream error".to_string()))],
                },
                |state| {
                    if !state.completed {
                        Some(Ok(StreamEvent::Completed {
                            metadata: CompletionMetadata {
                                model: state.model.clone(),
                                stop_reason: Some(StopReason::EndTurn),
                                usage: state.usage,
                            },
                        }))
                    } else {
                        None
                    }
                },
            );

            Ok(StreamHandle::new(
                Box::pin(event_stream),
                ctx.cancellation,
                ctx.correlation_id,
            ))
        })
    }

    fn list_models(
        &self,
        _ctx: ProviderContext,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<ModelInfo>, ProviderError>> + Send + '_>> {
        Box::pin(async move {
            // Query Ollama for local models
            match self.client.list_local_models().await {
                Ok(models) => {
                    let infos = models
                        .into_iter()
                        .map(|m| ModelInfo {
                            id: ModelId::new(&m.name)
                                // Safety: "unknown" is a valid model ID (alphanumeric)
                                .unwrap_or_else(|_| ModelId::new("unknown").expect("hardcoded valid model ID")),
                            name: m.name.clone(),
                            context_window: None, // Ollama doesn't report this in list
                        })
                        .collect();
                    Ok(infos)
                }
                Err(e) => {
                    // Fall back to common models if we can't query
                    tracing::warn!("Failed to list Ollama models: {}", e);
                    // Safety: all model IDs below are hardcoded valid strings
                    Ok(vec![
                        ModelInfo {
                            id: ModelId::new("llama3.2").expect("hardcoded valid model ID"),
                            name: "Llama 3.2".to_string(),
                            context_window: Some(8192),
                        },
                        ModelInfo {
                            id: ModelId::new("qwen3:14b").expect("hardcoded valid model ID"),
                            name: "Qwen3 14B".to_string(),
                            context_window: Some(32768),
                        },
                        ModelInfo {
                            id: ModelId::new("mistral").expect("hardcoded valid model ID"),
                            name: "Mistral".to_string(),
                            context_window: Some(8192),
                        },
                    ])
                }
            }
        })
    }
}

/// State for tracking stream parsing
struct StreamState {
    model: Option<ModelId>,
    started: bool,
    completed: bool,
    usage: Option<TokenUsage>,
}

impl StreamState {
    fn new(model: String) -> Self {
        Self {
            model: ModelId::new(model).ok(),
            started: false,
            completed: false,
            usage: None,
        }
    }
}

/// Parse an Ollama streaming response
fn parse_ollama_response(
    response: ollama_rs::generation::chat::ChatMessageResponse,
    state: &mut StreamState,
) -> Vec<Result<StreamEvent, StreamError>> {
    let mut events = Vec::new();

    // Send started event on first chunk
    if !state.started {
        state.started = true;
        events.push(Ok(StreamEvent::Started {
            metadata: CompletionMetadata {
                model: state.model.clone(),
                stop_reason: None,
                usage: None,
            },
        }));
        events.push(Ok(StreamEvent::ContentBlockStart {
            index: 0,
            block_type: ContentBlockType::Text,
        }));
    }

    // Extract text content from the message (message is not an Option)
    let content = &response.message.content;
    if !content.is_empty() {
        events.push(Ok(StreamEvent::Delta(StreamDelta::TextDelta {
            text: content.clone(),
        })));
    }

    // Check if this is the final response
    if response.done {
        // Build usage from final_data if available
        if let Some(ref final_data) = response.final_data {
            state.usage = Some(TokenUsage {
                input_tokens: final_data.prompt_eval_count as u32,
                output_tokens: final_data.eval_count as u32,
            });
        }

        events.push(Ok(StreamEvent::ContentBlockStop { index: 0 }));
    }

    events
}

//! OpenAI provider implementation using the `async-openai` crate.
//!
//! Supports GPT-4o, GPT-5.x, and o-series reasoning models. Handles:
//! - Streaming chat completions with tool/function calling
//! - Reasoning effort configuration for models that support it
//! - Non-streaming fallback for o3/o3-mini models

use std::future::Future;
use std::pin::Pin;

use async_openai::config::OpenAIConfig;
use async_openai::types::{
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
    ChatCompletionRequestUserMessageArgs, ChatCompletionToolArgs, ChatCompletionToolType,
    CreateChatCompletionRequestArgs, FunctionObjectArgs,
};
use async_openai::Client;

use crate::config::OpenAiConfig;
use crate::error::{
    is_content_policy_message, is_context_window_message, AuthErrorKind, ProviderError,
    StreamError, StreamErrorKind,
};
use crate::streaming::{
    CompletionMetadata, ContentBlockType, StopReason, StreamDelta, StreamEvent, StreamHandle,
    TokenUsage,
};
use crate::types::{ContentBlock, Message, ModelId, Role, ToolCallId, ToolName};

use super::{
    CompletionRequest, ModelInfo, Provider, ProviderCapabilities, ProviderContext, ProviderInfo,
};

/// OpenAI provider
pub struct OpenAiProvider {
    config: OpenAiConfig,
    client: Client<OpenAIConfig>,
    info: ProviderInfo,
}

impl OpenAiProvider {
    /// Create a new OpenAI provider
    pub fn new(config: OpenAiConfig) -> Result<Self, ProviderError> {
        let supports_streaming = config.model.supports_streaming();

        // Create async-openai client with API key
        let openai_config = OpenAIConfig::new().with_api_key(config.api_key.as_str());
        let client = Client::with_config(openai_config);

        let info = ProviderInfo {
            kind: super::ProviderKind::OpenAi,
            name: "OpenAI",
            capabilities: ProviderCapabilities {
                streaming: supports_streaming,
                tools: true,
                vision: true,
                extended_thinking: config.model.supports_reasoning(),
                max_context_tokens: Some(128_000),
            },
        };

        Ok(Self {
            config,
            client,
            info,
        })
    }

    /// Convert our messages to OpenAI format
    fn convert_messages(
        &self,
        messages: &[Message],
    ) -> Result<Vec<ChatCompletionRequestMessage>, ProviderError> {
        let mut openai_messages = Vec::new();

        for msg in messages {
            match msg.role {
                Role::System => {
                    // Extract text from content blocks
                    let text = msg
                        .content
                        .iter()
                        .filter_map(|b| match b {
                            ContentBlock::Text { text } => Some(text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");

                    if !text.is_empty() {
                        openai_messages.push(
                            ChatCompletionRequestSystemMessageArgs::default()
                                .content(text)
                                .build()
                                .map_err(|e| ProviderError::InvalidRequest {
                                    provider: super::ProviderKind::OpenAi,
                                    message: format!("Failed to build system message: {}", e),
                                })?
                                .into(),
                        );
                    }
                }
                Role::User => {
                    let text = msg
                        .content
                        .iter()
                        .filter_map(|b| match b {
                            ContentBlock::Text { text } => Some(text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");

                    if !text.is_empty() {
                        openai_messages.push(
                            ChatCompletionRequestUserMessageArgs::default()
                                .content(text)
                                .build()
                                .map_err(|e| ProviderError::InvalidRequest {
                                    provider: super::ProviderKind::OpenAi,
                                    message: format!("Failed to build user message: {}", e),
                                })?
                                .into(),
                        );
                    }
                }
                Role::Assistant => {
                    let text = msg
                        .content
                        .iter()
                        .filter_map(|b| match b {
                            ContentBlock::Text { text } => Some(text.as_str()),
                            ContentBlock::Thinking { text } => Some(text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");

                    // Check for tool calls
                    let tool_calls: Vec<_> = msg
                        .content
                        .iter()
                        .filter_map(|b| match b {
                            ContentBlock::ToolUse { id, name, input } => {
                                Some((id.clone(), name.clone(), input.clone()))
                            }
                            _ => None,
                        })
                        .collect();

                    let mut builder = ChatCompletionRequestAssistantMessageArgs::default();

                    if !text.is_empty() {
                        builder.content(text);
                    }

                    if !tool_calls.is_empty() {
                        let tc: Vec<_> = tool_calls
                            .into_iter()
                            .map(|(id, name, input)| {
                                async_openai::types::ChatCompletionMessageToolCall {
                                    id: id.as_str().to_owned(),
                                    r#type: ChatCompletionToolType::Function,
                                    function: async_openai::types::FunctionCall {
                                        name: name.as_str().to_owned(),
                                        arguments: serde_json::to_string(&input)
                                            .unwrap_or_default(),
                                    },
                                }
                            })
                            .collect();
                        builder.tool_calls(tc);
                    }

                    openai_messages.push(
                        builder
                            .build()
                            .map_err(|e| ProviderError::InvalidRequest {
                                provider: super::ProviderKind::OpenAi,
                                message: format!("Failed to build assistant message: {}", e),
                            })?
                            .into(),
                    );
                }
                Role::Tool => {
                    // Tool results
                    for block in &msg.content {
                        if let ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        } = block
                        {
                            let content_str = match content {
                                crate::types::ToolResultContent::Text(s) => s.clone(),
                            };
                            openai_messages.push(
                                ChatCompletionRequestToolMessageArgs::default()
                                    .tool_call_id(tool_use_id.as_str())
                                    .content(content_str)
                                    .build()
                                    .map_err(|e| ProviderError::InvalidRequest {
                                        provider: super::ProviderKind::OpenAi,
                                        message: format!("Failed to build tool message: {}", e),
                                    })?
                                    .into(),
                            );
                        }
                    }
                }
            }
        }

        Ok(openai_messages)
    }

    /// Convert our tools to OpenAI format
    fn convert_tools(
        &self,
        tools: &[crate::tool::ToolDefinition],
    ) -> Result<Vec<async_openai::types::ChatCompletionTool>, ProviderError> {
        tools
            .iter()
            .map(|t| {
                let function = FunctionObjectArgs::default()
                    .name(t.name.as_str())
                    .description(t.description.clone())
                    .parameters(t.input_schema.to_value())
                    .build()
                    .map_err(|e| ProviderError::InvalidRequest {
                        provider: super::ProviderKind::OpenAi,
                        message: format!("Failed to build function: {}", e),
                    })?;

                ChatCompletionToolArgs::default()
                    .r#type(ChatCompletionToolType::Function)
                    .function(function)
                    .build()
                    .map_err(|e| ProviderError::InvalidRequest {
                        provider: super::ProviderKind::OpenAi,
                        message: format!("Failed to build tool: {}", e),
                    })
            })
            .collect()
    }
}

impl Provider for OpenAiProvider {
    fn info(&self) -> &ProviderInfo {
        &self.info
    }

    fn complete_stream(
        &self,
        request: CompletionRequest,
        ctx: ProviderContext,
    ) -> Pin<Box<dyn Future<Output = Result<StreamHandle, ProviderError>> + Send + '_>> {
        Box::pin(async move {
            // Check if model supports streaming
            if !self.config.model.supports_streaming() {
                return Err(ProviderError::StreamingNotSupported {
                    provider: super::ProviderKind::OpenAi,
                    model: self.config.model.as_str().to_owned(),
                });
            }

            let model = self.config.model.as_str();

            // Convert messages
            let mut messages = Vec::new();

            // Add system prompt if present
            if let Some(ref system) = request.system {
                if !system.as_str().is_empty() {
                    messages.push(
                        ChatCompletionRequestSystemMessageArgs::default()
                            .content(system.as_str())
                            .build()
                            .map_err(|e| ProviderError::InvalidRequest {
                                provider: super::ProviderKind::OpenAi,
                                message: format!("Failed to build system message: {}", e),
                            })?
                            .into(),
                    );
                }
            }

            // Add conversation messages
            messages.extend(self.convert_messages(&request.messages)?);

            // Build request
            let mut req_builder = CreateChatCompletionRequestArgs::default();
            req_builder
                .model(model)
                .messages(messages)
                .max_completion_tokens(request.config.max_tokens.get())
                .stream(true);

            // Add temperature if supported
            if let Some(temp) = request.config.temperature {
                if self.config.model.supports_temperature() {
                    req_builder.temperature(temp.get());
                }
            }

            // Add tools if present
            if !request.tools.is_empty() {
                let tools = self.convert_tools(&request.tools)?;
                req_builder.tools(tools);
            }

            // Add stop sequences
            if !request.config.stop_sequences.is_empty() {
                req_builder.stop(request.config.stop_sequences.clone());
            }

            let openai_request =
                req_builder
                    .build()
                    .map_err(|e| ProviderError::InvalidRequest {
                        provider: super::ProviderKind::OpenAi,
                        message: format!("Failed to build request: {}", e),
                    })?;

            // Create stream
            let stream = self
                .client
                .chat()
                .create_stream(openai_request)
                .await
                .map_err(|e| {
                    let msg = e.to_string();
                    if msg.contains("401") || msg.contains("Unauthorized") {
                        ProviderError::Auth {
                            provider: super::ProviderKind::OpenAi,
                            kind: AuthErrorKind::Rejected,
                            message: msg,
                        }
                    } else if msg.contains("429") || msg.contains("rate") {
                        ProviderError::RateLimited {
                            provider: super::ProviderKind::OpenAi,
                            retry_after: None,
                        }
                    } else if msg.contains("model") && msg.contains("not found") {
                        ProviderError::ModelNotFound {
                            provider: super::ProviderKind::OpenAi,
                            model: model.to_owned(),
                        }
                    } else if is_context_window_message(&msg) {
                        ProviderError::ContextWindowExceeded {
                            provider: super::ProviderKind::OpenAi,
                            message: msg,
                            context_window: None,
                            tokens_used: None,
                        }
                    } else if is_content_policy_message(&msg) {
                        ProviderError::ContentPolicyViolation {
                            provider: super::ProviderKind::OpenAi,
                            message: msg,
                        }
                    } else {
                        ProviderError::InvalidRequest {
                            provider: super::ProviderKind::OpenAi,
                            message: msg,
                        }
                    }
                })?;

            let event_stream = super::stream_adapter::buffered_sdk_stream(
                stream,
                ctx.cancellation.clone(),
                StreamState::default(),
                |item, state| match item {
                    Ok(response) => parse_openai_chunk(response, state),
                    Err(e) => vec![Err(StreamError::ConnectionLost {
                        kind: StreamErrorKind::TransportError,
                        message: e.to_string(),
                    })],
                },
                |state| {
                    if !state.completed {
                        Some(Ok(StreamEvent::Completed {
                            metadata: CompletionMetadata {
                                model: state.model.clone(),
                                stop_reason: state.stop_reason,
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
            // Safety: all model IDs below are hardcoded valid strings (alphanumeric + hyphens)
            Ok(vec![
                ModelInfo {
                    id: ModelId::new("gpt-4o").expect("hardcoded valid model ID"),
                    name: "GPT-4o".to_owned(),
                    context_window: Some(128_000),
                },
                ModelInfo {
                    id: ModelId::new("gpt-4o-mini").expect("hardcoded valid model ID"),
                    name: "GPT-4o Mini".to_owned(),
                    context_window: Some(128_000),
                },
                ModelInfo {
                    id: ModelId::new("o1").expect("hardcoded valid model ID"),
                    name: "o1".to_owned(),
                    context_window: Some(200_000),
                },
                ModelInfo {
                    id: ModelId::new("o1-mini").expect("hardcoded valid model ID"),
                    name: "o1 Mini".to_owned(),
                    context_window: Some(128_000),
                },
            ])
        })
    }
}

/// State for tracking stream parsing
#[derive(Default)]
struct StreamState {
    model: Option<ModelId>,
    stop_reason: Option<StopReason>,
    usage: Option<TokenUsage>,
    current_tool_call_id: Option<String>,
    current_tool_name: Option<String>,
    started: bool,
    completed: bool,
}

/// Parse an OpenAI streaming chunk
fn parse_openai_chunk(
    response: async_openai::types::CreateChatCompletionStreamResponse,
    state: &mut StreamState,
) -> Vec<Result<StreamEvent, StreamError>> {
    let mut events = Vec::new();

    // Track model
    if state.model.is_none() {
        state.model = ModelId::new(&response.model).ok();
    }

    // Send started event on first chunk
    if !state.started {
        state.started = true;
        events.push(Ok(StreamEvent::Started {
            metadata: CompletionMetadata {
                model: ModelId::new(&response.model).ok(),
                stop_reason: None,
                usage: None,
            },
        }));
        events.push(Ok(StreamEvent::ContentBlockStart {
            index: 0,
            block_type: ContentBlockType::Text,
        }));
    }

    for choice in &response.choices {
        // Check finish reason
        if let Some(ref reason) = choice.finish_reason {
            #[allow(unreachable_patterns)] // forward-compat: async-openai may add variants
            let reason = match reason {
                async_openai::types::FinishReason::Stop => StopReason::EndTurn,
                async_openai::types::FinishReason::Length => StopReason::MaxTokens,
                async_openai::types::FinishReason::ToolCalls => StopReason::ToolUse,
                async_openai::types::FinishReason::ContentFilter => StopReason::ContentFilter,
                async_openai::types::FinishReason::FunctionCall => StopReason::ToolUse,
                _ => StopReason::EndTurn,
            };
            state.stop_reason = Some(reason);
        }

        // Process delta
        let delta = &choice.delta;

        // Text content
        if let Some(ref content) = delta.content {
            if !content.is_empty() {
                events.push(Ok(StreamEvent::Delta(StreamDelta::TextDelta {
                    text: content.clone(),
                })));
            }
        }

        // Tool calls
        if let Some(ref tool_calls) = delta.tool_calls {
            for tc in tool_calls {
                // Tool call start
                if let Some(ref id) = tc.id {
                    state.current_tool_call_id = Some(id.clone());
                }

                if let Some(ref function) = tc.function {
                    // Function name (tool start)
                    if let Some(ref name) = function.name {
                        state.current_tool_name = Some(name.clone());

                        events.push(Ok(StreamEvent::ContentBlockStart {
                            index: tc.index as usize,
                            block_type: ContentBlockType::ToolUse,
                        }));

                        if let Some(ref id) = state.current_tool_call_id {
                            events.push(Ok(StreamEvent::Delta(StreamDelta::ToolUseStart {
                                id: ToolCallId::new(id),
                                // Safety: "unknown" is a valid tool name (alphanumeric)
                                name: ToolName::new(name).unwrap_or_else(|_| {
                                    ToolName::new("unknown").expect("hardcoded valid tool name")
                                }),
                            })));
                        }
                    }

                    // Function arguments (tool input delta)
                    if let Some(ref args) = function.arguments {
                        if !args.is_empty() {
                            if let Some(ref id) = state.current_tool_call_id {
                                events.push(Ok(StreamEvent::Delta(StreamDelta::ToolInputDelta {
                                    id: ToolCallId::new(id),
                                    partial_json: args.clone(),
                                })));
                            }
                        }
                    }
                }
            }
        }
    }

    // Check for usage in response
    if let Some(ref usage) = response.usage {
        state.usage = Some(TokenUsage {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
        });
    }

    events
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_chunk(content: &str) -> async_openai::types::CreateChatCompletionStreamResponse {
        serde_json::from_value(serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1234567890_u64,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "delta": { "content": content },
                "finish_reason": null
            }]
        }))
        .expect("valid text chunk JSON")
    }

    fn finish_chunk(reason: &str) -> async_openai::types::CreateChatCompletionStreamResponse {
        serde_json::from_value(serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1234567890_u64,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": reason
            }]
        }))
        .expect("valid finish chunk JSON")
    }

    fn tool_call_chunk(
        id: &str,
        name: &str,
        args: &str,
    ) -> async_openai::types::CreateChatCompletionStreamResponse {
        serde_json::from_value(serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1234567890_u64,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": id,
                        "function": { "name": name, "arguments": args }
                    }]
                },
                "finish_reason": null
            }]
        }))
        .expect("valid tool call chunk JSON")
    }

    #[test]
    fn parse_text_content() {
        let mut state = StreamState::default();
        let events = parse_openai_chunk(text_chunk("Hello"), &mut state);

        // First chunk: Started + ContentBlockStart + TextDelta
        assert!(state.started);
        assert!(events.iter().any(|e| matches!(
            e,
            Ok(StreamEvent::Delta(StreamDelta::TextDelta { text })) if text == "Hello"
        )));
    }

    #[test]
    fn parse_finish_reason_stop() {
        let mut state = StreamState {
            started: true,
            ..Default::default()
        };

        drop(parse_openai_chunk(finish_chunk("stop"), &mut state));
        assert_eq!(state.stop_reason, Some(StopReason::EndTurn));
    }

    #[test]
    fn parse_finish_reason_tool_calls() {
        let mut state = StreamState {
            started: true,
            ..Default::default()
        };

        drop(parse_openai_chunk(finish_chunk("tool_calls"), &mut state));
        assert_eq!(state.stop_reason, Some(StopReason::ToolUse));
    }

    #[test]
    fn parse_tool_call_start_and_arguments() {
        let mut state = StreamState {
            started: true,
            ..Default::default()
        };

        let events = parse_openai_chunk(
            tool_call_chunk("call_abc", "read_file", r#"{"path":"/tmp"}"#),
            &mut state,
        );

        assert!(events.iter().any(|e| matches!(
            e,
            Ok(StreamEvent::Delta(StreamDelta::ToolUseStart { name, .. }))
                if name.as_str() == "read_file"
        )));
        assert!(events.iter().any(|e| matches!(
            e,
            Ok(StreamEvent::Delta(StreamDelta::ToolInputDelta { partial_json, .. }))
                if partial_json == r#"{"path":"/tmp"}"#
        )));
    }
}

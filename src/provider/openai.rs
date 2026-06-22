//! OpenAI provider implementation using the `async-openai` crate.
//!
//! Supports GPT-4o, GPT-5.x, and o-series reasoning models. Handles:
//! - Streaming chat completions with tool/function calling
//! - Reasoning effort configuration for models that support it
//! - Non-streaming fallback for o3/o3-mini models

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
    ChatCompletionRequestUserMessageArgs, ChatCompletionTool, ChatCompletionTools,
    CreateChatCompletionRequestArgs, CreateChatCompletionStreamResponse, FinishReason,
    FunctionCall, FunctionObjectArgs, StopConfiguration,
};

use crate::config::OpenAiConfig;
use crate::error::{
    AuthErrorKind, ProviderError, StreamError, StreamErrorKind, is_content_policy_message,
    is_context_window_message,
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

    /// Convert our messages to OpenAI format.
    ///
    /// System messages are excluded because OpenAI only accepts a single system
    /// message. The caller is responsible for extracting the system prompt from
    /// `request.system` or from a `Role::System` message and adding it once.
    fn convert_messages(
        &self,
        messages: &[Message],
    ) -> Result<Vec<ChatCompletionRequestMessage>, ProviderError> {
        let mut openai_messages = Vec::new();

        for msg in messages {
            match msg.role {
                Role::System => {
                    // System messages are handled separately in `complete_stream`
                    // to enforce a single system message.
                }
                Role::User => {
                    let text = msg
                        .content
                        .iter()
                        .filter_map(|b| match b {
                            ContentBlock::Text { text } => Some(text.as_str()),
                            ContentBlock::Thinking { .. }
                            | ContentBlock::ToolUse { .. }
                            | ContentBlock::ToolResult { .. } => None,
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
                                    message: format!("Failed to build user message: {e}"),
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
                            ContentBlock::ToolUse { .. } | ContentBlock::ToolResult { .. } => None,
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
                            ContentBlock::Text { .. }
                            | ContentBlock::Thinking { .. }
                            | ContentBlock::ToolResult { .. } => None,
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
                                ChatCompletionMessageToolCalls::Function(
                                    ChatCompletionMessageToolCall {
                                        id: id.as_str().to_owned(),
                                        function: FunctionCall {
                                            name: name.as_str().to_owned(),
                                            arguments: serde_json::to_string(&input)
                                                .unwrap_or_default(),
                                        },
                                    },
                                )
                            })
                            .collect();
                        builder.tool_calls(tc);
                    }

                    openai_messages.push(
                        builder
                            .build()
                            .map_err(|e| ProviderError::InvalidRequest {
                                provider: super::ProviderKind::OpenAi,
                                message: format!("Failed to build assistant message: {e}"),
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
                                        message: format!("Failed to build tool message: {e}"),
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

    /// Build the complete OpenAI message list, including a single system message.
    ///
    /// OpenAI only accepts one system message. We use `request.system` when it is
    /// set, otherwise fall back to the first `Role::System` message in the
    /// conversation. Any remaining system messages are dropped so they are not
    /// duplicated.
    fn build_messages(
        &self,
        request: &CompletionRequest,
    ) -> Result<Vec<ChatCompletionRequestMessage>, ProviderError> {
        let mut messages = Vec::new();

        let system_text = request
            .system
            .as_ref()
            .filter(|s| !s.as_str().is_empty())
            .map(|s| s.as_str().to_owned())
            .or_else(|| {
                request
                    .messages
                    .iter()
                    .find(|m| m.role == Role::System)
                    .and_then(|m| {
                        let text = m
                            .content
                            .iter()
                            .filter_map(|b| match b {
                                ContentBlock::Text { text } => Some(text.as_str()),
                                ContentBlock::Thinking { .. }
                                | ContentBlock::ToolUse { .. }
                                | ContentBlock::ToolResult { .. } => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n");
                        if text.is_empty() { None } else { Some(text) }
                    })
            });

        if let Some(system) = system_text {
            messages.push(
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(system)
                    .build()
                    .map_err(|e| ProviderError::InvalidRequest {
                        provider: super::ProviderKind::OpenAi,
                        message: format!("Failed to build system message: {e}"),
                    })?
                    .into(),
            );
        }

        let conversation_messages: Vec<_> = request
            .messages
            .iter()
            .filter(|m| m.role != Role::System)
            .cloned()
            .collect();
        messages.extend(self.convert_messages(&conversation_messages)?);

        Ok(messages)
    }

    /// Convert our tools to OpenAI format
    fn convert_tools(
        &self,
        tools: &[crate::tool::ToolDefinition],
    ) -> Result<Vec<ChatCompletionTools>, ProviderError> {
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
                        message: format!("Failed to build function: {e}"),
                    })?;

                Ok(ChatCompletionTools::Function(ChatCompletionTool {
                    function,
                }))
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

            let messages = self.build_messages(&request)?;

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
                req_builder.stop(StopConfiguration::StringArray(
                    request.config.stop_sequences.clone(),
                ));
            }

            let openai_request =
                req_builder
                    .build()
                    .map_err(|e| ProviderError::InvalidRequest {
                        provider: super::ProviderKind::OpenAi,
                        message: format!("Failed to build request: {e}"),
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
        #[allow(
            clippy::expect_used,
            reason = "listed OpenAI model IDs are hardcoded valid constants"
        )]
        Box::pin(async move {
            // All model IDs below are hardcoded valid strings
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
    tool_call_ids: HashMap<usize, String>,
    started: bool,
    completed: bool,
}

/// Parse an OpenAI streaming chunk
fn parse_openai_chunk(
    response: CreateChatCompletionStreamResponse,
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
            #[allow(
                unreachable_patterns,
                reason = "async-openai FinishReason may add variants"
            )]
            let reason = match reason {
                FinishReason::Stop => StopReason::EndTurn,
                FinishReason::Length => StopReason::MaxTokens,
                FinishReason::ToolCalls => StopReason::ToolUse,
                FinishReason::ContentFilter => StopReason::ContentFilter,
                FinishReason::FunctionCall => StopReason::ToolUse,
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
                let tc_index = tc.index as usize;

                // Tool call start
                if let Some(ref id) = tc.id {
                    state.tool_call_ids.insert(tc_index, id.clone());
                }

                if let Some(ref function) = tc.function {
                    // Function name (tool start)
                    if let Some(ref name) = function.name {
                        let name = match ToolName::new(name) {
                            Ok(name) => name,
                            Err(e) => {
                                events.push(Err(StreamError::Deserialize {
                                    message: format!("invalid OpenAI tool name: {e}"),
                                    raw_data: None,
                                }));
                                continue;
                            }
                        };

                        events.push(Ok(StreamEvent::ContentBlockStart {
                            index: tc_index,
                            block_type: ContentBlockType::ToolUse,
                        }));

                        if let Some(id) = state.tool_call_ids.get(&tc_index) {
                            events.push(Ok(StreamEvent::Delta(StreamDelta::ToolUseStart {
                                id: ToolCallId::new(id),
                                name,
                            })));
                        }
                    }

                    // Function arguments (tool input delta)
                    if let Some(ref args) = function.arguments {
                        if !args.is_empty() {
                            if let Some(id) = state.tool_call_ids.get(&tc_index) {
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

    fn text_chunk(content: &str) -> CreateChatCompletionStreamResponse {
        serde_json::from_value(serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1_234_567_890_u64,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "delta": { "content": content },
                "finish_reason": null
            }]
        }))
        .expect("valid text chunk JSON")
    }

    fn finish_chunk(reason: &str) -> CreateChatCompletionStreamResponse {
        serde_json::from_value(serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1_234_567_890_u64,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": reason
            }]
        }))
        .expect("valid finish chunk JSON")
    }

    fn tool_call_chunk(id: &str, name: &str, args: &str) -> CreateChatCompletionStreamResponse {
        serde_json::from_value(serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1_234_567_890_u64,
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

    /// OpenAI only accepts one system message. If both `request.system` and a
    /// `Role::System` message are supplied, we must not emit a duplicate.
    #[test]
    fn no_duplicate_system_message() {
        let config = OpenAiConfig {
            api_key: crate::config::ApiKey::new("test-key").unwrap(),
            model: crate::config::OpenAiModel::Gpt4o,
            max_tokens: crate::types::MaxTokens::new(100).unwrap(),
            temperature: None,
            reasoning: None,
            structured_output: false,
        };
        let provider = OpenAiProvider::new(config).unwrap();

        let request = CompletionRequest::new(
            crate::types::ModelId::new("gpt-4o").unwrap(),
            vec![Message::new(
                Role::System,
                vec![ContentBlock::Text {
                    text: "from message".into(),
                }],
            )],
        )
        .with_system(crate::types::SystemPrompt::new("from request"));

        let messages = provider.build_messages(&request).unwrap();

        let system_count = messages
            .iter()
            .filter(|m| matches!(m, ChatCompletionRequestMessage::System(_)))
            .count();
        assert_eq!(system_count, 1, "expected exactly one system message");

        let user_count = messages
            .iter()
            .filter(|m| matches!(m, ChatCompletionRequestMessage::User(_)))
            .count();
        assert_eq!(
            user_count, 0,
            "system message should not be converted to user"
        );
    }
}

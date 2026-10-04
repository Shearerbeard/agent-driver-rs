//! Chat Completions API surface of the OpenAI provider.
//!
//! Streaming chat completions with tool/function calling, reasoning
//! deltas from OpenAI-compatible endpoints, and the BYOT chunk wrapper
//! that keeps third-party reasoning fields alive through deserialization.

use std::collections::HashMap;
use std::pin::Pin;

use crate::error::{ProviderError, StreamError, StreamErrorKind};
use crate::provider::{CompletionRequest, ProviderContext, ProviderKind};
use crate::streaming::{
    CompletionMetadata, ContentBlockType, StopReason, StreamDelta, StreamEvent, StreamHandle,
    TokenUsage,
};
use crate::types::{ContentBlock, Message, ModelId, Role, ToolCallId, ToolName};
use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
    ChatCompletionRequestUserMessageArgs, ChatCompletionStreamOptions,
    ChatCompletionStreamResponseDelta, ChatCompletionTool, ChatCompletionTools, CompletionUsage,
    CreateChatCompletionRequestArgs, FinishReason, FunctionCall, FunctionObjectArgs,
    StopConfiguration,
};

use super::OpenAiProvider;

/// A streamed chat-completion chunk that also carries reasoning deltas.
///
/// The OpenAI-spec delta struct has no `reasoning_content` field, so the
/// typed parse silently drops the field vLLM-compatible servers (BaseTen,
/// OpenRouter, local vLLM) stream for reasoning models. This wrapper keeps
/// the spec fields via `#[serde(flatten)]` and captures both reasoning
/// spellings seen in the wild: `reasoning_content` (vLLM, DeepSeek) and
/// `reasoning` (OpenRouter).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ReasoningAwareChunk {
    /// The model that produced the chunk (drives the Started metadata)
    pub model: String,
    /// The per-choice payloads; OpenAI-compatible servers stream one
    pub choices: Vec<ReasoningAwareChoice>,
    /// Only present when the request sets `stream_options.include_usage`
    #[serde(default)]
    pub usage: Option<CompletionUsage>,
}

/// One streamed choice: spec fields plus the reasoning capture.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ReasoningAwareChoice {
    /// The streamed content for this choice
    pub delta: ReasoningAwareDelta,
    /// The terminal reason, present only on the final chunk of a choice
    #[serde(default)]
    pub finish_reason: Option<FinishReason>,
}

/// The streamed delta: reasoning capture plus the flattened spec struct.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ReasoningAwareDelta {
    /// Chain-of-thought text (vLLM / DeepSeek convention)
    #[serde(default)]
    pub reasoning_content: Option<String>,
    /// Alternate reasoning spelling (OpenRouter convention)
    #[serde(default)]
    pub reasoning: Option<String>,
    #[serde(flatten)]
    pub inner: ChatCompletionStreamResponseDelta,
}

#[allow(
    clippy::multiple_inherent_impl,
    reason = "the OpenAI impl is split per API surface (mod, chat, responses); the lint targets accidental fragmentation"
)]
impl OpenAiProvider {
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
                            | ContentBlock::EncryptedReasoning { .. }
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
                                    provider: ProviderKind::OpenAi,
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
                            ContentBlock::EncryptedReasoning { .. }
                            | ContentBlock::ToolUse { .. }
                            | ContentBlock::ToolResult { .. } => None,
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
                            | ContentBlock::EncryptedReasoning { .. }
                            | ContentBlock::ToolResult { .. } => None,
                        })
                        .collect();

                    // An assistant message with neither content nor tool
                    // calls (every block dropped in serialization, e.g.
                    // an encrypted-reasoning-only turn) is omitted.
                    if text.is_empty() && tool_calls.is_empty() {
                        continue;
                    }

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
                                provider: ProviderKind::OpenAi,
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
                                        provider: ProviderKind::OpenAi,
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
                                | ContentBlock::EncryptedReasoning { .. }
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
                        provider: ProviderKind::OpenAi,
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
                        provider: ProviderKind::OpenAi,
                        message: format!("Failed to build function: {e}"),
                    })?;

                Ok(ChatCompletionTools::Function(ChatCompletionTool {
                    function,
                }))
            })
            .collect()
    }

    /// Build the Chat Completions request body, enforcing the
    /// profile's model and capability contract.
    ///
    /// Separate from [`Self::complete_stream_chat`] so tests can pin the
    /// wire shape and so the request-time validation seam is single.
    fn build_chat_request(
        &self,
        request: &CompletionRequest,
    ) -> Result<async_openai::types::chat::CreateChatCompletionRequest, ProviderError> {
        let profile = &self.profile;

        // The provider pin is authoritative; a request naming a
        // different model must fail rather than be silently ignored.
        if request.model.as_str() != profile.model_id() {
            return Err(ProviderError::InvalidRequest {
                provider: ProviderKind::OpenAi,
                message: format!(
                    "request model {:?} does not match configured model {:?}",
                    request.model.as_str(),
                    profile.model_id()
                ),
            });
        }

        // Capability comes from the validated profile, not the model
        // enum, so a custom declaration is honored on this surface.
        if !profile.supports_streaming() {
            return Err(ProviderError::StreamingNotSupported {
                provider: ProviderKind::OpenAi,
                model: profile.model_id().to_owned(),
            });
        }

        let messages = self.build_messages(request)?;

        let mut req_builder = CreateChatCompletionRequestArgs::default();
        req_builder
            .model(profile.model_id())
            .messages(messages)
            .max_completion_tokens(request.config.max_tokens.get())
            .stream(true)
            // OpenAI-compatible servers (vLLM, BaseTen) stream the
            // token-usage chunk only when this is set; without it the
            // shim's context-usage metering would read zeros.
            // `include_obfuscation` stays unset: it is OpenAI-dialect
            // only, and compatible endpoints (Fireworks, BaseTen BYOT,
            // vLLM) reject the unknown field with a 400 before the
            // model runs (adr/A19; live-verified 2026-09-25).
            .stream_options(ChatCompletionStreamOptions {
                include_usage: Some(true),
                include_obfuscation: None,
            });

        // A supplied temperature is either sent or rejected; the profile
        // already guaranteed capability, so this cannot silently discard
        // the request.
        if let Some(temp) = request.config.temperature {
            if !profile.supports_temperature() {
                return Err(ProviderError::InvalidRequest {
                    provider: ProviderKind::OpenAi,
                    message: format!(
                        "model {:?} on the Chat Completions surface does not support temperature",
                        profile.model_id()
                    ),
                });
            }
            req_builder.temperature(temp.get());
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

        req_builder
            .build()
            .map_err(|e| ProviderError::InvalidRequest {
                provider: ProviderKind::OpenAi,
                message: format!("Failed to build request: {e}"),
            })
    }

    /// Run one chat-completions streaming request (the `Provider` trait
    /// dispatches here when the API surface is chat completions).
    pub(super) fn complete_stream_chat(
        &self,
        request: CompletionRequest,
        ctx: ProviderContext,
    ) -> Pin<Box<dyn Future<Output = Result<StreamHandle, ProviderError>> + Send + '_>> {
        Box::pin(async move {
            let profile = &self.profile;
            let model = profile.model_id();
            let openai_request = self.build_chat_request(&request)?;

            // Create stream. The BYOT form is what keeps reasoning alive:
            // the spec-typed `create_stream` would deserialize each chunk
            // into `CreateChatCompletionStreamResponse`, whose delta struct
            // has no `reasoning_content` field, and serde would silently
            // drop it. `create_stream_byot` deserializes into
            // [`ReasoningAwareChunk`] instead; the request is the same
            // spec type (stream(true) is already set on the builder).
            let stream = self
                .client
                .chat()
                .create_stream_byot::<_, ReasoningAwareChunk>(openai_request)
                .await
                .map_err(|e| super::map_request_error(e.to_string(), model))?;

            let event_stream = crate::provider::stream_adapter::buffered_sdk_stream(
                stream,
                ctx.cancellation.clone(),
                StreamState {
                    reasoning_output: profile.chat_reasoning_output(),
                    ..StreamState::default()
                },
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
    /// Preferred Chat reasoning output field for this connection.
    reasoning_output: crate::config::OpenAiChatReasoningOutput,
}

/// Parse an OpenAI streaming chunk
fn parse_openai_chunk(
    response: ReasoningAwareChunk,
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

        // Reasoning deltas (vLLM `reasoning_content`, OpenRouter
        // `reasoning`). Mapped to the driver's ThinkingDelta so the agent
        // loop folds them as thinking blocks; they never surface as
        // answer text. The connection's selected output field, when set,
        // decides which spelling wins if a server populates both; `None`
        // and the absence of a selected field fall back to the
        // vLLM-canonical `reasoning_content`, preserving the tolerant
        // BYOT behavior.
        let prefer_alt = matches!(
            state.reasoning_output,
            crate::config::OpenAiChatReasoningOutput::Reasoning
        );
        let (first, second) = if prefer_alt {
            (&delta.reasoning, &delta.reasoning_content)
        } else {
            (&delta.reasoning_content, &delta.reasoning)
        };
        let reasoning_text = match (first, second) {
            (Some(a), _) if !a.is_empty() => Some(a),
            (_, Some(b)) if !b.is_empty() => Some(b),
            _ => None,
        };
        if let Some(rc) = reasoning_text {
            events.push(Ok(StreamEvent::Delta(StreamDelta::ThinkingDelta {
                thinking: rc.clone(),
            })));
        }

        // Text content
        if let Some(ref content) = delta.inner.content {
            if !content.is_empty() {
                events.push(Ok(StreamEvent::Delta(StreamDelta::TextDelta {
                    text: content.clone(),
                })));
            }
        }

        // Tool calls
        if let Some(ref tool_calls) = delta.inner.tool_calls {
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
    use crate::config::OpenAiConfig;

    fn text_chunk(content: &str) -> ReasoningAwareChunk {
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

    fn finish_chunk(reason: &str) -> ReasoningAwareChunk {
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

    fn tool_call_chunk(id: &str, name: &str, args: &str) -> ReasoningAwareChunk {
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

    /// vLLM-style reasoning deltas (`reasoning_content`) surface as
    /// ThinkingDelta and never as answer text.
    #[test]
    fn parse_reasoning_content_deltas() {
        let mut state = StreamState {
            started: true,
            ..Default::default()
        };

        let chunk: ReasoningAwareChunk = serde_json::from_value(serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1_234_567_890_u64,
            "model": "deepseek-ai/DeepSeek-V4-Pro",
            "choices": [{
                "index": 0,
                "delta": { "reasoning_content": "Let me think step by step" },
                "finish_reason": null
            }]
        }))
        .expect("valid reasoning chunk JSON");

        let events = parse_openai_chunk(chunk, &mut state);

        assert!(events.iter().any(|e| matches!(
            e,
            Ok(StreamEvent::Delta(StreamDelta::ThinkingDelta { thinking }))
                if thinking == "Let me think step by step"
        )));
        // The reasoning must not surface as answer text.
        assert!(
            events
                .iter()
                .all(|e| !matches!(e, Ok(StreamEvent::Delta(StreamDelta::TextDelta { .. }))))
        );
    }

    /// OpenRouter-style spelling (`reasoning`) maps the same way.
    #[test]
    fn parse_reasoning_alt_spelling() {
        let mut state = StreamState {
            started: true,
            ..Default::default()
        };

        let chunk: ReasoningAwareChunk = serde_json::from_value(serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1_234_567_890_u64,
            "model": "zai-org/GLM-5.2",
            "choices": [{
                "index": 0,
                "delta": { "reasoning": "Consider the constraints" },
                "finish_reason": null
            }]
        }))
        .expect("valid reasoning chunk JSON");

        let events = parse_openai_chunk(chunk, &mut state);
        assert!(events.iter().any(|e| matches!(
            e,
            Ok(StreamEvent::Delta(StreamDelta::ThinkingDelta { thinking }))
                if thinking == "Consider the constraints"
        )));
    }

    /// A reasoning chunk may also carry text in the same delta; both
    /// mappings fire and keep their own channels.
    #[test]
    fn parse_reasoning_and_text_coexist() {
        let mut state = StreamState {
            started: true,
            ..Default::default()
        };

        let chunk: ReasoningAwareChunk = serde_json::from_value(serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1_234_567_890_u64,
            "model": "zai-org/GLM-5.2",
            "choices": [{
                "index": 0,
                "delta": { "reasoning_content": "hmm", "content": "Answer" },
                "finish_reason": null
            }]
        }))
        .expect("valid mixed chunk JSON");

        let events = parse_openai_chunk(chunk, &mut state);
        assert!(events.iter().any(|e| matches!(
            e,
            Ok(StreamEvent::Delta(StreamDelta::ThinkingDelta { thinking }))
                if thinking == "hmm"
        )));
        assert!(events.iter().any(|e| matches!(
            e,
            Ok(StreamEvent::Delta(StreamDelta::TextDelta { text })) if text == "Answer"
        )));
    }

    /// The usage-only final chunk (vLLM sends `stream_options.include_usage`
    /// this way): empty choices, usage present. Locks in the wrapper's
    /// leniency on both the default `usage` and the empty-choices path.
    #[test]
    fn parse_usage_only_chunk() {
        let mut state = StreamState {
            started: true,
            ..Default::default()
        };

        let chunk: ReasoningAwareChunk = serde_json::from_value(serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1_234_567_890_u64,
            "model": "gpt-4o",
            "choices": [],
            "usage": { "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120 }
        }))
        .expect("valid usage-only chunk JSON");

        let events = parse_openai_chunk(chunk, &mut state);
        assert_eq!(
            state.usage.map(|u| (u.input_tokens, u.output_tokens)),
            Some((100, 20))
        );
        assert!(
            events
                .iter()
                .all(|e| !matches!(e, Ok(StreamEvent::Delta(..))))
        );
    }

    /// A choice that omits `finish_reason` entirely must parse (the
    /// wrapper's `#[serde(default)]` is new surface).
    #[test]
    fn parse_chunk_with_missing_finish_reason() {
        let mut state = StreamState {
            started: true,
            ..Default::default()
        };

        let chunk: ReasoningAwareChunk = serde_json::from_value(serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1_234_567_890_u64,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "delta": { "content": "hi" }
            }]
        }))
        .expect("valid chunk without finish_reason JSON");

        let events = parse_openai_chunk(chunk, &mut state);
        assert_eq!(state.stop_reason, None);
        assert!(events.iter().any(|e| matches!(
            e,
            Ok(StreamEvent::Delta(StreamDelta::TextDelta { text })) if text == "hi"
        )));
    }

    /// Empty reasoning strings are guarded: no ThinkingDelta fires.
    #[test]
    fn parse_empty_reasoning_string_is_silent() {
        let mut state = StreamState {
            started: true,
            ..Default::default()
        };

        let chunk: ReasoningAwareChunk = serde_json::from_value(serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1_234_567_890_u64,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "delta": { "reasoning_content": "" },
                "finish_reason": null
            }]
        }))
        .expect("valid empty-reasoning chunk JSON");

        let events = parse_openai_chunk(chunk, &mut state);
        assert!(
            events
                .iter()
                .all(|e| !matches!(e, Ok(StreamEvent::Delta(StreamDelta::ThinkingDelta { .. }))))
        );
    }

    /// Third-party extension keys on the delta must stay tolerated — the
    /// BYOT switch exists so servers richer than the spec still work, and
    /// a future `deny_unknown_fields` would break them.
    #[test]
    fn parse_chunk_tolerates_extension_fields() {
        let mut state = StreamState {
            started: true,
            ..Default::default()
        };

        let chunk: ReasoningAwareChunk = serde_json::from_value(serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1_234_567_890_u64,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "delta": { "content": "ok", "rejected_prediction_ids": [1] },
                "finish_reason": null
            }]
        }))
        .expect("valid extension-field chunk JSON");

        let events = parse_openai_chunk(chunk, &mut state);
        assert!(events.iter().any(|e| matches!(
            e,
            Ok(StreamEvent::Delta(StreamDelta::TextDelta { text })) if text == "ok"
        )));
    }

    /// A server sending both reasoning spellings yields ONE ThinkingDelta,
    /// preferring the vLLM-canonical field (Gate A round 1, finding 4).
    #[test]
    fn parse_both_reasoning_spellings_emit_one_delta() {
        let mut state = StreamState {
            started: true,
            ..Default::default()
        };

        let chunk: ReasoningAwareChunk = serde_json::from_value(serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1_234_567_890_u64,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "delta": { "reasoning_content": "vllm text", "reasoning": "alt text" },
                "finish_reason": null
            }]
        }))
        .expect("valid dual-spelling chunk JSON");

        let events = parse_openai_chunk(chunk, &mut state);
        let thinking: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                Ok(StreamEvent::Delta(StreamDelta::ThinkingDelta { thinking })) => {
                    Some(thinking.as_str())
                }
                _ => None,
            })
            .collect();
        assert_eq!(thinking, ["vllm text"]);
    }

    fn dual_spelling_chunk() -> ReasoningAwareChunk {
        serde_json::from_value(serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1_234_567_890_u64,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "delta": { "reasoning_content": "vllm text", "reasoning": "alt text" },
                "finish_reason": null
            }]
        }))
        .expect("valid dual-spelling chunk JSON")
    }

    fn single_thinking(events: &[Result<StreamEvent, StreamError>]) -> Option<&str> {
        let mut thinking = events.iter().filter_map(|e| match e {
            Ok(StreamEvent::Delta(StreamDelta::ThinkingDelta { thinking })) => {
                Some(thinking.as_str())
            }
            _ => None,
        });
        let first = thinking.next();
        assert!(thinking.next().is_none(), "expected one thinking delta");
        first
    }

    /// Selecting the `reasoning` output field flips the preference when a
    /// server populates both spellings.
    #[test]
    fn parse_both_spellings_prefers_selected_reasoning() {
        let mut state = StreamState {
            started: true,
            reasoning_output: crate::config::OpenAiChatReasoningOutput::Reasoning,
            ..Default::default()
        };
        let events = parse_openai_chunk(dual_spelling_chunk(), &mut state);
        assert_eq!(single_thinking(&events), Some("alt text"));
    }

    /// Selecting `reasoning_content` explicitly prefers the vLLM field.
    #[test]
    fn parse_both_spellings_prefers_selected_reasoning_content() {
        let mut state = StreamState {
            started: true,
            reasoning_output: crate::config::OpenAiChatReasoningOutput::ReasoningContent,
            ..Default::default()
        };
        let events = parse_openai_chunk(dual_spelling_chunk(), &mut state);
        assert_eq!(single_thinking(&events), Some("vllm text"));
    }

    /// With no selected preference (`None`), an unselected spelling still
    /// appears as thinking — the tolerant BYOT parser is preserved.
    #[test]
    fn parse_unselected_spelling_still_surfaces() {
        let mut state = StreamState {
            started: true,
            reasoning_output: crate::config::OpenAiChatReasoningOutput::None,
            ..Default::default()
        };
        let chunk: ReasoningAwareChunk = serde_json::from_value(serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1_234_567_890_u64,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "delta": { "reasoning": "only alt" },
                "finish_reason": null
            }]
        }))
        .expect("valid chunk JSON");
        let events = parse_openai_chunk(chunk, &mut state);
        assert_eq!(single_thinking(&events), Some("only alt"));
    }

    /// OpenAI only accepts one system message. If both `request.system` and a
    /// `Role::System` message are supplied, we must not emit a duplicate.
    #[test]
    fn no_duplicate_system_message() {
        let config = OpenAiConfig {
            api_key: crate::config::ApiKey::new("test-key").unwrap(),
            api: crate::config::OpenAiApi::default(),
            base_url: None,
            model: crate::config::OpenAiModel::Gpt4o,
            max_tokens: crate::types::MaxTokens::new(100).unwrap(),
            temperature: None,
            reasoning: None,
            structured_output: false,
            chat_reasoning_output: crate::config::OpenAiChatReasoningOutput::None,
            custom_capabilities: None,
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

    /// A19: the stream options this provider sends must not carry
    /// `include_obfuscation` — it is OpenAI-dialect only, and compatible
    /// endpoints (Fireworks, BaseTen BYOT, vLLM) reject the unknown field
    /// with a 400 before the model runs. Live-verified 2026-09-25: with
    /// the field absent and `max_completion_tokens` kept, Fireworks
    /// answered 200 and a full orchestrated DAG completed.
    #[test]
    fn stream_options_carry_usage_without_the_obfuscation_dialect() {
        let options = ChatCompletionStreamOptions {
            include_usage: Some(true),
            include_obfuscation: None,
        };
        let wire = serde_json::to_value(options).expect("stream options serialize");
        assert_eq!(wire["include_usage"], true);
        assert!(
            wire.get("include_obfuscation").is_none(),
            "the wire body must not name include_obfuscation, got: {wire}"
        );
    }

    fn chat_provider(
        model: crate::config::OpenAiModel,
        output: crate::config::OpenAiChatReasoningOutput,
        capabilities: Option<crate::config::CustomModelCapabilities>,
    ) -> OpenAiProvider {
        let config = OpenAiConfig {
            api_key: crate::config::ApiKey::new("test-key").unwrap(),
            api: crate::config::OpenAiApi::ChatCompletions,
            base_url: None,
            model,
            max_tokens: crate::types::MaxTokens::new(100).unwrap(),
            temperature: None,
            reasoning: None,
            structured_output: false,
            chat_reasoning_output: output,
            custom_capabilities: capabilities,
        };
        OpenAiProvider::new(config).unwrap()
    }

    /// Ordinary Chat wire body, asserted in full. The request-level
    /// `max_tokens` default (4096) is what reaches the wire, not the
    /// provider config's value.
    #[test]
    fn chat_request_body_is_streaming_with_usage() {
        let provider = chat_provider(
            crate::config::OpenAiModel::Gpt4o,
            crate::config::OpenAiChatReasoningOutput::None,
            None,
        );
        let request = CompletionRequest::new(
            ModelId::new("gpt-4o").unwrap(),
            vec![Message::user("hello")],
        );

        let wire = serde_json::to_value(provider.build_chat_request(&request).unwrap())
            .expect("request serializes");
        assert_eq!(
            wire,
            serde_json::json!({
                "messages": [{ "role": "user", "content": "hello" }],
                "model": "gpt-4o",
                "max_completion_tokens": 4096,
                "stream": true,
                "stream_options": { "include_usage": true }
            })
        );
    }

    /// A custom capability declaration is honored on the request body:
    /// the full body shows a temperature, which is absent by default.
    #[test]
    fn chat_request_body_sends_temperature_honoring_custom_capability() {
        let provider = chat_provider(
            crate::config::OpenAiModel::Custom("org/local".into()),
            crate::config::OpenAiChatReasoningOutput::None,
            Some(crate::config::CustomModelCapabilities::new(
                true, false, true,
            )),
        );
        let request = CompletionRequest::new(
            ModelId::new("org/local").unwrap(),
            vec![Message::user("hello")],
        )
        .with_config(
            crate::provider::CompletionConfig::new(crate::types::MaxTokens::new(64).unwrap())
                .with_temperature(crate::types::Temperature::new(0.5).unwrap()),
        );

        let wire = serde_json::to_value(provider.build_chat_request(&request).unwrap()).unwrap();
        assert_eq!(
            wire,
            serde_json::json!({
                "messages": [{ "role": "user", "content": "hello" }],
                "model": "org/local",
                "max_completion_tokens": 64,
                "stream": true,
                "stream_options": { "include_usage": true },
                "temperature": 0.5
            })
        );
    }

    /// The selected reasoning output preference is an output-side
    /// concern: both Chat output modes produce the same wire body. The
    /// full body is asserted, not a comparison of two implementation
    /// outputs.
    #[test]
    fn chat_request_body_fixture_across_output_modes() {
        let expected = serde_json::json!({
            "messages": [{ "role": "user", "content": "hello" }],
            "model": "gpt-4o",
            "max_completion_tokens": 4096,
            "stream": true,
            "stream_options": { "include_usage": true }
        });
        for output in [
            crate::config::OpenAiChatReasoningOutput::None,
            crate::config::OpenAiChatReasoningOutput::ReasoningContent,
            crate::config::OpenAiChatReasoningOutput::Reasoning,
        ] {
            let provider = chat_provider(crate::config::OpenAiModel::Gpt4o, output, None);
            let request = CompletionRequest::new(
                ModelId::new("gpt-4o").unwrap(),
                vec![Message::user("hello")],
            );
            let wire =
                serde_json::to_value(provider.build_chat_request(&request).unwrap()).unwrap();
            assert_eq!(
                wire, expected,
                "output mode {output:?} changed the wire body"
            );
        }
    }

    /// Full body with system prompt, assistant text, a tool call, and a
    /// tool result, showing message conversion on the wire.
    #[test]
    fn chat_request_body_fixture_with_tools_and_history() {
        let provider = chat_provider(
            crate::config::OpenAiModel::Gpt4o,
            crate::config::OpenAiChatReasoningOutput::None,
            None,
        );
        let request = CompletionRequest::new(
            ModelId::new("gpt-4o").unwrap(),
            vec![
                Message::user("what is 2+2?"),
                Message::new(
                    Role::Assistant,
                    vec![
                        ContentBlock::Text {
                            text: "Let me check.".into(),
                        },
                        ContentBlock::ToolUse {
                            id: ToolCallId::new("call_abc"),
                            name: ToolName::new("calculator").unwrap(),
                            input: serde_json::json!({"x": 2, "y": 2, "operation": "add"}),
                        },
                    ],
                ),
                Message::new(
                    Role::Tool,
                    vec![ContentBlock::ToolResult {
                        tool_use_id: ToolCallId::new("call_abc"),
                        content: crate::types::ToolResultContent::Text("4".into()),
                        is_error: false,
                    }],
                ),
            ],
        )
        .with_system(crate::types::SystemPrompt::new("be brief"));

        let wire = serde_json::to_value(provider.build_chat_request(&request).unwrap()).unwrap();
        assert_eq!(
            wire,
            serde_json::json!({
                "messages": [
                    { "role": "system", "content": "be brief" },
                    { "role": "user", "content": "what is 2+2?" },
                    {
                        "role": "assistant",
                        "content": "Let me check.",
                        "tool_calls": [{
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": "{\"x\":2,\"y\":2,\"operation\":\"add\"}"
                            }
                        }]
                    },
                    { "role": "tool", "tool_call_id": "call_abc", "content": "4" }
                ],
                "model": "gpt-4o",
                "max_completion_tokens": 4096,
                "stream": true,
                "stream_options": { "include_usage": true }
            })
        );
    }

    /// The request-time guard rejects a temperature the selected model
    /// and surface do not allow.
    #[test]
    fn chat_request_temperature_unsupported_fails_loud() {
        let provider = chat_provider(
            crate::config::OpenAiModel::Gpt5,
            crate::config::OpenAiChatReasoningOutput::None,
            None,
        );
        let request =
            CompletionRequest::new(ModelId::new("gpt-5").unwrap(), vec![Message::user("hello")])
                .with_config(
                    crate::provider::CompletionConfig::new(
                        crate::types::MaxTokens::new(64).unwrap(),
                    )
                    .with_temperature(crate::types::Temperature::new(1.0).unwrap()),
                );
        let err = provider.build_chat_request(&request).unwrap_err();
        assert!(
            matches!(err, ProviderError::InvalidRequest { .. }),
            "unsupported temperature must fail loud, got: {err:?}"
        );
    }
}

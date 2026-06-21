//! Ollama provider implementation using the `ollama-rs` crate.
//!
//! Connects to a local Ollama instance for running open-source models like Llama,
//! Qwen, Mistral, and DeepSeek. Supports streaming chat completions with configurable
//! context window size (`num_ctx`) and model keep-alive settings.
//!
//! ## Tool calling
//!
//! Ollama delivers tool calls fully-formed in the final chunk (`done: true`) with
//! `message.tool_calls` populated. This provider converts them to the standard
//! `StreamEvent::Delta(ToolUseStart / ToolInputDelta)` sequence with synthetic
//! `ToolCallId`s (`ollama_call_0`, `ollama_call_1`, ...).
//!
//! ## Thinking
//!
//! When `OllamaConfig::think` is `Some(true)`, the request is sent with `.think(true)`.
//! Thinking content arrives in `message.thinking` and is emitted as `ThinkingDelta`.

use std::future::Future;
use std::pin::Pin;

use ollama_rs::Ollama;
use ollama_rs::generation::chat::request::ChatMessageRequest;
use ollama_rs::generation::chat::{ChatMessage, ChatMessageResponseStream, MessageRole};
use ollama_rs::generation::tools::{
    ToolCall, ToolCallFunction, ToolFunctionInfo, ToolInfo, ToolType,
};
use ollama_rs::models::ModelOptions;

use crate::config::OllamaConfig;
use crate::error::{ProviderError, StreamError, StreamErrorKind, is_context_window_message};
use crate::streaming::{
    CompletionMetadata, ContentBlockType, StopReason, StreamDelta, StreamEvent, StreamHandle,
    TokenUsage,
};
use crate::tool::ToolDefinition;
use crate::types::{ContentBlock, Message, ModelId, Role, ToolCallId, ToolName};

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
        let host = config.base_url.host_str().unwrap_or("localhost").to_owned();
        let port = config.base_url.port().unwrap_or(11434);

        let client = Ollama::new(format!("http://{host}"), port);

        let info = ProviderInfo {
            kind: super::ProviderKind::Ollama,
            name: "Ollama",
            capabilities: ProviderCapabilities {
                streaming: true,
                tools: true,
                vision: false,
                extended_thinking: config.think.unwrap_or(false),
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

    /// Convert our messages to Ollama format, preserving tool call round-trips
    fn convert_messages(&self, messages: &[Message]) -> Vec<ChatMessage> {
        messages
            .iter()
            .filter_map(|msg| match msg.role {
                Role::User => {
                    let text = extract_text(&msg.content);
                    if text.is_empty() {
                        None
                    } else {
                        Some(ChatMessage::new(MessageRole::User, text))
                    }
                }
                Role::Assistant => {
                    let text = extract_text(&msg.content);

                    // Extract tool calls from ContentBlock::ToolUse
                    let tool_calls: Vec<ToolCall> = msg
                        .content
                        .iter()
                        .filter_map(|b| {
                            if let ContentBlock::ToolUse { name, input, .. } = b {
                                Some(ToolCall {
                                    function: ToolCallFunction {
                                        name: name.as_str().to_owned(),
                                        arguments: input.clone(),
                                    },
                                })
                            } else {
                                None
                            }
                        })
                        .collect();

                    // Ollama requires non-empty content or tool_calls
                    if text.is_empty() && tool_calls.is_empty() {
                        return None;
                    }

                    let mut cm = ChatMessage::assistant(text);
                    if !tool_calls.is_empty() {
                        cm.tool_calls = tool_calls;
                    }
                    Some(cm)
                }
                Role::Tool => {
                    let text = extract_tool_result_text(&msg.content);
                    if text.is_empty() {
                        None
                    } else {
                        Some(ChatMessage::tool(text))
                    }
                }
                Role::System => {
                    let text = extract_text(&msg.content);
                    if text.is_empty() {
                        None
                    } else {
                        Some(ChatMessage::system(text))
                    }
                }
            })
            .collect()
    }
}

/// Extract text (and thinking) content from content blocks
fn extract_text(content: &[ContentBlock]) -> String {
    let mut result = String::new();
    for block in content {
        match block {
            ContentBlock::Text { text } | ContentBlock::Thinking { text } => {
                if !result.is_empty() {
                    result.push('\n');
                }
                result.push_str(text);
            }
            ContentBlock::ToolUse { .. } | ContentBlock::ToolResult { .. } => {}
        }
    }
    result
}

/// Extract text from ToolResult content blocks
fn extract_tool_result_text(content: &[ContentBlock]) -> String {
    let mut result = String::new();
    for block in content {
        if let ContentBlock::ToolResult { content, .. } = block {
            match content {
                crate::types::ToolResultContent::Text(s) => {
                    if !result.is_empty() {
                        result.push('\n');
                    }
                    result.push_str(s);
                }
            }
        }
    }
    result
}

/// Convert our ToolDefinitions to Ollama's ToolInfo format.
///
/// Uses serde round-trip to convert our `Map<String, Value>` schema into
/// `schemars::Schema` (required by ollama-rs), avoiding a direct dependency
/// on the `schemars` crate.
fn convert_tools(tools: &[ToolDefinition]) -> Vec<ToolInfo> {
    tools
        .iter()
        .map(|t| {
            // Round-trip through serde: Map -> Value::Object -> Schema
            let schema_value = t.input_schema.to_value();
            let parameters = serde_json::from_value(schema_value).unwrap_or_default();
            ToolInfo {
                tool_type: ToolType::Function,
                function: ToolFunctionInfo {
                    name: t.name.as_str().to_owned(),
                    description: t.description.clone(),
                    parameters,
                },
            }
        })
        .collect()
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
            let model = self.config.model.as_str().to_owned();

            // Convert messages
            let mut messages = Vec::new();

            // Add system prompt if present
            if let Some(ref system) = request.system {
                if !system.as_str().is_empty() {
                    messages.push(ChatMessage::new(
                        MessageRole::System,
                        system.as_str().to_owned(),
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
            let mut chat_request =
                ChatMessageRequest::new(model.clone(), messages).options(options);

            // Wire tools if provided
            if !request.tools.is_empty() {
                chat_request = chat_request.tools(convert_tools(&request.tools));
            }

            // Wire thinking if configured
            if self.config.think == Some(true) {
                chat_request = chat_request.think(true);
            }

            // Create streaming request
            let stream: ChatMessageResponseStream = self
                .client
                .send_chat_messages_stream(chat_request)
                .await
                .map_err(|e| {
                    let msg = format!("Ollama error: {e}");
                    if is_context_window_message(&msg) {
                        ProviderError::ContextWindowExceeded {
                            provider: super::ProviderKind::Ollama,
                            message: msg,
                            context_window: None,
                            tokens_used: None,
                        }
                    } else {
                        ProviderError::InvalidRequest {
                            provider: super::ProviderKind::Ollama,
                            message: msg,
                        }
                    }
                })?;

            let event_stream = super::stream_adapter::buffered_sdk_stream(
                stream,
                ctx.cancellation.clone(),
                StreamState::new(model.clone()),
                |item, state| match item {
                    Ok(response) => parse_ollama_response(response, state),
                    Err(()) => vec![Err(StreamError::ConnectionLost {
                        kind: StreamErrorKind::TransportError,
                        message: "Stream error".to_owned(),
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
            reason = "fallback Ollama model IDs are hardcoded valid constants"
        )]
        Box::pin(async move {
            // Query Ollama for local models
            match self.client.list_local_models().await {
                Ok(models) => {
                    let infos = models
                        .into_iter()
                        .map(|m| ModelInfo {
                            id: ModelId::new(&m.name)
                                // Safety: "unknown" is a valid model ID (alphanumeric)
                                .unwrap_or_else(|_| {
                                    ModelId::new("unknown").expect("hardcoded valid model ID")
                                }),
                            name: m.name,
                            context_window: None, // Ollama doesn't report this in list
                        })
                        .collect();
                    Ok(infos)
                }
                Err(e) => {
                    // Fall back to common models if we can't query
                    tracing::warn!("Failed to list Ollama models: {}", e);
                    // All model IDs below are hardcoded valid strings
                    Ok(vec![
                        ModelInfo {
                            id: ModelId::new("llama3.2").expect("hardcoded valid model ID"),
                            name: "Llama 3.2".to_owned(),
                            context_window: Some(8192),
                        },
                        ModelInfo {
                            id: ModelId::new("qwen3:14b").expect("hardcoded valid model ID"),
                            name: "Qwen3 14B".to_owned(),
                            context_window: Some(32768),
                        },
                        ModelInfo {
                            id: ModelId::new("mistral").expect("hardcoded valid model ID"),
                            name: "Mistral".to_owned(),
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
    /// Current content block index (incremented per block)
    block_index: usize,
    /// Whether a text block is currently open
    text_block_open: bool,
    /// Whether thinking content has been emitted (deduplicate across chunks)
    thinking_emitted: bool,
    /// Counter for generating synthetic ToolCallIds
    tool_call_counter: u32,
    /// Determined stop reason (ToolUse if tool calls present, else EndTurn)
    stop_reason: Option<StopReason>,
}

impl StreamState {
    fn new(model: String) -> Self {
        Self {
            model: ModelId::new(model).ok(),
            started: false,
            completed: false,
            usage: None,
            block_index: 0,
            text_block_open: false,
            thinking_emitted: false,
            tool_call_counter: 0,
            stop_reason: None,
        }
    }
}

/// Parse an Ollama streaming response into stream events.
///
/// Ollama delivers tool calls fully-formed in the final chunk (`done: true`).
/// Text and thinking content stream incrementally across chunks.
#[allow(
    clippy::expect_used,
    reason = "fallback tool name is a hardcoded valid sentinel"
)]
fn parse_ollama_response(
    response: ollama_rs::generation::chat::ChatMessageResponse,
    state: &mut StreamState,
) -> Vec<Result<StreamEvent, StreamError>> {
    let mut events = Vec::new();

    // First chunk: emit Started
    if !state.started {
        state.started = true;
        events.push(Ok(StreamEvent::Started {
            metadata: CompletionMetadata {
                model: state.model.clone(),
                stop_reason: None,
                usage: None,
            },
        }));
    }

    // Thinking content (arrives in message.thinking)
    if let Some(ref thinking) = response.message.thinking {
        if !thinking.is_empty() && !state.thinking_emitted {
            state.thinking_emitted = true;
            events.push(Ok(StreamEvent::ContentBlockStart {
                index: state.block_index,
                block_type: ContentBlockType::Thinking,
            }));
            events.push(Ok(StreamEvent::Delta(StreamDelta::ThinkingDelta {
                thinking: thinking.clone(),
            })));
            events.push(Ok(StreamEvent::ContentBlockStop {
                index: state.block_index,
            }));
            state.block_index += 1;
        }
    }

    // Text content
    if !response.message.content.is_empty() {
        if !state.text_block_open {
            events.push(Ok(StreamEvent::ContentBlockStart {
                index: state.block_index,
                block_type: ContentBlockType::Text,
            }));
            state.text_block_open = true;
        }
        events.push(Ok(StreamEvent::Delta(StreamDelta::TextDelta {
            text: response.message.content.clone(),
        })));
    }

    // Tool calls (arrive fully-formed in final chunk)
    if !response.message.tool_calls.is_empty() {
        // Close text block if open
        if state.text_block_open {
            events.push(Ok(StreamEvent::ContentBlockStop {
                index: state.block_index,
            }));
            state.block_index += 1;
            state.text_block_open = false;
        }

        for tc in &response.message.tool_calls {
            let id = ToolCallId::new(format!("ollama_call_{}", state.tool_call_counter));
            state.tool_call_counter += 1;

            let name = ToolName::new(&tc.function.name)
                .unwrap_or_else(|_| ToolName::new("unknown").expect("hardcoded valid"));

            let input_json =
                serde_json::to_string(&tc.function.arguments).unwrap_or_else(|_| "{}".to_owned());

            events.push(Ok(StreamEvent::ContentBlockStart {
                index: state.block_index,
                block_type: ContentBlockType::ToolUse,
            }));
            events.push(Ok(StreamEvent::Delta(StreamDelta::ToolUseStart {
                id: id.clone(),
                name,
            })));
            events.push(Ok(StreamEvent::Delta(StreamDelta::ToolInputDelta {
                id,
                partial_json: input_json,
            })));
            events.push(Ok(StreamEvent::ContentBlockStop {
                index: state.block_index,
            }));
            state.block_index += 1;
        }

        state.stop_reason = Some(StopReason::ToolUse);
    }

    // Final chunk
    if response.done {
        if state.text_block_open {
            events.push(Ok(StreamEvent::ContentBlockStop {
                index: state.block_index,
            }));
            state.text_block_open = false;
        }

        // Track usage from final_data
        if let Some(ref final_data) = response.final_data {
            state.usage = Some(TokenUsage {
                input_tokens: final_data.prompt_eval_count as u32,
                output_tokens: final_data.eval_count as u32,
            });
        }

        if state.stop_reason.is_none() {
            state.stop_reason = Some(StopReason::EndTurn);
        }
    }

    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use ollama_rs::generation::chat::ChatMessageResponse;
    use serde_json::json;

    fn streaming_chunk(content: &str) -> ChatMessageResponse {
        serde_json::from_value(json!({
            "model": "llama3.2",
            "created_at": "2024-01-01T00:00:00Z",
            "message": { "role": "assistant", "content": content },
            "done": false
        }))
        .expect("valid streaming chunk JSON")
    }

    fn final_chunk() -> ChatMessageResponse {
        serde_json::from_value(json!({
            "model": "llama3.2",
            "created_at": "2024-01-01T00:00:01Z",
            "message": { "role": "assistant", "content": "" },
            "done": true,
            "total_duration": 1000000,
            "load_duration": 500000,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 100000,
            "eval_count": 20,
            "eval_duration": 200000
        }))
        .expect("valid final chunk JSON")
    }

    fn tool_call_chunk(tool_calls: Vec<serde_json::Value>) -> ChatMessageResponse {
        serde_json::from_value(json!({
            "model": "llama3.2",
            "created_at": "2024-01-01T00:00:01Z",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": tool_calls
            },
            "done": true,
            "total_duration": 1000000,
            "load_duration": 500000,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 100000,
            "eval_count": 20,
            "eval_duration": 200000
        }))
        .expect("valid tool call chunk JSON")
    }

    fn thinking_chunk(thinking: &str, content: &str) -> ChatMessageResponse {
        serde_json::from_value(json!({
            "model": "qwen3:14b",
            "created_at": "2024-01-01T00:00:00Z",
            "message": {
                "role": "assistant",
                "content": content,
                "thinking": thinking
            },
            "done": false
        }))
        .expect("valid thinking chunk JSON")
    }

    // --- Existing text tests ---

    #[test]
    fn parse_text_content() {
        let mut state = StreamState::new("llama3.2".to_string());
        let events = parse_ollama_response(streaming_chunk("Hello"), &mut state);

        // First chunk: Started + ContentBlockStart + TextDelta
        assert!(state.started);
        assert!(events.iter().any(|e| matches!(
            e,
            Ok(StreamEvent::Delta(StreamDelta::TextDelta { text })) if text == "Hello"
        )));
    }

    #[test]
    fn parse_empty_content_skipped() {
        let mut state = StreamState::new("llama3.2".to_string());
        state.started = true;

        let events = parse_ollama_response(streaming_chunk(""), &mut state);
        // Empty content should not produce a TextDelta
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Ok(StreamEvent::Delta(StreamDelta::TextDelta { .. }))))
        );
    }

    #[test]
    fn parse_final_chunk_emits_stop() {
        let mut state = StreamState::new("llama3.2".to_string());
        state.started = true;
        // Simulate a text block was opened
        state.text_block_open = true;

        let events = parse_ollama_response(final_chunk(), &mut state);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Ok(StreamEvent::ContentBlockStop { index: 0 })))
        );
        // Usage should be tracked from final_data
        assert!(state.usage.is_some());
        assert_eq!(state.stop_reason, Some(StopReason::EndTurn));
    }

    // --- Tool calling tests ---

    #[test]
    fn parse_tool_call_response() {
        let mut state = StreamState::new("llama3.2".to_string());

        let chunk = tool_call_chunk(vec![json!({
            "function": {
                "name": "read_file",
                "arguments": {"path": "/tmp/test.txt"}
            }
        })]);

        let events = parse_ollama_response(chunk, &mut state);

        // Should have: Started, ContentBlockStart(ToolUse), ToolUseStart, ToolInputDelta, ContentBlockStop
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Ok(StreamEvent::Started { .. })))
        );
        assert!(events.iter().any(|e| matches!(
            e,
            Ok(StreamEvent::ContentBlockStart {
                block_type: ContentBlockType::ToolUse,
                ..
            })
        )));
        assert!(events.iter().any(|e| matches!(
            e,
            Ok(StreamEvent::Delta(StreamDelta::ToolUseStart { id, name }))
            if id.as_str() == "ollama_call_0" && name.as_str() == "read_file"
        )));
        assert!(events.iter().any(|e| matches!(
            e,
            Ok(StreamEvent::Delta(StreamDelta::ToolInputDelta { partial_json, .. }))
            if partial_json.contains("path")
        )));
    }

    #[test]
    fn parse_multi_tool_calls() {
        let mut state = StreamState::new("llama3.2".to_string());

        let chunk = tool_call_chunk(vec![
            json!({
                "function": {
                    "name": "read_file",
                    "arguments": {"path": "/a.txt"}
                }
            }),
            json!({
                "function": {
                    "name": "list_dir",
                    "arguments": {"path": "/tmp"}
                }
            }),
        ]);

        let events = parse_ollama_response(chunk, &mut state);

        // Should have two ToolUseStart deltas with different IDs and block indices
        let tool_starts: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, Ok(StreamEvent::Delta(StreamDelta::ToolUseStart { .. }))))
            .collect();
        assert_eq!(tool_starts.len(), 2);

        // Verify unique IDs
        assert_eq!(state.tool_call_counter, 2);

        // Block indices should be 0 and 1
        let block_starts: Vec<usize> = events
            .iter()
            .filter_map(|e| match e {
                Ok(StreamEvent::ContentBlockStart {
                    index,
                    block_type: ContentBlockType::ToolUse,
                }) => Some(*index),
                _ => None,
            })
            .collect();
        assert_eq!(block_starts.len(), 2);
        assert_ne!(block_starts[0], block_starts[1]);
    }

    #[test]
    fn synthetic_tool_call_ids_unique() {
        let mut state = StreamState::new("llama3.2".to_string());

        let chunk = tool_call_chunk(vec![
            json!({ "function": { "name": "a", "arguments": {} } }),
            json!({ "function": { "name": "b", "arguments": {} } }),
        ]);

        parse_ollama_response(chunk, &mut state);

        assert_eq!(state.tool_call_counter, 2);
        // IDs are "ollama_call_0" and "ollama_call_1"
    }

    #[test]
    fn tool_call_sets_stop_reason() {
        let mut state = StreamState::new("llama3.2".to_string());

        let chunk = tool_call_chunk(vec![json!({
            "function": { "name": "test", "arguments": {} }
        })]);

        parse_ollama_response(chunk, &mut state);

        assert_eq!(state.stop_reason, Some(StopReason::ToolUse));
    }

    #[test]
    fn convert_tools_produces_valid_tool_info() {
        let def = ToolDefinition::new(
            ToolName::new("read_file").unwrap(),
            "Read a file from disk",
            crate::tool::ToolSchema::from_value(json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"}
                },
                "required": ["path"]
            }))
            .unwrap(),
        );

        let tools = convert_tools(&[def]);
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "read_file");
        assert_eq!(tools[0].function.description, "Read a file from disk");
        assert!(matches!(tools[0].tool_type, ToolType::Function));
    }

    #[test]
    fn convert_messages_preserves_tool_calls() {
        let provider = make_test_provider();

        let msg = Message::with_content(
            Role::Assistant,
            vec![
                ContentBlock::Text {
                    text: "I'll read that file.".into(),
                },
                ContentBlock::ToolUse {
                    id: ToolCallId::new("call_1"),
                    name: ToolName::new("read_file").unwrap(),
                    input: json!({"path": "/test.txt"}),
                },
            ],
        );

        let converted = provider.convert_messages(&[msg]);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].role, MessageRole::Assistant);
        assert!(!converted[0].tool_calls.is_empty());
        assert_eq!(converted[0].tool_calls[0].function.name, "read_file");
    }

    #[test]
    fn convert_messages_tool_role() {
        let provider = make_test_provider();

        let msg = Message::tool_result(ToolCallId::new("call_1"), "file contents here", false);

        let converted = provider.convert_messages(&[msg]);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].role, MessageRole::Tool);
        assert_eq!(converted[0].content, "file contents here");
    }

    // --- Thinking tests ---

    #[test]
    fn parse_thinking_response() {
        let mut state = StreamState::new("qwen3:14b".to_string());

        let chunk = thinking_chunk("Let me analyze this...", "Here is the answer.");
        let events = parse_ollama_response(chunk, &mut state);

        // Should have thinking block events
        assert!(events.iter().any(|e| matches!(
            e,
            Ok(StreamEvent::ContentBlockStart {
                block_type: ContentBlockType::Thinking,
                ..
            })
        )));
        assert!(events.iter().any(|e| matches!(
            e,
            Ok(StreamEvent::Delta(StreamDelta::ThinkingDelta { thinking }))
            if thinking == "Let me analyze this..."
        )));
        // And text events
        assert!(events.iter().any(|e| matches!(
            e,
            Ok(StreamEvent::Delta(StreamDelta::TextDelta { text }))
            if text == "Here is the answer."
        )));
    }

    #[test]
    fn thinking_emitted_once() {
        let mut state = StreamState::new("qwen3:14b".to_string());

        // First chunk with thinking
        let chunk1 = thinking_chunk("thinking...", "text1");
        let events1 = parse_ollama_response(chunk1, &mut state);
        let thinking_count_1 = events1
            .iter()
            .filter(|e| matches!(e, Ok(StreamEvent::Delta(StreamDelta::ThinkingDelta { .. }))))
            .count();
        assert_eq!(thinking_count_1, 1);

        // Second chunk with same thinking — should NOT emit again
        let chunk2 = thinking_chunk("thinking...", "text2");
        let events2 = parse_ollama_response(chunk2, &mut state);
        let thinking_count_2 = events2
            .iter()
            .filter(|e| matches!(e, Ok(StreamEvent::Delta(StreamDelta::ThinkingDelta { .. }))))
            .count();
        assert_eq!(thinking_count_2, 0);
    }

    #[test]
    fn config_think_from_env() {
        // Test that the think field deserializes properly
        let config: OllamaConfig = serde_json::from_value(json!({
            "model": "qwen3:14b",
            "num_ctx": 32768,
            "think": true
        }))
        .unwrap();
        assert_eq!(config.think, Some(true));

        let config_no_think: OllamaConfig = serde_json::from_value(json!({
            "model": "qwen3:14b",
            "num_ctx": 32768
        }))
        .unwrap();
        assert_eq!(config_no_think.think, None);
    }

    // --- Helper ---

    fn make_test_provider() -> OllamaProvider {
        let config: OllamaConfig = serde_json::from_value(json!({
            "model": "llama3.2",
            "num_ctx": 8192
        }))
        .unwrap();
        OllamaProvider::new(config).unwrap()
    }
}

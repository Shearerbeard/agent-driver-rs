//! Anthropic/Claude provider implementation using the Messages API with SSE streaming.
//!
//! Communicates directly with the Anthropic API (`https://api.anthropic.com/v1/messages`)
//! using `reqwest` + `reqwest_eventsource` for server-sent events. Supports:
//! - Streaming text and thinking deltas
//! - Tool/function calling in Claude's native format
//! - Extended thinking (when configured via [`ThinkingConfig`](crate::config::ThinkingConfig))

use std::future::Future;
use std::pin::Pin;

use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use reqwest_eventsource::EventSource;
use serde::Deserialize;
use serde_json::Value as JsonValue;

use crate::config::AnthropicConfig;
use crate::error::{AuthErrorKind, ProviderError, StreamError, StreamErrorKind};
use crate::streaming::{
    CompletionMetadata, ContentBlockType, StopReason, StreamDelta, StreamEvent, StreamHandle,
    TokenUsage,
};
use crate::tool::ToolFormat;
use crate::types::{ContentBlock, ModelId, ToolCallId, ToolName};

use super::{
    CompletionRequest, ModelInfo, Provider, ProviderCapabilities, ProviderContext, ProviderInfo,
};

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Anthropic/Claude provider
pub struct AnthropicProvider {
    config: AnthropicConfig,
    client: reqwest::Client,
    info: ProviderInfo,
}

impl AnthropicProvider {
    /// Create a new Anthropic provider
    pub fn new(config: AnthropicConfig) -> Result<Self, ProviderError> {
        let client = reqwest::Client::new();

        let info = ProviderInfo {
            kind: super::ProviderKind::Anthropic,
            name: "Anthropic",
            capabilities: ProviderCapabilities {
                streaming: true,
                tools: true,
                vision: true,
                extended_thinking: config.thinking.is_some(),
                max_context_tokens: Some(200_000),
            },
        };

        Ok(Self {
            config,
            client,
            info,
        })
    }

    /// Build the request body for the API
    fn build_request_body(&self, request: &CompletionRequest) -> JsonValue {
        let mut body = serde_json::json!({
            "model": request.model.as_str(),
            "max_tokens": request.config.max_tokens.get(),
            "stream": true,
        });

        // Add system prompt
        if let Some(ref system) = request.system {
            if !system.is_empty() {
                body["system"] = JsonValue::String(system.as_str().to_string());
            }
        }

        // Add temperature
        if let Some(temp) = request.config.temperature {
            body["temperature"] = JsonValue::Number(
                serde_json::Number::from_f64(temp.get() as f64).unwrap_or_else(|| 1.into()),
            );
        }

        // Add stop sequences
        if !request.config.stop_sequences.is_empty() {
            body["stop_sequences"] = JsonValue::Array(
                request
                    .config
                    .stop_sequences
                    .iter()
                    .map(|s| JsonValue::String(s.clone()))
                    .collect(),
            );
        }

        // Add tools
        if !request.tools.is_empty() {
            body["tools"] = ToolFormat::claude().serialize_tools(&request.tools);
        }

        // Add messages
        let messages: Vec<JsonValue> = request
            .messages
            .iter()
            .map(|msg| self.serialize_message(msg))
            .collect();
        body["messages"] = JsonValue::Array(messages);

        // Add extended thinking if configured
        if let Some(ref thinking) = self.config.thinking {
            body["thinking"] = serde_json::json!({
                "type": "enabled",
                "budget_tokens": thinking.budget_tokens()
            });
        }

        body
    }

    /// Serialize a message for the API
    fn serialize_message(&self, msg: &crate::types::Message) -> JsonValue {
        let role = match msg.role {
            crate::types::Role::User => "user",
            crate::types::Role::Assistant => "assistant",
            crate::types::Role::Tool => "user", // Tool results come as user messages
            crate::types::Role::System => "user", // System should be handled separately
        };

        let content: Vec<JsonValue> = msg
            .content
            .iter()
            .map(|block| match block {
                ContentBlock::Text { text } => serde_json::json!({
                    "type": "text",
                    "text": text
                }),
                ContentBlock::Thinking { text } => serde_json::json!({
                    "type": "thinking",
                    "thinking": text
                }),
                ContentBlock::ToolUse { id, name, input } => serde_json::json!({
                    "type": "tool_use",
                    "id": id.as_str(),
                    "name": name.as_str(),
                    "input": input
                }),
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => {
                    let content_str = match content {
                        crate::types::ToolResultContent::Text(s) => s.clone(),
                    };
                    serde_json::json!({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id.as_str(),
                        "content": content_str,
                        "is_error": is_error
                    })
                }
            })
            .collect();

        serde_json::json!({
            "role": role,
            "content": content
        })
    }

    /// Build headers for the API request.
    ///
    /// Returns an error if the API key contains characters invalid for HTTP headers
    /// (e.g. non-visible ASCII). This fails loudly rather than silently sending
    /// an empty auth header.
    fn build_headers(&self) -> Result<HeaderMap, ProviderError> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            "x-api-key",
            HeaderValue::from_str(self.config.api_key.as_str()).map_err(|_| {
                ProviderError::Auth {
                    provider: super::ProviderKind::Anthropic,
                    kind: AuthErrorKind::InvalidApiKey,
                    message: "ANTHROPIC_API_KEY contains invalid header characters".to_string(),
                }
            })?,
        );
        headers.insert(
            "anthropic-version",
            HeaderValue::from_static(ANTHROPIC_VERSION),
        );
        Ok(headers)
    }
}

impl Provider for AnthropicProvider {
    fn info(&self) -> &ProviderInfo {
        &self.info
    }

    fn complete_stream(
        &self,
        request: CompletionRequest,
        ctx: ProviderContext,
    ) -> Pin<Box<dyn Future<Output = Result<StreamHandle, ProviderError>> + Send + '_>> {
        Box::pin(async move {
            let body = self.build_request_body(&request);
            let headers = self.build_headers()?;

            let request_builder = self
                .client
                .post(ANTHROPIC_API_URL)
                .headers(headers)
                .json(&body);

            let event_source = EventSource::new(request_builder).map_err(|e| {
                ProviderError::Stream(StreamError::ConnectionLost {
                    kind: StreamErrorKind::ConnectionDropped,
                    message: e.to_string(),
                })
            })?;

            let stream = create_anthropic_stream(event_source, ctx.cancellation.clone());

            Ok(StreamHandle::new(
                Box::pin(stream),
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
                    id: ModelId::new("claude-opus-4-20250514").expect("hardcoded valid model ID"),
                    name: "Claude Opus 4".to_string(),
                    context_window: Some(200_000),
                },
                ModelInfo {
                    id: ModelId::new("claude-sonnet-4-20250514").expect("hardcoded valid model ID"),
                    name: "Claude Sonnet 4".to_string(),
                    context_window: Some(200_000),
                },
                ModelInfo {
                    id: ModelId::new("claude-3-5-haiku-20241022")
                        .expect("hardcoded valid model ID"),
                    name: "Claude 3.5 Haiku".to_string(),
                    context_window: Some(200_000),
                },
            ])
        })
    }
}

/// Create a stream from Anthropic SSE events.
///
/// Uses the shared [`buffered_sse_stream`](super::stream_adapter::buffered_sse_stream)
/// adapter with Anthropic-specific event parsing. Anthropic doesn't use a done
/// marker — the stream terminates when the `EventSource` closes.
fn create_anthropic_stream(
    event_source: EventSource,
    cancellation: tokio_util::sync::CancellationToken,
) -> impl futures::Stream<Item = Result<StreamEvent, StreamError>> {
    super::stream_adapter::buffered_sse_stream(
        event_source,
        cancellation,
        StreamState::default(),
        None, // Anthropic has no done marker
        parse_anthropic_event,
        |_state| None, // No done marker → no final event
    )
}

/// State for tracking stream parsing
#[derive(Default)]
struct StreamState {
    current_block_index: usize,
    current_block_type: Option<ContentBlockType>,
    /// Tool call ID per block index, carried from ContentBlockStart to
    /// subsequent ContentBlockDelta events so that `input_json_delta`
    /// can reference the correct `ToolCallId` instead of using an empty string.
    tool_call_ids: std::collections::HashMap<usize, String>,
}

/// Parse an Anthropic SSE event
fn parse_anthropic_event(
    data: &str,
    state: &mut StreamState,
) -> Option<Result<Vec<StreamEvent>, StreamError>> {
    let parsed: AnthropicStreamEvent = match serde_json::from_str(data) {
        Ok(event) => event,
        Err(e) => {
            tracing::warn!(data = %data, error = %e, "Failed to parse Anthropic SSE event");
            return None;
        }
    };

    let events = match parsed {
        AnthropicStreamEvent::MessageStart { message } => {
            vec![StreamEvent::Started {
                metadata: CompletionMetadata {
                    model: ModelId::new(message.model).ok(),
                    stop_reason: None,
                    usage: message.usage.map(|u| TokenUsage {
                        input_tokens: u.input_tokens,
                        output_tokens: u.output_tokens,
                    }),
                },
            }]
        }
        AnthropicStreamEvent::ContentBlockStart {
            index,
            content_block,
        } => {
            state.current_block_index = index;
            let block_type = match content_block.r#type.as_str() {
                "text" => ContentBlockType::Text,
                "thinking" => ContentBlockType::Thinking,
                "tool_use" => ContentBlockType::ToolUse,
                _ => ContentBlockType::Text,
            };
            state.current_block_type = Some(block_type);

            let mut events = vec![StreamEvent::ContentBlockStart { index, block_type }];

            // For tool_use, store the ID and emit the start delta
            if block_type == ContentBlockType::ToolUse {
                if let (Some(id), Some(name)) = (content_block.id, content_block.name) {
                    state.tool_call_ids.insert(index, id.clone());
                    events.push(StreamEvent::Delta(StreamDelta::ToolUseStart {
                        id: ToolCallId::new(id),
                        // Safety: "unknown" is a valid tool name (alphanumeric)
                        name: ToolName::new(name).unwrap_or_else(|_| {
                            ToolName::new("unknown").expect("hardcoded valid tool name")
                        }),
                    }));
                }
            }

            events
        }
        AnthropicStreamEvent::ContentBlockDelta { index, delta } => {
            match delta.r#type.as_str() {
                "text_delta" => {
                    if let Some(text) = delta.text {
                        vec![StreamEvent::Delta(StreamDelta::TextDelta { text })]
                    } else {
                        vec![]
                    }
                }
                "thinking_delta" => {
                    if let Some(thinking) = delta.thinking {
                        vec![StreamEvent::Delta(StreamDelta::ThinkingDelta { thinking })]
                    } else {
                        vec![]
                    }
                }
                "input_json_delta" => {
                    if let Some(partial_json) = delta.partial_json {
                        // Look up the tool call ID stored from the preceding
                        // ContentBlockStart event for this block index.
                        let id = state
                            .tool_call_ids
                            .get(&index)
                            .map(ToolCallId::new)
                            .unwrap_or_else(|| ToolCallId::new(""));
                        vec![StreamEvent::Delta(StreamDelta::ToolInputDelta {
                            id,
                            partial_json,
                        })]
                    } else {
                        vec![]
                    }
                }
                "signature_delta" => {
                    if let Some(signature) = delta.signature {
                        vec![StreamEvent::Delta(StreamDelta::SignatureDelta {
                            signature,
                        })]
                    } else {
                        vec![]
                    }
                }
                _ => vec![],
            }
        }
        AnthropicStreamEvent::ContentBlockStop { index } => {
            let _block_type = state
                .current_block_type
                .take()
                .unwrap_or(ContentBlockType::Text);
            // Clean up stored tool call ID for this block index
            state.tool_call_ids.remove(&index);
            vec![StreamEvent::ContentBlockStop { index }]
        }
        AnthropicStreamEvent::MessageDelta { delta, usage } => {
            let stop_reason = delta.stop_reason.and_then(|r| match r.as_str() {
                "end_turn" => Some(StopReason::EndTurn),
                "max_tokens" => Some(StopReason::MaxTokens),
                "tool_use" => Some(StopReason::ToolUse),
                "stop_sequence" => Some(StopReason::StopSequence),
                _ => None,
            });
            vec![StreamEvent::Completed {
                metadata: CompletionMetadata {
                    model: None,
                    stop_reason,
                    usage: usage.map(|u| TokenUsage {
                        input_tokens: 0,
                        output_tokens: u.output_tokens,
                    }),
                },
            }]
        }
        AnthropicStreamEvent::MessageStop => {
            vec![]
        }
        AnthropicStreamEvent::Ping => vec![],
        AnthropicStreamEvent::Error { error } => {
            let message = match (&error.error_type, &error.message) {
                (Some(t), Some(m)) => format!("{}: {}", t, m),
                (None, Some(m)) => m.clone(),
                (Some(t), None) => t.clone(),
                (None, None) => "Unknown error".into(),
            };
            return Some(Err(StreamError::ConnectionLost {
                kind: StreamErrorKind::ProviderError,
                message,
            }));
        }
    };

    if events.is_empty() {
        None
    } else {
        Some(Ok(events))
    }
}

// Anthropic streaming event types

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicStreamEvent {
    MessageStart {
        message: MessageStartData,
    },
    ContentBlockStart {
        index: usize,
        content_block: ContentBlockData,
    },
    ContentBlockDelta {
        index: usize,
        delta: DeltaData,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        delta: MessageDeltaData,
        usage: Option<UsageData>,
    },
    MessageStop,
    Ping,
    Error {
        error: ErrorData,
    },
}

#[derive(Debug, Deserialize)]
struct MessageStartData {
    model: String,
    usage: Option<UsageData>,
}

#[derive(Debug, Deserialize)]
struct ContentBlockData {
    r#type: String,
    id: Option<String>,
    name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DeltaData {
    r#type: String,
    text: Option<String>,
    thinking: Option<String>,
    partial_json: Option<String>,
    signature: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MessageDeltaData {
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UsageData {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct ErrorData {
    #[serde(rename = "type")]
    error_type: Option<String>,
    message: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_message_start() {
        let data = r#"{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":100,"output_tokens":0}}}"#;

        let mut state = StreamState::default();
        let result = parse_anthropic_event(data, &mut state);

        assert!(result.is_some());
        let events = result.unwrap().unwrap();
        assert!(!events.is_empty());
        assert!(matches!(events[0], StreamEvent::Started { .. }));
    }

    #[test]
    fn parse_text_delta() {
        let data = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#;

        let mut state = StreamState::default();
        let result = parse_anthropic_event(data, &mut state);

        assert!(result.is_some());
        let events = result.unwrap().unwrap();
        assert!(!events.is_empty());
        assert!(matches!(
            &events[0],
            StreamEvent::Delta(StreamDelta::TextDelta { text }) if text == "Hello"
        ));
    }

    #[test]
    fn parse_tool_use_content_block_start() {
        let data = r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01abc","name":"read_file"}}"#;
        let mut state = StreamState::default();
        let result = parse_anthropic_event(data, &mut state);

        let events = result.unwrap().unwrap();
        assert_eq!(events.len(), 2);
        assert!(matches!(
            events[0],
            StreamEvent::ContentBlockStart {
                index: 1,
                block_type: ContentBlockType::ToolUse
            }
        ));
        assert!(matches!(
            &events[1],
            StreamEvent::Delta(StreamDelta::ToolUseStart { id, name })
                if id.as_str() == "toolu_01abc" && name.as_str() == "read_file"
        ));
        // State should track the tool call ID for subsequent input deltas
        assert_eq!(state.tool_call_ids.get(&1).unwrap(), "toolu_01abc");
    }

    #[test]
    fn parse_tool_input_json_delta() {
        let data = r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"partial"}}"#;
        let mut state = StreamState::default();
        state.tool_call_ids.insert(1, "toolu_01abc".to_string());

        let result = parse_anthropic_event(data, &mut state);
        let events = result.unwrap().unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            StreamEvent::Delta(StreamDelta::ToolInputDelta { id, partial_json })
                if id.as_str() == "toolu_01abc" && partial_json == "partial"
        ));
    }

    #[test]
    fn parse_thinking_delta() {
        let data = r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me think..."}}"#;
        let mut state = StreamState::default();
        let result = parse_anthropic_event(data, &mut state);

        let events = result.unwrap().unwrap();
        assert!(matches!(
            &events[0],
            StreamEvent::Delta(StreamDelta::ThinkingDelta { thinking })
                if thinking == "Let me think..."
        ));
    }

    #[test]
    fn parse_signature_delta() {
        let data = r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig_abc123"}}"#;
        let mut state = StreamState::default();
        let result = parse_anthropic_event(data, &mut state);

        let events = result.unwrap().unwrap();
        assert!(matches!(
            &events[0],
            StreamEvent::Delta(StreamDelta::SignatureDelta { signature })
                if signature == "sig_abc123"
        ));
    }

    #[test]
    fn parse_error_event() {
        let data = r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#;
        let mut state = StreamState::default();
        let result = parse_anthropic_event(data, &mut state);

        assert!(matches!(
            result.unwrap(),
            Err(StreamError::ConnectionLost { message: ref msg, .. }) if msg == "overloaded_error: Overloaded"
        ));
    }

    #[test]
    fn parse_message_delta_with_tool_use_stop_reason() {
        let data = r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":42}}"#;
        let mut state = StreamState::default();
        let result = parse_anthropic_event(data, &mut state);

        let events = result.unwrap().unwrap();
        assert!(matches!(
            &events[0],
            StreamEvent::Completed { metadata }
                if metadata.stop_reason == Some(StopReason::ToolUse)
                && metadata.usage.map(|u| u.output_tokens) == Some(42)
        ));
    }

    #[test]
    fn parse_content_block_stop_cleans_tool_id() {
        let data = r#"{"type":"content_block_stop","index":1}"#;
        let mut state = StreamState::default();
        state.tool_call_ids.insert(1, "toolu_01abc".to_string());
        state.current_block_type = Some(ContentBlockType::ToolUse);

        let result = parse_anthropic_event(data, &mut state);
        let events = result.unwrap().unwrap();
        assert!(matches!(
            events[0],
            StreamEvent::ContentBlockStop { index: 1 }
        ));
        assert!(!state.tool_call_ids.contains_key(&1));
    }

    #[test]
    fn parse_malformed_json_returns_none() {
        let mut state = StreamState::default();
        assert!(parse_anthropic_event("not json", &mut state).is_none());
    }
}

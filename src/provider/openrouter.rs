//! OpenRouter provider implementation
//!
//! OpenRouter provides access to multiple LLM providers through a unified API.
//! Uses OpenAI-compatible format with SSE streaming.

use std::future::Future;
use std::pin::Pin;

use futures::StreamExt;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use reqwest_eventsource::{Event, EventSource};
use serde::Deserialize;
use serde_json::Value as JsonValue;

use crate::config::OpenRouterConfig;
use crate::error::{ProviderError, StreamError};
use crate::streaming::{
    CompletionMetadata, ContentBlockType, StopReason, StreamDelta, StreamEvent, StreamHandle,
    TokenUsage,
};
use crate::tool::ToolFormat;
use crate::types::{ContentBlock, ModelId, ToolCallId, ToolName};

use super::{
    CompletionRequest, ModelInfo, Provider, ProviderCapabilities, ProviderContext, ProviderInfo,
};

const OPENROUTER_API_URL: &str = "https://openrouter.ai/api/v1/chat/completions";

/// OpenRouter provider
pub struct OpenRouterProvider {
    config: OpenRouterConfig,
    client: reqwest::Client,
    info: ProviderInfo,
}

impl OpenRouterProvider {
    /// Create a new OpenRouter provider
    pub fn new(config: OpenRouterConfig) -> Result<Self, ProviderError> {
        let client = reqwest::Client::new();

        let info = ProviderInfo {
            id: "openrouter",
            name: "OpenRouter",
            capabilities: ProviderCapabilities {
                streaming: true,
                tools: true,
                vision: true,
                extended_thinking: false,
                max_context_tokens: None, // Varies by model
            },
        };

        Ok(Self {
            config,
            client,
            info,
        })
    }

    /// Build the request body for the API (OpenAI-compatible format)
    fn build_request_body(&self, request: &CompletionRequest) -> JsonValue {
        let mut body = serde_json::json!({
            "model": request.model.as_str(),
            "max_tokens": request.config.max_tokens.get(),
            "stream": true,
        });

        // Add temperature
        if let Some(temp) = request.config.temperature {
            body["temperature"] = JsonValue::Number(
                serde_json::Number::from_f64(temp.get() as f64).unwrap_or_else(|| 1.into()),
            );
        }

        // Add stop sequences
        if !request.config.stop_sequences.is_empty() {
            body["stop"] = JsonValue::Array(
                request
                    .config
                    .stop_sequences
                    .iter()
                    .map(|s| JsonValue::String(s.clone()))
                    .collect(),
            );
        }

        // Add tools (OpenAI function format)
        if !request.tools.is_empty() {
            body["tools"] = ToolFormat::openai().serialize_tools(&request.tools);
        }

        // Add provider preferences if configured
        if let Some(ref prefs) = self.config.provider_preferences {
            let mut provider_obj = serde_json::json!({});
            if !prefs.allow.is_empty() {
                provider_obj["allow"] = JsonValue::Array(
                    prefs.allow.iter().map(|s| JsonValue::String(s.clone())).collect(),
                );
            }
            if !prefs.deny.is_empty() {
                provider_obj["deny"] = JsonValue::Array(
                    prefs.deny.iter().map(|s| JsonValue::String(s.clone())).collect(),
                );
            }
            if prefs.require_primary {
                provider_obj["require_primary"] = JsonValue::Bool(true);
            }
            body["provider"] = provider_obj;
        }

        // Build messages array (OpenAI format)
        let mut messages = Vec::new();

        // Add system prompt as first message
        if let Some(ref system) = request.system {
            if !system.is_empty() {
                messages.push(serde_json::json!({
                    "role": "system",
                    "content": system.as_str()
                }));
            }
        }

        // Add conversation messages
        for msg in &request.messages {
            messages.push(self.serialize_message(msg));
        }

        body["messages"] = JsonValue::Array(messages);
        body
    }

    /// Serialize a message for the OpenAI-compatible API
    fn serialize_message(&self, msg: &crate::types::Message) -> JsonValue {
        let role = match msg.role {
            crate::types::Role::User => "user",
            crate::types::Role::Assistant => "assistant",
            crate::types::Role::Tool => "tool",
            crate::types::Role::System => "system",
        };

        // Handle tool messages specially
        if msg.role == crate::types::Role::Tool {
            // Tool results in OpenAI format
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
                    return serde_json::json!({
                        "role": "tool",
                        "tool_call_id": tool_use_id.as_str(),
                        "content": content_str
                    });
                }
            }
        }

        // Handle assistant messages with tool calls
        if msg.role == crate::types::Role::Assistant {
            let text: String = msg
                .content
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    ContentBlock::Thinking { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("");

            let tool_calls: Vec<JsonValue> = msg
                .content
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::ToolUse { id, name, input } => Some(serde_json::json!({
                        "id": id.as_str(),
                        "type": "function",
                        "function": {
                            "name": name.as_str(),
                            "arguments": serde_json::to_string(input).unwrap_or_default()
                        }
                    })),
                    _ => None,
                })
                .collect();

            let mut message = serde_json::json!({ "role": role });
            if !text.is_empty() {
                message["content"] = JsonValue::String(text);
            }
            if !tool_calls.is_empty() {
                message["tool_calls"] = JsonValue::Array(tool_calls);
            }
            return message;
        }

        // Regular user/system messages
        let content: String = msg
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");

        serde_json::json!({
            "role": role,
            "content": content
        })
    }

    /// Build headers for the API request
    fn build_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        // Bearer token auth
        let auth_value = format!("Bearer {}", self.config.api_key.as_str());
        if let Ok(value) = HeaderValue::from_str(&auth_value) {
            headers.insert(AUTHORIZATION, value);
        }

        // OpenRouter recommended headers
        headers.insert(
            "HTTP-Referer",
            HeaderValue::from_static("https://github.com/agent-driver-rs"),
        );
        headers.insert("X-Title", HeaderValue::from_static("agent-driver-rs"));

        headers
    }
}

impl Provider for OpenRouterProvider {
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
            let headers = self.build_headers();

            let request_builder = self
                .client
                .post(OPENROUTER_API_URL)
                .headers(headers)
                .json(&body);

            let event_source = EventSource::new(request_builder)
                .map_err(|e| ProviderError::Stream(StreamError::ConnectionLost(e.to_string())))?;

            let stream = create_openrouter_stream(event_source, ctx.cancellation.clone());

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
            // OpenRouter has a models endpoint, but for now return common ones
            Ok(vec![
                ModelInfo {
                    id: ModelId::new("anthropic/claude-sonnet-4").unwrap(),
                    name: "Claude Sonnet 4".to_string(),
                    context_window: Some(200_000),
                },
                ModelInfo {
                    id: ModelId::new("anthropic/claude-opus-4").unwrap(),
                    name: "Claude Opus 4".to_string(),
                    context_window: Some(200_000),
                },
                ModelInfo {
                    id: ModelId::new("openai/gpt-4o").unwrap(),
                    name: "GPT-4o".to_string(),
                    context_window: Some(128_000),
                },
                ModelInfo {
                    id: ModelId::new("google/gemini-2.0-flash").unwrap(),
                    name: "Gemini 2.0 Flash".to_string(),
                    context_window: Some(1_000_000),
                },
                ModelInfo {
                    id: ModelId::new("meta-llama/llama-3.3-70b").unwrap(),
                    name: "Llama 3.3 70B".to_string(),
                    context_window: Some(128_000),
                },
            ])
        })
    }
}

/// Create a stream from OpenRouter SSE events
fn create_openrouter_stream(
    event_source: EventSource,
    cancellation: tokio_util::sync::CancellationToken,
) -> impl futures::Stream<Item = Result<StreamEvent, StreamError>> {
    futures::stream::unfold(
        (event_source, cancellation, StreamState::default()),
        |(mut es, cancel, mut state)| async move {
            loop {
                if cancel.is_cancelled() {
                    return None;
                }

                tokio::select! {
                    biased;

                    _ = cancel.cancelled() => {
                        return None;
                    }

                    event = es.next() => {
                        match event {
                            Some(Ok(Event::Open)) => continue,
                            Some(Ok(Event::Message(msg))) => {
                                // OpenAI-style [DONE] marker
                                if msg.data == "[DONE]" {
                                    // Send completion event
                                    let event = StreamEvent::Completed {
                                        metadata: CompletionMetadata {
                                            model: state.model.clone(),
                                            stop_reason: state.stop_reason,
                                            usage: state.usage,
                                        },
                                    };
                                    return Some((Ok(event), (es, cancel, state)));
                                }

                                match parse_openrouter_event(&msg.data, &mut state) {
                                    Some(Ok(events)) => {
                                        if let Some(first) = events.into_iter().next() {
                                            return Some((Ok(first), (es, cancel, state)));
                                        }
                                        continue;
                                    }
                                    Some(Err(e)) => {
                                        return Some((Err(e), (es, cancel, state)));
                                    }
                                    None => continue,
                                }
                            }
                            Some(Err(e)) => {
                                let err = StreamError::ConnectionLost(e.to_string());
                                return Some((Err(err), (es, cancel, state)));
                            }
                            None => return None,
                        }
                    }
                }
            }
        },
    )
}

/// State for tracking stream parsing
#[derive(Default)]
struct StreamState {
    model: Option<String>,
    stop_reason: Option<StopReason>,
    usage: Option<TokenUsage>,
    current_tool_call_id: Option<String>,
    current_tool_name: Option<String>,
    started: bool,
}

/// Parse an OpenRouter SSE event (OpenAI-compatible format)
fn parse_openrouter_event(
    data: &str,
    state: &mut StreamState,
) -> Option<Result<Vec<StreamEvent>, StreamError>> {
    let parsed: OpenRouterStreamChunk = serde_json::from_str(data)
        .map_err(|e| StreamError::Deserialize(e.to_string()))
        .ok()?;

    let mut events = Vec::new();

    // Track model
    if state.model.is_none() {
        state.model = Some(parsed.model.clone());
    }

    // Send started event on first chunk
    if !state.started {
        state.started = true;
        events.push(StreamEvent::Started {
            metadata: CompletionMetadata {
                model: Some(parsed.model.clone()),
                stop_reason: None,
                usage: None,
            },
        });
        events.push(StreamEvent::ContentBlockStart {
            index: 0,
            block_type: ContentBlockType::Text,
        });
    }

    for choice in &parsed.choices {
        // Check finish reason
        if let Some(ref reason) = choice.finish_reason {
            state.stop_reason = Some(match reason.as_str() {
                "stop" => StopReason::EndTurn,
                "length" => StopReason::MaxTokens,
                "tool_calls" => StopReason::ToolUse,
                "content_filter" => StopReason::EndTurn,
                _ => StopReason::EndTurn,
            });
        }

        // Process delta
        if let Some(ref delta) = choice.delta {
            // Text content
            if let Some(ref content) = delta.content {
                if !content.is_empty() {
                    events.push(StreamEvent::Delta(StreamDelta::TextDelta {
                        text: content.clone(),
                    }));
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

                            events.push(StreamEvent::ContentBlockStart {
                                index: tc.index.unwrap_or(0),
                                block_type: ContentBlockType::ToolUse,
                            });

                            if let Some(ref id) = state.current_tool_call_id {
                                events.push(StreamEvent::Delta(StreamDelta::ToolUseStart {
                                    id: ToolCallId::new(id),
                                    name: ToolName::new(name)
                                        .unwrap_or_else(|_| ToolName::new("unknown").unwrap()),
                                }));
                            }
                        }

                        // Function arguments (tool input delta)
                        if let Some(ref args) = function.arguments {
                            if !args.is_empty() {
                                if let Some(ref id) = state.current_tool_call_id {
                                    events.push(StreamEvent::Delta(StreamDelta::ToolInputDelta {
                                        id: ToolCallId::new(id),
                                        partial_json: args.clone(),
                                    }));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Check for usage in response
    if let Some(ref usage) = parsed.usage {
        state.usage = Some(TokenUsage {
            input_tokens: usage.prompt_tokens.unwrap_or(0),
            output_tokens: usage.completion_tokens.unwrap_or(0),
        });
    }

    if events.is_empty() {
        None
    } else {
        Some(Ok(events))
    }
}

// OpenRouter/OpenAI streaming types

#[derive(Debug, Deserialize)]
struct OpenRouterStreamChunk {
    #[serde(default)]
    model: String,
    choices: Vec<ChunkChoice>,
    usage: Option<UsageData>,
}

#[derive(Debug, Deserialize)]
struct ChunkChoice {
    delta: Option<DeltaData>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DeltaData {
    content: Option<String>,
    tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
struct ToolCallDelta {
    index: Option<usize>,
    id: Option<String>,
    function: Option<FunctionDelta>,
}

#[derive(Debug, Deserialize)]
struct FunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UsageData {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_text_chunk() {
        let data = r#"{"id":"gen-123","model":"anthropic/claude-sonnet-4","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#;

        let mut state = StreamState::default();
        let result = parse_openrouter_event(data, &mut state);

        assert!(result.is_some());
        let events = result.unwrap().unwrap();
        // First chunk: Started + ContentBlockStart + TextDelta
        assert!(events.len() >= 1);
    }

    #[test]
    fn parse_finish_chunk() {
        let data = r#"{"id":"gen-123","model":"anthropic/claude-sonnet-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#;

        let mut state = StreamState::default();
        state.started = true; // Simulate already started
        let _ = parse_openrouter_event(data, &mut state);

        // Should update stop_reason in state
        assert_eq!(state.stop_reason, Some(StopReason::EndTurn));
    }

    #[test]
    fn parse_tool_call_chunk() {
        let data = r#"{"id":"gen-123","model":"anthropic/claude-sonnet-4","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_abc","function":{"name":"read_file","arguments":""}}]},"finish_reason":null}]}"#;

        let mut state = StreamState::default();
        state.started = true;
        let result = parse_openrouter_event(data, &mut state);

        assert!(result.is_some());
        let events = result.unwrap().unwrap();
        assert!(!events.is_empty());
    }
}

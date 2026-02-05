//! Streaming types for LLM completions
//!
//! This module provides types for handling streaming responses from LLM providers.

use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use tokio_util::sync::CancellationToken;

use crate::error::StreamError;
use crate::types::{ContentBlock, CorrelationId, ToolCallId, ToolName};

/// Incremental content during streaming
#[derive(Debug, Clone)]
pub enum StreamDelta {
    /// Text content being generated
    TextDelta { text: String },

    /// Thinking/reasoning content (Claude extended thinking, OpenAI o1/o3)
    ThinkingDelta { thinking: String },

    /// Thinking block signature (Claude extended thinking)
    SignatureDelta { signature: String },

    /// Start of a tool use block
    ToolUseStart { id: ToolCallId, name: ToolName },

    /// Incremental JSON input for a tool call
    ToolInputDelta { id: ToolCallId, partial_json: String },
}

/// Stream events with proper block lifecycle
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Stream started, initial metadata
    Started { metadata: CompletionMetadata },

    /// Content block started (needed to know block type before deltas)
    ContentBlockStart {
        index: usize,
        block_type: ContentBlockType,
    },

    /// Incremental content
    Delta(StreamDelta),

    /// Content block finished
    ContentBlockStop { index: usize },

    /// Full block available (accumulated from deltas)
    BlockComplete { index: usize, block: ContentBlock },

    /// Stream completed successfully
    Completed { metadata: CompletionMetadata },

    /// Stream error
    Error { error: StreamError },
}

/// Type of content block
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentBlockType {
    Text,
    Thinking,
    ToolUse,
}

/// Metadata about a completion
#[derive(Debug, Clone, Default)]
pub struct CompletionMetadata {
    pub model: Option<String>,
    pub stop_reason: Option<StopReason>,
    pub usage: Option<TokenUsage>,
}

/// Reason the completion stopped
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    ToolUse,
    StopSequence,
}

/// Token usage information
#[derive(Debug, Clone, Copy, Default)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// Pending tool use accumulator
#[derive(Debug, Clone)]
struct PendingToolUse {
    id: ToolCallId,
    name: ToolName,
    input_json: String,
}

/// Fully collected response from streaming
#[derive(Debug, Clone, Default)]
pub struct CollectedResponse {
    pub content: Vec<ContentBlock>,
    pub metadata: CompletionMetadata,
    // Internal: accumulation state for streaming
    pending_text: String,
    pending_thinking: String,
    pending_tool_use: Option<PendingToolUse>,
}

impl CollectedResponse {
    /// Create an empty response
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply a stream delta to accumulate content
    pub fn apply_delta(&mut self, delta: StreamDelta) {
        match delta {
            StreamDelta::TextDelta { text } => {
                self.pending_text.push_str(&text);
            }
            StreamDelta::ThinkingDelta { thinking } => {
                self.pending_thinking.push_str(&thinking);
            }
            StreamDelta::ToolUseStart { id, name } => {
                self.pending_tool_use = Some(PendingToolUse {
                    id,
                    name,
                    input_json: String::new(),
                });
            }
            StreamDelta::ToolInputDelta { partial_json, .. } => {
                if let Some(ref mut pending) = self.pending_tool_use {
                    pending.input_json.push_str(&partial_json);
                }
            }
            StreamDelta::SignatureDelta { .. } => {
                // Signatures are metadata, not content - ignore for collection
            }
        }
    }

    /// Finalize a content block (called on ContentBlockStop)
    pub fn finalize_block(&mut self, block_type: ContentBlockType) {
        match block_type {
            ContentBlockType::Text if !self.pending_text.is_empty() => {
                self.content.push(ContentBlock::Text {
                    text: std::mem::take(&mut self.pending_text),
                });
            }
            ContentBlockType::Thinking if !self.pending_thinking.is_empty() => {
                self.content.push(ContentBlock::Thinking {
                    text: std::mem::take(&mut self.pending_thinking),
                });
            }
            ContentBlockType::ToolUse => {
                if let Some(pending) = self.pending_tool_use.take() {
                    let input = serde_json::from_str(&pending.input_json)
                        .unwrap_or(JsonValue::Null);
                    self.content.push(ContentBlock::ToolUse {
                        id: pending.id,
                        name: pending.name,
                        input,
                    });
                }
            }
            _ => {}
        }
    }

    /// Get text content as a single string (convenience method)
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Get thinking content as a single string
    pub fn thinking(&self) -> String {
        self.content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Thinking { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Get all tool use blocks
    pub fn tool_uses(&self) -> Vec<(&ToolCallId, &ToolName, &JsonValue)> {
        self.content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolUse { id, name, input } => Some((id, name, input)),
                _ => None,
            })
            .collect()
    }

    /// Check if the response contains any tool use requests
    pub fn has_tool_use(&self) -> bool {
        self.content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolUse { .. }))
    }
}

/// Type alias for the stream of completion events
pub type CompletionStream = Pin<Box<dyn Stream<Item = Result<StreamEvent, StreamError>> + Send>>;

/// Handle to streaming completion with cancellation
pub struct StreamHandle {
    stream: CompletionStream,
    cancellation: CancellationToken,
    correlation_id: CorrelationId,
}

impl std::fmt::Debug for StreamHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamHandle")
            .field("correlation_id", &self.correlation_id)
            .field("is_cancelled", &self.cancellation.is_cancelled())
            .finish_non_exhaustive()
    }
}

impl StreamHandle {
    /// Create a new stream handle
    pub fn new(
        stream: CompletionStream,
        cancellation: CancellationToken,
        correlation_id: CorrelationId,
    ) -> Self {
        Self {
            stream,
            cancellation,
            correlation_id,
        }
    }

    /// Request cancellation of the stream
    pub fn cancel(&self) {
        self.cancellation.cancel();
    }

    /// Check if cancellation has been requested
    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }

    /// Get the correlation ID
    pub fn correlation_id(&self) -> CorrelationId {
        self.correlation_id
    }

    /// Get a reference to the cancellation token
    pub fn cancellation_token(&self) -> &CancellationToken {
        &self.cancellation
    }

    /// Convert to the inner stream
    pub fn into_stream(self) -> CompletionStream {
        self.stream
    }

    /// Collect entire stream into response (blocks until complete or cancelled)
    ///
    /// Uses tokio::select! to properly handle cancellation during await.
    pub async fn collect(self) -> Result<CollectedResponse, StreamError> {
        use futures::StreamExt;

        // Destructure self to avoid partial move issues
        let Self {
            mut stream,
            cancellation,
            correlation_id: _,
        } = self;

        let mut response = CollectedResponse::default();
        let mut current_block_type: Option<ContentBlockType> = None;

        loop {
            // Use select! to race stream polling against cancellation
            tokio::select! {
                biased; // Check cancellation first

                _ = cancellation.cancelled() => {
                    return Err(StreamError::Cancelled);
                }

                event_opt = stream.next() => {
                    match event_opt {
                        Some(Ok(event)) => match event {
                            StreamEvent::ContentBlockStart { block_type, .. } => {
                                current_block_type = Some(block_type);
                            }
                            StreamEvent::Delta(delta) => {
                                response.apply_delta(delta);
                            }
                            StreamEvent::ContentBlockStop { .. } => {
                                if let Some(block_type) = current_block_type.take() {
                                    response.finalize_block(block_type);
                                }
                            }
                            StreamEvent::BlockComplete { .. } => {
                                // Ignored - we use finalize_block on ContentBlockStop instead
                            }
                            StreamEvent::Completed { metadata } => {
                                response.metadata = metadata;
                                return Ok(response);
                            }
                            StreamEvent::Started { .. } => {}
                            StreamEvent::Error { error } => {
                                return Err(error);
                            }
                        },
                        Some(Err(e)) => return Err(e),
                        None => return Ok(response), // Stream ended
                    }
                }
            }
        }
    }
}

// Implement Stream trait for StreamHandle with cancellation checking
impl Stream for StreamHandle {
    type Item = Result<StreamEvent, StreamError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Check cancellation first - if cancelled, return error and end stream
        if self.cancellation.is_cancelled() {
            return Poll::Ready(None); // End stream cleanly; collect() returns Cancelled error
        }

        // Safe to project: CompletionStream is Pin<Box<...>> which is Unpin
        Pin::new(&mut self.stream).poll_next(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collected_response_apply_text_delta() {
        let mut response = CollectedResponse::new();
        response.apply_delta(StreamDelta::TextDelta {
            text: "Hello ".into(),
        });
        response.apply_delta(StreamDelta::TextDelta {
            text: "world!".into(),
        });
        response.finalize_block(ContentBlockType::Text);

        assert_eq!(response.text(), "Hello world!");
    }

    #[test]
    fn collected_response_apply_thinking_delta() {
        let mut response = CollectedResponse::new();
        response.apply_delta(StreamDelta::ThinkingDelta {
            thinking: "Let me think...".into(),
        });
        response.finalize_block(ContentBlockType::Thinking);

        assert_eq!(response.thinking(), "Let me think...");
    }

    #[test]
    fn collected_response_tool_use() {
        let mut response = CollectedResponse::new();
        response.apply_delta(StreamDelta::ToolUseStart {
            id: ToolCallId::new("call_123"),
            name: ToolName::new("read_file").unwrap(),
        });
        response.apply_delta(StreamDelta::ToolInputDelta {
            id: ToolCallId::new("call_123"),
            partial_json: r#"{"path": "/test"}"#.into(),
        });
        response.finalize_block(ContentBlockType::ToolUse);

        assert!(response.has_tool_use());
        let tool_uses = response.tool_uses();
        assert_eq!(tool_uses.len(), 1);
        assert_eq!(tool_uses[0].0.as_str(), "call_123");
        assert_eq!(tool_uses[0].1.as_str(), "read_file");
    }
}

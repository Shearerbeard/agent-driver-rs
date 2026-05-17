//! Streaming types for LLM completions.
//!
//! This module provides the event-driven streaming infrastructure:
//! - [`StreamEvent`] -- lifecycle events emitted during streaming (started, delta, completed)
//! - [`StreamDelta`] -- incremental content updates (text, thinking, tool use)
//! - [`StreamHandle`] -- owned handle to a running stream with cancellation support
//! - [`CollectedResponse`] -- accumulator that assembles deltas into complete content blocks
//! - [`CompletionStream`] -- the underlying `Pin<Box<dyn Stream>>` type alias
//!
//! All providers emit the same `StreamEvent` types, so consumers write
//! provider-agnostic streaming code.

use std::collections::HashMap;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use tokio_util::sync::CancellationToken;

use crate::error::StreamError;
use crate::types::{ContentBlock, CorrelationId, ModelId, ToolCallId, ToolName};

/// Incremental content during streaming
#[derive(Debug, Clone)]
#[non_exhaustive]
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
    ToolInputDelta {
        id: ToolCallId,
        partial_json: String,
    },
}

/// Stream events with proper block lifecycle
#[derive(Debug, Clone)]
#[non_exhaustive]
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

/// Metadata about a completion, available at stream start and end.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct CompletionMetadata {
    /// The model that generated this completion (provider-reported).
    ///
    /// Uses `ModelId` to maintain the newtype pattern. Providers convert
    /// API-returned model strings via `ModelId::new().ok()`, so this is
    /// `None` if the provider doesn't report a model or if the reported
    /// string fails validation.
    pub model: Option<ModelId>,
    /// Why the completion stopped (end of turn, max tokens, tool use, etc.).
    pub stop_reason: Option<StopReason>,
    /// Token usage statistics (input and output token counts).
    pub usage: Option<TokenUsage>,
}

/// Reason the completion stopped
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    ToolUse,
    StopSequence,
    /// The provider's content filter triggered, blocking further output.
    ContentFilter,
}

/// Token usage information reported by the provider.
#[derive(Debug, Clone, Copy, Default)]
pub struct TokenUsage {
    /// Number of tokens in the input (prompt + message history).
    pub input_tokens: u32,
    /// Number of tokens generated in the output.
    pub output_tokens: u32,
}

/// Pending tool use accumulator
#[derive(Debug, Clone)]
struct PendingToolUse {
    id: ToolCallId,
    name: ToolName,
    input_json: String,
}

/// Fully collected response from streaming.
///
/// Built by accumulating [`StreamDelta`]s via [`apply_delta`](Self::apply_delta) and
/// [`finalize_block`](Self::finalize_block), or by calling [`StreamHandle::collect`].
///
/// # Example
///
/// ```
/// use agent_driver_rs::streaming::{CollectedResponse, StreamDelta, ContentBlockType};
///
/// let mut response = CollectedResponse::new();
/// response.apply_delta(StreamDelta::TextDelta { text: "Hello!".into() });
/// response.finalize_block(ContentBlockType::Text);
/// assert_eq!(response.text(), "Hello!");
/// ```
#[derive(Debug, Clone, Default)]
pub struct CollectedResponse {
    /// The accumulated content blocks (text, thinking, tool use).
    pub content: Vec<ContentBlock>,
    /// Metadata from the completion (model, stop reason, token usage).
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
                // If there's already a pending tool use, finalize it first.
                // This handles parallel tool calls from providers that don't
                // emit ContentBlockStop between tool calls (e.g., OpenRouter).
                if let Some(prev) = self.pending_tool_use.take() {
                    // Fallback to empty object if accumulated JSON fragments are incomplete.
                    let input = serde_json::from_str(&prev.input_json)
                        .unwrap_or_else(|_| serde_json::json!({}));
                    self.content.push(ContentBlock::ToolUse {
                        id: prev.id,
                        name: prev.name,
                        input,
                    });
                }
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
                    // Fallback to empty object if accumulated JSON fragments are incomplete
                    // (e.g., stream interrupted mid-tool-input).
                    let input = serde_json::from_str(&pending.input_json)
                        .unwrap_or_else(|_| serde_json::json!({}));
                    self.content.push(ContentBlock::ToolUse {
                        id: pending.id,
                        name: pending.name,
                        input,
                    });
                }
            }
            // Guard-failed fallthrough: empty pending text/thinking
            ContentBlockType::Text | ContentBlockType::Thinking => {}
        }
    }

    /// Flush any pending (unfinalized) content blocks.
    ///
    /// This should be called when the stream ends (e.g., on `Completed`) to ensure
    /// any accumulated content that wasn't explicitly closed with `ContentBlockStop`
    /// is captured. Some providers (e.g., OpenRouter) don't emit `ContentBlockStop`.
    pub fn flush_pending(&mut self) {
        if !self.pending_text.is_empty() {
            self.content.push(ContentBlock::Text {
                text: std::mem::take(&mut self.pending_text),
            });
        }
        if !self.pending_thinking.is_empty() {
            self.content.push(ContentBlock::Thinking {
                text: std::mem::take(&mut self.pending_thinking),
            });
        }
        if let Some(pending) = self.pending_tool_use.take() {
            // Fallback to empty object if accumulated JSON fragments are incomplete
            // (e.g., stream ended without explicit ContentBlockStop).
            let input =
                serde_json::from_str(&pending.input_json).unwrap_or_else(|_| serde_json::json!({}));
            self.content.push(ContentBlock::ToolUse {
                id: pending.id,
                name: pending.name,
                input,
            });
        }
    }

    /// Get text content as a single string (convenience method)
    #[must_use]
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                ContentBlock::Thinking { .. }
                | ContentBlock::ToolUse { .. }
                | ContentBlock::ToolResult { .. } => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Get thinking content as a single string
    #[must_use]
    pub fn thinking(&self) -> String {
        self.content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Thinking { text } => Some(text.as_str()),
                ContentBlock::Text { .. }
                | ContentBlock::ToolUse { .. }
                | ContentBlock::ToolResult { .. } => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Get all tool use blocks
    #[must_use]
    pub fn tool_uses(&self) -> Vec<(&ToolCallId, &ToolName, &JsonValue)> {
        self.content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolUse { id, name, input } => Some((id, name, input)),
                ContentBlock::Text { .. }
                | ContentBlock::Thinking { .. }
                | ContentBlock::ToolResult { .. } => None,
            })
            .collect()
    }

    /// Check if the response contains any tool use requests
    #[must_use]
    pub fn has_tool_use(&self) -> bool {
        self.content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolUse { .. }))
    }

    /// Scan text blocks for embedded tool calls and convert them to `ToolUse` blocks.
    ///
    /// This is a safety net for models that emit tool calls as text instead of using
    /// the native structured format (common with Qwen, GLM, Ministral, etc.).
    ///
    /// Returns the number of tool calls extracted. Extracted calls are appended to
    /// `self.content` and the source text is cleaned up in the original text block.
    ///
    /// Supported patterns (checked in order of specificity):
    /// 1. `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` — XML-tagged
    /// 2. `` ```json\n{"name": "...", "arguments": {...}}\n``` `` — fenced code block
    /// 3. Bare JSON object with `"name"` + `"arguments"` keys at end of text
    pub fn extract_fallback_tool_calls(&mut self) -> usize {
        let mut extracted = Vec::new();
        let mut counter = 0u32;

        // Process each text block, collecting tool calls and cleaned text
        let mut new_content = Vec::new();
        for block in self.content.drain(..) {
            match block {
                ContentBlock::Text { ref text } => {
                    let (calls, remaining) = parse_embedded_tool_calls(text, &mut counter);
                    extracted.extend(calls);
                    if !remaining.trim().is_empty() {
                        new_content.push(ContentBlock::Text { text: remaining });
                    }
                }
                ContentBlock::Thinking { .. }
                | ContentBlock::ToolUse { .. }
                | ContentBlock::ToolResult { .. } => new_content.push(block),
            }
        }

        let count = extracted.len();
        self.content = new_content;
        self.content.extend(extracted);

        if count > 0 {
            // Update stop reason to ToolUse since we found tool calls
            self.metadata.stop_reason = Some(StopReason::ToolUse);
        }

        count
    }
}

/// Parse embedded tool calls from a text string.
///
/// Returns (extracted_tool_use_blocks, remaining_text).
#[allow(clippy::string_slice)] // offsets from .find() on ASCII delimiters are always valid
fn parse_embedded_tool_calls(
    text: &str,
    counter: &mut u32,
) -> (Vec<ContentBlock>, String) {
    let mut tool_calls = Vec::new();
    let mut remaining = text.to_owned();

    // Pattern 1: <tool_call>...</tool_call>
    while let Some(start) = remaining.find("<tool_call>") {
        if let Some(end) = remaining[start..].find("</tool_call>") {
            let json_start = start + "<tool_call>".len();
            let json_end = start + end;
            let json_str = remaining[json_start..json_end].trim();

            if let Some(block) = try_parse_tool_call_json(json_str, counter) {
                tool_calls.push(block);
                let tag_end = json_end + "</tool_call>".len();
                remaining = format!("{}{}", &remaining[..start], &remaining[tag_end..]);
                continue;
            }
        }
        break;
    }

    // Pattern 2: ```json\n{...}\n``` (fenced code blocks)
    while let Some(fence_start) = remaining.find("```json") {
        let content_start = fence_start + "```json".len();
        let fence_end = if let Some(pos) = remaining[content_start..].find("```") {
            content_start + pos
        } else {
            break;
        };

        let json_str = remaining[content_start..fence_end].trim();
        if let Some(block) = try_parse_tool_call_json(json_str, counter) {
            tool_calls.push(block);
            let tag_end = fence_end + "```".len();
            remaining = format!("{}{}", &remaining[..fence_start], &remaining[tag_end..]);
        } else {
            break;
        }
    }

    // Pattern 3: Bare JSON object at end of text with "name" + "arguments" keys.
    // Scan from right to left for '{' positions, trying each as a potential
    // tool call object start. We search right-to-left because tool calls
    // typically appear at the end of text.
    if tool_calls.is_empty() {
        let trimmed = remaining.trim();
        let bytes = trimmed.as_bytes();
        let mut pos = bytes.len();
        while pos > 0 {
            pos -= 1;
            if bytes[pos] == b'{' {
                let candidate = &trimmed[pos..];
                if let Some(block) = try_parse_tool_call_json(candidate, counter) {
                    tool_calls.push(block);
                    remaining = trimmed[..pos].to_string();
                    break;
                }
            }
        }
    }

    (tool_calls, remaining)
}

/// Try to parse a JSON string as a tool call with "name" and "arguments" fields.
fn try_parse_tool_call_json(json_str: &str, counter: &mut u32) -> Option<ContentBlock> {
    let parsed: serde_json::Value = serde_json::from_str(json_str).ok()?;
    let obj = parsed.as_object()?;

    let name_str = obj.get("name")?.as_str()?;
    let arguments = obj.get("arguments")?;

    // Validate name is a valid ToolName
    let name = ToolName::new(name_str).ok()?;

    let id = ToolCallId::new(format!("fallback_call_{}", counter));
    *counter += 1;

    Some(ContentBlock::ToolUse {
        id,
        name,
        input: arguments.clone(),
    })
}

/// Type alias for the stream of completion events
pub type CompletionStream = Pin<Box<dyn Stream<Item = Result<StreamEvent, StreamError>> + Send>>;

/// Handle to a streaming completion with built-in cancellation support.
///
/// Obtained from [`Provider::complete_stream`](crate::provider::Provider::complete_stream)
/// or [`Session::send_streaming`](crate::session::Session::send_streaming). Implements
/// the `Stream` trait, so you can consume events one at a time, or call
/// [`collect`](Self::collect) to accumulate everything into a [`CollectedResponse`].
///
/// # Cancellation latency
///
/// The [`Stream`] implementation performs a **synchronous** `is_cancelled()`
/// check at the start of each [`poll_next`](futures::Stream::poll_next) call.
/// This means that if the inner stream is blocked waiting for a network read
/// (i.e., the inner `poll_next` returned `Poll::Pending` and registered a
/// waker), cancellation will **not** be observed until the inner stream
/// yields its next item or error, which then triggers another `poll_next`
/// on this `StreamHandle`.
///
/// In contrast, [`collect()`](Self::collect) uses `tokio::select!` with
/// `biased` to race the cancellation future against the stream, so it
/// responds to cancellation even while the inner stream is parked. **If
/// responsive cancellation is important, prefer `collect()` or use
/// `tokio::select!` manually when consuming the stream.**
///
/// Provider implementations further mitigate this by checking
/// `ctx.cancellation` inside their own stream loops (see the
/// [cancellation contract](crate::provider::Provider#cancellation-contract)),
/// so in practice the inner stream also exits promptly on cancellation.
///
/// # Example
///
/// ```no_run
/// # async fn example(session: &agent_driver_rs::Session) -> Result<(), Box<dyn std::error::Error>> {
/// use futures::StreamExt;
///
/// let mut handle = session.send_streaming("Hello!").await?;
///
/// // Option A: consume events one at a time
/// while let Some(event) = handle.next().await {
///     println!("{:?}", event?);
/// }
///
/// // Option B: collect all events into a response
/// // let response = handle.collect().await?;
/// // println!("{}", response.text());
/// # Ok(())
/// # }
/// ```
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
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }

    /// Get the correlation ID
    #[must_use]
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
    /// Block types are tracked per index via a HashMap so that interleaved
    /// ContentBlockStart/ContentBlockStop pairs (across different indices)
    /// are finalized with the correct type.
    pub async fn collect(self) -> Result<CollectedResponse, StreamError> {
        use futures::StreamExt as _;

        // Destructure self to avoid partial move issues
        let Self {
            mut stream,
            cancellation,
            correlation_id: _,
        } = self;

        let mut response = CollectedResponse::default();
        let mut block_types: HashMap<usize, ContentBlockType> = HashMap::new();

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
                            StreamEvent::ContentBlockStart { index, block_type } => {
                                block_types.insert(index, block_type);
                            }
                            StreamEvent::Delta(delta) => {
                                response.apply_delta(delta);
                            }
                            StreamEvent::ContentBlockStop { index } => {
                                if let Some(block_type) = block_types.remove(&index) {
                                    response.finalize_block(block_type);
                                }
                            }
                            StreamEvent::BlockComplete { .. } => {
                                // Ignored - we use finalize_block on ContentBlockStop instead
                            }
                            StreamEvent::Completed { metadata } => {
                                response.flush_pending();
                                response.metadata = metadata;
                                return Ok(response);
                            }
                            StreamEvent::Started { .. } => {}
                            StreamEvent::Error { error } => {
                                return Err(error);
                            }
                        },
                        Some(Err(e)) => return Err(e),
                        None => {
                            response.flush_pending();
                            return Ok(response); // Stream ended
                        }
                    }
                }
            }
        }
    }
}

// Implement Stream trait for StreamHandle with cancellation checking.
//
// NOTE: The `is_cancelled()` check here is synchronous and only runs when
// `poll_next` is called. If the inner stream has returned `Pending` and
// registered a waker for a network read, cancellation won't be noticed
// until the waker fires and this method is called again. For responsive
// cancellation while awaiting, use `collect()` or `tokio::select!`
// externally. See the "Cancellation latency" section on `StreamHandle`.
impl Stream for StreamHandle {
    type Item = Result<StreamEvent, StreamError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Check cancellation first - if cancelled, end stream cleanly.
        // This is a synchronous (non-blocking) check, so it only catches
        // cancellation that occurred *before* this poll_next invocation.
        // Cancellation that occurs while the inner stream is Pending will
        // not be observed until the next poll_next call.
        if self.cancellation.is_cancelled() {
            return Poll::Ready(None);
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

    /// Verify that malformed/empty tool input JSON produces `{}` (empty object),
    /// not `null`. This ensures `ContentBlock::ToolUse { input }` always contains
    /// a valid object that passes MCP schema validation.
    #[test]
    fn tool_use_with_empty_input_produces_empty_object() {
        let mut response = CollectedResponse::new();
        response.apply_delta(StreamDelta::ToolUseStart {
            id: ToolCallId::new("call_empty"),
            name: ToolName::new("list_dirs").unwrap(),
        });
        // No ToolInputDelta — simulates a tool with no arguments
        response.finalize_block(ContentBlockType::ToolUse);

        let tool_uses = response.tool_uses();
        assert_eq!(tool_uses.len(), 1);
        assert!(
            tool_uses[0].2.is_object(),
            "tool input should be an object, got: {:?}",
            tool_uses[0].2,
        );
    }

    /// Verify that interrupted/malformed tool input JSON produces `{}`,
    /// not `null`, when flushed at stream end.
    #[test]
    fn tool_use_with_malformed_input_produces_empty_object() {
        let mut response = CollectedResponse::new();
        response.apply_delta(StreamDelta::ToolUseStart {
            id: ToolCallId::new("call_bad"),
            name: ToolName::new("broken_tool").unwrap(),
        });
        response.apply_delta(StreamDelta::ToolInputDelta {
            id: ToolCallId::new("call_bad"),
            partial_json: "{\"truncated\":".into(),
        });
        // Flush without finalize — simulates stream ending abruptly
        response.flush_pending();

        let tool_uses = response.tool_uses();
        assert_eq!(tool_uses.len(), 1);
        assert!(
            tool_uses[0].2.is_object(),
            "malformed input should fall back to empty object, got: {:?}",
            tool_uses[0].2,
        );
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

    // --- Fallback tool parsing tests ---

    #[test]
    fn extract_xml_tagged_tool_call() {
        let mut response = CollectedResponse::new();
        response.content.push(ContentBlock::Text {
            text: "I'll read that file for you.\n<tool_call>{\"name\": \"read_file\", \"arguments\": {\"path\": \"/tmp/test.txt\"}}</tool_call>".into(),
        });

        let count = response.extract_fallback_tool_calls();
        assert_eq!(count, 1);
        assert!(response.has_tool_use());

        let tool_uses = response.tool_uses();
        assert_eq!(tool_uses[0].1.as_str(), "read_file");
        assert_eq!(tool_uses[0].0.as_str(), "fallback_call_0");

        // Text should be preserved without the tag
        let text = response.text();
        assert!(text.contains("read that file"));
        assert!(!text.contains("<tool_call>"));
    }

    #[test]
    fn extract_json_code_block() {
        let mut response = CollectedResponse::new();
        response.content.push(ContentBlock::Text {
            text: "Let me list the directory.\n```json\n{\"name\": \"list_dir\", \"arguments\": {\"path\": \"/tmp\"}}\n```".into(),
        });

        let count = response.extract_fallback_tool_calls();
        assert_eq!(count, 1);
        assert!(response.has_tool_use());

        let tool_uses = response.tool_uses();
        assert_eq!(tool_uses[0].1.as_str(), "list_dir");
    }

    #[test]
    fn extract_bare_json() {
        let mut response = CollectedResponse::new();
        response.content.push(ContentBlock::Text {
            text: "I'll read the file now.\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"/a.txt\"}}".into(),
        });

        let count = response.extract_fallback_tool_calls();
        assert_eq!(count, 1);
        assert!(response.has_tool_use());
    }

    #[test]
    fn extract_multiple_calls() {
        let mut response = CollectedResponse::new();
        response.content.push(ContentBlock::Text {
            text: "<tool_call>{\"name\": \"read_file\", \"arguments\": {\"path\": \"/a.txt\"}}</tool_call>\n<tool_call>{\"name\": \"list_dir\", \"arguments\": {\"path\": \"/tmp\"}}</tool_call>".into(),
        });

        let count = response.extract_fallback_tool_calls();
        assert_eq!(count, 2);

        let tool_uses = response.tool_uses();
        assert_eq!(tool_uses.len(), 2);
        assert_eq!(tool_uses[0].1.as_str(), "read_file");
        assert_eq!(tool_uses[1].1.as_str(), "list_dir");
    }

    #[test]
    fn no_false_positives() {
        let mut response = CollectedResponse::new();
        response.content.push(ContentBlock::Text {
            text: "Here's some normal text about JSON objects and tool calls. Nothing to extract here.".into(),
        });

        let count = response.extract_fallback_tool_calls();
        assert_eq!(count, 0);
        assert!(!response.has_tool_use());
        assert!(!response.text().is_empty());
    }

    #[test]
    fn remaining_text_preserved() {
        let mut response = CollectedResponse::new();
        response.content.push(ContentBlock::Text {
            text: "Before the call.\n<tool_call>{\"name\": \"echo\", \"arguments\": {}}</tool_call>\nAfter the call.".into(),
        });

        response.extract_fallback_tool_calls();

        let text = response.text();
        assert!(text.contains("Before the call."));
        assert!(text.contains("After the call."));
        assert!(!text.contains("<tool_call>"));
    }
}

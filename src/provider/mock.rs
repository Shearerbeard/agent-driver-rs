//! Mock provider for deterministic testing without network calls.
//!
//! [`MockProvider`] pops pre-configured response sequences from a queue on each
//! `complete_stream()` call. Helper functions create the standard event sequences
//! for text and tool-call responses.

use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Mutex;

use futures::Future;

use crate::error::ProviderError;
use crate::streaming::{
    CompletionMetadata, CompletionStream, ContentBlockType, StopReason, StreamDelta, StreamEvent,
    StreamHandle,
};
use crate::types::{ModelId, ToolCallId, ToolName};

use super::{
    CompletionRequest, ModelInfo, Provider, ProviderCapabilities, ProviderContext, ProviderInfo,
    ProviderKind,
};

/// A deterministic provider for testing.
///
/// Each call to `complete_stream()` pops the next response sequence from an
/// internal queue. Panics if the queue is exhausted (indicates a test bug).
pub struct MockProvider {
    responses: Mutex<VecDeque<Vec<StreamEvent>>>,
    info: ProviderInfo,
}

impl MockProvider {
    /// Create a mock provider with the given response sequences.
    ///
    /// Each inner `Vec<StreamEvent>` represents one complete stream response.
    pub fn new(responses: Vec<Vec<StreamEvent>>) -> Self {
        Self {
            responses: Mutex::new(VecDeque::from(responses)),
            info: ProviderInfo {
                kind: ProviderKind::Anthropic,
                name: "Mock",
                capabilities: ProviderCapabilities {
                    streaming: true,
                    tools: true,
                    vision: false,
                    extended_thinking: false,
                    max_context_tokens: Some(200_000),
                },
            },
        }
    }
}

impl Provider for MockProvider {
    fn info(&self) -> &ProviderInfo {
        &self.info
    }

    fn complete_stream(
        &self,
        _request: CompletionRequest,
        ctx: ProviderContext,
    ) -> Pin<Box<dyn Future<Output = Result<StreamHandle, ProviderError>> + Send + '_>> {
        let events = self
            .responses
            .lock()
            .expect("MockProvider lock poisoned")
            .pop_front()
            .expect("MockProvider: no more responses queued (test bug)");

        Box::pin(async move {
            let stream: CompletionStream =
                Box::pin(futures::stream::iter(events.into_iter().map(Ok)));
            Ok(StreamHandle::new(
                stream,
                ctx.cancellation,
                ctx.correlation_id,
            ))
        })
    }

    fn list_models(
        &self,
        _ctx: ProviderContext,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<ModelInfo>, ProviderError>> + Send + '_>> {
        Box::pin(async { Ok(vec![]) })
    }
}

/// Create a full event sequence for a simple text response.
///
/// Produces: Started -> ContentBlockStart(Text) -> TextDelta -> ContentBlockStop -> Completed
pub fn mock_text_response(text: &str) -> Vec<StreamEvent> {
    vec![
        StreamEvent::Started {
            metadata: CompletionMetadata {
                model: ModelId::new("mock-model").ok(),
                stop_reason: None,
                usage: None,
            },
        },
        StreamEvent::ContentBlockStart {
            index: 0,
            block_type: ContentBlockType::Text,
        },
        StreamEvent::Delta(StreamDelta::TextDelta {
            text: text.to_string(),
        }),
        StreamEvent::ContentBlockStop { index: 0 },
        StreamEvent::Completed {
            metadata: CompletionMetadata {
                model: ModelId::new("mock-model").ok(),
                stop_reason: Some(StopReason::EndTurn),
                usage: None,
            },
        },
    ]
}

/// Create a full event sequence for a tool call response.
///
/// Produces: Started -> ContentBlockStart(ToolUse) -> ToolUseStart -> ToolInputDelta
///           -> ContentBlockStop -> Completed(stop_reason=ToolUse)
pub fn mock_tool_call_response(id: &str, name: &str, input_json: &str) -> Vec<StreamEvent> {
    vec![
        StreamEvent::Started {
            metadata: CompletionMetadata {
                model: ModelId::new("mock-model").ok(),
                stop_reason: None,
                usage: None,
            },
        },
        StreamEvent::ContentBlockStart {
            index: 0,
            block_type: ContentBlockType::ToolUse,
        },
        StreamEvent::Delta(StreamDelta::ToolUseStart {
            id: ToolCallId::new(id),
            name: ToolName::new(name).expect("invalid tool name in mock"),
        }),
        StreamEvent::Delta(StreamDelta::ToolInputDelta {
            id: ToolCallId::new(id),
            partial_json: input_json.to_string(),
        }),
        StreamEvent::ContentBlockStop { index: 0 },
        StreamEvent::Completed {
            metadata: CompletionMetadata {
                model: ModelId::new("mock-model").ok(),
                stop_reason: Some(StopReason::ToolUse),
                usage: None,
            },
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    #[tokio::test]
    async fn mock_provider_returns_queued_responses() {
        let provider = MockProvider::new(vec![
            mock_text_response("first"),
            mock_text_response("second"),
        ]);
        let ctx = ProviderContext::default();
        let request = CompletionRequest::new(
            ModelId::new("mock-model").unwrap(),
            vec![],
        );

        let handle1 = provider.complete_stream(request.clone(), ctx.clone()).await.unwrap();
        let events1: Vec<_> = handle1.into_stream().collect().await;
        assert!(events1.iter().any(|e| matches!(
            e,
            Ok(StreamEvent::Delta(StreamDelta::TextDelta { text })) if text == "first"
        )));

        let handle2 = provider.complete_stream(request, ctx).await.unwrap();
        let events2: Vec<_> = handle2.into_stream().collect().await;
        assert!(events2.iter().any(|e| matches!(
            e,
            Ok(StreamEvent::Delta(StreamDelta::TextDelta { text })) if text == "second"
        )));
    }

    #[test]
    fn mock_text_response_has_correct_structure() {
        let events = mock_text_response("hello");
        assert_eq!(events.len(), 5);
        assert!(matches!(&events[0], StreamEvent::Started { .. }));
        assert!(matches!(&events[1], StreamEvent::ContentBlockStart { block_type: ContentBlockType::Text, .. }));
        assert!(matches!(&events[2], StreamEvent::Delta(StreamDelta::TextDelta { text }) if text == "hello"));
        assert!(matches!(&events[3], StreamEvent::ContentBlockStop { .. }));
        assert!(matches!(&events[4], StreamEvent::Completed { metadata } if metadata.stop_reason == Some(StopReason::EndTurn)));
    }

    #[test]
    fn mock_tool_call_response_has_correct_structure() {
        let events = mock_tool_call_response("call_1", "my_tool", "{\"key\": \"val\"}");
        assert_eq!(events.len(), 6);
        assert!(matches!(&events[0], StreamEvent::Started { .. }));
        assert!(matches!(&events[1], StreamEvent::ContentBlockStart { block_type: ContentBlockType::ToolUse, .. }));
        assert!(matches!(&events[4], StreamEvent::ContentBlockStop { .. }));
        assert!(matches!(&events[5], StreamEvent::Completed { metadata } if metadata.stop_reason == Some(StopReason::ToolUse)));
    }
}

//! Mock provider for deterministic testing without network calls.
//!
//! [`MockProvider`] pops pre-configured response sequences from a queue on each
//! `complete_stream()` call. Helper functions create the standard event sequences
//! for text and tool-call responses.

#![allow(
    clippy::expect_used,
    reason = "mock provider panics identify invalid test fixtures or exhausted test queues"
)]

use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Mutex;

use futures::Future;

use crate::error::{ProviderError, StreamError};
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
            text: text.to_owned(),
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
            partial_json: input_json.to_owned(),
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

/// Create a full event sequence with multiple tool_use blocks in one response.
///
/// Each tuple is `(id, name, input_json)`. Produces one ContentBlockStart/ToolUseStart/
/// ToolInputDelta/ContentBlockStop sequence per tool, then a single Completed event.
pub fn mock_multi_tool_response(tools: &[(&str, &str, &str)]) -> Vec<StreamEvent> {
    let mut events = vec![StreamEvent::Started {
        metadata: CompletionMetadata {
            model: ModelId::new("mock-model").ok(),
            stop_reason: None,
            usage: None,
        },
    }];

    for (i, (id, name, input_json)) in tools.iter().enumerate() {
        events.push(StreamEvent::ContentBlockStart {
            index: i,
            block_type: ContentBlockType::ToolUse,
        });
        events.push(StreamEvent::Delta(StreamDelta::ToolUseStart {
            id: ToolCallId::new(*id),
            name: ToolName::new(*name).expect("invalid tool name in mock"),
        }));
        events.push(StreamEvent::Delta(StreamDelta::ToolInputDelta {
            id: ToolCallId::new(*id),
            partial_json: (*input_json).to_owned(),
        }));
        events.push(StreamEvent::ContentBlockStop { index: i });
    }

    events.push(StreamEvent::Completed {
        metadata: CompletionMetadata {
            model: ModelId::new("mock-model").ok(),
            stop_reason: Some(StopReason::ToolUse),
            usage: None,
        },
    });

    events
}

/// Create a event sequence that yields a `StreamEvent::Error` mid-stream.
///
/// Produces: Started -> Error. The stream ends after the error event.
pub fn mock_error_response(error: StreamError) -> Vec<StreamEvent> {
    vec![
        StreamEvent::Started {
            metadata: CompletionMetadata {
                model: ModelId::new("mock-model").ok(),
                stop_reason: None,
                usage: None,
            },
        },
        StreamEvent::Error { error },
    ]
}

/// Create a full event sequence with a thinking block followed by a text block.
///
/// Produces: Started -> ContentBlockStart(Thinking) -> ThinkingDelta -> ContentBlockStop
///           -> ContentBlockStart(Text) -> TextDelta -> ContentBlockStop -> Completed
pub fn mock_thinking_response(thinking: &str, text: &str) -> Vec<StreamEvent> {
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
            block_type: ContentBlockType::Thinking,
        },
        StreamEvent::Delta(StreamDelta::ThinkingDelta {
            thinking: thinking.to_owned(),
        }),
        StreamEvent::ContentBlockStop { index: 0 },
        StreamEvent::ContentBlockStart {
            index: 1,
            block_type: ContentBlockType::Text,
        },
        StreamEvent::Delta(StreamDelta::TextDelta {
            text: text.to_owned(),
        }),
        StreamEvent::ContentBlockStop { index: 1 },
        StreamEvent::Completed {
            metadata: CompletionMetadata {
                model: ModelId::new("mock-model").ok(),
                stop_reason: Some(StopReason::EndTurn),
                usage: None,
            },
        },
    ]
}

/// Create a full event sequence for a text response that ends with `StopReason::ContentFilter`.
///
/// Simulates the provider's content filter triggering after producing some partial text.
/// Produces: Started -> ContentBlockStart(Text) -> TextDelta -> ContentBlockStop -> Completed(ContentFilter)
pub fn mock_content_filter_response(partial_text: &str) -> Vec<StreamEvent> {
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
            text: partial_text.to_owned(),
        }),
        StreamEvent::ContentBlockStop { index: 0 },
        StreamEvent::Completed {
            metadata: CompletionMetadata {
                model: ModelId::new("mock-model").ok(),
                stop_reason: Some(StopReason::ContentFilter),
                usage: None,
            },
        },
    ]
}

/// Create a full event sequence with interleaved text and tool_use blocks.
///
/// Produces: Started -> ContentBlockStart(Text) -> TextDelta -> ContentBlockStop
///           -> ContentBlockStart(ToolUse) -> ToolUseStart -> ToolInputDelta
///           -> ContentBlockStop -> Completed(stop_reason=ToolUse)
pub fn mock_mixed_text_tool_response(
    text: &str,
    tool_id: &str,
    tool_name: &str,
    tool_input: &str,
) -> Vec<StreamEvent> {
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
            text: text.to_owned(),
        }),
        StreamEvent::ContentBlockStop { index: 0 },
        StreamEvent::ContentBlockStart {
            index: 1,
            block_type: ContentBlockType::ToolUse,
        },
        StreamEvent::Delta(StreamDelta::ToolUseStart {
            id: ToolCallId::new(tool_id),
            name: ToolName::new(tool_name).expect("invalid tool name in mock"),
        }),
        StreamEvent::Delta(StreamDelta::ToolInputDelta {
            id: ToolCallId::new(tool_id),
            partial_json: tool_input.to_owned(),
        }),
        StreamEvent::ContentBlockStop { index: 1 },
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
        let request = CompletionRequest::new(ModelId::new("mock-model").unwrap(), vec![]);

        let handle1 = provider
            .complete_stream(request.clone(), ctx.clone())
            .await
            .unwrap();
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
        assert!(matches!(
            &events[1],
            StreamEvent::ContentBlockStart {
                block_type: ContentBlockType::Text,
                ..
            }
        ));
        assert!(
            matches!(&events[2], StreamEvent::Delta(StreamDelta::TextDelta { text }) if text == "hello")
        );
        assert!(matches!(&events[3], StreamEvent::ContentBlockStop { .. }));
        assert!(
            matches!(&events[4], StreamEvent::Completed { metadata } if metadata.stop_reason == Some(StopReason::EndTurn))
        );
    }

    #[test]
    fn mock_tool_call_response_has_correct_structure() {
        let events = mock_tool_call_response("call_1", "my_tool", "{\"key\": \"val\"}");
        assert_eq!(events.len(), 6);
        assert!(matches!(&events[0], StreamEvent::Started { .. }));
        assert!(matches!(
            &events[1],
            StreamEvent::ContentBlockStart {
                block_type: ContentBlockType::ToolUse,
                ..
            }
        ));
        assert!(matches!(&events[4], StreamEvent::ContentBlockStop { .. }));
        assert!(
            matches!(&events[5], StreamEvent::Completed { metadata } if metadata.stop_reason == Some(StopReason::ToolUse))
        );
    }

    #[test]
    fn mock_multi_tool_response_has_correct_structure() {
        let events = mock_multi_tool_response(&[
            ("call_1", "tool_a", "{}"),
            ("call_2", "tool_b", "{\"x\": 1}"),
        ]);
        // Started + 2*(BlockStart + ToolUseStart + ToolInputDelta + BlockStop) + Completed = 10
        assert_eq!(events.len(), 10);
        assert!(matches!(&events[0], StreamEvent::Started { .. }));
        assert!(
            matches!(&events[9], StreamEvent::Completed { metadata } if metadata.stop_reason == Some(StopReason::ToolUse))
        );

        // First tool block at indices 1-4
        assert!(matches!(
            &events[1],
            StreamEvent::ContentBlockStart {
                index: 0,
                block_type: ContentBlockType::ToolUse
            }
        ));
        assert!(matches!(
            &events[4],
            StreamEvent::ContentBlockStop { index: 0 }
        ));

        // Second tool block at indices 5-8
        assert!(matches!(
            &events[5],
            StreamEvent::ContentBlockStart {
                index: 1,
                block_type: ContentBlockType::ToolUse
            }
        ));
        assert!(matches!(
            &events[8],
            StreamEvent::ContentBlockStop { index: 1 }
        ));
    }

    #[test]
    fn mock_error_response_has_correct_structure() {
        let events = mock_error_response(StreamError::ConnectionLost {
            kind: crate::error::StreamErrorKind::TransportError,
            message: "gone".into(),
        });
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], StreamEvent::Started { .. }));
        assert!(
            matches!(&events[1], StreamEvent::Error { error } if matches!(error, StreamError::ConnectionLost { .. }))
        );
    }

    #[test]
    fn mock_thinking_response_has_correct_structure() {
        let events = mock_thinking_response("let me think", "the answer");
        assert_eq!(events.len(), 8);
        assert!(matches!(&events[0], StreamEvent::Started { .. }));
        assert!(matches!(
            &events[1],
            StreamEvent::ContentBlockStart {
                index: 0,
                block_type: ContentBlockType::Thinking
            }
        ));
        assert!(
            matches!(&events[2], StreamEvent::Delta(StreamDelta::ThinkingDelta { thinking }) if thinking == "let me think")
        );
        assert!(matches!(
            &events[3],
            StreamEvent::ContentBlockStop { index: 0 }
        ));
        assert!(matches!(
            &events[4],
            StreamEvent::ContentBlockStart {
                index: 1,
                block_type: ContentBlockType::Text
            }
        ));
        assert!(
            matches!(&events[5], StreamEvent::Delta(StreamDelta::TextDelta { text }) if text == "the answer")
        );
        assert!(matches!(
            &events[6],
            StreamEvent::ContentBlockStop { index: 1 }
        ));
        assert!(
            matches!(&events[7], StreamEvent::Completed { metadata } if metadata.stop_reason == Some(StopReason::EndTurn))
        );
    }

    #[test]
    fn mock_content_filter_response_has_correct_structure() {
        let events = mock_content_filter_response("partial output");
        assert_eq!(events.len(), 5);
        assert!(matches!(&events[0], StreamEvent::Started { .. }));
        assert!(matches!(
            &events[1],
            StreamEvent::ContentBlockStart {
                block_type: ContentBlockType::Text,
                ..
            }
        ));
        assert!(
            matches!(&events[2], StreamEvent::Delta(StreamDelta::TextDelta { text }) if text == "partial output")
        );
        assert!(matches!(&events[3], StreamEvent::ContentBlockStop { .. }));
        assert!(
            matches!(&events[4], StreamEvent::Completed { metadata } if metadata.stop_reason == Some(StopReason::ContentFilter))
        );
    }

    #[test]
    fn mock_mixed_text_tool_response_has_correct_structure() {
        let events = mock_mixed_text_tool_response("thinking out loud", "call_1", "my_tool", "{}");
        assert_eq!(events.len(), 9);
        assert!(matches!(&events[0], StreamEvent::Started { .. }));
        // Text block
        assert!(matches!(
            &events[1],
            StreamEvent::ContentBlockStart {
                index: 0,
                block_type: ContentBlockType::Text
            }
        ));
        assert!(
            matches!(&events[2], StreamEvent::Delta(StreamDelta::TextDelta { text }) if text == "thinking out loud")
        );
        assert!(matches!(
            &events[3],
            StreamEvent::ContentBlockStop { index: 0 }
        ));
        // Tool block
        assert!(matches!(
            &events[4],
            StreamEvent::ContentBlockStart {
                index: 1,
                block_type: ContentBlockType::ToolUse
            }
        ));
        assert!(matches!(
            &events[7],
            StreamEvent::ContentBlockStop { index: 1 }
        ));
        assert!(
            matches!(&events[8], StreamEvent::Completed { metadata } if metadata.stop_reason == Some(StopReason::ToolUse))
        );
    }
}

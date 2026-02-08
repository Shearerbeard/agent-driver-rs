//! Integration tests for the streaming system.
//!
//! These tests exercise `StreamHandle`, `CollectedResponse`, and the mock provider
//! helpers end-to-end, covering error propagation, multi-tool streams, mixed content
//! blocks, cancellation, incomplete streams, and manual event consumption.

use agent_driver_rs::error::StreamError;
use agent_driver_rs::provider::mock::*;
use agent_driver_rs::provider::{CompletionRequest, Provider, ProviderContext};
use agent_driver_rs::streaming::{
    CompletionMetadata, CompletionStream, ContentBlockType, StopReason, StreamDelta, StreamEvent,
    StreamHandle,
};
use agent_driver_rs::types::{CorrelationId, ModelId};
use futures::StreamExt;
use tokio_util::sync::CancellationToken;

/// Helper: create a `CompletionRequest` with the mock model and no messages.
fn mock_request() -> CompletionRequest {
    CompletionRequest::new(ModelId::new("mock-model").unwrap(), vec![])
}

// ---------------------------------------------------------------------------
// 1. mid_stream_error_propagates
// ---------------------------------------------------------------------------

#[tokio::test]
async fn mid_stream_error_propagates() {
    let events = mock_error_response(StreamError::ConnectionLost {
        kind: agent_driver_rs::error::StreamErrorKind::TransportError,
        message: "gone".into(),
    });
    let provider = MockProvider::new(vec![events]);
    let ctx = ProviderContext::default();

    let handle = provider.complete_stream(mock_request(), ctx).await.unwrap();
    let result = handle.collect().await;

    assert!(
        result.is_err(),
        "collect() should return Err on stream error"
    );
    assert!(
        matches!(result.unwrap_err(), StreamError::ConnectionLost { ref message, .. } if message == "gone"),
        "error should be ConnectionLost with the original message",
    );
}

// ---------------------------------------------------------------------------
// 2. multiple_tool_uses_one_stream
// ---------------------------------------------------------------------------

#[tokio::test]
async fn multiple_tool_uses_one_stream() {
    let events = mock_multi_tool_response(&[
        ("call_1", "tool_a", "{}"),
        ("call_2", "tool_b", "{\"x\": 1}"),
    ]);
    let provider = MockProvider::new(vec![events]);
    let ctx = ProviderContext::default();

    let handle = provider.complete_stream(mock_request(), ctx).await.unwrap();
    let response = handle.collect().await.expect("collect should succeed");

    let tool_uses = response.tool_uses();
    assert_eq!(tool_uses.len(), 2, "should have exactly 2 tool uses");
    assert_eq!(tool_uses[0].1.as_str(), "tool_a");
    assert_eq!(tool_uses[1].1.as_str(), "tool_b");
}

// ---------------------------------------------------------------------------
// 3. mixed_text_and_tool_blocks
// ---------------------------------------------------------------------------

#[tokio::test]
async fn mixed_text_and_tool_blocks() {
    let events = mock_mixed_text_tool_response("thinking out loud", "call_1", "my_tool", "{}");
    let provider = MockProvider::new(vec![events]);
    let ctx = ProviderContext::default();

    let handle = provider.complete_stream(mock_request(), ctx).await.unwrap();
    let response = handle.collect().await.expect("collect should succeed");

    assert_eq!(response.text(), "thinking out loud");
    assert!(response.has_tool_use(), "response should contain tool use");
    assert_eq!(response.tool_uses().len(), 1);
}

// ---------------------------------------------------------------------------
// 4. thinking_plus_text_blocks
// ---------------------------------------------------------------------------

#[tokio::test]
async fn thinking_plus_text_blocks() {
    let events = mock_thinking_response("let me think about this", "the answer is 42");
    let provider = MockProvider::new(vec![events]);
    let ctx = ProviderContext::default();

    let handle = provider.complete_stream(mock_request(), ctx).await.unwrap();
    let response = handle.collect().await.expect("collect should succeed");

    assert_eq!(response.thinking(), "let me think about this");
    assert_eq!(response.text(), "the answer is 42");
}

// ---------------------------------------------------------------------------
// 5. stream_handle_cancellation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stream_handle_cancellation() {
    // Create a cancellation token and cancel it immediately.
    let cancel = CancellationToken::new();
    cancel.cancel();

    // Build a stream from a text response -- the events will never be consumed
    // because collect() should detect cancellation first.
    let stream: CompletionStream = Box::pin(futures::stream::iter(
        mock_text_response("never seen").into_iter().map(Ok),
    ));
    let handle = StreamHandle::new(stream, cancel, CorrelationId::generate());

    let result = handle.collect().await;
    assert!(
        matches!(result, Err(StreamError::Cancelled)),
        "collect() on a pre-cancelled handle should return StreamError::Cancelled",
    );
}

// ---------------------------------------------------------------------------
// 6. flush_pending_on_incomplete
// ---------------------------------------------------------------------------

#[tokio::test]
async fn flush_pending_on_incomplete() {
    // Build a custom event sequence with no ContentBlockStop before Completed.
    // The collect() method calls flush_pending() on Completed, so the partial
    // text should still be captured.
    let mut start_meta = CompletionMetadata::default();
    start_meta.model = ModelId::new("mock-model").ok();

    let mut end_meta = CompletionMetadata::default();
    end_meta.model = ModelId::new("mock-model").ok();
    end_meta.stop_reason = Some(StopReason::EndTurn);

    let events = vec![
        StreamEvent::Started {
            metadata: start_meta,
        },
        StreamEvent::ContentBlockStart {
            index: 0,
            block_type: ContentBlockType::Text,
        },
        StreamEvent::Delta(StreamDelta::TextDelta {
            text: "partial".to_string(),
        }),
        // Intentionally no ContentBlockStop here.
        StreamEvent::Completed { metadata: end_meta },
    ];

    let provider = MockProvider::new(vec![events]);
    let ctx = ProviderContext::default();

    let handle = provider.complete_stream(mock_request(), ctx).await.unwrap();
    let response = handle.collect().await.expect("collect should succeed");

    assert_eq!(
        response.text(),
        "partial",
        "flush_pending should capture text even without ContentBlockStop",
    );
}

// ---------------------------------------------------------------------------
// 7. manual_stream_consumption
// ---------------------------------------------------------------------------

#[tokio::test]
async fn manual_stream_consumption() {
    let events = mock_text_response("hello");
    let provider = MockProvider::new(vec![events]);
    let ctx = ProviderContext::default();

    let mut handle = provider.complete_stream(mock_request(), ctx).await.unwrap();

    // Event 1: Started
    let ev1 = handle.next().await.expect("should have Started event");
    assert!(
        matches!(ev1, Ok(StreamEvent::Started { .. })),
        "first event should be Started, got: {ev1:?}",
    );

    // Event 2: ContentBlockStart(Text)
    let ev2 = handle
        .next()
        .await
        .expect("should have ContentBlockStart event");
    assert!(
        matches!(
            ev2,
            Ok(StreamEvent::ContentBlockStart {
                block_type: ContentBlockType::Text,
                ..
            })
        ),
        "second event should be ContentBlockStart(Text), got: {ev2:?}",
    );

    // Event 3: Delta(TextDelta)
    let ev3 = handle.next().await.expect("should have Delta event");
    assert!(
        matches!(
            &ev3,
            Ok(StreamEvent::Delta(StreamDelta::TextDelta { text })) if text == "hello"
        ),
        "third event should be TextDelta(\"hello\"), got: {ev3:?}",
    );

    // Event 4: ContentBlockStop
    let ev4 = handle
        .next()
        .await
        .expect("should have ContentBlockStop event");
    assert!(
        matches!(ev4, Ok(StreamEvent::ContentBlockStop { .. })),
        "fourth event should be ContentBlockStop, got: {ev4:?}",
    );

    // Event 5: Completed
    let ev5 = handle.next().await.expect("should have Completed event");
    assert!(
        matches!(ev5, Ok(StreamEvent::Completed { .. })),
        "fifth event should be Completed, got: {ev5:?}",
    );

    // Stream should be exhausted
    let ev_none = handle.next().await;
    assert!(
        ev_none.is_none(),
        "stream should be exhausted after Completed, got: {ev_none:?}",
    );
}

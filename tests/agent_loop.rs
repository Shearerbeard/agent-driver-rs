//! Integration tests for the agent loop.
//!
//! These tests exercise the full agent loop through the public API using
//! `MockProvider` for deterministic, network-free testing.
//!
//! Requires `--features test-support` to compile because `MockProvider` is
//! behind `#[cfg(any(test, feature = "test-support"))]` in the library, and
//! integration tests are a separate crate where `cfg(test)` does not apply
//! to the library.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::FutureExt;
use tokio_util::sync::CancellationToken;

use agent_driver_rs::agent::{
    AgentEvent, AgentLoop, AgentLoopConfig, AgentObserver, LoopStopReason, MaxToolDepth,
};
use agent_driver_rs::error::AgentLoopError;
use agent_driver_rs::provider::mock::*;
use agent_driver_rs::session::SessionBuilder;
use agent_driver_rs::tool::{DynTool, FnTool, ToolDefinition, ToolResult, ToolSchema};
use agent_driver_rs::types::{ModelId, ToolName};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn model() -> ModelId {
    ModelId::new("mock-model").unwrap()
}

/// Observer that records all events as short string tags for test assertions.
struct RecordingObserver {
    events: Arc<Mutex<Vec<String>>>,
}

impl RecordingObserver {
    fn new() -> (Self, Arc<Mutex<Vec<String>>>) {
        let events = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                events: events.clone(),
            },
            events,
        )
    }
}

#[async_trait]
impl AgentObserver for RecordingObserver {
    async fn on_event(&self, event: &AgentEvent) {
        let tag = match event {
            AgentEvent::TextDelta { text } => format!("TextDelta:{}", text),
            AgentEvent::ThinkingDelta { .. } => "ThinkingDelta".to_string(),
            AgentEvent::IterationStart { iteration } => format!("IterationStart:{}", iteration),
            AgentEvent::ToolCallStart { name, .. } => format!("ToolCallStart:{}", name),
            AgentEvent::ToolCallComplete { name, .. } => format!("ToolCallComplete:{}", name),
            AgentEvent::IterationComplete { iteration, .. } => {
                format!("IterationComplete:{}", iteration)
            }
            AgentEvent::LoopComplete { reason, .. } => format!("LoopComplete:{}", reason),
            _ => "Unknown".to_string(),
        };
        self.events.lock().unwrap().push(tag);
    }
}

/// Create a simple echo tool that returns "echoed!" for any input.
fn echo_tool() -> DynTool {
    let definition = ToolDefinition::new(
        ToolName::new("echo").unwrap(),
        "Echoes back",
        ToolSchema::empty(),
    );
    Arc::new(FnTool::new(definition, |_input| {
        async { Ok(ToolResult::text("echoed!")) }.boxed()
    }))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Three consecutive tool call rounds followed by a text response.
/// Verifies the loop counts iterations correctly and the final text is returned.
#[tokio::test]
async fn multi_round_tool_chain_depth_3() {
    let provider = MockProvider::new(vec![
        mock_tool_call_response("call_1", "echo", "{}"),
        mock_tool_call_response("call_2", "echo", "{}"),
        mock_tool_call_response("call_3", "echo", "{}"),
        mock_text_response("All done after 3 rounds"),
    ]);

    let session = SessionBuilder::new()
        .with_provider(provider)
        .model(model())
        .tool(echo_tool())
        .build()
        .await
        .unwrap();

    let config = AgentLoopConfig {
        max_tool_depth: MaxToolDepth::new(5).unwrap(),
        continue_on_tool_error: true,
    };

    let outcome = AgentLoop::new(&session)
        .with_config(config)
        .run("go")
        .await
        .unwrap();

    assert_eq!(outcome.iterations, 3);
    assert_eq!(outcome.final_response.text(), "All done after 3 rounds");
    assert!(matches!(outcome.stop_reason, LoopStopReason::EndTurn));

    // History: user + (assistant_tool_use + tool_result)*3 + assistant_text = 1 + 6 + 1 = 8
    let msgs = session.messages().await;
    assert_eq!(
        msgs.len(),
        8,
        "expected 8 messages in history, got {}",
        msgs.len()
    );
}

/// A single response with two parallel tool calls (both "echo").
/// Verifies both tool calls are executed and the loop completes in one iteration.
#[tokio::test]
async fn parallel_tool_calls_single_response() {
    let provider = MockProvider::new(vec![
        mock_multi_tool_response(&[
            ("call_a", "echo", "{}"),
            ("call_b", "echo", "{}"),
        ]),
        mock_text_response("Done!"),
    ]);

    let session = SessionBuilder::new()
        .with_provider(provider)
        .model(model())
        .tool(echo_tool())
        .build()
        .await
        .unwrap();

    let outcome = AgentLoop::new(&session).run("parallel").await.unwrap();

    assert_eq!(outcome.iterations, 1);
    assert_eq!(outcome.final_response.text(), "Done!");
    assert!(matches!(outcome.stop_reason, LoopStopReason::EndTurn));

    // History: user + assistant(2 tool_use) + 2 tool_results + assistant(text) = 1 + 1 + 2 + 1 = 5
    let msgs = session.messages().await;
    assert_eq!(
        msgs.len(),
        5,
        "expected 5 messages in history, got {}",
        msgs.len()
    );
}

/// A tool that returns an error stops the loop when `continue_on_tool_error` is false.
#[tokio::test]
async fn tool_error_stops_loop() {
    let fail_def = ToolDefinition::new(
        ToolName::new("fail_tool").unwrap(),
        "Always fails",
        ToolSchema::empty(),
    );
    let fail_tool: DynTool = Arc::new(FnTool::new(fail_def, |_input| {
        async { Ok(ToolResult::error("boom")) }.boxed()
    }));

    let provider = MockProvider::new(vec![
        mock_tool_call_response("call_1", "fail_tool", "{}"),
        // This text response should NOT be reached
        mock_text_response("should not see this"),
    ]);

    let session = SessionBuilder::new()
        .with_provider(provider)
        .model(model())
        .tool(fail_tool)
        .build()
        .await
        .unwrap();

    let config = AgentLoopConfig {
        max_tool_depth: MaxToolDepth::new(5).unwrap(),
        continue_on_tool_error: false,
    };

    let outcome = AgentLoop::new(&session)
        .with_config(config)
        .run("fail please")
        .await
        .unwrap();

    assert!(
        matches!(outcome.stop_reason, LoopStopReason::ToolError { .. }),
        "expected ToolError stop reason, got: {:?}",
        outcome.stop_reason
    );
    assert_eq!(outcome.iterations, 1);
}

/// A tool that returns an error does NOT stop the loop when `continue_on_tool_error` is true.
/// The loop continues and the model can recover.
#[tokio::test]
async fn tool_error_continues_when_configured() {
    let fail_def = ToolDefinition::new(
        ToolName::new("fail_tool").unwrap(),
        "Always fails",
        ToolSchema::empty(),
    );
    let fail_tool: DynTool = Arc::new(FnTool::new(fail_def, |_input| {
        async { Ok(ToolResult::error("boom")) }.boxed()
    }));

    let provider = MockProvider::new(vec![
        mock_tool_call_response("call_1", "fail_tool", "{}"),
        mock_text_response("recovered"),
    ]);

    let session = SessionBuilder::new()
        .with_provider(provider)
        .model(model())
        .tool(fail_tool)
        .build()
        .await
        .unwrap();

    let config = AgentLoopConfig {
        max_tool_depth: MaxToolDepth::new(5).unwrap(),
        continue_on_tool_error: true,
    };

    let outcome = AgentLoop::new(&session)
        .with_config(config)
        .run("fail then recover")
        .await
        .unwrap();

    assert_eq!(outcome.final_response.text(), "recovered");
    assert!(matches!(outcome.stop_reason, LoopStopReason::EndTurn));
    assert_eq!(outcome.iterations, 1);
}

/// Pre-cancelling the token causes the first `collect_with_observer` to return
/// `Err(AgentLoopError::Cancelled)` immediately, before the loop body executes.
#[tokio::test]
async fn cancellation_stops_loop() {
    let provider = MockProvider::new(vec![
        mock_tool_call_response("call_1", "echo", "{}"),
    ]);

    let session = SessionBuilder::new()
        .with_provider(provider)
        .model(model())
        .tool(echo_tool())
        .build()
        .await
        .unwrap();

    let token = CancellationToken::new();
    token.cancel(); // Cancel BEFORE running

    let result = AgentLoop::new(&session)
        .with_cancellation(token)
        .run("should be cancelled")
        .await;

    assert!(
        matches!(&result, Err(AgentLoopError::Cancelled)),
        "expected Cancelled error, got: {:?}",
        result
    );
}

/// Verifies the full event sequence emitted by the observer during a single
/// tool call round followed by a text response.
#[tokio::test]
async fn observer_event_sequence_full_round() {
    let provider = MockProvider::new(vec![
        mock_tool_call_response("call_1", "echo", "{}"),
        mock_text_response("final answer"),
    ]);

    let session = SessionBuilder::new()
        .with_provider(provider)
        .model(model())
        .tool(echo_tool())
        .build()
        .await
        .unwrap();

    let (observer, events) = RecordingObserver::new();

    let outcome = AgentLoop::new(&session)
        .with_observer(observer)
        .run("observe me")
        .await
        .unwrap();

    assert_eq!(outcome.final_response.text(), "final answer");

    let recorded = events.lock().unwrap();

    // Verify all expected events are present
    assert!(
        recorded.iter().any(|e| e == "IterationStart:1"),
        "missing IterationStart:1, got: {:?}",
        *recorded
    );
    assert!(
        recorded.iter().any(|e| e == "ToolCallStart:echo"),
        "missing ToolCallStart:echo, got: {:?}",
        *recorded
    );
    assert!(
        recorded.iter().any(|e| e == "ToolCallComplete:echo"),
        "missing ToolCallComplete:echo, got: {:?}",
        *recorded
    );
    assert!(
        recorded.iter().any(|e| e == "IterationComplete:1"),
        "missing IterationComplete:1, got: {:?}",
        *recorded
    );
    assert!(
        recorded
            .iter()
            .any(|e| e.starts_with("LoopComplete:end_turn")),
        "missing LoopComplete:end_turn, got: {:?}",
        *recorded
    );
    assert!(
        recorded
            .iter()
            .any(|e| e.starts_with("TextDelta:final answer")),
        "missing TextDelta for final answer, got: {:?}",
        *recorded
    );
}

/// Verifies that ThinkingDelta events from extended thinking are forwarded
/// to the observer.
#[tokio::test]
async fn observer_receives_thinking_deltas() {
    let provider = MockProvider::new(vec![mock_thinking_response(
        "let me think about this",
        "the answer is 42",
    )]);

    let session = SessionBuilder::new()
        .with_provider(provider)
        .model(model())
        .build()
        .await
        .unwrap();

    let (observer, events) = RecordingObserver::new();

    let outcome = AgentLoop::new(&session)
        .with_observer(observer)
        .run("think hard")
        .await
        .unwrap();

    assert_eq!(outcome.final_response.text(), "the answer is 42");

    let recorded = events.lock().unwrap();
    assert!(
        recorded.iter().any(|e| e == "ThinkingDelta"),
        "expected ThinkingDelta event, got: {:?}",
        *recorded
    );
}

/// With `max_tool_depth: 1`, the loop executes one tool round and then stops
/// even if the model wants to call more tools.
#[tokio::test]
async fn max_tool_depth_1() {
    let provider = MockProvider::new(vec![
        mock_tool_call_response("call_1", "echo", "{}"),
        // The loop should stop after depth 1, so this second tool call
        // response will be fetched but the loop will exit at the depth check.
        mock_tool_call_response("call_2", "echo", "{}"),
    ]);

    let session = SessionBuilder::new()
        .with_provider(provider)
        .model(model())
        .tool(echo_tool())
        .build()
        .await
        .unwrap();

    let config = AgentLoopConfig {
        max_tool_depth: MaxToolDepth::new(1).unwrap(),
        continue_on_tool_error: true,
    };

    let outcome = AgentLoop::new(&session)
        .with_config(config)
        .run("only one round")
        .await
        .unwrap();

    assert_eq!(outcome.iterations, 1);
    assert!(
        matches!(outcome.stop_reason, LoopStopReason::MaxToolDepthReached),
        "expected MaxToolDepthReached, got: {:?}",
        outcome.stop_reason
    );
}

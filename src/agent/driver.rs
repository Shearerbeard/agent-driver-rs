//! Agent loop driver — the core orchestrator

use std::collections::HashMap;

use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use crate::error::AgentLoopError;
use crate::session::Session;
use crate::streaming::{
    CollectedResponse, CompletionMetadata, ContentBlockType, StopReason, StreamDelta, StreamEvent,
};
use crate::types::{ContentBlock, Message, Role, ToolName};

use super::config::AgentLoopConfig;
use super::observer::{AgentEvent, AgentObserver, LoopStopReason, NoOpObserver};

/// Outcome of running the agent loop
#[derive(Debug, Clone)]
pub struct AgentOutcome {
    /// The final response from the last model turn
    pub final_response: CollectedResponse,
    /// All responses from every iteration (including intermediate tool-calling turns)
    pub responses: Vec<CollectedResponse>,
    /// Why the loop stopped
    pub stop_reason: LoopStopReason,
    /// Number of tool execution rounds completed
    pub iterations: u32,
}

/// Agent loop that drives multi-turn tool-calling conversations.
///
/// Borrows a `&Session` and orchestrates the send -> tool_use -> execute -> continue cycle.
/// Streaming events are forwarded to an optional observer in real-time.
///
/// The loop counts *tool execution rounds*, not model responses. A single round may
/// contain multiple tool calls executed in sequence. The loop stops when:
/// - The model responds without any `tool_use` blocks (normal end-of-turn)
/// - [`MaxToolDepth`](super::MaxToolDepth) is reached
/// - The cancellation token fires
///
/// # Example
///
/// ```no_run
/// # use agent_driver_rs::agent::{AgentLoop, AgentLoopConfig, MaxToolDepth};
/// # use agent_driver_rs::Session;
/// # async fn example(session: &Session) -> Result<(), agent_driver_rs::AgentLoopError> {
/// // Basic usage
/// let outcome = AgentLoop::new(session)
///     .run("What time is it?")
///     .await?;
/// println!("Response: {}", outcome.final_response.text());
///
/// // With configuration and cancellation
/// let config = AgentLoopConfig {
///     max_tool_depth: MaxToolDepth::new(10)?,
///     continue_on_tool_error: true,
/// };
/// let outcome = AgentLoop::new(session)
///     .with_config(config)
///     .run("Search for recent news")
///     .await?;
/// println!("Completed in {} tool rounds", outcome.iterations);
/// # Ok(())
/// # }
/// ```
pub struct AgentLoop<'s> {
    session: &'s Session,
    config: AgentLoopConfig,
    observer: Box<dyn AgentObserver>,
    cancellation: Option<CancellationToken>,
}

impl<'s> AgentLoop<'s> {
    /// Create a new agent loop bound to a session
    #[must_use]
    pub fn new(session: &'s Session) -> Self {
        Self {
            session,
            config: AgentLoopConfig::default(),
            observer: Box::new(NoOpObserver),
            cancellation: None,
        }
    }

    /// Set the agent loop configuration
    #[must_use]
    pub fn with_config(mut self, config: AgentLoopConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the observer for real-time events
    #[must_use]
    pub fn with_observer(mut self, observer: impl AgentObserver + 'static) -> Self {
        self.observer = Box::new(observer);
        self
    }

    /// Set a cancellation token for the loop
    #[must_use]
    pub fn with_cancellation(mut self, token: CancellationToken) -> Self {
        self.cancellation = Some(token);
        self
    }

    /// Run the agent loop with an initial user message
    ///
    /// Sends the message and then enters the tool execution loop:
    /// 1. Stream the model response, forwarding deltas to the observer
    /// 2. If the response contains tool_use blocks, execute them
    /// 3. Continue streaming with tool results in history
    /// 4. Repeat until no more tool calls, or a limit is hit
    pub async fn run(self, message: impl Into<String>) -> Result<AgentOutcome, AgentLoopError> {
        let cancellation = self
            .cancellation
            .as_ref()
            .cloned()
            .unwrap_or_else(|| self.session.child_token());

        let mut responses: Vec<CollectedResponse> = Vec::new();
        let mut tool_depth: u32 = 0;

        // Step 1: Send the initial user message
        let handle = self.session.send_streaming(message).await?;

        // Step 2: Collect the first response while forwarding events
        let mut response =
            collect_with_observer(handle, &cancellation, self.observer.as_ref()).await?;

        // Add assistant response to history
        add_assistant_to_history(self.session, &response).await;

        // Step 3: Tool execution loop
        loop {
            // Check cancellation
            if cancellation.is_cancelled() {
                return self
                    .complete_loop(response, &mut responses, LoopStopReason::Cancelled, tool_depth)
                    .await;
            }

            // If no tool use, we're done
            if !response.has_tool_use() {
                let reason = stop_reason_from_metadata(&response.metadata);
                return self
                    .complete_loop(response, &mut responses, reason, tool_depth)
                    .await;
            }

            // Check tool depth limit
            if tool_depth >= self.config.max_tool_depth.get() {
                return self
                    .complete_loop(response, &mut responses, LoopStopReason::MaxToolDepthReached, tool_depth)
                    .await;
            }

            tool_depth += 1;

            self.observer
                .on_event(&AgentEvent::IterationStart {
                    iteration: tool_depth,
                })
                .await;

            // Execute tool calls
            let tool_error =
                execute_tools(self.session, self.observer.as_ref(), &response).await;

            responses.push(response);

            // Handle tool errors when continue_on_tool_error is false
            if let Some((failed_tool, err_msg)) = tool_error {
                if !self.config.continue_on_tool_error {
                    let reason = LoopStopReason::ToolError {
                        tool_name: failed_tool,
                        message: err_msg,
                    };
                    self.observer
                        .on_event(&AgentEvent::LoopComplete {
                            reason: reason.clone(),
                            total_iterations: tool_depth,
                        })
                        .await;
                    return Ok(AgentOutcome {
                        final_response: responses.last().cloned().unwrap_or_default(),
                        responses,
                        stop_reason: reason,
                        iterations: tool_depth,
                    });
                }
            }

            // Continue streaming (tool results are already in history)
            let handle = self.session.continue_streaming().await?;

            response =
                collect_with_observer(handle, &cancellation, self.observer.as_ref()).await?;

            // Add assistant response to history
            add_assistant_to_history(self.session, &response).await;

            self.observer
                .on_event(&AgentEvent::IterationComplete {
                    iteration: tool_depth,
                    response: response.clone(),
                })
                .await;
        }
    }

    /// Fire the LoopComplete observer event, push the final response, and return the outcome.
    async fn complete_loop(
        &self,
        response: CollectedResponse,
        responses: &mut Vec<CollectedResponse>,
        reason: LoopStopReason,
        tool_depth: u32,
    ) -> Result<AgentOutcome, AgentLoopError> {
        self.observer
            .on_event(&AgentEvent::LoopComplete {
                reason: reason.clone(),
                total_iterations: tool_depth,
            })
            .await;
        responses.push(response.clone());
        Ok(AgentOutcome {
            final_response: response,
            responses: std::mem::take(responses),
            stop_reason: reason,
            iterations: tool_depth,
        })
    }
}

/// Collect a stream handle into a response while forwarding events to the observer.
///
/// Block types are tracked per index via a HashMap so that interleaved
/// ContentBlockStart/ContentBlockStop pairs (across different indices)
/// are finalized with the correct type.
async fn collect_with_observer(
    handle: crate::streaming::StreamHandle,
    cancellation: &CancellationToken,
    observer: &dyn AgentObserver,
) -> Result<CollectedResponse, AgentLoopError> {
    let stream = handle.into_stream();
    futures::pin_mut!(stream);

    let mut response = CollectedResponse::new();
    let mut block_types: HashMap<usize, ContentBlockType> = HashMap::new();

    loop {
        tokio::select! {
            biased;

            _ = cancellation.cancelled() => {
                return Err(AgentLoopError::Cancelled);
            }

            event_opt = stream.next() => {
                match event_opt {
                    Some(Ok(event)) => {
                        match &event {
                            StreamEvent::Delta(StreamDelta::TextDelta { text }) => {
                                observer
                                    .on_event(&AgentEvent::TextDelta {
                                        text: text.clone(),
                                    })
                                    .await;
                            }
                            StreamEvent::Delta(StreamDelta::ThinkingDelta { thinking }) => {
                                observer
                                    .on_event(&AgentEvent::ThinkingDelta {
                                        thinking: thinking.clone(),
                                    })
                                    .await;
                            }
                            _ => {}
                        }

                        // Apply to response accumulator
                        match event {
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
                            StreamEvent::Completed { metadata } => {
                                response.flush_pending();
                                response.metadata = metadata;
                                return Ok(response);
                            }
                            StreamEvent::Error { error } => {
                                return Err(AgentLoopError::Session(
                                    crate::error::SessionError::Stream(error),
                                ));
                            }
                            _ => {}
                        }
                    }
                    Some(Err(e)) => {
                        return Err(AgentLoopError::Session(
                            crate::error::SessionError::Stream(e),
                        ));
                    }
                    None => {
                        // Stream ended without Completed event
                        response.flush_pending();
                        return Ok(response);
                    }
                }
            }
        }
    }
}

/// Execute tool calls from a response, emitting observer events.
/// Returns `Some((tool_name, error_message))` if any tool execution failed.
async fn execute_tools(
    session: &Session,
    observer: &dyn AgentObserver,
    response: &CollectedResponse,
) -> Option<(ToolName, String)> {
    let mut error_info = None;

    for block in &response.content {
        if let ContentBlock::ToolUse { id, name, input } = block {
            observer
                .on_event(&AgentEvent::ToolCallStart {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                })
                .await;

            let tool_input = match crate::tool::ToolInput::from_value(input.clone()) {
                Ok(ti) => ti,
                Err(e) => {
                    let err = format!("Invalid tool input: {}", e);
                    // Add error result to history so model can see it
                    session
                        .add_message(Message::tool_result(id.clone(), &err, true))
                        .await;
                    observer
                        .on_event(&AgentEvent::ToolCallComplete {
                            id: id.clone(),
                            name: name.clone(),
                            result: err.clone(),
                            is_error: true,
                        })
                        .await;
                    error_info = Some((name.clone(), err));
                    continue;
                }
            };

            match session.execute_tool(id.clone(), name, tool_input).await {
                Ok(result) => {
                    let is_error = result.is_error();
                    let content = result.content().to_string();
                    observer
                        .on_event(&AgentEvent::ToolCallComplete {
                            id: id.clone(),
                            name: name.clone(),
                            result: content,
                            is_error,
                        })
                        .await;
                    if is_error {
                        error_info = Some((name.clone(), result.content().to_string()));
                    }
                }
                Err(e) => {
                    let err = format!("Tool execution error: {}", e);
                    // execute_tool already adds to history on success,
                    // but on error we need to add a result manually
                    session
                        .add_message(Message::tool_result(id.clone(), &err, true))
                        .await;
                    observer
                        .on_event(&AgentEvent::ToolCallComplete {
                            id: id.clone(),
                            name: name.clone(),
                            result: err.clone(),
                            is_error: true,
                        })
                        .await;
                    error_info = Some((name.clone(), err));
                }
            }
        }
    }

    error_info
}

/// Add assistant response content to session history
async fn add_assistant_to_history(session: &Session, response: &CollectedResponse) {
    if !response.content.is_empty() {
        session
            .add_message(Message::with_content(
                Role::Assistant,
                response.content.clone(),
            ))
            .await;
    }
}

/// Map completion metadata stop_reason to LoopStopReason
fn stop_reason_from_metadata(metadata: &CompletionMetadata) -> LoopStopReason {
    match metadata.stop_reason {
        Some(StopReason::EndTurn) | Some(StopReason::ToolUse) => LoopStopReason::EndTurn,
        Some(StopReason::MaxTokens) => LoopStopReason::MaxTokens,
        Some(StopReason::StopSequence) => LoopStopReason::StopSequence,
        Some(StopReason::ContentFilter) => LoopStopReason::EndTurn,
        None => LoopStopReason::EndTurn,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_reason_mapping() {
        let meta = CompletionMetadata {
            stop_reason: Some(StopReason::EndTurn),
            ..Default::default()
        };
        assert!(matches!(
            stop_reason_from_metadata(&meta),
            LoopStopReason::EndTurn
        ));

        let meta = CompletionMetadata {
            stop_reason: Some(StopReason::MaxTokens),
            ..Default::default()
        };
        assert!(matches!(
            stop_reason_from_metadata(&meta),
            LoopStopReason::MaxTokens
        ));

        let meta = CompletionMetadata {
            stop_reason: None,
            ..Default::default()
        };
        assert!(matches!(
            stop_reason_from_metadata(&meta),
            LoopStopReason::EndTurn
        ));
    }

    #[test]
    fn agent_outcome_default_iterations() {
        let outcome = AgentOutcome {
            final_response: CollectedResponse::default(),
            responses: vec![],
            stop_reason: LoopStopReason::EndTurn,
            iterations: 0,
        };
        assert_eq!(outcome.iterations, 0);
    }

    // --- Integration tests using MockProvider ---

    use crate::provider::{mock_text_response, mock_tool_call_response, MockProvider};
    use crate::session::SessionBuilder;
    use crate::tool::{FnTool, ToolDefinition, ToolResult, ToolSchema};
    use crate::types::ModelId;
    use async_trait::async_trait;
    use std::sync::{Arc, Mutex};

    fn model() -> ModelId {
        ModelId::new("mock-model").unwrap()
    }

    /// Observer that records all events for test assertions.
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
                AgentEvent::IterationStart { iteration } => {
                    format!("IterationStart:{}", iteration)
                }
                AgentEvent::ToolCallStart { name, .. } => {
                    format!("ToolCallStart:{}", name)
                }
                AgentEvent::ToolCallComplete { name, .. } => {
                    format!("ToolCallComplete:{}", name)
                }
                AgentEvent::IterationComplete { iteration, .. } => {
                    format!("IterationComplete:{}", iteration)
                }
                AgentEvent::LoopComplete { reason, .. } => {
                    format!("LoopComplete:{}", reason)
                }
            };
            self.events.lock().unwrap().push(tag);
        }
    }

    #[tokio::test]
    async fn single_turn_no_tools() {
        let session = SessionBuilder::new()
            .with_provider(MockProvider::new(vec![mock_text_response("Hello!")]))
            .model(model())
            .build()
            .await
            .unwrap();

        let outcome = AgentLoop::new(&session).run("hi").await.unwrap();

        assert_eq!(outcome.final_response.text(), "Hello!");
        assert_eq!(outcome.iterations, 0);
        assert!(matches!(outcome.stop_reason, LoopStopReason::EndTurn));
    }

    #[tokio::test]
    async fn one_tool_call_round() {
        use futures::FutureExt;

        // First response: tool call, second response: text
        let provider = MockProvider::new(vec![
            mock_tool_call_response("call_1", "echo", "{}"),
            mock_text_response("Done!"),
        ]);

        let definition = ToolDefinition::new(
            ToolName::new("echo").unwrap(),
            "Echoes back",
            ToolSchema::empty(),
        );
        let tool: crate::tool::DynTool = Arc::new(FnTool::new(definition, |_input, _ctx| {
            async { Ok(ToolResult::text("echoed!")) }.boxed()
        }));

        let session = SessionBuilder::new()
            .with_provider(provider)
            .model(model())
            .tool(tool)
            .build()
            .await
            .unwrap();

        let outcome = AgentLoop::new(&session).run("please echo").await.unwrap();

        assert_eq!(outcome.iterations, 1);
        assert_eq!(outcome.final_response.text(), "Done!");
        assert!(matches!(outcome.stop_reason, LoopStopReason::EndTurn));

        // Verify history has: user, assistant(tool_use), tool_result, assistant(text)
        let msgs = session.messages().await;
        assert!(msgs.len() >= 4, "expected at least 4 messages, got {}", msgs.len());
    }

    #[tokio::test]
    async fn max_tool_depth_enforced() {
        use futures::FutureExt;

        // Provider always returns tool calls — the loop should stop at depth 2
        let provider = MockProvider::new(vec![
            mock_tool_call_response("call_1", "echo", "{}"),
            mock_tool_call_response("call_2", "echo", "{}"),
            mock_tool_call_response("call_3", "echo", "{}"),
        ]);

        let definition = ToolDefinition::new(
            ToolName::new("echo").unwrap(),
            "Echoes back",
            ToolSchema::empty(),
        );
        let tool: crate::tool::DynTool = Arc::new(FnTool::new(definition, |_input, _ctx| {
            async { Ok(ToolResult::text("ok")) }.boxed()
        }));

        let config = AgentLoopConfig {
            max_tool_depth: super::super::config::MaxToolDepth::new(2).unwrap(),
            continue_on_tool_error: true,
        };

        let session = SessionBuilder::new()
            .with_provider(provider)
            .model(model())
            .tool(tool)
            .build()
            .await
            .unwrap();

        let outcome = AgentLoop::new(&session)
            .with_config(config)
            .run("loop forever")
            .await
            .unwrap();

        assert_eq!(outcome.iterations, 2);
        assert!(matches!(
            outcome.stop_reason,
            LoopStopReason::MaxToolDepthReached
        ));
    }

    #[tokio::test]
    async fn observer_receives_events() {
        let session = SessionBuilder::new()
            .with_provider(MockProvider::new(vec![mock_text_response("world")]))
            .model(model())
            .build()
            .await
            .unwrap();

        let (observer, events) = RecordingObserver::new();

        let outcome = AgentLoop::new(&session)
            .with_observer(observer)
            .run("hello")
            .await
            .unwrap();

        assert_eq!(outcome.final_response.text(), "world");

        let recorded = events.lock().unwrap();
        assert!(
            recorded.iter().any(|e| e.starts_with("TextDelta:")),
            "expected TextDelta event, got: {:?}",
            *recorded
        );
        assert!(
            recorded.iter().any(|e| e.starts_with("LoopComplete:")),
            "expected LoopComplete event, got: {:?}",
            *recorded
        );
    }
}

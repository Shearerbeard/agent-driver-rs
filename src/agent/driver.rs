//! Agent loop driver — the core orchestrator

use std::collections::HashMap;

use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use crate::error::AgentLoopError;
use crate::session::Session;
use crate::streaming::{
    CollectedResponse, CompletionMetadata, ContentBlockType, StopReason, StreamDelta, StreamEvent,
};
use crate::types::{ContentBlock, Message, Role};

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
                let reason = LoopStopReason::Cancelled;
                self.observer
                    .on_event(&AgentEvent::LoopComplete {
                        reason: reason.clone(),
                        total_iterations: tool_depth,
                    })
                    .await;
                responses.push(response.clone());
                return Ok(AgentOutcome {
                    final_response: response,
                    responses,
                    stop_reason: reason,
                    iterations: tool_depth,
                });
            }

            // If no tool use, we're done
            if !response.has_tool_use() {
                let reason = stop_reason_from_metadata(&response.metadata);
                self.observer
                    .on_event(&AgentEvent::LoopComplete {
                        reason: reason.clone(),
                        total_iterations: tool_depth,
                    })
                    .await;
                responses.push(response.clone());
                return Ok(AgentOutcome {
                    final_response: response,
                    responses,
                    stop_reason: reason,
                    iterations: tool_depth,
                });
            }

            // Check tool depth limit
            if tool_depth >= self.config.max_tool_depth.get() {
                let reason = LoopStopReason::MaxToolDepthReached;
                self.observer
                    .on_event(&AgentEvent::LoopComplete {
                        reason: reason.clone(),
                        total_iterations: tool_depth,
                    })
                    .await;
                responses.push(response.clone());
                return Ok(AgentOutcome {
                    final_response: response,
                    responses,
                    stop_reason: reason,
                    iterations: tool_depth,
                });
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
            if let Some(err_msg) = tool_error {
                if !self.config.continue_on_tool_error {
                    let reason = LoopStopReason::ToolError(err_msg);
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
/// Returns Some(error_message) if any tool execution failed.
async fn execute_tools(
    session: &Session,
    observer: &dyn AgentObserver,
    response: &CollectedResponse,
) -> Option<String> {
    let mut error_msg = None;

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
                    error_msg = Some(err);
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
                        error_msg = Some(result.content().to_string());
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
                    error_msg = Some(err);
                }
            }
        }
    }

    error_msg
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
}

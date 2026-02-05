//! Agent observer trait and event types for real-time loop monitoring.
//!
//! The [`AgentObserver`] trait receives [`AgentEvent`]s as the loop executes,
//! enabling streaming output, progress indicators, logging, and cancellation hooks.
//! The default implementation is a no-op, so implementors only handle events they need.

use async_trait::async_trait;
use serde_json::Value as JsonValue;

use crate::streaming::CollectedResponse;
use crate::types::{ToolCallId, ToolName};

/// Events emitted by the agent loop
///
/// Single enum keeps the vtable small and is forward-compatible:
/// new events don't break existing implementors.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// A new tool execution iteration is starting
    IterationStart { iteration: u32 },

    /// Streaming text content from the model
    TextDelta { text: String },

    /// Streaming thinking/reasoning content from the model
    ThinkingDelta { thinking: String },

    /// A tool call is about to be executed
    ToolCallStart {
        id: ToolCallId,
        name: ToolName,
        input: JsonValue,
    },

    /// A tool call has completed
    ToolCallComplete {
        id: ToolCallId,
        name: ToolName,
        result: String,
        is_error: bool,
    },

    /// A model response iteration has completed
    IterationComplete {
        iteration: u32,
        response: CollectedResponse,
    },

    /// The agent loop has finished
    LoopComplete {
        reason: LoopStopReason,
        total_iterations: u32,
    },
}

/// Why the agent loop stopped
#[derive(Debug, Clone)]
pub enum LoopStopReason {
    /// Model chose to stop (end_turn)
    EndTurn,
    /// Model hit max_tokens limit
    MaxTokens,
    /// Model hit a stop sequence
    StopSequence,
    /// Safety limit: too many tool execution rounds
    MaxToolDepthReached,
    /// Cancellation token was triggered
    Cancelled,
    /// A tool error stopped the loop (when continue_on_tool_error is false)
    ToolError(String),
}

impl std::fmt::Display for LoopStopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EndTurn => write!(f, "end_turn"),
            Self::MaxTokens => write!(f, "max_tokens"),
            Self::StopSequence => write!(f, "stop_sequence"),
            Self::MaxToolDepthReached => write!(f, "max_tool_depth_reached"),
            Self::Cancelled => write!(f, "cancelled"),
            Self::ToolError(e) => write!(f, "tool_error: {}", e),
        }
    }
}

/// Observer for agent loop events
///
/// Implement this trait to receive real-time events from the agent loop.
/// The default implementation does nothing, so implementors only need to
/// handle the events they care about.
///
/// # Example
///
/// ```no_run
/// use agent_driver_rs::agent::{AgentObserver, AgentEvent};
///
/// struct PrintObserver;
///
/// #[async_trait::async_trait]
/// impl AgentObserver for PrintObserver {
///     async fn on_event(&self, event: &AgentEvent) {
///         match event {
///             AgentEvent::TextDelta { text } => print!("{}", text),
///             AgentEvent::ToolCallStart { name, .. } => {
///                 eprintln!("\n[calling tool: {}]", name);
///             }
///             _ => {}
///         }
///     }
/// }
/// ```
#[async_trait]
pub trait AgentObserver: Send + Sync {
    /// Called for each event in the agent loop
    async fn on_event(&self, event: &AgentEvent) {
        let _ = event;
    }
}

/// No-op observer that discards all events
pub struct NoOpObserver;

#[async_trait]
impl AgentObserver for NoOpObserver {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_reason_display() {
        assert_eq!(LoopStopReason::EndTurn.to_string(), "end_turn");
        assert_eq!(LoopStopReason::Cancelled.to_string(), "cancelled");
        assert_eq!(
            LoopStopReason::ToolError("oops".into()).to_string(),
            "tool_error: oops"
        );
    }
}

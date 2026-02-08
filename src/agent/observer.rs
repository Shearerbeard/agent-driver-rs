//! Agent observer trait and event types for real-time loop monitoring.
//!
//! The [`AgentObserver`] trait receives [`AgentEvent`]s as the loop executes,
//! enabling streaming output, progress indicators, logging, and cancellation hooks.
//! The default implementation is a no-op, so implementors only handle events they need.
//!
//! ## Event lifecycle
//!
//! For a single tool-calling round, events arrive in this order:
//!
//! 1. `TextDelta` / `ThinkingDelta` — streamed as the model generates (first response)
//! 2. `IterationStart { iteration: 1 }` — tool execution round begins
//! 3. `ToolCallStart { id, name, input }` — for each tool in the response
//! 4. `ToolCallComplete { id, name, result, is_error }` — after each tool executes
//! 5. `TextDelta` — streamed from the follow-up response
//! 6. `IterationComplete { iteration: 1, response }` — round finished
//! 7. `LoopComplete { reason, total_iterations }` — loop done
//!
//! If the model doesn't call tools, only `TextDelta`s and `LoopComplete` are emitted.
//!
//! ## Custom observer example
//!
//! ```no_run
//! use agent_driver_rs::agent::{AgentObserver, AgentEvent};
//!
//! struct StreamingPrinter;
//!
//! #[async_trait::async_trait]
//! impl AgentObserver for StreamingPrinter {
//!     async fn on_event(&self, event: &AgentEvent) {
//!         match event {
//!             AgentEvent::TextDelta { text } => print!("{}", text),
//!             AgentEvent::ThinkingDelta { thinking } => {
//!                 eprint!("[thinking] {}", thinking);
//!             }
//!             AgentEvent::ToolCallStart { name, .. } => {
//!                 eprintln!("\n> Calling tool: {}", name);
//!             }
//!             AgentEvent::ToolCallComplete { name, is_error, .. } => {
//!                 if *is_error {
//!                     eprintln!("> Tool {} failed", name);
//!                 }
//!             }
//!             AgentEvent::LoopComplete { reason, total_iterations } => {
//!                 eprintln!("\n[done: {} after {} rounds]", reason, total_iterations);
//!             }
//!             _ => {}
//!         }
//!     }
//! }
//! ```

use async_trait::async_trait;
use serde_json::Value as JsonValue;

use crate::streaming::CollectedResponse;
use crate::types::{ToolCallId, ToolName};

/// Events emitted by the agent loop
///
/// Single enum keeps the vtable small and is forward-compatible:
/// new events don't break existing implementors.
#[derive(Debug, Clone)]
#[non_exhaustive]
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
#[non_exhaustive]
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
    /// The provider's content filter triggered, blocking further output
    ContentFilter,
    /// A tool error stopped the loop (when continue_on_tool_error is false)
    ToolError {
        tool_name: ToolName,
        message: String,
    },
}

impl std::fmt::Display for LoopStopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EndTurn => write!(f, "end_turn"),
            Self::MaxTokens => write!(f, "max_tokens"),
            Self::StopSequence => write!(f, "stop_sequence"),
            Self::ContentFilter => write!(f, "content_filter"),
            Self::MaxToolDepthReached => write!(f, "max_tool_depth_reached"),
            Self::Cancelled => write!(f, "cancelled"),
            Self::ToolError { tool_name, message } => {
                write!(f, "tool_error({}): {}", tool_name, message)
            }
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
        assert_eq!(LoopStopReason::ContentFilter.to_string(), "content_filter");
        assert_eq!(
            LoopStopReason::ToolError {
                tool_name: ToolName::new("my_tool").unwrap(),
                message: "oops".into(),
            }
            .to_string(),
            "tool_error(my_tool): oops"
        );
    }
}

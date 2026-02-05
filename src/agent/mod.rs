//! Agentic tool loop for driving multi-turn tool-calling conversations
//!
//! This module provides a typed orchestrator that drives the tool call cycle:
//! send → detect tool_use → execute → continue. Higher-level strategies like
//! ReAct compose on top of this infrastructure.
//!
//! # Example
//!
//! ```no_run
//! use agent_driver_rs::agent::{AgentLoop, AgentLoopConfig};
//! use agent_driver_rs::Session;
//!
//! # async fn example(session: &Session) -> Result<(), agent_driver_rs::AgentLoopError> {
//! let outcome = AgentLoop::new(session)
//!     .run("List all files in /tmp")
//!     .await?;
//!
//! println!("Final: {}", outcome.final_response.text());
//! println!("Iterations: {}", outcome.iterations);
//! # Ok(())
//! # }
//! ```

mod config;
mod driver;
mod observer;

pub use config::{AgentLoopConfig, MaxToolDepth};
pub use driver::{AgentLoop, AgentOutcome};
pub use observer::{AgentEvent, AgentObserver, LoopStopReason};

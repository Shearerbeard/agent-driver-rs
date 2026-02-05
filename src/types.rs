//! Core types for agent-driver-rs
//!
//! This module contains fundamental types used throughout the library.

mod correlation;
mod message;
mod model;

// Re-export all types
pub use correlation::{CorrelationContext, CorrelationId};
pub use message::{
    ContentBlock, Message, Role, SystemPrompt, ToolCallId, ToolName, ToolResultContent,
};
pub use model::{MaxTokens, ModelId, Temperature};

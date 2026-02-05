//! Configuration system for agent-driver-rs
//!
//! This module provides provider-specific configuration types with environment-based loading.

mod anthropic;
mod bedrock;
mod common;
mod ollama;
mod openai;
mod openrouter;
mod provider;

// Re-export configuration types
pub use anthropic::{AnthropicConfig, AnthropicModel, ThinkingConfig, WellKnownAnthropicModel};
pub use bedrock::{BedrockConfig, BedrockModel};
pub use common::{ApiKey, AwsRegion};
pub use ollama::{KeepAlive, NumCtx, OllamaConfig, OllamaModel, WellKnownOllamaModel};
pub use openai::{
    OpenAiConfig, OpenAiModel, ReasoningConfig, ReasoningEffort, ReasoningSummary,
};
pub use openrouter::{
    OpenRouterConfig, OpenRouterModel, ProviderPreferences, WellKnownOpenRouterModel,
};
pub use provider::ProviderConfig;

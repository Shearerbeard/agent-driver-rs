//! Top-level provider configuration discriminator.
//!
//! [`ProviderConfig`] is a tagged enum that selects a provider and holds its
//! configuration. Use [`ProviderConfig::from_env()`] to load from environment
//! variables with the `PROVIDER` env var as the discriminator.

use serde::{Deserialize, Serialize};

use crate::error::ConfigError;

use super::{
    AnthropicConfig, BedrockConfig, OllamaConfig, OpenAiConfig, OpenRouterConfig,
};

/// Top-level provider configuration enum
///
/// This enum represents the configuration for a single provider.
/// Use `from_env()` to load from environment variables with PROVIDER as discriminator.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "provider", rename_all = "snake_case")]
pub enum ProviderConfig {
    Anthropic(AnthropicConfig),
    OpenAi(OpenAiConfig),
    Bedrock(BedrockConfig),
    OpenRouter(OpenRouterConfig),
    Ollama(OllamaConfig),
}

impl ProviderConfig {
    /// Load from environment using dotenvy + env vars
    ///
    /// Uses PROVIDER env var as discriminator, then loads provider-specific vars.
    pub fn from_env() -> Result<Self, ConfigError> {
        // Load .env if present
        dotenvy::dotenv().ok();

        let provider = std::env::var("PROVIDER")
            .map_err(|_| ConfigError::MissingField { field: "PROVIDER" })?;

        match provider.to_lowercase().as_str() {
            "anthropic" => Ok(Self::Anthropic(AnthropicConfig::from_env()?)),
            "openai" => Ok(Self::OpenAi(OpenAiConfig::from_env()?)),
            "bedrock" => Ok(Self::Bedrock(BedrockConfig::from_env()?)),
            "openrouter" => Ok(Self::OpenRouter(OpenRouterConfig::from_env()?)),
            "ollama" => Ok(Self::Ollama(OllamaConfig::from_env()?)),
            other => Err(ConfigError::UnknownProvider(other.to_string())),
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), ConfigError> {
        match self {
            Self::Anthropic(c) => c.validate(),
            Self::OpenAi(c) => c.validate(),
            Self::Bedrock(c) => c.validate(),
            Self::OpenRouter(c) => c.validate(),
            Self::Ollama(c) => c.validate(),
        }
    }

    /// Get the provider name
    #[must_use]
    pub fn provider_name(&self) -> &'static str {
        match self {
            Self::Anthropic(_) => "anthropic",
            Self::OpenAi(_) => "openai",
            Self::Bedrock(_) => "bedrock",
            Self::OpenRouter(_) => "openrouter",
            Self::Ollama(_) => "ollama",
        }
    }
}

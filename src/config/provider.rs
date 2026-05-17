//! Top-level provider configuration discriminator.
//!
//! [`ProviderConfig`] is a tagged enum that selects a provider and holds its
//! configuration. Use [`ProviderConfig::from_env()`] to load from environment
//! variables with the `PROVIDER` env var as the discriminator.

use serde::{Deserialize, Serialize};

use crate::error::ConfigError;
use crate::provider::ProviderKind;

use super::{AnthropicConfig, BedrockConfig, OllamaConfig, OpenAiConfig, OpenRouterConfig};

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
        let _env_loaded = dotenvy::dotenv();

        let provider_str = std::env::var("PROVIDER")
            .map_err(|_| ConfigError::MissingField { field: "PROVIDER" })?;

        let kind: ProviderKind = provider_str
            .parse()
            .map_err(|_| ConfigError::UnknownProvider(provider_str))?;

        match kind {
            ProviderKind::Anthropic => Ok(Self::Anthropic(AnthropicConfig::from_env()?)),
            ProviderKind::OpenAi => Ok(Self::OpenAi(OpenAiConfig::from_env()?)),
            ProviderKind::Bedrock => Ok(Self::Bedrock(BedrockConfig::from_env()?)),
            ProviderKind::OpenRouter => Ok(Self::OpenRouter(OpenRouterConfig::from_env()?)),
            ProviderKind::Ollama => Ok(Self::Ollama(OllamaConfig::from_env()?)),
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

    /// Get the provider kind
    #[must_use]
    pub fn provider_kind(&self) -> ProviderKind {
        match self {
            Self::Anthropic(_) => ProviderKind::Anthropic,
            Self::OpenAi(_) => ProviderKind::OpenAi,
            Self::Bedrock(_) => ProviderKind::Bedrock,
            Self::OpenRouter(_) => ProviderKind::OpenRouter,
            Self::Ollama(_) => ProviderKind::Ollama,
        }
    }
}

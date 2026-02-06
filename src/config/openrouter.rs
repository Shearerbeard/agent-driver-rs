//! OpenRouter provider configuration

use serde::{Deserialize, Serialize};

use crate::error::ConfigError;
use crate::types::{MaxTokens, Temperature};

use super::common::ApiKey;

/// OpenRouter configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenRouterConfig {
    /// API key
    pub api_key: ApiKey,

    /// Model (can be any OpenRouter model path)
    pub model: OpenRouterModel,

    /// Max tokens
    pub max_tokens: MaxTokens,

    /// Temperature
    #[serde(default)]
    pub temperature: Option<Temperature>,

    /// Provider preferences
    #[serde(default)]
    pub provider_preferences: Option<ProviderPreferences>,
}

/// OpenRouter model specification
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum OpenRouterModel {
    WellKnown(WellKnownOpenRouterModel),
    Custom(String), // e.g., "anthropic/claude-sonnet-4"
}

/// Well-known OpenRouter models
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub enum WellKnownOpenRouterModel {
    #[serde(rename = "anthropic/claude-sonnet-4")]
    AnthropicClaudeSonnet4,
    #[serde(rename = "openai/gpt-4o")]
    OpenaiGpt4o,
    #[serde(rename = "google/gemini-2.0-flash")]
    GoogleGemini2Flash,
    #[serde(rename = "meta-llama/llama-3.3-70b")]
    MetaLlama3_3_70b,
}

impl OpenRouterModel {
    /// Get the model path string
    pub fn as_str(&self) -> &str {
        match self {
            Self::WellKnown(m) => match m {
                WellKnownOpenRouterModel::AnthropicClaudeSonnet4 => "anthropic/claude-sonnet-4",
                WellKnownOpenRouterModel::OpenaiGpt4o => "openai/gpt-4o",
                WellKnownOpenRouterModel::GoogleGemini2Flash => "google/gemini-2.0-flash",
                WellKnownOpenRouterModel::MetaLlama3_3_70b => "meta-llama/llama-3.3-70b",
            },
            Self::Custom(s) => s,
        }
    }
}

impl Default for OpenRouterModel {
    fn default() -> Self {
        Self::WellKnown(WellKnownOpenRouterModel::AnthropicClaudeSonnet4)
    }
}

/// Provider preferences for routing
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ProviderPreferences {
    /// Preferred providers (in order of preference)
    #[serde(default)]
    pub allow: Vec<String>,
    /// Providers to avoid
    #[serde(default)]
    pub deny: Vec<String>,
    /// Require the primary provider
    #[serde(default)]
    pub require_primary: bool,
}

impl OpenRouterConfig {
    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self, ConfigError> {
        let api_key = ApiKey::new(
            std::env::var("OPENROUTER_API_KEY")
                .map_err(|_| ConfigError::MissingField { field: "OPENROUTER_API_KEY" })?,
        );

        let model = OpenRouterModel::Custom(
            std::env::var("OPENROUTER_MODEL")
                .unwrap_or_else(|_| "anthropic/claude-sonnet-4".into()),
        );

        let max_tokens = std::env::var("OPENROUTER_MAX_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok())
            .and_then(MaxTokens::new)
            // Safe: 4096 is a hardcoded non-zero constant
            .unwrap_or_else(|| MaxTokens::new(4096).expect("4096 is non-zero"));

        let temperature = std::env::var("OPENROUTER_TEMPERATURE")
            .ok()
            .and_then(|s| s.parse().ok())
            .and_then(|t| Temperature::new(t).ok());

        Ok(Self {
            api_key,
            model,
            max_tokens,
            temperature,
            provider_preferences: None,
        })
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Custom model strings must be non-empty and contain a `/` (provider/model format)
        if let OpenRouterModel::Custom(ref s) = self.model {
            if s.is_empty() {
                return Err(ConfigError::InvalidValue {
                    field: "model",
                    reason: "custom model string must not be empty".into(),
                });
            }
            if !s.contains('/') {
                return Err(ConfigError::InvalidValue {
                    field: "model",
                    reason: format!(
                        "OpenRouter model must be in 'provider/model' format, got: {}",
                        s
                    ),
                });
            }
        }

        // Validate provider preferences entries are non-empty
        if let Some(ref prefs) = self.provider_preferences {
            for entry in &prefs.allow {
                if entry.is_empty() {
                    return Err(ConfigError::InvalidValue {
                        field: "provider_preferences.allow",
                        reason: "entries must not be empty".into(),
                    });
                }
            }
            for entry in &prefs.deny {
                if entry.is_empty() {
                    return Err(ConfigError::InvalidValue {
                        field: "provider_preferences.deny",
                        reason: "entries must not be empty".into(),
                    });
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_as_str() {
        assert_eq!(
            OpenRouterModel::WellKnown(WellKnownOpenRouterModel::AnthropicClaudeSonnet4).as_str(),
            "anthropic/claude-sonnet-4"
        );
        assert_eq!(
            OpenRouterModel::Custom("custom/model".into()).as_str(),
            "custom/model"
        );
    }
}

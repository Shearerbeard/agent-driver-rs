//! Anthropic provider configuration

use serde::{Deserialize, Serialize};

use crate::error::ConfigError;
use crate::types::{MaxTokens, Temperature};

use super::common::{ApiKey, env_parse_opt, env_parse_or_default};

/// Anthropic API configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnthropicConfig {
    /// API key (required)
    pub api_key: ApiKey,

    /// Model to use
    pub model: AnthropicModel,

    /// Max response tokens
    pub max_tokens: MaxTokens,

    /// Sampling temperature (0.0 - 1.0)
    #[serde(default)]
    pub temperature: Option<Temperature>,

    /// Extended thinking configuration (optional)
    #[serde(default)]
    pub thinking: Option<ThinkingConfig>,
}

/// Extended thinking configuration (matches Claude API)
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ThinkingConfig {
    /// Budget tokens for thinking (minimum 1024, must be < max_tokens)
    budget_tokens: u32,
}

impl ThinkingConfig {
    /// Minimum allowed thinking budget tokens (per Anthropic API requirements).
    pub const MIN_BUDGET: u32 = 1024;

    /// Create a new thinking config with validation
    pub fn new(budget_tokens: u32) -> Result<Self, ConfigError> {
        if budget_tokens < Self::MIN_BUDGET {
            return Err(ConfigError::InvalidValue {
                field: "budget_tokens".to_owned(),
                reason: format!("must be >= {}", Self::MIN_BUDGET),
            });
        }
        Ok(Self { budget_tokens })
    }

    /// Get the thinking budget token count.
    pub fn budget_tokens(&self) -> u32 {
        self.budget_tokens
    }
}

impl AnthropicConfig {
    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self, ConfigError> {
        let api_key = ApiKey::new(std::env::var("ANTHROPIC_API_KEY").map_err(|_| {
            ConfigError::MissingField {
                field: "ANTHROPIC_API_KEY",
            }
        })?)?;

        let model_str = std::env::var("ANTHROPIC_MODEL")
            .unwrap_or_else(|_| "claude-sonnet-4-5-20250929".into());

        let max_tokens = MaxTokens::new(env_parse_or_default::<u32>("ANTHROPIC_MAX_TOKENS")?)
            .unwrap_or_default();

        let temperature: Option<Temperature> = env_parse_opt::<f32>("ANTHROPIC_TEMPERATURE")?
            .map(Temperature::new)
            .transpose()?;

        let thinking: Option<ThinkingConfig> = env_parse_opt::<u32>("ANTHROPIC_THINKING_BUDGET")?
            .map(ThinkingConfig::new)
            .transpose()?;

        Ok(Self {
            api_key,
            model: model_str.parse::<AnthropicModel>()?,
            max_tokens,
            temperature,
            thinking,
        })
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), ConfigError> {
        if let AnthropicModel::Custom(ref s) = self.model {
            if s.is_empty() {
                return Err(ConfigError::InvalidValue {
                    field: "model".to_owned(),
                    reason: "custom model string must not be empty".into(),
                });
            }
        }

        if let Some(thinking) = &self.thinking {
            if thinking.budget_tokens() < ThinkingConfig::MIN_BUDGET {
                return Err(ConfigError::InvalidValue {
                    field: "thinking.budget_tokens".to_owned(),
                    reason: format!("must be >= {}", ThinkingConfig::MIN_BUDGET),
                });
            }
            if thinking.budget_tokens() >= self.max_tokens.get() {
                return Err(ConfigError::InvalidValue {
                    field: "thinking.budget_tokens".to_owned(),
                    reason: "must be < max_tokens".into(),
                });
            }
        }
        Ok(())
    }
}

/// Anthropic model specification
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
#[non_exhaustive]
pub enum AnthropicModel {
    WellKnown(WellKnownAnthropicModel),
    Custom(String),
}

/// Well-known Anthropic models
#[derive(Debug, Clone, Deserialize, Serialize)]
#[non_exhaustive]
pub enum WellKnownAnthropicModel {
    #[serde(rename = "claude-opus-4-20250514")]
    ClaudeOpus4_20250514,
    #[serde(rename = "claude-sonnet-4-20250514")]
    ClaudeSonnet4_20250514,
    #[serde(rename = "claude-sonnet-4-5-20250929")]
    ClaudeSonnet4_5_20250929,
    #[serde(rename = "claude-3-5-haiku-20241022")]
    ClaudeHaiku3_5_20241022,
}

impl AnthropicModel {
    /// Get the model ID string
    pub fn as_str(&self) -> &str {
        match self {
            Self::WellKnown(m) => match m {
                WellKnownAnthropicModel::ClaudeOpus4_20250514 => "claude-opus-4-20250514",
                WellKnownAnthropicModel::ClaudeSonnet4_20250514 => "claude-sonnet-4-20250514",
                WellKnownAnthropicModel::ClaudeSonnet4_5_20250929 => "claude-sonnet-4-5-20250929",
                WellKnownAnthropicModel::ClaudeHaiku3_5_20241022 => "claude-3-5-haiku-20241022",
            },
            Self::Custom(s) => s,
        }
    }

    /// Check if extended thinking is supported
    pub fn supports_thinking(&self) -> bool {
        match self {
            Self::WellKnown(m) => matches!(
                m,
                WellKnownAnthropicModel::ClaudeOpus4_20250514
                    | WellKnownAnthropicModel::ClaudeSonnet4_20250514
                    | WellKnownAnthropicModel::ClaudeSonnet4_5_20250929
            ),
            Self::Custom(s) => s.contains("claude-opus-4") || s.contains("claude-sonnet-4"),
        }
    }
}

impl Default for AnthropicModel {
    fn default() -> Self {
        Self::WellKnown(WellKnownAnthropicModel::ClaudeSonnet4_5_20250929)
    }
}

impl std::str::FromStr for AnthropicModel {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "claude-opus-4-20250514" => {
                Self::WellKnown(WellKnownAnthropicModel::ClaudeOpus4_20250514)
            }
            "claude-sonnet-4-20250514" => {
                Self::WellKnown(WellKnownAnthropicModel::ClaudeSonnet4_20250514)
            }
            "claude-sonnet-4-5-20250929" => {
                Self::WellKnown(WellKnownAnthropicModel::ClaudeSonnet4_5_20250929)
            }
            "claude-3-5-haiku-20241022" => {
                Self::WellKnown(WellKnownAnthropicModel::ClaudeHaiku3_5_20241022)
            }
            other => Self::Custom(other.to_owned()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thinking_config_validation() {
        ThinkingConfig::new(1024).unwrap();
        ThinkingConfig::new(2048).unwrap();
        ThinkingConfig::new(1023).unwrap_err();
        ThinkingConfig::new(0).unwrap_err();
    }

    #[test]
    fn model_as_str() {
        assert_eq!(
            AnthropicModel::WellKnown(WellKnownAnthropicModel::ClaudeSonnet4_5_20250929).as_str(),
            "claude-sonnet-4-5-20250929"
        );
        assert_eq!(
            AnthropicModel::Custom("claude-3-opus".into()).as_str(),
            "claude-3-opus"
        );
    }

    #[test]
    fn model_from_str_recognizes_well_known() {
        let parsed: AnthropicModel = "claude-sonnet-4-5-20250929".parse().unwrap();
        assert!(
            matches!(
                parsed,
                AnthropicModel::WellKnown(WellKnownAnthropicModel::ClaudeSonnet4_5_20250929)
            ),
            "well-known env strings should parse to WellKnown variants, got {parsed:?}"
        );

        let parsed: AnthropicModel = "custom-model".parse().unwrap();
        assert!(
            matches!(parsed, AnthropicModel::Custom(s) if s == "custom-model"),
            "unknown strings should fall back to Custom"
        );
    }
}

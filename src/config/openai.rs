//! OpenAI provider configuration

use serde::{Deserialize, Serialize};

use crate::error::ConfigError;
use crate::types::{MaxTokens, Temperature};

use super::common::ApiKey;

/// OpenAI API configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiConfig {
    /// API key (required)
    pub api_key: ApiKey,

    /// Model to use
    pub model: OpenAiModel,

    /// Max response tokens
    pub max_tokens: MaxTokens,

    /// Temperature (not supported on reasoning models like GPT-5)
    #[serde(default)]
    pub temperature: Option<Temperature>,

    /// Reasoning effort for models that support it (GPT-5, o1, o3)
    #[serde(default)]
    pub reasoning: Option<ReasoningConfig>,

    /// Structured output mode
    #[serde(default)]
    pub structured_output: bool,
}

/// OpenAI model specification
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum OpenAiModel {
    // GPT-5 series (reasoning models, no temperature)
    Gpt5,
    #[serde(rename = "gpt-5.1")]
    Gpt5_1,
    #[serde(rename = "gpt-5.2")]
    Gpt5_2,

    // GPT-4 series (supports temperature)
    #[default]
    Gpt4o,
    Gpt4oMini,

    // o-series reasoning models
    O1,
    O1Mini,
    O3,     // NOTE: o3 does NOT support streaming
    O3Mini, // NOTE: o3-mini does NOT support streaming

    // Custom model string
    #[serde(untagged)]
    Custom(String),
}

impl OpenAiModel {
    /// Check if the model supports temperature parameter
    pub fn supports_temperature(&self) -> bool {
        matches!(self, Self::Gpt4o | Self::Gpt4oMini | Self::Custom(_))
    }

    /// Check if the model supports reasoning configuration
    pub fn supports_reasoning(&self) -> bool {
        matches!(
            self,
            Self::Gpt5
                | Self::Gpt5_1
                | Self::Gpt5_2
                | Self::O1
                | Self::O1Mini
                | Self::O3
                | Self::O3Mini
        )
    }

    /// Check if the model supports streaming
    ///
    /// o3 and o3-mini don't support streaming
    pub fn supports_streaming(&self) -> bool {
        !matches!(self, Self::O3 | Self::O3Mini)
    }

    /// Get the model ID string
    pub fn as_str(&self) -> &str {
        match self {
            Self::Gpt5 => "gpt-5",
            Self::Gpt5_1 => "gpt-5.1",
            Self::Gpt5_2 => "gpt-5.2",
            Self::Gpt4o => "gpt-4o",
            Self::Gpt4oMini => "gpt-4o-mini",
            Self::O1 => "o1",
            Self::O1Mini => "o1-mini",
            Self::O3 => "o3",
            Self::O3Mini => "o3-mini",
            Self::Custom(s) => s,
        }
    }
}

/// Reasoning configuration for reasoning models
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ReasoningConfig {
    /// Reasoning effort level
    pub effort: ReasoningEffort,
    /// Summary mode for reasoning
    #[serde(default)]
    pub summary: ReasoningSummary,
}

/// Reasoning effort levels (per OpenAI API)
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    #[default]
    Medium,
    High,
    XHigh,
}

/// Reasoning summary mode
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningSummary {
    #[default]
    Auto,
    Concise,
    Detailed,
}

impl OpenAiConfig {
    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self, ConfigError> {
        let api_key = ApiKey::new(std::env::var("OPENAI_API_KEY").map_err(|_| {
            ConfigError::MissingField {
                field: "OPENAI_API_KEY",
            }
        })?);

        let model_str = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o".into());
        let model = match model_str.as_str() {
            "gpt-5" => OpenAiModel::Gpt5,
            "gpt-5.1" => OpenAiModel::Gpt5_1,
            "gpt-5.2" => OpenAiModel::Gpt5_2,
            "gpt-4o" => OpenAiModel::Gpt4o,
            "gpt-4o-mini" => OpenAiModel::Gpt4oMini,
            "o1" => OpenAiModel::O1,
            "o1-mini" => OpenAiModel::O1Mini,
            "o3" => OpenAiModel::O3,
            "o3-mini" => OpenAiModel::O3Mini,
            other => OpenAiModel::Custom(other.to_owned()),
        };

        let max_tokens = std::env::var("OPENAI_MAX_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok())
            .and_then(MaxTokens::new)
            .unwrap_or_default();

        let temperature = if model.supports_temperature() {
            std::env::var("OPENAI_TEMPERATURE")
                .ok()
                .and_then(|s| s.parse().ok())
                .and_then(|t| Temperature::new(t).ok())
        } else {
            None
        };

        let reasoning = if model.supports_reasoning() {
            let effort = std::env::var("OPENAI_REASONING_EFFORT")
                .ok()
                .and_then(|s| match s.as_str() {
                    "none" => Some(ReasoningEffort::None),
                    "minimal" => Some(ReasoningEffort::Minimal),
                    "low" => Some(ReasoningEffort::Low),
                    "medium" => Some(ReasoningEffort::Medium),
                    "high" => Some(ReasoningEffort::High),
                    "xhigh" => Some(ReasoningEffort::XHigh),
                    _ => None,
                })
                .unwrap_or(ReasoningEffort::Medium);
            Some(ReasoningConfig {
                effort,
                summary: ReasoningSummary::Auto,
            })
        } else {
            None
        };

        Ok(Self {
            api_key,
            model,
            max_tokens,
            temperature,
            reasoning,
            structured_output: false,
        })
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), ConfigError> {
        if let OpenAiModel::Custom(ref s) = self.model {
            if s.is_empty() {
                return Err(ConfigError::InvalidValue {
                    field: "model",
                    reason: "custom model string must not be empty".into(),
                });
            }
        }

        // Temperature not allowed on reasoning models
        if self.temperature.is_some() && !self.model.supports_temperature() {
            return Err(ConfigError::InvalidValue {
                field: "temperature",
                reason: format!("{:?} does not support temperature", self.model),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_streaming_support() {
        assert!(OpenAiModel::Gpt4o.supports_streaming());
        assert!(OpenAiModel::Gpt5.supports_streaming());
        assert!(OpenAiModel::O1.supports_streaming());
        assert!(!OpenAiModel::O3.supports_streaming());
        assert!(!OpenAiModel::O3Mini.supports_streaming());
    }

    #[test]
    fn model_temperature_support() {
        assert!(OpenAiModel::Gpt4o.supports_temperature());
        assert!(OpenAiModel::Gpt4oMini.supports_temperature());
        assert!(!OpenAiModel::Gpt5.supports_temperature());
        assert!(!OpenAiModel::O1.supports_temperature());
    }

    #[test]
    fn model_as_str() {
        assert_eq!(OpenAiModel::Gpt4o.as_str(), "gpt-4o");
        assert_eq!(OpenAiModel::Gpt5_2.as_str(), "gpt-5.2");
        assert_eq!(OpenAiModel::O3Mini.as_str(), "o3-mini");
    }
}

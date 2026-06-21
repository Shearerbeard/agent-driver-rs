//! AWS Bedrock provider configuration

use serde::{Deserialize, Serialize};

use crate::error::ConfigError;
use crate::types::{MaxTokens, Temperature};

use super::common::AwsRegion;

/// AWS Bedrock configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BedrockConfig {
    /// AWS region
    pub region: AwsRegion,

    /// Model ID
    pub model: BedrockModel,

    /// Max tokens
    pub max_tokens: MaxTokens,

    /// Temperature
    #[serde(default)]
    pub temperature: Option<Temperature>,

    /// Inference profile ARN (for cross-region inference)
    #[serde(default)]
    pub inference_profile: Option<String>,
}

/// Bedrock model specification
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum BedrockModel {
    ClaudeSonnet4,
    #[default]
    ClaudeSonnet4_5,
    ClaudeOpus4,
    ClaudeOpus4_5,
    ClaudeHaiku4_5,
    ClaudeHaiku3_5,
    // Custom model ID
    #[serde(untagged)]
    Custom(String),
}

impl BedrockModel {
    /// Get the full model ID for Bedrock API
    pub fn model_id(&self) -> &str {
        match self {
            Self::ClaudeSonnet4 => "anthropic.claude-sonnet-4-20250514-v1:0",
            Self::ClaudeSonnet4_5 => "anthropic.claude-sonnet-4-5-20250929-v1:0",
            Self::ClaudeOpus4 => "anthropic.claude-opus-4-1-20250805-v1:0",
            Self::ClaudeOpus4_5 => "anthropic.claude-opus-4-5-20251101-v1:0",
            Self::ClaudeHaiku4_5 => "anthropic.claude-haiku-4-5-20251001-v1:0",
            Self::ClaudeHaiku3_5 => "anthropic.claude-3-5-haiku-20241022-v1:0",
            Self::Custom(s) => s,
        }
    }

    /// Get a short display name
    pub fn display_name(&self) -> &str {
        match self {
            Self::ClaudeSonnet4 => "claude-sonnet-4",
            Self::ClaudeSonnet4_5 => "claude-sonnet-4.5",
            Self::ClaudeOpus4 => "claude-opus-4",
            Self::ClaudeOpus4_5 => "claude-opus-4.5",
            Self::ClaudeHaiku4_5 => "claude-haiku-4.5",
            Self::ClaudeHaiku3_5 => "claude-3.5-haiku",
            Self::Custom(s) => s,
        }
    }
}

impl BedrockConfig {
    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self, ConfigError> {
        let region = std::env::var("AWS_REGION")
            .map(AwsRegion::new)
            .unwrap_or_else(|_| AwsRegion::US_EAST_1);

        let model_str =
            std::env::var("BEDROCK_MODEL").unwrap_or_else(|_| "claude-sonnet-4.5".into());
        let model = match model_str.as_str() {
            "claude-sonnet-4" => BedrockModel::ClaudeSonnet4,
            "claude-sonnet-4.5" => BedrockModel::ClaudeSonnet4_5,
            "claude-opus-4" => BedrockModel::ClaudeOpus4,
            "claude-opus-4.5" => BedrockModel::ClaudeOpus4_5,
            "claude-haiku-4.5" => BedrockModel::ClaudeHaiku4_5,
            "claude-haiku-3.5" | "claude-3.5-haiku" => BedrockModel::ClaudeHaiku3_5,
            other => BedrockModel::Custom(other.to_owned()),
        };

        let max_tokens = std::env::var("BEDROCK_MAX_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok())
            .and_then(MaxTokens::new)
            .unwrap_or_default();

        let temperature = std::env::var("BEDROCK_TEMPERATURE")
            .ok()
            .and_then(|s| s.parse().ok())
            .and_then(|t| Temperature::new(t).ok());

        let inference_profile = std::env::var("BEDROCK_INFERENCE_PROFILE").ok();

        Ok(Self {
            region,
            model,
            max_tokens,
            temperature,
            inference_profile,
        })
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), ConfigError> {
        if let BedrockModel::Custom(ref s) = self.model {
            if s.is_empty() {
                return Err(ConfigError::InvalidValue {
                    field: "model",
                    reason: "custom model string must not be empty".into(),
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_id() {
        assert_eq!(
            BedrockModel::ClaudeSonnet4_5.model_id(),
            "anthropic.claude-sonnet-4-5-20250929-v1:0"
        );
        assert_eq!(
            BedrockModel::ClaudeOpus4_5.model_id(),
            "anthropic.claude-opus-4-5-20251101-v1:0"
        );
        assert_eq!(
            BedrockModel::Custom("custom-model".into()).model_id(),
            "custom-model"
        );
    }
}

//! AWS Bedrock provider configuration

use serde::{Deserialize, Serialize};

use crate::error::ConfigError;
use crate::types::{MaxTokens, Temperature};

use super::common::{AwsRegion, env_parse_opt, env_parse_or_default};

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

    /// Extended thinking configuration (Claude family). When set, the
    /// request carries the `additionalModelRequestFields` thinking block
    /// and reasoning deltas flow as `ThinkingDelta`/`SignatureDelta`.
    /// Models that reason by default (DeepSeek class) return
    /// `reasoningContent` with no request field; leave this `None` for
    /// them.
    #[serde(default)]
    pub thinking: Option<BedrockThinkingConfig>,
}

/// Extended thinking configuration for Bedrock (mirrors the direct
/// Anthropic `ThinkingConfig` budget semantics).
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
pub struct BedrockThinkingConfig {
    budget_tokens: u32,
}

impl BedrockThinkingConfig {
    /// Minimum allowed thinking budget tokens (per the Bedrock Converse
    /// API requirements, same floor as Anthropic's).
    pub const MIN_BUDGET: u32 = 1024;

    /// Create a thinking config with validation.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::InvalidValue`] when the budget is below
    /// [`Self::MIN_BUDGET`].
    pub fn new(budget_tokens: u32) -> Result<Self, ConfigError> {
        if budget_tokens < Self::MIN_BUDGET {
            return Err(ConfigError::InvalidValue {
                field: "budget_tokens".to_owned(),
                reason: format!("must be >= {}", Self::MIN_BUDGET),
            });
        }
        Ok(Self { budget_tokens })
    }

    /// The thinking budget token count.
    pub fn budget_tokens(self) -> u32 {
        self.budget_tokens
    }
}

/// Bedrock model specification
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum BedrockModel {
    ClaudeSonnet4,
    #[default]
    ClaudeSonnet4_5,
    ClaudeSonnet4_6,
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
            Self::ClaudeSonnet4_6 => "us.anthropic.claude-sonnet-4-6",
            Self::ClaudeOpus4 => "anthropic.claude-opus-4-1-20250805-v1:0",
            Self::ClaudeOpus4_5 => "anthropic.claude-opus-4-5-20251101-v1:0",
            Self::ClaudeHaiku4_5 => "anthropic.claude-haiku-4-5-20251001-v1:0",
            Self::ClaudeHaiku3_5 => "anthropic.claude-3-5-haiku-20241022-v1:0",
            Self::Custom(s) => s,
        }
    }

    /// Returns true if the model is a cross-region inference profile model.
    ///
    /// Modern Claude models such as `claude-sonnet-4.6` require a
    /// `BEDROCK_INFERENCE_PROFILE` ARN to be configured.
    pub fn is_cross_region(&self) -> bool {
        self.model_id().starts_with("us.")
    }

    /// Get a short display name
    pub fn display_name(&self) -> &str {
        match self {
            Self::ClaudeSonnet4 => "claude-sonnet-4",
            Self::ClaudeSonnet4_5 => "claude-sonnet-4.5",
            Self::ClaudeSonnet4_6 => "claude-sonnet-4.6",
            Self::ClaudeOpus4 => "claude-opus-4",
            Self::ClaudeOpus4_5 => "claude-opus-4.5",
            Self::ClaudeHaiku4_5 => "claude-haiku-4.5",
            Self::ClaudeHaiku3_5 => "claude-3.5-haiku",
            Self::Custom(s) => s,
        }
    }
}

impl std::str::FromStr for BedrockModel {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "claude-sonnet-4" => Self::ClaudeSonnet4,
            "claude-sonnet-4.5" => Self::ClaudeSonnet4_5,
            "claude-sonnet-4.6" => Self::ClaudeSonnet4_6,
            "claude-opus-4" => Self::ClaudeOpus4,
            "claude-opus-4.5" => Self::ClaudeOpus4_5,
            "claude-haiku-4.5" => Self::ClaudeHaiku4_5,
            "claude-haiku-3.5" | "claude-3.5-haiku" => Self::ClaudeHaiku3_5,
            other => Self::Custom(other.to_owned()),
        })
    }
}

impl BedrockConfig {
    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self, ConfigError> {
        let region = match std::env::var("AWS_REGION") {
            Ok(s) => AwsRegion::new(s)?,
            Err(_) => AwsRegion::US_EAST_1,
        };

        let model_str =
            std::env::var("BEDROCK_MODEL").unwrap_or_else(|_| "claude-sonnet-4.5".into());
        let model = model_str.parse::<BedrockModel>()?;

        let max_tokens =
            MaxTokens::new(env_parse_or_default::<u32>("BEDROCK_MAX_TOKENS")?).unwrap_or_default();

        let temperature: Option<Temperature> = env_parse_opt::<f32>("BEDROCK_TEMPERATURE")?
            .map(Temperature::new)
            .transpose()?;

        let inference_profile = std::env::var("BEDROCK_INFERENCE_PROFILE").ok();

        let thinking = env_parse_opt::<u32>("BEDROCK_THINKING_BUDGET")?
            .map(BedrockThinkingConfig::new)
            .transpose()?;

        Ok(Self {
            region,
            model,
            max_tokens,
            temperature,
            inference_profile,
            thinking,
        })
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), ConfigError> {
        if let BedrockModel::Custom(ref s) = self.model {
            if s.is_empty() {
                return Err(ConfigError::InvalidValue {
                    field: "model".to_owned(),
                    reason: "custom model string must not be empty".into(),
                });
            }
        }

        if self.model.is_cross_region() && self.inference_profile.is_none() {
            return Err(ConfigError::InvalidValue {
                field: "inference_profile".to_owned(),
                reason: format!(
                    "model {} requires an inference profile ARN",
                    self.model.display_name()
                ),
            });
        }

        if let Some(thinking) = &self.thinking {
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

    #[test]
    fn model_from_str_recognizes_well_known() {
        let parsed: BedrockModel = "claude-sonnet-4.5".parse().unwrap();
        assert_eq!(parsed, BedrockModel::ClaudeSonnet4_5);

        let parsed: BedrockModel = "custom-model".parse().unwrap();
        assert_eq!(parsed, BedrockModel::Custom("custom-model".into()));
    }

    #[test]
    fn cross_region_model_requires_inference_profile() {
        let config = BedrockConfig {
            region: AwsRegion::US_EAST_1,
            model: BedrockModel::ClaudeSonnet4_6,
            max_tokens: crate::types::MaxTokens::new(100).unwrap(),
            temperature: None,
            inference_profile: None,
            thinking: None,
        };
        config.validate().unwrap_err();

        let config = BedrockConfig {
            region: AwsRegion::US_EAST_1,
            model: BedrockModel::ClaudeSonnet4_6,
            max_tokens: crate::types::MaxTokens::new(100).unwrap(),
            temperature: None,
            inference_profile: Some("arn:aws:bedrock:us-east-1:123:profile/foo".into()),
            thinking: None,
        };
        config.validate().unwrap();
    }

    #[test]
    fn non_cross_region_model_does_not_require_inference_profile() {
        let config = BedrockConfig {
            region: AwsRegion::US_EAST_1,
            model: BedrockModel::ClaudeSonnet4_5,
            max_tokens: crate::types::MaxTokens::new(100).unwrap(),
            temperature: None,
            inference_profile: None,
            thinking: None,
        };
        config.validate().unwrap();
    }
}

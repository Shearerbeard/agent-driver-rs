//! Ollama provider configuration

use serde::{Deserialize, Serialize};
use std::num::NonZeroU32;

use crate::error::ConfigError;
use crate::types::{MaxTokens, Temperature};

use super::common::env_parse_opt;

fn default_ollama_url() -> url::Url {
    let Ok(url) = url::Url::parse("http://localhost:11434") else {
        unreachable!("default Ollama URL is valid")
    };
    url
}

/// Ollama configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OllamaConfig {
    /// Base URL (default: http://localhost:11434)
    #[serde(default = "default_ollama_url")]
    pub base_url: url::Url,

    /// Model name
    pub model: OllamaModel,

    /// Context window size (critical for Ollama)
    pub num_ctx: NumCtx,

    /// Max response tokens
    #[serde(default)]
    pub max_tokens: Option<MaxTokens>,

    /// Temperature
    #[serde(default)]
    pub temperature: Option<Temperature>,

    /// Keep model loaded in memory
    #[serde(default)]
    pub keep_alive: Option<KeepAlive>,

    /// Enable thinking/reasoning mode (for models that support it)
    #[serde(default)]
    pub think: Option<bool>,
}

/// Context window size for Ollama (must be > 0)
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(transparent)]
pub struct NumCtx(NonZeroU32);

impl NumCtx {
    /// Create a new NumCtx with validation
    pub fn new(ctx: u32) -> Result<Self, ConfigError> {
        NonZeroU32::new(ctx)
            .map(Self)
            .ok_or_else(|| ConfigError::InvalidValue {
                field: "num_ctx".to_owned(),
                reason: "context window size must be greater than 0".into(),
            })
    }

    /// Get the context window size
    pub fn get(&self) -> u32 {
        self.0.get()
    }
}

impl std::fmt::Display for NumCtx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<'de> Deserialize<'de> for NumCtx {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let n = u32::deserialize(deserializer)?;
        Self::new(n).map_err(serde::de::Error::custom)
    }
}

/// Ollama model specification
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(untagged)]
#[non_exhaustive]
pub enum OllamaModel {
    WellKnown(WellKnownOllamaModel),
    Custom(String),
}

/// Well-known Ollama models
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum WellKnownOllamaModel {
    // Llama models
    #[serde(rename = "llama3.2")]
    Llama3_2,
    #[serde(rename = "llama3.2:3b")]
    Llama3_2_3b,

    // Qwen3 models (good for 24GB VRAM)
    #[serde(rename = "qwen3:30b")]
    Qwen3_30b,
    #[serde(rename = "qwen3:14b")]
    Qwen3_14b,
    #[serde(rename = "qwen3-coder:14b")]
    Qwen3Coder14b,

    // Other common models
    #[serde(rename = "deepseek-r1")]
    DeepseekR1,
    Mistral,
}

impl OllamaModel {
    /// Get the model name string
    pub fn as_str(&self) -> &str {
        match self {
            Self::WellKnown(m) => match m {
                WellKnownOllamaModel::Llama3_2 => "llama3.2",
                WellKnownOllamaModel::Llama3_2_3b => "llama3.2:3b",
                WellKnownOllamaModel::Qwen3_30b => "qwen3:30b",
                WellKnownOllamaModel::Qwen3_14b => "qwen3:14b",
                WellKnownOllamaModel::Qwen3Coder14b => "qwen3-coder:14b",
                WellKnownOllamaModel::DeepseekR1 => "deepseek-r1",
                WellKnownOllamaModel::Mistral => "mistral",
            },
            Self::Custom(s) => s,
        }
    }
}

impl Default for OllamaModel {
    fn default() -> Self {
        Self::WellKnown(WellKnownOllamaModel::Llama3_2)
    }
}

impl std::str::FromStr for OllamaModel {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "llama3.2" => Self::WellKnown(WellKnownOllamaModel::Llama3_2),
            "llama3.2:3b" => Self::WellKnown(WellKnownOllamaModel::Llama3_2_3b),
            "qwen3:30b" => Self::WellKnown(WellKnownOllamaModel::Qwen3_30b),
            "qwen3:14b" => Self::WellKnown(WellKnownOllamaModel::Qwen3_14b),
            "qwen3-coder:14b" => Self::WellKnown(WellKnownOllamaModel::Qwen3Coder14b),
            "deepseek-r1" => Self::WellKnown(WellKnownOllamaModel::DeepseekR1),
            "mistral" => Self::WellKnown(WellKnownOllamaModel::Mistral),
            other => Self::Custom(other.to_owned()),
        })
    }
}

/// Keep-alive configuration
///
/// `Minutes` wraps a [`NonZeroU32`] so that `0` (unload) is unrepresentable in
/// that variant, making `Unload` the only valid way to request immediate
/// unloading.
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum KeepAlive {
    /// Keep model loaded indefinitely (-1)
    Indefinite,
    /// Keep model loaded for specified minutes
    Minutes(NonZeroU32),
    /// Unload model immediately after request (0)
    Unload,
}

impl KeepAlive {
    /// Convert to the API value
    pub fn as_api_value(&self) -> String {
        match self {
            Self::Indefinite => "-1".to_owned(),
            Self::Minutes(m) => format!("{m}m"),
            Self::Unload => "0".to_owned(),
        }
    }
}

impl OllamaConfig {
    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self, ConfigError> {
        let base_url = std::env::var("OLLAMA_BASE_URL")
            .unwrap_or_else(|_| "http://localhost:11434".into())
            .parse()
            .map_err(|_| ConfigError::InvalidValue {
                field: "OLLAMA_BASE_URL".to_owned(),
                reason: "Invalid URL".into(),
            })?;

        let model = std::env::var("OLLAMA_MODEL")
            .unwrap_or_else(|_| "llama3.2".into())
            .parse::<OllamaModel>()?;

        // num_ctx is critical for Ollama - require it
        let num_ctx_str =
            std::env::var("OLLAMA_NUM_CTX").map_err(|_| ConfigError::MissingField {
                field: "OLLAMA_NUM_CTX",
            })?;
        let num_ctx_val: u32 = num_ctx_str.parse().map_err(|_| ConfigError::InvalidValue {
            field: "OLLAMA_NUM_CTX".to_owned(),
            reason: "must be a positive integer".into(),
        })?;
        let num_ctx = NumCtx::new(num_ctx_val)?;

        let max_tokens = env_parse_opt::<u32>("OLLAMA_MAX_TOKENS")?.and_then(MaxTokens::new);

        let temperature: Option<Temperature> = env_parse_opt::<f32>("OLLAMA_TEMPERATURE")?
            .map(Temperature::new)
            .transpose()?;

        let keep_alive = std::env::var("OLLAMA_KEEP_ALIVE")
            .ok()
            .and_then(|s| match s.as_str() {
                "indefinite" | "-1" => Some(KeepAlive::Indefinite),
                "unload" | "0" => Some(KeepAlive::Unload),
                _ => s
                    .parse::<u32>()
                    .ok()
                    .and_then(NonZeroU32::new)
                    .map(KeepAlive::Minutes),
            });

        let think =
            std::env::var("OLLAMA_THINK")
                .ok()
                .and_then(|s| match s.to_lowercase().as_str() {
                    "true" | "1" | "yes" => Some(true),
                    "false" | "0" | "no" => Some(false),
                    _ => None,
                });

        Ok(Self {
            base_url,
            model,
            num_ctx,
            max_tokens,
            temperature,
            keep_alive,
            think,
        })
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), ConfigError> {
        if let OllamaModel::Custom(ref s) = self.model {
            if s.is_empty() {
                return Err(ConfigError::InvalidValue {
                    field: "model".to_owned(),
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
    fn num_ctx_validation() {
        NumCtx::new(8192).unwrap();
        NumCtx::new(1).unwrap();
        NumCtx::new(0).unwrap_err();
    }

    #[test]
    fn keep_alive_api_value() {
        assert_eq!(KeepAlive::Indefinite.as_api_value(), "-1");
        assert_eq!(
            KeepAlive::Minutes(NonZeroU32::new(30).unwrap()).as_api_value(),
            "30m"
        );
        assert_eq!(KeepAlive::Unload.as_api_value(), "0");
    }

    #[test]
    fn keep_alive_minutes_rejects_zero() {
        assert!(NonZeroU32::new(0).is_none());
    }

    #[test]
    fn model_as_str() {
        assert_eq!(
            OllamaModel::WellKnown(WellKnownOllamaModel::Llama3_2).as_str(),
            "llama3.2"
        );
        assert_eq!(
            OllamaModel::Custom("custom:latest".into()).as_str(),
            "custom:latest"
        );
    }

    #[test]
    fn model_from_str_recognizes_well_known() {
        let parsed: OllamaModel = "llama3.2".parse().unwrap();
        assert_eq!(
            parsed,
            OllamaModel::WellKnown(WellKnownOllamaModel::Llama3_2)
        );

        let parsed: OllamaModel = "custom:latest".parse().unwrap();
        assert_eq!(parsed, OllamaModel::Custom("custom:latest".into()));
    }
}

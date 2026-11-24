//! OpenAI provider configuration

use serde::{Deserialize, Serialize};

use crate::error::ConfigError;
use crate::types::{MaxTokens, Temperature};

use super::common::{ApiKey, env_parse_opt, env_parse_or_default};

/// OpenAI API surface the provider targets
///
/// The Responses API is the reasoning-native surface for GPT-5.x models
/// (`ReasoningConfig` maps to its `reasoning: { effort, summary }`
/// parameter); Chat Completions is the long-standing default. Reasoning
/// config is not sent on the Chat Completions surface.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum OpenAiApi {
    /// Chat Completions API (default)
    #[default]
    ChatCompletions,
    /// Responses API
    Responses,
}

impl std::str::FromStr for OpenAiApi {
    type Err = ConfigError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "chat-completions" => Ok(Self::ChatCompletions),
            "responses" => Ok(Self::Responses),
            other => Err(ConfigError::InvalidValue {
                field: "api".to_owned(),
                reason: format!(
                    "unknown API surface {other:?}; expected \"chat-completions\" or \"responses\""
                ),
            }),
        }
    }
}

/// Preferred Chat Completions reasoning output field.
///
/// This selects which response field the provider prefers when both
/// `reasoning_content` and `reasoning` are present on a streaming chunk.
/// `None` keeps the tolerant BYOT parser: either spelling is accepted,
/// with `reasoning_content` preferred when both appear.
///
/// This field is only meaningful for [`OpenAiApi::ChatCompletions`];
/// selecting it for [`OpenAiApi::Responses`] is rejected at profile
/// construction.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum OpenAiChatReasoningOutput {
    /// No preferred field; tolerate either spelling.
    #[default]
    None,
    /// Prefer the `reasoning_content` field (vLLM / DeepSeek convention).
    ReasoningContent,
    /// Prefer the `reasoning` field (OpenRouter convention).
    Reasoning,
}

impl std::str::FromStr for OpenAiChatReasoningOutput {
    type Err = ConfigError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "none" => Ok(Self::None),
            "reasoning_content" => Ok(Self::ReasoningContent),
            "reasoning" => Ok(Self::Reasoning),
            other => Err(ConfigError::InvalidValue {
                field: "chat_reasoning_output".to_owned(),
                reason: format!(
                    "unknown Chat reasoning output {other:?}; expected \"none\", \"reasoning_content\", or \"reasoning\""
                ),
            }),
        }
    }
}

/// OpenAI API configuration
///
/// # Invariants
///
/// This struct is a deserializable input bag. Public struct literals are
/// not proof of validation: invalid combinations are caught when the
/// configuration is parsed into an [`OpenAiConnectionProfile`] by
/// [`OpenAiProvider::new`](crate::provider::OpenAiProvider::new).
///
/// - `chat_reasoning_output` is only valid for [`OpenAiApi::ChatCompletions`].
/// - `custom_capabilities` is required when `model` is [`OpenAiModel::Custom`]
///   and forbidden for well-known models.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiConfig {
    /// API key (required)
    pub api_key: ApiKey,

    /// API surface to target (chat completions by default)
    #[serde(default)]
    pub api: OpenAiApi,

    /// Base URL for OpenAI-compatible endpoints (BaseTen, OpenRouter,
    /// vLLM servers). When unset, the official `api.openai.com` base
    /// applies. The URL must include the endpoint path prefix (usually
    /// `/v1`); malformed URLs are rejected without guessing a path.
    #[serde(default)]
    pub base_url: Option<String>,

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

    /// Preferred Chat Completions reasoning output field.
    #[serde(default)]
    pub chat_reasoning_output: OpenAiChatReasoningOutput,

    /// Explicit capability declaration for a custom model.
    ///
    /// Required for [`OpenAiModel::Custom`]; must be `None` for well-known
    /// models so that capability declarations are properties of the
    /// selected endpoint/model, not inferred brand IDs.
    #[serde(default)]
    pub custom_capabilities: Option<CustomModelCapabilities>,
}

/// Explicit capability declaration for a custom endpoint/model pair.
///
/// Custom models never inherit capabilities from their ID. All three
/// flags must be supplied together; a partial declaration is rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub struct CustomModelCapabilities {
    /// Whether the model accepts a temperature parameter.
    pub supports_temperature: bool,
    /// Whether the model supports reasoning configuration.
    pub supports_reasoning: bool,
    /// Whether the model supports streaming on the selected surface.
    pub supports_streaming: bool,
}

impl CustomModelCapabilities {
    /// Create an explicit custom capability declaration.
    pub const fn new(
        supports_temperature: bool,
        supports_reasoning: bool,
        supports_streaming: bool,
    ) -> Self {
        Self {
            supports_temperature,
            supports_reasoning,
            supports_streaming,
        }
    }
}

/// Builder for a programmatic custom-model [`OpenAiConfig`].
///
/// Produced by [`OpenAiConfig::custom`]; call [`Self::build`] to obtain
/// the configuration. Kept small so the required capability declaration
/// cannot be omitted.
#[derive(Debug, Clone)]
#[must_use]
pub struct CustomOpenAiConfigBuilder {
    api_key: ApiKey,
    api: OpenAiApi,
    base_url: Option<String>,
    model: String,
    max_tokens: MaxTokens,
    capabilities: CustomModelCapabilities,
}

impl CustomOpenAiConfigBuilder {
    /// Target the Responses API instead of Chat Completions.
    pub fn with_responses_api(mut self) -> Self {
        self.api = OpenAiApi::Responses;
        self
    }

    /// Set a compatible base URL (must include the endpoint path prefix).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Produce the configuration.
    pub fn build(self) -> OpenAiConfig {
        OpenAiConfig {
            api_key: self.api_key,
            api: self.api,
            base_url: self.base_url,
            model: OpenAiModel::Custom(self.model),
            max_tokens: self.max_tokens,
            temperature: None,
            reasoning: None,
            structured_output: false,
            chat_reasoning_output: OpenAiChatReasoningOutput::None,
            custom_capabilities: Some(self.capabilities),
        }
    }
}

/// OpenAI model specification
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
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
    O3,     // NOTE: o3 does NOT support streaming on Chat Completions
    O3Mini, // NOTE: o3-mini does NOT support streaming on Chat Completions

    // Custom model string
    #[serde(untagged)]
    Custom(String),
}

impl OpenAiModel {
    /// Check if the well-known model supports a temperature parameter.
    ///
    /// Custom models never infer capabilities from their ID; this method
    /// returns `false` for [`Self::Custom`]. Use [`OpenAiConnectionProfile`]
    /// for the validated capability of a selected endpoint/model pair.
    pub fn supports_temperature(&self) -> bool {
        matches!(self, Self::Gpt4o | Self::Gpt4oMini)
    }

    /// Check if the well-known model supports reasoning configuration.
    ///
    /// Custom models never infer capabilities from their ID; this method
    /// returns `false` for [`Self::Custom`]. Use [`OpenAiConnectionProfile`]
    /// for the validated capability of a selected endpoint/model pair.
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

    /// Check if the well-known model supports streaming.
    ///
    /// o3 and o3-mini don't support streaming on the Chat Completions
    /// surface. Custom models never infer capabilities from their ID; this
    /// method returns `false` for [`Self::Custom`]. Use
    /// [`OpenAiConnectionProfile`] for the validated capability of a selected
    /// endpoint/model pair.
    pub fn supports_streaming(&self) -> bool {
        matches!(
            self,
            Self::Gpt5
                | Self::Gpt5_1
                | Self::Gpt5_2
                | Self::Gpt4o
                | Self::Gpt4oMini
                | Self::O1
                | Self::O1Mini
        )
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

    /// Classify a model ID string.
    ///
    /// Well-known IDs map to their variant; anything else becomes
    /// [`Self::Custom`]. This constructor is infallible: an unknown ID
    /// is a custom model, and capability is never inferred from the
    /// string.
    pub fn from_id(s: &str) -> Self {
        match s {
            "gpt-5" => Self::Gpt5,
            "gpt-5.1" => Self::Gpt5_1,
            "gpt-5.2" => Self::Gpt5_2,
            "gpt-4o" => Self::Gpt4o,
            "gpt-4o-mini" => Self::Gpt4oMini,
            "o1" => Self::O1,
            "o1-mini" => Self::O1Mini,
            "o3" => Self::O3,
            "o3-mini" => Self::O3Mini,
            other => Self::Custom(other.to_owned()),
        }
    }
}

impl std::str::FromStr for OpenAiModel {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self::from_id(s))
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

/// Validated OpenAI connection profile.
///
/// This is the narrow, invariant-protecting view of [`OpenAiConfig`].
/// It is composed once in [`OpenAiProvider::new`](crate::provider::OpenAiProvider::new)
/// and consumed by request builders. The profile keeps cross-field
/// invariants private: a Chat-only output field cannot be selected for
/// Responses, custom models must declare explicit capabilities, and
/// malformed base URLs are rejected without path guessing.
#[derive(Debug, Clone)]
pub(crate) struct OpenAiConnectionProfile {
    api: OpenAiApi,
    model_id: String,
    capabilities: ModelCapabilities,
    chat_reasoning_output: OpenAiChatReasoningOutput,
    base_url: Option<String>,
    reasoning: Option<ReasoningConfig>,
}

/// Capability snapshot for a selected endpoint/model pair.
///
/// Well-known models get explicit per-surface capabilities; custom models
/// use the declared [`CustomModelCapabilities`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ModelCapabilities {
    supports_temperature: bool,
    supports_reasoning: bool,
    supports_streaming: bool,
}

impl OpenAiConnectionProfile {
    /// Compose a validated profile from an [`OpenAiConfig`].
    ///
    /// Rejects invalid combinations that the public input struct cannot
    /// forbid at the type level.
    pub fn new(config: &OpenAiConfig) -> Result<Self, ConfigError> {
        Self::validate_base_url(config.base_url.as_deref())?;

        let model_id = config.model.as_str().to_owned();
        if let OpenAiModel::Custom(ref s) = config.model {
            if s.is_empty() {
                return Err(ConfigError::InvalidValue {
                    field: "model".to_owned(),
                    reason: "custom model string must not be empty".into(),
                });
            }
        }

        let capabilities = Self::resolve_capabilities(config)?;

        if config.api == OpenAiApi::Responses
            && config.chat_reasoning_output != OpenAiChatReasoningOutput::None
        {
            return Err(ConfigError::InvalidValue {
                field: "chat_reasoning_output".to_owned(),
                reason: "Chat reasoning output field cannot be selected for the Responses API"
                    .into(),
            });
        }

        if config.temperature.is_some() && !capabilities.supports_temperature {
            return Err(ConfigError::InvalidValue {
                field: "temperature".to_owned(),
                reason: format!(
                    "model {:?} on {:?} does not support temperature",
                    config.model, config.api
                ),
            });
        }

        if config.reasoning.is_some() {
            if !capabilities.supports_reasoning {
                return Err(ConfigError::InvalidValue {
                    field: "reasoning".to_owned(),
                    reason: format!(
                        "model {:?} on {:?} does not support reasoning",
                        config.model, config.api
                    ),
                });
            }
            if matches!(config.model, OpenAiModel::Custom(_))
                && config.api == OpenAiApi::ChatCompletions
            {
                return Err(ConfigError::InvalidValue {
                    field: "reasoning".to_owned(),
                    reason: "custom Chat models do not support reasoning effort until a verified request dialect is supplied".into(),
                });
            }
        }

        Ok(Self {
            api: config.api,
            model_id,
            capabilities,
            chat_reasoning_output: config.chat_reasoning_output,
            base_url: config.base_url.clone(),
            reasoning: config.reasoning.clone(),
        })
    }

    /// Selected API surface.
    pub const fn api(&self) -> OpenAiApi {
        self.api
    }

    /// Canonical model ID for this connection.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Whether the selected model/surface supports temperature.
    pub const fn supports_temperature(&self) -> bool {
        self.capabilities.supports_temperature
    }

    /// Whether the selected model/surface supports reasoning configuration.
    pub const fn supports_reasoning(&self) -> bool {
        self.capabilities.supports_reasoning
    }

    /// Whether the selected model/surface supports streaming.
    pub const fn supports_streaming(&self) -> bool {
        self.capabilities.supports_streaming
    }

    /// Preferred Chat Completions reasoning output field.
    pub const fn chat_reasoning_output(&self) -> OpenAiChatReasoningOutput {
        self.chat_reasoning_output
    }

    /// Configured base URL, if any.
    pub fn base_url(&self) -> Option<&str> {
        self.base_url.as_deref()
    }

    /// Configured reasoning effort, if any.
    pub fn reasoning(&self) -> Option<&ReasoningConfig> {
        self.reasoning.as_ref()
    }

    fn resolve_capabilities(config: &OpenAiConfig) -> Result<ModelCapabilities, ConfigError> {
        match &config.model {
            OpenAiModel::Custom(_) => {
                if let Some(capabilities) = config.custom_capabilities {
                    Ok(ModelCapabilities {
                        supports_temperature: capabilities.supports_temperature,
                        supports_reasoning: capabilities.supports_reasoning,
                        supports_streaming: capabilities.supports_streaming,
                    })
                } else {
                    Err(ConfigError::InvalidValue {
                        field: "custom_capabilities".to_owned(),
                        reason: "custom model requires an explicit capability declaration".into(),
                    })
                }
            }
            OpenAiModel::Gpt5
            | OpenAiModel::Gpt5_1
            | OpenAiModel::Gpt5_2
            | OpenAiModel::Gpt4o
            | OpenAiModel::Gpt4oMini
            | OpenAiModel::O1
            | OpenAiModel::O1Mini
            | OpenAiModel::O3
            | OpenAiModel::O3Mini => {
                if config.custom_capabilities.is_some() {
                    return Err(ConfigError::InvalidValue {
                        field: "custom_capabilities".to_owned(),
                        reason: "well-known model cannot have a custom capability declaration"
                            .into(),
                    });
                }
                Ok(Self::known_capabilities(&config.model, config.api))
            }
        }
    }

    fn known_capabilities(model: &OpenAiModel, api: OpenAiApi) -> ModelCapabilities {
        let supports_temperature = matches!(model, OpenAiModel::Gpt4o | OpenAiModel::Gpt4oMini);
        let supports_reasoning = matches!(
            model,
            OpenAiModel::Gpt5
                | OpenAiModel::Gpt5_1
                | OpenAiModel::Gpt5_2
                | OpenAiModel::O1
                | OpenAiModel::O1Mini
                | OpenAiModel::O3
                | OpenAiModel::O3Mini
        );
        let supports_streaming = match api {
            OpenAiApi::ChatCompletions => !matches!(model, OpenAiModel::O3 | OpenAiModel::O3Mini),
            OpenAiApi::Responses => true,
        };
        ModelCapabilities {
            supports_temperature,
            supports_reasoning,
            supports_streaming,
        }
    }

    fn validate_base_url(base_url: Option<&str>) -> Result<(), ConfigError> {
        let Some(base) = base_url else {
            return Ok(());
        };
        if base.trim().is_empty() {
            return Err(ConfigError::InvalidValue {
                field: "base_url".to_owned(),
                reason: "base_url must not be empty".into(),
            });
        }
        let parsed = url::Url::parse(base).map_err(|e| ConfigError::InvalidValue {
            field: "base_url".to_owned(),
            reason: format!("base_url is not a valid URL: {e}"),
        })?;
        if parsed.scheme() != "http" && parsed.scheme() != "https" {
            return Err(ConfigError::InvalidValue {
                field: "base_url".to_owned(),
                reason: format!(
                    "base_url scheme must be http:// or https://, got {:?}",
                    parsed.scheme()
                ),
            });
        }
        Ok(())
    }
}

impl OpenAiConfig {
    /// Build a configuration for a programmatic custom model.
    ///
    /// The capability declaration is required here (not inferred from the
    /// model string) so a caller cannot silently register a custom model
    /// without an explicit endpoint/model contract. The result still
    /// passes through [`OpenAiProvider::new`](crate::provider::OpenAiProvider::new),
    /// which composes and validates the connection profile.
    pub fn custom(
        api_key: ApiKey,
        model: impl Into<String>,
        max_tokens: MaxTokens,
        capabilities: CustomModelCapabilities,
    ) -> CustomOpenAiConfigBuilder {
        CustomOpenAiConfigBuilder {
            api_key,
            api: OpenAiApi::ChatCompletions,
            base_url: None,
            model: model.into(),
            max_tokens,
            capabilities,
        }
    }

    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self, ConfigError> {
        let api_key = ApiKey::new(std::env::var("OPENAI_API_KEY").map_err(|_| {
            ConfigError::MissingField {
                field: "OPENAI_API_KEY",
            }
        })?)?;

        let base_url = match std::env::var("OPENAI_BASE_URL") {
            Ok(v) => Some(v.trim().to_owned()).filter(|v| !v.is_empty()),
            Err(std::env::VarError::NotPresent) => None,
            Err(e) => return Err(ConfigError::from(e)),
        };

        let api = match std::env::var("OPENAI_API") {
            Ok(v) => v.trim().parse::<OpenAiApi>()?,
            Err(std::env::VarError::NotPresent) => OpenAiApi::default(),
            Err(e) => return Err(ConfigError::from(e)),
        };

        let model = resolve_model(std::env::var("OPENAI_MODEL"))?;

        let chat_reasoning_output = match std::env::var("OPENAI_CHAT_REASONING_OUTPUT") {
            Ok(v) => v.trim().parse::<OpenAiChatReasoningOutput>()?,
            Err(std::env::VarError::NotPresent) => OpenAiChatReasoningOutput::default(),
            Err(e) => return Err(ConfigError::from(e)),
        };

        let custom_capabilities = parse_custom_capabilities_from_env(&model)?;

        let max_tokens =
            MaxTokens::new(env_parse_or_default::<u32>("OPENAI_MAX_TOKENS")?).unwrap_or_default();

        // Parse temperature independently of capability; unsupported
        // combinations are rejected when the profile is composed.
        let temperature: Option<Temperature> = match env_parse_opt::<f32>("OPENAI_TEMPERATURE")? {
            Some(value) => Some(Temperature::new(value)?),
            None => None,
        };

        // Parse reasoning effort independently of capability; unknown
        // values fail loud. Known reasoning models default to Medium
        // when absent; custom models never auto-default.
        let reasoning = match parse_reasoning_effort_from_env()? {
            Some(effort) => Some(ReasoningConfig {
                effort,
                summary: ReasoningSummary::Auto,
            }),
            None => {
                if !matches!(model, OpenAiModel::Custom(_)) && model.supports_reasoning() {
                    Some(ReasoningConfig {
                        effort: ReasoningEffort::Medium,
                        summary: ReasoningSummary::Auto,
                    })
                } else {
                    None
                }
            }
        };

        Ok(Self {
            api_key,
            api,
            base_url,
            model,
            max_tokens,
            temperature,
            reasoning,
            structured_output: false,
            chat_reasoning_output,
            custom_capabilities,
        })
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), ConfigError> {
        OpenAiConnectionProfile::new(self).map(|_| ())
    }
}

/// Resolve `OPENAI_MODEL` from its raw environment result.
///
/// A missing variable takes the default model. Any other environment
/// error (for example a non-Unicode value) is surfaced rather than
/// silently defaulted, so a mis-set environment cannot be masked.
fn resolve_model(result: Result<String, std::env::VarError>) -> Result<OpenAiModel, ConfigError> {
    match result {
        Ok(value) => Ok(OpenAiModel::from_id(value.trim())),
        Err(std::env::VarError::NotPresent) => Ok(OpenAiModel::default()),
        Err(e) => Err(ConfigError::from(e)),
    }
}

/// Parse the three explicit custom capability env vars.
///
/// A custom model requires all three to be set; a known model must not
/// have any of them set.
fn parse_custom_capabilities_from_env(
    model: &OpenAiModel,
) -> Result<Option<CustomModelCapabilities>, ConfigError> {
    let temperature = env_parse_strict_bool("OPENAI_CUSTOM_TEMPERATURE")?;
    let reasoning = env_parse_strict_bool("OPENAI_CUSTOM_REASONING")?;
    let streaming = env_parse_strict_bool("OPENAI_CUSTOM_STREAMING")?;

    if let OpenAiModel::Custom(_) = model {
        match (temperature, reasoning, streaming) {
            (Some(t), Some(r), Some(s)) => Ok(Some(CustomModelCapabilities::new(t, r, s))),
            (None, None, None) => Err(ConfigError::InvalidValue {
                field: "OPENAI_CUSTOM_*".to_owned(),
                reason: "custom model requires OPENAI_CUSTOM_TEMPERATURE, OPENAI_CUSTOM_REASONING, and OPENAI_CUSTOM_STREAMING".into(),
            }),
            _ => Err(ConfigError::InvalidValue {
                field: "OPENAI_CUSTOM_*".to_owned(),
                reason: "all three custom capability variables must be set for a custom model".into(),
            }),
        }
    } else {
        for (var, value) in [
            ("OPENAI_CUSTOM_TEMPERATURE", temperature),
            ("OPENAI_CUSTOM_REASONING", reasoning),
            ("OPENAI_CUSTOM_STREAMING", streaming),
        ] {
            if value.is_some() {
                return Err(ConfigError::InvalidValue {
                    field: var.to_owned(),
                    reason: "custom capability variables are only valid with a custom model".into(),
                });
            }
        }
        Ok(None)
    }
}

/// Parse an optional boolean env var, accepting only strict `true`/`false`.
fn env_parse_strict_bool(var: &str) -> Result<Option<bool>, ConfigError> {
    match std::env::var(var) {
        Ok(v) => match v.trim() {
            "true" => Ok(Some(true)),
            "false" => Ok(Some(false)),
            other => Err(ConfigError::InvalidValue {
                field: var.to_owned(),
                reason: format!("expected \"true\" or \"false\", got {other:?}"),
            }),
        },
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(ConfigError::from(e)),
    }
}

/// Parse `OPENAI_REASONING_EFFORT` strictly, rejecting unknown values.
fn parse_reasoning_effort_from_env() -> Result<Option<ReasoningEffort>, ConfigError> {
    match std::env::var("OPENAI_REASONING_EFFORT") {
        Ok(v) => match v.trim() {
            "none" => Ok(Some(ReasoningEffort::None)),
            "minimal" => Ok(Some(ReasoningEffort::Minimal)),
            "low" => Ok(Some(ReasoningEffort::Low)),
            "medium" => Ok(Some(ReasoningEffort::Medium)),
            "high" => Ok(Some(ReasoningEffort::High)),
            "xhigh" => Ok(Some(ReasoningEffort::XHigh)),
            other => Err(ConfigError::InvalidValue {
                field: "OPENAI_REASONING_EFFORT".to_owned(),
                reason: format!(
                    "unknown reasoning effort {other:?}; expected none, minimal, low, medium, high, or xhigh"
                ),
            }),
        },
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(ConfigError::from(e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpt4o_config() -> OpenAiConfig {
        OpenAiConfig {
            api_key: ApiKey::new("test-key").unwrap(),
            api: OpenAiApi::ChatCompletions,
            base_url: None,
            model: OpenAiModel::Gpt4o,
            max_tokens: MaxTokens::new(100).unwrap(),
            temperature: None,
            reasoning: None,
            structured_output: false,
            chat_reasoning_output: OpenAiChatReasoningOutput::None,
            custom_capabilities: None,
        }
    }

    fn custom_config(capabilities: CustomModelCapabilities) -> OpenAiConfig {
        OpenAiConfig {
            api_key: ApiKey::new("test-key").unwrap(),
            api: OpenAiApi::ChatCompletions,
            base_url: None,
            model: OpenAiModel::Custom("org/model".into()),
            max_tokens: MaxTokens::new(100).unwrap(),
            temperature: None,
            reasoning: None,
            structured_output: false,
            chat_reasoning_output: OpenAiChatReasoningOutput::None,
            custom_capabilities: Some(capabilities),
        }
    }

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

    #[test]
    fn model_from_str_recognizes_well_known() {
        let parsed: OpenAiModel = "gpt-4o".parse().unwrap();
        assert_eq!(parsed, OpenAiModel::Gpt4o);

        let parsed: OpenAiModel = "custom-model".parse().unwrap();
        assert_eq!(parsed, OpenAiModel::Custom("custom-model".into()));
    }

    #[test]
    fn model_capabilities_never_infer_custom() {
        assert!(!OpenAiModel::Custom("gpt-5.6-terra".into()).supports_temperature());
        assert!(!OpenAiModel::Custom("gpt-5.6-terra".into()).supports_reasoning());
        assert!(!OpenAiModel::Custom("o4-mini".into()).supports_streaming());
        assert!(!OpenAiModel::Gpt4o.supports_reasoning());
    }

    #[test]
    fn api_defaults_to_chat_completions() {
        assert_eq!(OpenAiApi::default(), OpenAiApi::ChatCompletions);

        let cfg: OpenAiConfig = serde_json::from_str(
            "{\"api_key\": \"k\", \"model\": \"gpt-4o\", \"max_tokens\": 1024}",
        )
        .expect("config without an api field deserializes");
        assert_eq!(cfg.api, OpenAiApi::ChatCompletions);
    }

    #[test]
    fn api_serde_round_trips_kebab_case() {
        let api: OpenAiApi = serde_json::from_str("\"responses\"").unwrap();
        assert_eq!(api, OpenAiApi::Responses);
        assert_eq!(serde_json::to_string(&api).unwrap(), "\"responses\"");

        let api: OpenAiApi = serde_json::from_str("\"chat-completions\"").unwrap();
        assert_eq!(api, OpenAiApi::ChatCompletions);
        assert_eq!(serde_json::to_string(&api).unwrap(), "\"chat-completions\"");
    }

    #[test]
    fn api_from_str_parses_both_surfaces_and_fails_loud() {
        let api: OpenAiApi = "responses".parse().unwrap();
        assert_eq!(api, OpenAiApi::Responses);
        let api: OpenAiApi = "chat-completions".parse().unwrap();
        assert_eq!(api, OpenAiApi::ChatCompletions);

        let err = "completions".parse::<OpenAiApi>().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("chat-completions") && msg.contains("responses"),
            "error must name both accepted values: {msg}"
        );
    }

    #[test]
    fn chat_reasoning_output_defaults_to_none() {
        assert_eq!(
            OpenAiChatReasoningOutput::default(),
            OpenAiChatReasoningOutput::None
        );
    }

    #[test]
    fn chat_reasoning_output_from_str_accepts_values() {
        assert_eq!(
            "none".parse::<OpenAiChatReasoningOutput>().unwrap(),
            OpenAiChatReasoningOutput::None
        );
        assert_eq!(
            "reasoning_content"
                .parse::<OpenAiChatReasoningOutput>()
                .unwrap(),
            OpenAiChatReasoningOutput::ReasoningContent
        );
        assert_eq!(
            "reasoning".parse::<OpenAiChatReasoningOutput>().unwrap(),
            OpenAiChatReasoningOutput::Reasoning
        );
    }

    #[test]
    fn chat_reasoning_output_from_str_rejects_unknown() {
        let err = "reasoningContent"
            .parse::<OpenAiChatReasoningOutput>()
            .unwrap_err();
        assert!(err.to_string().contains("reasoning_content"));
    }

    #[test]
    fn chat_reasoning_output_serde_round_trips() {
        for value in [
            OpenAiChatReasoningOutput::None,
            OpenAiChatReasoningOutput::ReasoningContent,
            OpenAiChatReasoningOutput::Reasoning,
        ] {
            let json = serde_json::to_string(&value).unwrap();
            let parsed: OpenAiChatReasoningOutput = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, value);
        }
    }

    #[test]
    fn custom_capabilities_serde_round_trips() {
        let caps = CustomModelCapabilities::new(true, false, true);
        let json = serde_json::to_string(&caps).unwrap();
        let parsed: CustomModelCapabilities = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, caps);
    }

    #[test]
    fn profile_known_gpt4o_chat_capabilities() {
        let profile = OpenAiConnectionProfile::new(&gpt4o_config()).unwrap();
        assert!(profile.supports_temperature());
        assert!(!profile.supports_reasoning());
        assert!(profile.supports_streaming());
    }

    #[test]
    fn profile_known_gpt5_chat_capabilities() {
        let mut config = gpt4o_config();
        config.model = OpenAiModel::Gpt5;
        let profile = OpenAiConnectionProfile::new(&config).unwrap();
        assert!(!profile.supports_temperature());
        assert!(profile.supports_reasoning());
        assert!(profile.supports_streaming());
    }

    #[test]
    fn profile_known_o3_chat_no_streaming() {
        let mut config = gpt4o_config();
        config.model = OpenAiModel::O3;
        let profile = OpenAiConnectionProfile::new(&config).unwrap();
        assert!(!profile.supports_temperature());
        assert!(profile.supports_reasoning());
        assert!(!profile.supports_streaming());
    }

    #[test]
    fn profile_known_o3_responses_streams() {
        let mut config = gpt4o_config();
        config.api = OpenAiApi::Responses;
        config.model = OpenAiModel::O3;
        let profile = OpenAiConnectionProfile::new(&config).unwrap();
        assert!(!profile.supports_temperature());
        assert!(profile.supports_reasoning());
        assert!(profile.supports_streaming());
    }

    #[test]
    fn profile_custom_requires_capabilities() {
        let mut config = gpt4o_config();
        config.model = OpenAiModel::Custom("org/model".into());
        let err = OpenAiConnectionProfile::new(&config).unwrap_err();
        assert!(err.to_string().contains("custom_capabilities"));
    }

    #[test]
    fn profile_custom_capabilities_accepted() {
        let config = custom_config(CustomModelCapabilities::new(false, true, true));
        let profile = OpenAiConnectionProfile::new(&config).unwrap();
        assert!(!profile.supports_temperature());
        assert!(profile.supports_reasoning());
        assert!(profile.supports_streaming());
    }

    #[test]
    fn profile_known_model_with_custom_capabilities_rejected() {
        let mut config = gpt4o_config();
        config.custom_capabilities = Some(CustomModelCapabilities::new(true, true, true));
        let err = OpenAiConnectionProfile::new(&config).unwrap_err();
        assert!(err.to_string().contains("well-known model"));
    }

    #[test]
    fn profile_responses_rejects_chat_reasoning_output() {
        let mut config = gpt4o_config();
        config.api = OpenAiApi::Responses;
        config.chat_reasoning_output = OpenAiChatReasoningOutput::ReasoningContent;
        let err = OpenAiConnectionProfile::new(&config).unwrap_err();
        assert!(err.to_string().contains("Responses"));
    }

    #[test]
    fn profile_custom_chat_rejects_reasoning() {
        let mut config = custom_config(CustomModelCapabilities::new(false, true, true));
        config.reasoning = Some(ReasoningConfig {
            effort: ReasoningEffort::Medium,
            summary: ReasoningSummary::Auto,
        });
        let err = OpenAiConnectionProfile::new(&config).unwrap_err();
        assert!(err.to_string().contains("custom Chat"));
    }

    #[test]
    fn profile_rejects_temperature_on_reasoning_model() {
        let mut config = gpt4o_config();
        config.model = OpenAiModel::Gpt5;
        config.temperature = Some(Temperature::new(1.0).unwrap());
        let err = OpenAiConnectionProfile::new(&config).unwrap_err();
        assert!(err.to_string().contains("temperature"));
    }

    #[test]
    fn profile_rejects_malformed_base_url() {
        let mut config = gpt4o_config();
        config.base_url = Some("not a url".into());
        let err = OpenAiConnectionProfile::new(&config).unwrap_err();
        assert!(err.to_string().contains("base_url"));
    }

    #[test]
    fn profile_rejects_non_http_base_url() {
        let mut config = gpt4o_config();
        config.base_url = Some("ftp://example.com".into());
        let err = OpenAiConnectionProfile::new(&config).unwrap_err();
        assert!(err.to_string().contains("http"));
    }

    #[test]
    fn profile_accepts_valid_base_url() {
        let mut config = gpt4o_config();
        config.base_url = Some("http://localhost:11434/v1".into());
        let profile = OpenAiConnectionProfile::new(&config).unwrap();
        assert_eq!(profile.base_url(), Some("http://localhost:11434/v1"));
    }

    #[test]
    fn config_validate_delegates_to_profile() {
        let mut config = gpt4o_config();
        config.model = OpenAiModel::Gpt5;
        config.temperature = Some(Temperature::new(1.0).unwrap());
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("temperature"));
    }

    #[test]
    fn resolve_model_defaults_when_absent() {
        let model = resolve_model(Err(std::env::VarError::NotPresent)).unwrap();
        assert_eq!(model, OpenAiModel::default());
    }

    #[test]
    fn resolve_model_parses_present_value() {
        let model = resolve_model(Ok("  gpt-5.2  ".to_owned())).unwrap();
        assert_eq!(model, OpenAiModel::Gpt5_2);
    }

    /// A non-Unicode value must not silently default; it surfaces as an
    /// environment error. The value is constructed directly so the test
    /// is portable across platforms.
    #[test]
    fn resolve_model_surfaces_non_unicode_error() {
        #[cfg(unix)]
        let err = {
            use std::os::unix::ffi::OsStringExt as _;
            std::env::VarError::NotUnicode(std::ffi::OsString::from_vec(vec![0xff, 0xfe]))
        };
        #[cfg(windows)]
        let err = std::env::VarError::NotUnicode(std::ffi::OsString::from("bad\u{fffd}"));

        let result = resolve_model(Err(err));
        let Err(err) = result else {
            panic!("non-Unicode env value must not default");
        };
        assert!(
            matches!(err, ConfigError::Env(_)),
            "expected an environment error, got: {err:?}"
        );
    }
}

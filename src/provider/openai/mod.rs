//! OpenAI provider implementation using the `async-openai` crate.
//!
//! Supports GPT-4o, GPT-5.x, and o-series reasoning models over two
//! config-selected API surfaces ([`crate::config::OpenAiApi`]):
//! - Chat Completions (default): streaming with tool/function calling
//!   and third-party reasoning deltas from compatible endpoints
//! - Responses: the reasoning-native surface for GPT-5.x models, run
//!   statelessly (`store: false`) with the session owning history

mod chat;
mod responses;

pub use chat::{ReasoningAwareChoice, ReasoningAwareChunk, ReasoningAwareDelta};

use std::future::Future;
use std::pin::Pin;

use async_openai::Client;
use async_openai::config::OpenAIConfig;

use crate::config::{OpenAiConfig, OpenAiConnectionProfile};
use crate::error::{
    AuthErrorKind, ProviderError, is_content_policy_message, is_context_window_message,
};
use crate::provider::{
    CompletionRequest, ModelInfo, Provider, ProviderCapabilities, ProviderContext, ProviderInfo,
};
use crate::streaming::StreamHandle;
use crate::types::ModelId;

/// Official OpenAI API base URL.
///
/// Pinned explicitly when [`OpenAiConfig::base_url`] is unset so the
/// `async-openai` SDK's `OPENAI_BASE_URL` environment default cannot
/// override an explicitly configured, programmatic connection.
pub(super) const OFFICIAL_API_BASE: &str = "https://api.openai.com/v1";

/// Map an async-openai request error to the driver's provider errors.
///
/// Shared by both API surfaces (chat completions and responses); the
/// SDK's error type stringifies uniformly, so the mapping is
/// message-based either way.
pub(super) fn map_request_error(msg: String, model: &str) -> ProviderError {
    if msg.contains("401") || msg.contains("Unauthorized") {
        ProviderError::Auth {
            provider: crate::provider::ProviderKind::OpenAi,
            kind: AuthErrorKind::Rejected,
            message: msg,
        }
    } else if msg.contains("429") || msg.contains("rate") {
        ProviderError::RateLimited {
            provider: crate::provider::ProviderKind::OpenAi,
            retry_after: None,
        }
    } else if msg.contains("model") && msg.contains("not found") {
        ProviderError::ModelNotFound {
            provider: crate::provider::ProviderKind::OpenAi,
            model: model.to_owned(),
        }
    } else if is_context_window_message(&msg) {
        ProviderError::ContextWindowExceeded {
            provider: crate::provider::ProviderKind::OpenAi,
            message: msg,
            context_window: None,
            tokens_used: None,
        }
    } else if is_content_policy_message(&msg) {
        ProviderError::ContentPolicyViolation {
            provider: crate::provider::ProviderKind::OpenAi,
            message: msg,
        }
    } else {
        ProviderError::InvalidRequest {
            provider: crate::provider::ProviderKind::OpenAi,
            message: msg,
        }
    }
}

/// OpenAI provider
pub struct OpenAiProvider {
    profile: OpenAiConnectionProfile,
    client: Client<OpenAIConfig>,
    info: ProviderInfo,
}

#[allow(
    clippy::multiple_inherent_impl,
    reason = "the OpenAI impl is split per API surface (mod, chat, responses); the lint targets accidental fragmentation"
)]
impl OpenAiProvider {
    /// Create a new OpenAI-compatible provider.
    ///
    /// `config.base_url` points the client at any OpenAI-compatible
    /// endpoint (BaseTen Model APIs, OpenRouter, a local vLLM server);
    /// unset means the official OpenAI base.
    pub fn new(config: OpenAiConfig) -> Result<Self, ProviderError> {
        let profile =
            OpenAiConnectionProfile::new(&config).map_err(|e| ProviderError::InvalidRequest {
                provider: crate::provider::ProviderKind::OpenAi,
                message: e.to_string(),
            })?;

        // The profile is the sole source for the selected connection. A
        // configured base URL wins; otherwise pin the official base
        // explicitly so the SDK's `OPENAI_BASE_URL` environment default
        // cannot silently redirect a programmatic configuration.
        let base = profile
            .base_url()
            .unwrap_or(crate::provider::openai::OFFICIAL_API_BASE);
        let openai_config = OpenAIConfig::new()
            .with_api_key(config.api_key.as_str())
            .with_api_base(base);
        let client = Client::with_config(openai_config);

        let info = ProviderInfo {
            kind: crate::provider::ProviderKind::OpenAi,
            name: "OpenAI",
            capabilities: ProviderCapabilities {
                streaming: profile.supports_streaming(),
                tools: true,
                vision: true,
                extended_thinking: profile.supports_reasoning(),
                max_context_tokens: Some(128_000),
            },
        };

        Ok(Self {
            profile,
            client,
            info,
        })
    }
}

impl Provider for OpenAiProvider {
    fn info(&self) -> &ProviderInfo {
        &self.info
    }

    fn complete_stream(
        &self,
        request: CompletionRequest,
        ctx: ProviderContext,
    ) -> Pin<Box<dyn Future<Output = Result<StreamHandle, ProviderError>> + Send + '_>> {
        match self.profile.api() {
            crate::config::OpenAiApi::ChatCompletions => self.complete_stream_chat(request, ctx),
            crate::config::OpenAiApi::Responses => self.complete_stream_responses(request, ctx),
        }
    }

    fn list_models(
        &self,
        _ctx: ProviderContext,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<ModelInfo>, ProviderError>> + Send + '_>> {
        #[allow(
            clippy::expect_used,
            reason = "listed OpenAI model IDs are hardcoded valid constants"
        )]
        Box::pin(async move {
            // All model IDs below are hardcoded valid strings
            Ok(vec![
                ModelInfo {
                    id: ModelId::new("gpt-4o").expect("hardcoded valid model ID"),
                    name: "GPT-4o".to_owned(),
                    context_window: Some(128_000),
                },
                ModelInfo {
                    id: ModelId::new("gpt-4o-mini").expect("hardcoded valid model ID"),
                    name: "GPT-4o Mini".to_owned(),
                    context_window: Some(128_000),
                },
                ModelInfo {
                    id: ModelId::new("o1").expect("hardcoded valid model ID"),
                    name: "o1".to_owned(),
                    context_window: Some(200_000),
                },
                ModelInfo {
                    id: ModelId::new("o1-mini").expect("hardcoded valid model ID"),
                    name: "o1 Mini".to_owned(),
                    context_window: Some(128_000),
                },
            ])
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        ApiKey, CustomModelCapabilities, OpenAiApi, OpenAiChatReasoningOutput, OpenAiConfig,
        OpenAiModel,
    };
    use crate::provider::CompletionConfig;
    use crate::types::{CorrelationId, MaxTokens, Message};
    use async_openai::config::Config as _;

    fn base_config(model: OpenAiModel, api: OpenAiApi) -> OpenAiConfig {
        OpenAiConfig {
            api_key: ApiKey::new("test-key").unwrap(),
            api,
            base_url: None,
            model,
            max_tokens: MaxTokens::new(100).unwrap(),
            temperature: None,
            reasoning: None,
            structured_output: false,
            chat_reasoning_output: OpenAiChatReasoningOutput::None,
            custom_capabilities: None,
        }
    }

    fn custom_config(
        model: &str,
        api: OpenAiApi,
        capabilities: CustomModelCapabilities,
    ) -> OpenAiConfig {
        let mut config = base_config(OpenAiModel::Custom(model.to_owned()), api);
        config.custom_capabilities = Some(capabilities);
        config
    }

    fn ctx() -> ProviderContext {
        ProviderContext::new(
            CorrelationId::generate(),
            tokio_util::sync::CancellationToken::new(),
            tokio_util::task::TaskTracker::new(),
        )
    }

    fn user_request(model: &str) -> CompletionRequest {
        CompletionRequest::new(ModelId::new(model).unwrap(), vec![Message::user("hi")])
    }

    #[test]
    fn programmatic_custom_config_builds_and_validates() {
        let config = OpenAiConfig::custom(
            ApiKey::new("k").unwrap(),
            "org/local-llm",
            MaxTokens::new(128).unwrap(),
            CustomModelCapabilities::new(true, false, true),
        )
        .build();
        let provider = OpenAiProvider::new(config).unwrap();
        let info = provider.info();
        assert!(info.capabilities.streaming);
        assert!(!info.capabilities.extended_thinking);
    }

    #[test]
    fn custom_capabilities_honored_in_info() {
        let config = custom_config(
            "org/local-llm",
            OpenAiApi::ChatCompletions,
            CustomModelCapabilities::new(true, false, false),
        );
        let provider = OpenAiProvider::new(config).unwrap();
        let info = provider.info();
        assert!(
            !info.capabilities.streaming,
            "declared non-streaming custom model must report no streaming"
        );
        assert!(!info.capabilities.extended_thinking);
    }

    #[test]
    fn custom_reasoning_capability_reflected_in_info() {
        let config = custom_config(
            "org/reasoner",
            OpenAiApi::Responses,
            CustomModelCapabilities::new(false, true, true),
        );
        let provider = OpenAiProvider::new(config).unwrap();
        assert!(provider.info().capabilities.extended_thinking);
    }

    #[test]
    fn o3_chat_info_reports_no_streaming_but_responses_does() {
        let chat = OpenAiProvider::new(base_config(OpenAiModel::O3, OpenAiApi::ChatCompletions));
        assert!(!chat.unwrap().info().capabilities.streaming);

        let responses =
            OpenAiProvider::new(base_config(OpenAiModel::O3, OpenAiApi::Responses)).unwrap();
        assert!(
            responses.info().capabilities.streaming,
            "o3 must be allowed to stream on the Responses surface"
        );
    }

    #[test]
    fn configured_base_url_is_used_verbatim() {
        let mut config = base_config(OpenAiModel::Gpt4o, OpenAiApi::ChatCompletions);
        config.base_url = Some("http://localhost:11434/v1".to_owned());
        let provider = OpenAiProvider::new(config).unwrap();
        assert_eq!(
            provider.client.config().api_base(),
            "http://localhost:11434/v1"
        );
    }

    /// A programmatic config with no base URL must pin the official base
    /// explicitly; the SDK's `OPENAI_BASE_URL` default must not redirect
    /// an explicitly configured connection.
    #[test]
    fn unset_base_url_pins_official_base_despite_ambient_env() {
        struct EnvGuard(Option<String>);
        impl Drop for EnvGuard {
            fn drop(&mut self) {
                // SAFETY: restores the process environment variable this
                // guard set. The guard runs even on panic.
                match &self.0 {
                    // SAFETY: test-only restore of a previously captured
                    // value.
                    Some(value) => unsafe { std::env::set_var("OPENAI_BASE_URL", value) },
                    // SAFETY: test-only removal of the variable this test
                    // introduced.
                    None => unsafe { std::env::remove_var("OPENAI_BASE_URL") },
                }
            }
        }

        let config = base_config(OpenAiModel::Gpt4o, OpenAiApi::ChatCompletions);
        let _guard = EnvGuard(std::env::var("OPENAI_BASE_URL").ok());
        // SAFETY: test-only mutation; `_guard` restores the previous
        // value on drop.
        unsafe {
            std::env::set_var("OPENAI_BASE_URL", "http://ambient.example/v1");
        }

        let provider = OpenAiProvider::new(config).unwrap();
        assert_eq!(
            provider.client.config().api_base(),
            OFFICIAL_API_BASE,
            "official base must win over the SDK env default"
        );
    }

    #[tokio::test]
    async fn mismatched_request_model_fails_at_request_time() {
        let provider =
            OpenAiProvider::new(base_config(OpenAiModel::Gpt4o, OpenAiApi::ChatCompletions))
                .unwrap();
        let request = user_request("gpt-4o-mini");
        let err = provider.complete_stream(request, ctx()).await.unwrap_err();
        assert!(
            matches!(err, ProviderError::InvalidRequest { .. }),
            "mismatched model must fail loud, got: {err:?}"
        );
    }

    #[tokio::test]
    async fn matching_request_model_reaches_transport() {
        // Point at a local port so the transport failure is fast and the
        // test stays hermetic.
        let mut config = base_config(OpenAiModel::Gpt4o, OpenAiApi::ChatCompletions);
        config.base_url = Some("http://127.0.0.1:1/v1".to_owned());
        let provider = OpenAiProvider::new(config).unwrap();
        let request = user_request("gpt-4o");
        // No server: the request must pass the local guards and fail in
        // transport, not on the model-match guard.
        let err = provider.complete_stream(request, ctx()).await.unwrap_err();
        let message = err.to_string();
        assert!(
            !message.contains("does not match configured model"),
            "matching model must not trip the model guard: {message}"
        );
    }

    #[tokio::test]
    async fn responses_mismatched_request_model_fails_at_request_time() {
        let provider =
            OpenAiProvider::new(base_config(OpenAiModel::Gpt5, OpenAiApi::Responses)).unwrap();
        let request = user_request("gpt-5.2");
        let err = provider.complete_stream(request, ctx()).await.unwrap_err();
        assert!(matches!(err, ProviderError::InvalidRequest { .. }));
    }

    /// A custom non-streaming Chat model must fail the request guard
    /// before any transport call.
    #[tokio::test]
    async fn custom_non_streaming_chat_fails_loud() {
        let config = custom_config(
            "org/local",
            OpenAiApi::ChatCompletions,
            CustomModelCapabilities::new(true, false, false),
        );
        let provider = OpenAiProvider::new(config).unwrap();
        let request = user_request("org/local");
        let err = provider.complete_stream(request, ctx()).await.unwrap_err();
        assert!(
            matches!(err, ProviderError::StreamingNotSupported { .. }),
            "declared non-streaming model must fail loud, got: {err:?}"
        );
    }

    #[tokio::test]
    async fn request_temperature_on_non_temperature_model_fails_loud() {
        let provider =
            OpenAiProvider::new(base_config(OpenAiModel::Gpt5, OpenAiApi::Responses)).unwrap();
        let request = user_request("gpt-5").with_config(
            CompletionConfig::new(MaxTokens::new(64).unwrap())
                .with_temperature(crate::types::Temperature::new(1.0).unwrap()),
        );
        let err = provider.complete_stream(request, ctx()).await.unwrap_err();
        assert!(
            matches!(err, ProviderError::InvalidRequest { .. }),
            "temperature on a reasoning model must fail loud, got: {err:?}"
        );
    }
}

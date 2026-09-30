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

use crate::config::OpenAiConfig;
use crate::error::{
    AuthErrorKind, ProviderError, is_content_policy_message, is_context_window_message,
};
use crate::provider::{
    CompletionRequest, ModelInfo, Provider, ProviderCapabilities, ProviderContext, ProviderInfo,
};
use crate::streaming::StreamHandle;
use crate::types::ModelId;

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
    config: OpenAiConfig,
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
        let supports_streaming = config.model.supports_streaming();

        // Create async-openai client with API key, and the configured
        // base URL when one is set.
        let mut openai_config = OpenAIConfig::new().with_api_key(config.api_key.as_str());
        if let Some(base) = &config.base_url {
            openai_config = openai_config.with_api_base(base);
        }
        let client = Client::with_config(openai_config);

        let info = ProviderInfo {
            kind: crate::provider::ProviderKind::OpenAi,
            name: "OpenAI",
            capabilities: ProviderCapabilities {
                streaming: supports_streaming,
                tools: true,
                vision: true,
                extended_thinking: config.model.supports_reasoning(),
                max_context_tokens: Some(128_000),
            },
        };

        Ok(Self {
            config,
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
        match self.config.api {
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

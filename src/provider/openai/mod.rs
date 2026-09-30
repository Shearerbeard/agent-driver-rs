//! OpenAI provider implementation using the `async-openai` crate.
//!
//! Supports GPT-4o, GPT-5.x, and o-series reasoning models. Handles:
//! - Streaming chat completions with tool/function calling
//! - Reasoning effort configuration for models that support it
//! - Non-streaming fallback for o3/o3-mini models

pub mod chat;

pub use chat::{ReasoningAwareChoice, ReasoningAwareChunk, ReasoningAwareDelta};

use std::future::Future;
use std::pin::Pin;

use async_openai::Client;
use async_openai::config::OpenAIConfig;

use crate::config::OpenAiConfig;
use crate::error::ProviderError;
use crate::provider::{
    CompletionRequest, ModelInfo, Provider, ProviderCapabilities, ProviderContext, ProviderInfo,
};
use crate::streaming::StreamHandle;
use crate::types::ModelId;

/// OpenAI provider
pub struct OpenAiProvider {
    config: OpenAiConfig,
    client: Client<OpenAIConfig>,
    info: ProviderInfo,
}

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
        self.complete_stream_chat(request, ctx)
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

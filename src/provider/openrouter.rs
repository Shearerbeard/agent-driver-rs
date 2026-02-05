//! OpenRouter provider implementation

use std::future::Future;
use std::pin::Pin;

use crate::config::OpenRouterConfig;
use crate::error::ProviderError;
use crate::streaming::StreamHandle;
use crate::types::ModelId;

use super::{
    CompletionRequest, ModelInfo, Provider, ProviderCapabilities, ProviderContext, ProviderInfo,
};

/// OpenRouter provider
pub struct OpenRouterProvider {
    #[allow(dead_code)]
    config: OpenRouterConfig,
    info: ProviderInfo,
}

impl OpenRouterProvider {
    /// Create a new OpenRouter provider
    pub fn new(config: OpenRouterConfig) -> Result<Self, ProviderError> {
        let info = ProviderInfo {
            id: "openrouter",
            name: "OpenRouter",
            capabilities: ProviderCapabilities {
                streaming: true,
                tools: true,
                vision: true,
                extended_thinking: false,
                max_context_tokens: None, // Varies by model
            },
        };

        Ok(Self { config, info })
    }
}

impl Provider for OpenRouterProvider {
    fn info(&self) -> &ProviderInfo {
        &self.info
    }

    fn complete_stream(
        &self,
        _request: CompletionRequest,
        _ctx: ProviderContext,
    ) -> Pin<Box<dyn Future<Output = Result<StreamHandle, ProviderError>> + Send + '_>> {
        Box::pin(async move {
            // TODO: Implement OpenRouter streaming using reqwest-eventsource
            Err(ProviderError::InvalidRequest(
                "OpenRouter provider not yet implemented".into(),
            ))
        })
    }

    fn list_models(
        &self,
        _ctx: ProviderContext,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<ModelInfo>, ProviderError>> + Send + '_>> {
        Box::pin(async move {
            // OpenRouter has a models endpoint, but for now return common ones
            Ok(vec![
                ModelInfo {
                    id: ModelId::new("anthropic/claude-sonnet-4").unwrap(),
                    name: "Claude Sonnet 4".to_string(),
                    context_window: Some(200_000),
                },
                ModelInfo {
                    id: ModelId::new("openai/gpt-4o").unwrap(),
                    name: "GPT-4o".to_string(),
                    context_window: Some(128_000),
                },
                ModelInfo {
                    id: ModelId::new("google/gemini-2.0-flash").unwrap(),
                    name: "Gemini 2.0 Flash".to_string(),
                    context_window: Some(1_000_000),
                },
            ])
        })
    }
}

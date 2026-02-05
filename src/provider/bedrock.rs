//! AWS Bedrock provider implementation

use std::future::Future;
use std::pin::Pin;

use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::types::{
    ContentBlock as BedrockContentBlock, ConversationRole, Message as BedrockMessage,
    SystemContentBlock, Tool, ToolConfiguration, ToolInputSchema, ToolResultBlock,
    ToolResultContentBlock, ToolSpecification, ToolUseBlock,
};
use aws_sdk_bedrockruntime::Client;
use aws_smithy_types::Document;

use crate::config::BedrockConfig;
use crate::error::{ProviderError, StreamError};
use crate::streaming::{
    CompletionMetadata, ContentBlockType, StopReason, StreamDelta, StreamEvent, StreamHandle,
    TokenUsage,
};
use crate::types::{ContentBlock, ModelId, ToolCallId, ToolName};

use super::{
    CompletionRequest, ModelInfo, Provider, ProviderCapabilities, ProviderContext, ProviderInfo,
};

/// AWS Bedrock provider
pub struct BedrockProvider {
    config: BedrockConfig,
    client: Client,
    info: ProviderInfo,
}

impl BedrockProvider {
    /// Create a new Bedrock provider
    ///
    /// This is async because it needs to load AWS credentials.
    pub async fn new(config: BedrockConfig) -> Result<Self, ProviderError> {
        // Load AWS config from environment
        let aws_config = aws_config::defaults(BehaviorVersion::latest())
            .region(aws_config::Region::new(config.region.as_str().to_string()))
            .load()
            .await;

        let client = Client::new(&aws_config);

        let info = ProviderInfo {
            id: "bedrock",
            name: "AWS Bedrock",
            capabilities: ProviderCapabilities {
                streaming: true,
                tools: true,
                vision: true,
                extended_thinking: false,
                max_context_tokens: Some(200_000),
            },
        };

        Ok(Self {
            config,
            client,
            info,
        })
    }

    /// Convert our messages to Bedrock format
    fn convert_messages(
        &self,
        messages: &[crate::types::Message],
    ) -> Result<Vec<BedrockMessage>, ProviderError> {
        let mut bedrock_messages = Vec::new();

        for msg in messages {
            let role = match msg.role {
                crate::types::Role::User => ConversationRole::User,
                crate::types::Role::Assistant => ConversationRole::Assistant,
                crate::types::Role::Tool => ConversationRole::User, // Tool results come as user
                crate::types::Role::System => continue, // System handled separately
            };

            let mut content_blocks = Vec::new();

            for block in &msg.content {
                match block {
                    ContentBlock::Text { text } => {
                        content_blocks.push(BedrockContentBlock::Text(text.clone()));
                    }
                    ContentBlock::Thinking { text } => {
                        // Bedrock doesn't have a thinking block, include as text
                        content_blocks.push(BedrockContentBlock::Text(format!(
                            "<thinking>{}</thinking>",
                            text
                        )));
                    }
                    ContentBlock::ToolUse { id, name, input } => {
                        content_blocks.push(BedrockContentBlock::ToolUse(
                            ToolUseBlock::builder()
                                .tool_use_id(id.as_str())
                                .name(name.as_str())
                                .input(Document::Object(
                                    input
                                        .as_object()
                                        .map(|o| {
                                            o.iter()
                                                .map(|(k, v)| (k.clone(), json_to_document(v)))
                                                .collect()
                                        })
                                        .unwrap_or_default(),
                                ))
                                .build()
                                .map_err(|e| {
                                    ProviderError::InvalidRequest(format!(
                                        "Failed to build tool use: {}",
                                        e
                                    ))
                                })?,
                        ));
                    }
                    ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    } => {
                        let content_str = match content {
                            crate::types::ToolResultContent::Text(s) => s.clone(),
                        };
                        content_blocks.push(BedrockContentBlock::ToolResult(
                            ToolResultBlock::builder()
                                .tool_use_id(tool_use_id.as_str())
                                .content(ToolResultContentBlock::Text(content_str))
                                .status(if *is_error {
                                    aws_sdk_bedrockruntime::types::ToolResultStatus::Error
                                } else {
                                    aws_sdk_bedrockruntime::types::ToolResultStatus::Success
                                })
                                .build()
                                .map_err(|e| {
                                    ProviderError::InvalidRequest(format!(
                                        "Failed to build tool result: {}",
                                        e
                                    ))
                                })?,
                        ));
                    }
                }
            }

            if !content_blocks.is_empty() {
                bedrock_messages.push(
                    BedrockMessage::builder()
                        .role(role)
                        .set_content(Some(content_blocks))
                        .build()
                        .map_err(|e| {
                            ProviderError::InvalidRequest(format!("Failed to build message: {}", e))
                        })?,
                );
            }
        }

        Ok(bedrock_messages)
    }

    /// Convert our tools to Bedrock format
    fn convert_tools(
        &self,
        tools: &[crate::tool::ToolDefinition],
    ) -> Result<Option<ToolConfiguration>, ProviderError> {
        if tools.is_empty() {
            return Ok(None);
        }

        let mut bedrock_tools = Vec::new();
        for t in tools {
            let schema_doc = json_to_document(&t.input_schema.to_value());

            let spec = ToolSpecification::builder()
                .name(t.name.as_str())
                .description(t.description.clone())
                .input_schema(ToolInputSchema::Json(schema_doc))
                .build()
                .map_err(|e| {
                    ProviderError::InvalidRequest(format!("Failed to build tool spec: {}", e))
                })?;

            bedrock_tools.push(Tool::ToolSpec(spec));
        }

        Ok(Some(
            ToolConfiguration::builder()
                .set_tools(Some(bedrock_tools))
                .build()
                .map_err(|e| {
                    ProviderError::InvalidRequest(format!("Failed to build tool config: {}", e))
                })?,
        ))
    }
}

/// Convert serde_json::Value to AWS Document
fn json_to_document(value: &serde_json::Value) -> Document {
    match value {
        serde_json::Value::Null => Document::Null,
        serde_json::Value::Bool(b) => Document::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Document::Number(aws_smithy_types::Number::NegInt(i))
            } else if let Some(f) = n.as_f64() {
                Document::Number(aws_smithy_types::Number::Float(f))
            } else {
                Document::Null
            }
        }
        serde_json::Value::String(s) => Document::String(s.clone()),
        serde_json::Value::Array(arr) => {
            Document::Array(arr.iter().map(json_to_document).collect())
        }
        serde_json::Value::Object(obj) => Document::Object(
            obj.iter()
                .map(|(k, v)| (k.clone(), json_to_document(v)))
                .collect(),
        ),
    }
}

impl Provider for BedrockProvider {
    fn info(&self) -> &ProviderInfo {
        &self.info
    }

    fn complete_stream(
        &self,
        request: CompletionRequest,
        ctx: ProviderContext,
    ) -> Pin<Box<dyn Future<Output = Result<StreamHandle, ProviderError>> + Send + '_>> {
        Box::pin(async move {
            // Build the model ID - use inference profile if provided
            let model_id = self
                .config
                .inference_profile
                .clone()
                .unwrap_or_else(|| self.config.model.model_id().to_string());

            // Convert messages
            let messages = self.convert_messages(&request.messages)?;

            // Build the request
            let mut req = self
                .client
                .converse_stream()
                .model_id(&model_id)
                .set_messages(Some(messages))
                .inference_config(
                    aws_sdk_bedrockruntime::types::InferenceConfiguration::builder()
                        .max_tokens(request.config.max_tokens.get() as i32)
                        .set_temperature(request.config.temperature.map(|t| t.get()))
                        .set_stop_sequences(if request.config.stop_sequences.is_empty() {
                            None
                        } else {
                            Some(request.config.stop_sequences.clone())
                        })
                        .build(),
                );

            // Add system prompt
            if let Some(ref system) = request.system {
                if !system.is_empty() {
                    req = req.system(SystemContentBlock::Text(system.as_str().to_string()));
                }
            }

            // Add tools if present
            if let Some(tool_config) = self.convert_tools(&request.tools)? {
                req = req.tool_config(tool_config);
            }

            // Send the request
            let response = req.send().await.map_err(|e| {
                let msg = e.to_string();
                if msg.contains("AccessDenied") || msg.contains("UnauthorizedException") {
                    ProviderError::Auth(msg)
                } else if msg.contains("ThrottlingException") {
                    ProviderError::RateLimited { retry_after: None }
                } else if msg.contains("ModelNotFound") || msg.contains("ResourceNotFoundException")
                {
                    ProviderError::ModelNotFound(model_id.clone())
                } else if msg.contains("inference profile") || msg.contains("InferenceProfile") {
                    ProviderError::InvalidRequest(format!(
                        "Model requires an inference profile. Set BEDROCK_INFERENCE_PROFILE env var. Error: {}",
                        msg
                    ))
                } else {
                    ProviderError::InvalidRequest(msg)
                }
            })?;

            // Create stream from response
            let stream = response.stream;
            let cancellation = ctx.cancellation.clone();

            let event_stream = futures::stream::unfold(
                (stream, cancellation.clone(), StreamState::default()),
                |(mut stream, cancel, mut state)| async move {
                    if cancel.is_cancelled() {
                        return None;
                    }

                    tokio::select! {
                        biased;

                        _ = cancel.cancelled() => {
                            None
                        }

                        event = stream.recv() => {
                            match event {
                                Ok(Some(event)) => {
                                    let events = parse_bedrock_event(event, &mut state);
                                    if let Some(first) = events.into_iter().next() {
                                        Some((first, (stream, cancel, state)))
                                    } else {
                                        // Return a placeholder event to keep the stream going
                                        Some((Ok(StreamEvent::Started { metadata: CompletionMetadata::default() }), (stream, cancel, state)))
                                    }
                                }
                                Ok(None) => None,
                                Err(e) => {
                                    Some((Err(StreamError::ConnectionLost(e.to_string())), (stream, cancel, state)))
                                }
                            }
                        }
                    }
                },
            );

            Ok(StreamHandle::new(
                Box::pin(event_stream),
                ctx.cancellation,
                ctx.correlation_id,
            ))
        })
    }

    fn list_models(
        &self,
        _ctx: ProviderContext,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<ModelInfo>, ProviderError>> + Send + '_>> {
        Box::pin(async move {
            Ok(vec![
                ModelInfo {
                    id: ModelId::new("anthropic.claude-opus-4-5-20251101-v1:0").unwrap(),
                    name: "Claude Opus 4.5 (Bedrock)".to_string(),
                    context_window: Some(200_000),
                },
                ModelInfo {
                    id: ModelId::new("anthropic.claude-sonnet-4-5-20250929-v1:0").unwrap(),
                    name: "Claude Sonnet 4.5 (Bedrock)".to_string(),
                    context_window: Some(200_000),
                },
                ModelInfo {
                    id: ModelId::new("anthropic.claude-haiku-4-5-20251001-v1:0").unwrap(),
                    name: "Claude Haiku 4.5 (Bedrock)".to_string(),
                    context_window: Some(200_000),
                },
                ModelInfo {
                    id: ModelId::new("anthropic.claude-sonnet-4-20250514-v1:0").unwrap(),
                    name: "Claude Sonnet 4 (Bedrock)".to_string(),
                    context_window: Some(200_000),
                },
            ])
        })
    }
}

/// State for tracking stream parsing
#[derive(Default)]
struct StreamState {
    current_tool_use_id: Option<String>,
    #[allow(dead_code)]
    current_tool_name: Option<String>,
}

/// Parse a Bedrock streaming event
fn parse_bedrock_event(
    event: aws_sdk_bedrockruntime::types::ConverseStreamOutput,
    state: &mut StreamState,
) -> Vec<Result<StreamEvent, StreamError>> {
    use aws_sdk_bedrockruntime::types::ConverseStreamOutput;

    match event {
        ConverseStreamOutput::MessageStart(_msg) => {
            vec![Ok(StreamEvent::Started {
                metadata: CompletionMetadata {
                    model: None,
                    stop_reason: None,
                    usage: None,
                },
            })]
        }
        ConverseStreamOutput::ContentBlockStart(block) => {
            let index = block.content_block_index() as usize;

            if let Some(start) = block.start() {
                match start {
                    aws_sdk_bedrockruntime::types::ContentBlockStart::ToolUse(tool) => {
                        state.current_tool_use_id = Some(tool.tool_use_id().to_string());
                        state.current_tool_name = Some(tool.name().to_string());

                        vec![
                            Ok(StreamEvent::ContentBlockStart {
                                index,
                                block_type: ContentBlockType::ToolUse,
                            }),
                            Ok(StreamEvent::Delta(StreamDelta::ToolUseStart {
                                id: ToolCallId::new(tool.tool_use_id()),
                                name: ToolName::new(tool.name())
                                    .unwrap_or_else(|_| ToolName::new("unknown").unwrap()),
                            })),
                        ]
                    }
                    _ => vec![Ok(StreamEvent::ContentBlockStart {
                        index,
                        block_type: ContentBlockType::Text,
                    })],
                }
            } else {
                vec![Ok(StreamEvent::ContentBlockStart {
                    index,
                    block_type: ContentBlockType::Text,
                })]
            }
        }
        ConverseStreamOutput::ContentBlockDelta(delta) => {
            if let Some(d) = delta.delta() {
                match d {
                    aws_sdk_bedrockruntime::types::ContentBlockDelta::Text(text) => {
                        vec![Ok(StreamEvent::Delta(StreamDelta::TextDelta {
                            text: text.clone(),
                        }))]
                    }
                    aws_sdk_bedrockruntime::types::ContentBlockDelta::ToolUse(tool) => {
                        vec![Ok(StreamEvent::Delta(StreamDelta::ToolInputDelta {
                            id: ToolCallId::new(
                                state.current_tool_use_id.clone().unwrap_or_default(),
                            ),
                            partial_json: tool.input().to_string(),
                        }))]
                    }
                    _ => vec![],
                }
            } else {
                vec![]
            }
        }
        ConverseStreamOutput::ContentBlockStop(stop) => {
            vec![Ok(StreamEvent::ContentBlockStop {
                index: stop.content_block_index() as usize,
            })]
        }
        ConverseStreamOutput::MessageStop(stop) => {
            let stop_reason = match stop.stop_reason() {
                aws_sdk_bedrockruntime::types::StopReason::EndTurn => Some(StopReason::EndTurn),
                aws_sdk_bedrockruntime::types::StopReason::MaxTokens => Some(StopReason::MaxTokens),
                aws_sdk_bedrockruntime::types::StopReason::ToolUse => Some(StopReason::ToolUse),
                aws_sdk_bedrockruntime::types::StopReason::StopSequence => {
                    Some(StopReason::StopSequence)
                }
                _ => Some(StopReason::EndTurn),
            };

            vec![Ok(StreamEvent::Completed {
                metadata: CompletionMetadata {
                    model: None,
                    stop_reason,
                    usage: None,
                },
            })]
        }
        ConverseStreamOutput::Metadata(meta) => {
            let usage = meta.usage().map(|u| TokenUsage {
                input_tokens: u.input_tokens() as u32,
                output_tokens: u.output_tokens() as u32,
            });

            vec![Ok(StreamEvent::Completed {
                metadata: CompletionMetadata {
                    model: None,
                    stop_reason: None,
                    usage,
                },
            })]
        }
        _ => vec![],
    }
}

//! AWS Bedrock provider implementation using the `converse_stream` API.
//!
//! Bedrock uses the AWS SDK's `ConverseStream` operation, which has its own event
//! format distinct from Anthropic's direct API. This module handles:
//! - AWS credential loading and region configuration
//! - Converting between library message types and Bedrock's `ContentBlock`/`Message` types
//! - Converting `serde_json::Value` to `aws_smithy_types::Document`
//! - Parsing `ConverseStreamOutput` events into the library's `StreamEvent` enum
//! - Inference profile support for cross-region routing

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::Client;
use aws_sdk_bedrockruntime::types::{
    ContentBlock as BedrockContentBlock, ConversationRole, Message as BedrockMessage,
    SystemContentBlock, Tool, ToolConfiguration, ToolInputSchema, ToolResultBlock,
    ToolResultContentBlock, ToolSpecification, ToolUseBlock,
};
use aws_smithy_types::Document;

use crate::config::BedrockConfig;
use crate::error::{
    AuthErrorKind, ProviderError, StreamError, StreamErrorKind, is_content_policy_message,
    is_context_window_message,
};
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
            .region(aws_config::Region::new(config.region.as_str().to_owned()))
            .load()
            .await;

        let client = Client::new(&aws_config);

        let info = ProviderInfo {
            kind: super::ProviderKind::Bedrock,
            name: "AWS Bedrock",
            capabilities: ProviderCapabilities {
                streaming: true,
                tools: true,
                vision: true,
                extended_thinking: config.thinking.is_some(),
                max_context_tokens: Some(200_000),
            },
        };

        Ok(Self {
            config,
            client,
            info,
        })
    }

    /// Convert our messages to Bedrock format, merging consecutive same-role messages.
    fn convert_messages(
        &self,
        messages: &[crate::types::Message],
    ) -> Result<Vec<BedrockMessage>, ProviderError> {
        let thinking_enabled = self.config.thinking.is_some();
        convert_messages(messages, thinking_enabled)
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
                .map_err(|e| ProviderError::InvalidRequest {
                    provider: super::ProviderKind::Bedrock,
                    message: format!("Failed to build tool spec: {e}"),
                })?;

            bedrock_tools.push(Tool::ToolSpec(spec));
        }

        Ok(Some(
            ToolConfiguration::builder()
                .set_tools(Some(bedrock_tools))
                .build()
                .map_err(|e| ProviderError::InvalidRequest {
                    provider: super::ProviderKind::Bedrock,
                    message: format!("Failed to build tool config: {e}"),
                })?,
        ))
    }
}

/// Convert library messages to Bedrock format, merging consecutive same-role messages.
///
/// Bedrock's converse API requires strictly alternating User/Assistant roles.
/// When the model makes multiple tool calls in one response, the agent loop
/// appends a separate `Role::Tool` message per result. Since `Role::Tool` maps
/// to `ConversationRole::User`, consecutive tool-result messages must be merged
/// into a single Bedrock User message to satisfy the alternation constraint.
///
/// Replayed `Thinking` blocks take the wire shape the configured mode
/// expects: with thinking configured they replay as `ReasoningContent`
/// blocks (Claude's round-trip shape); without it they flatten to
/// `<thinking>` text for models that do not accept reasoning input.
fn convert_messages(
    messages: &[crate::types::Message],
    thinking_enabled: bool,
) -> Result<Vec<BedrockMessage>, ProviderError> {
    // Phase 1: Convert content blocks, merging consecutive same-role entries.
    let mut pairs: Vec<(ConversationRole, Vec<BedrockContentBlock>)> = Vec::new();

    for msg in messages {
        let role = match msg.role {
            crate::types::Role::User => ConversationRole::User,
            crate::types::Role::Assistant => ConversationRole::Assistant,
            crate::types::Role::Tool => ConversationRole::User, // Tool results come as user
            crate::types::Role::System => continue,             // System handled separately
        };

        let mut content_blocks = Vec::new();

        for block in &msg.content {
            match block {
                ContentBlock::Text { text } => {
                    content_blocks.push(BedrockContentBlock::Text(text.clone()));
                }
                ContentBlock::Thinking { text } => {
                    match replay_thinking_block(text, thinking_enabled) {
                        Some(block) => content_blocks.push(block),
                        None => continue,
                    }
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
                            .map_err(|e| ProviderError::InvalidRequest {
                                provider: super::ProviderKind::Bedrock,
                                message: format!("Failed to build tool use: {e}"),
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
                            .map_err(|e| ProviderError::InvalidRequest {
                                provider: super::ProviderKind::Bedrock,
                                message: format!("Failed to build tool result: {e}"),
                            })?,
                    ));
                }
            }
        }

        if content_blocks.is_empty() {
            continue;
        }

        // Merge into previous entry if same role and compatible content types,
        // otherwise push new pair. Bedrock rejects messages that mix conversation
        // blocks (Text, ToolUse) with tool result blocks in the same turn.
        if let Some(last) = pairs.last_mut() {
            if last.0 == role && blocks_compatible(&last.1, &content_blocks) {
                last.1.extend(content_blocks);
                continue;
            }
        }
        pairs.push((role, content_blocks));
    }

    // Phase 2: Build BedrockMessages from accumulated pairs.
    pairs
        .into_iter()
        .map(|(role, blocks)| {
            BedrockMessage::builder()
                .role(role)
                .set_content(Some(blocks))
                .build()
                .map_err(|e| ProviderError::InvalidRequest {
                    provider: super::ProviderKind::Bedrock,
                    message: format!("Failed to build message: {e}"),
                })
        })
        .collect()
}

/// Convert serde_json::Value to AWS Document
fn json_to_document(value: &serde_json::Value) -> Document {
    match value {
        serde_json::Value::Null => Document::Null,
        serde_json::Value::Bool(b) => Document::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                Document::Number(aws_smithy_types::Number::PosInt(u))
            } else if let Some(i) = n.as_i64() {
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
            // ── Resolve model ID ────────────────────────────────────────
            let model_id = self
                .config
                .inference_profile
                .clone()
                .unwrap_or_else(|| self.config.model.model_id().to_owned());

            // ── Build the Bedrock converse_stream request ────────────────
            let messages = self.convert_messages(&request.messages)?;
            let mut req = self
                .client
                .converse_stream()
                .model_id(&model_id)
                .set_messages(Some(messages))
                .inference_config(
                    aws_sdk_bedrockruntime::types::InferenceConfiguration::builder()
                        .max_tokens(
                            i32::try_from(request.config.max_tokens.get()).unwrap_or(i32::MAX),
                        )
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
                    req = req.system(SystemContentBlock::Text(system.as_str().to_owned()));
                }
            }

            // Add tools if present
            if let Some(tool_config) = self.convert_tools(&request.tools)? {
                req = req.tool_config(tool_config);
            }

            // Add the thinking block when extended thinking is configured
            // (Claude family; default-reasoning models need no field).
            if let Some(thinking) = &self.config.thinking {
                req = req.additional_model_request_fields(thinking_request_fields(
                    thinking.budget_tokens(),
                ));
            }

            // ── Send request and classify errors ────────────────────────
            let response = req.send().await.map_err(|e| {
                let msg = e.to_string();
                if msg.contains("AccessDenied") || msg.contains("UnauthorizedException") {
                    ProviderError::Auth { provider: super::ProviderKind::Bedrock, kind: AuthErrorKind::Rejected, message: msg }
                } else if msg.contains("ThrottlingException") {
                    ProviderError::RateLimited { provider: super::ProviderKind::Bedrock, retry_after: None }
                } else if msg.contains("ModelNotFound") || msg.contains("ResourceNotFoundException")
                {
                    ProviderError::ModelNotFound { provider: super::ProviderKind::Bedrock, model: model_id.clone() }
                } else if msg.contains("inference profile") || msg.contains("InferenceProfile") {
                    ProviderError::InvalidRequest { provider: super::ProviderKind::Bedrock, message: format!(
                        "Model requires an inference profile. Set BEDROCK_INFERENCE_PROFILE env var. Error: {msg}"
                    ) }
                } else if is_context_window_message(&msg) {
                    ProviderError::ContextWindowExceeded { provider: super::ProviderKind::Bedrock, message: msg, context_window: None, tokens_used: None }
                } else if is_content_policy_message(&msg) {
                    ProviderError::ContentPolicyViolation { provider: super::ProviderKind::Bedrock, message: msg }
                } else {
                    ProviderError::InvalidRequest { provider: super::ProviderKind::Bedrock, message: msg }
                }
            })?;

            // ── Wrap the SDK stream as a cancellation-aware StreamHandle ─
            let stream = response.stream;
            let cancellation = ctx.cancellation.clone();

            // Buffer for events: parse_bedrock_event can return multiple events per
            // Bedrock stream event (e.g. ContentBlockStart + ToolUseStart), so we
            // drain the buffer before fetching the next raw event.
            //
            // Buffer bound: parse_bedrock_event returns at most 2
            // events per Bedrock stream event (the worst case is
            // ContentBlockStart for a tool_use, which emits
            // ContentBlockStart + ToolUseStart). All other event types produce
            // 0 or 1 events. The buffer is fully drained before fetching the
            // next raw event, so it never accumulates across events.
            let event_stream = futures::stream::unfold(
                (
                    stream,
                    cancellation,
                    StreamState::default(),
                    std::collections::VecDeque::new(),
                ),
                |(mut stream, cancel, mut state, mut pending)| async move {
                    // Drain pending events first
                    if let Some(event) = pending.pop_front() {
                        return Some((event, (stream, cancel, state, pending)));
                    }

                    loop {
                        if cancel.is_cancelled() {
                            return None;
                        }

                        tokio::select! {
                            biased;

                            _ = cancel.cancelled() => {
                                return None;
                            }

                            event = stream.recv() => {
                                match event {
                                    Ok(Some(event)) => {
                                        let mut events = parse_bedrock_event(event, &mut state);
                                        if events.is_empty() {
                                            // No events to emit (e.g. MessageStop); keep polling
                                            continue;
                                        }
                                        let first = events.remove(0);
                                        pending.extend(events);
                                        return Some((first, (stream, cancel, state, pending)));
                                    }
                                    Ok(None) => return None,
                                    Err(e) => {
                                        return Some((Err(StreamError::ConnectionLost { kind: StreamErrorKind::TransportError, message: e.to_string() }), (stream, cancel, state, pending)));
                                    }
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
        #[allow(
            clippy::expect_used,
            reason = "listed Bedrock model IDs are hardcoded valid constants"
        )]
        Box::pin(async move {
            // All model IDs below are hardcoded valid strings
            Ok(vec![
                ModelInfo {
                    id: ModelId::new("anthropic.claude-opus-4-5-20251101-v1:0")
                        .expect("hardcoded valid model ID"),
                    name: "Claude Opus 4.5 (Bedrock)".to_owned(),
                    context_window: Some(200_000),
                },
                ModelInfo {
                    id: ModelId::new("anthropic.claude-sonnet-4-5-20250929-v1:0")
                        .expect("hardcoded valid model ID"),
                    name: "Claude Sonnet 4.5 (Bedrock)".to_owned(),
                    context_window: Some(200_000),
                },
                ModelInfo {
                    id: ModelId::new("anthropic.claude-haiku-4-5-20251001-v1:0")
                        .expect("hardcoded valid model ID"),
                    name: "Claude Haiku 4.5 (Bedrock)".to_owned(),
                    context_window: Some(200_000),
                },
                ModelInfo {
                    id: ModelId::new("anthropic.claude-sonnet-4-20250514-v1:0")
                        .expect("hardcoded valid model ID"),
                    name: "Claude Sonnet 4 (Bedrock)".to_owned(),
                    context_window: Some(200_000),
                },
            ])
        })
    }
}

/// State for tracking stream parsing
#[derive(Default)]
struct StreamState {
    tool_call_ids: HashMap<usize, String>,
    /// Stored from MessageStop, emitted with Metadata for a single Completed event
    stop_reason: Option<StopReason>,
}

/// Check if two sets of content blocks can be merged into a single Bedrock message.
///
/// Bedrock rejects messages that mix "conversation blocks" (Text, ToolUse) with
/// "tool result blocks" (ToolResult) in the same turn.
fn blocks_compatible(existing: &[BedrockContentBlock], new: &[BedrockContentBlock]) -> bool {
    let existing_has_tool_result = existing
        .iter()
        .any(|b| matches!(b, BedrockContentBlock::ToolResult(_)));
    let new_has_tool_result = new
        .iter()
        .any(|b| matches!(b, BedrockContentBlock::ToolResult(_)));
    // Compatible if both are tool results, or neither is
    existing_has_tool_result == new_has_tool_result
}

/// Build the wire shape for one replayed `Thinking` block.
///
/// With thinking configured, the block replays as a `ReasoningContent`
/// block. Without it, the block flattens to `<thinking>` text so models
/// that reject reasoning input still receive the context. An empty
/// reasoning text returns `None`: Bedrock rejects a `ReasoningContent`
/// block with no content.
fn replay_thinking_block(text: &str, thinking_enabled: bool) -> Option<BedrockContentBlock> {
    if !thinking_enabled {
        return Some(BedrockContentBlock::Text(format!(
            "<thinking>{text}</thinking>"
        )));
    }
    if text.is_empty() {
        return None;
    }
    #[allow(
        clippy::expect_used,
        reason = "text is non-empty, the only required ReasoningTextBlock field"
    )]
    let reasoning = aws_sdk_bedrockruntime::types::ReasoningTextBlock::builder()
        .text(text)
        .build()
        .expect("text is set, the only required field");
    Some(BedrockContentBlock::ReasoningContent(
        aws_sdk_bedrockruntime::types::ReasoningContentBlock::ReasoningText(reasoning),
    ))
}

/// Build the `additionalModelRequestFields` document for a configured
/// thinking budget (Claude family).
fn thinking_request_fields(budget_tokens: u32) -> Document {
    let mut thinking = std::collections::HashMap::new();
    thinking.insert("type".to_owned(), Document::String("enabled".to_owned()));
    thinking.insert(
        "budget_tokens".to_owned(),
        Document::Number(aws_smithy_types::Number::PosInt(u64::from(budget_tokens))),
    );
    let mut fields = std::collections::HashMap::new();
    fields.insert("thinking".to_owned(), Document::Object(thinking));
    Document::Object(fields)
}

/// Check if a `BedrockContentBlock` is a `ToolResult` variant (for test assertions).
#[cfg(test)]
fn is_tool_result(block: &BedrockContentBlock) -> bool {
    matches!(block, BedrockContentBlock::ToolResult(_))
}

/// Parse a Bedrock streaming event
fn parse_bedrock_event(
    event: aws_sdk_bedrockruntime::types::ConverseStreamOutput,
    state: &mut StreamState,
) -> Vec<Result<StreamEvent, StreamError>> {
    use aws_sdk_bedrockruntime::types::ConverseStreamOutput;

    // ConverseStreamOutput is #[non_exhaustive] in the AWS SDK
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "AWS SDK enum is #[non_exhaustive]"
    )]
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
            let index = usize::try_from(block.content_block_index()).unwrap_or(0);

            if let Some(start) = block.start() {
                // ContentBlockStart is #[non_exhaustive] in the AWS SDK
                #[allow(
                    clippy::wildcard_enum_match_arm,
                    reason = "AWS SDK enum is #[non_exhaustive]"
                )]
                match start {
                    aws_sdk_bedrockruntime::types::ContentBlockStart::ToolUse(tool) => {
                        let name = match ToolName::new(tool.name()) {
                            Ok(name) => name,
                            Err(e) => {
                                return vec![Err(StreamError::Deserialize {
                                    message: format!("invalid Bedrock tool name: {e}"),
                                    raw_data: None,
                                })];
                            }
                        };

                        state
                            .tool_call_ids
                            .insert(index, tool.tool_use_id().to_owned());

                        vec![
                            Ok(StreamEvent::ContentBlockStart {
                                index,
                                block_type: ContentBlockType::ToolUse,
                            }),
                            Ok(StreamEvent::Delta(StreamDelta::ToolUseStart {
                                id: ToolCallId::new(tool.tool_use_id()),
                                name,
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
                // ContentBlockDelta is #[non_exhaustive] in the AWS SDK
                #[allow(
                    clippy::wildcard_enum_match_arm,
                    reason = "AWS SDK enum is #[non_exhaustive]"
                )]
                match d {
                    aws_sdk_bedrockruntime::types::ContentBlockDelta::Text(text) => {
                        vec![Ok(StreamEvent::Delta(StreamDelta::TextDelta {
                            text: text.clone(),
                        }))]
                    }
                    aws_sdk_bedrockruntime::types::ContentBlockDelta::ToolUse(tool) => {
                        let index = usize::try_from(delta.content_block_index()).unwrap_or(0);
                        let Some(id) = state.tool_call_ids.get(&index) else {
                            return vec![Err(StreamError::Deserialize {
                                message: format!(
                                    "Bedrock tool input delta for unknown block index {index}"
                                ),
                                raw_data: None,
                            })];
                        };
                        vec![Ok(StreamEvent::Delta(StreamDelta::ToolInputDelta {
                            id: ToolCallId::new(id),
                            partial_json: tool.input().to_owned(),
                        }))]
                    }
                    aws_sdk_bedrockruntime::types::ContentBlockDelta::ReasoningContent(
                        reasoning,
                    ) => {
                        // ReasoningContentBlockDelta is #[non_exhaustive] in the AWS SDK
                        #[allow(
                            clippy::wildcard_enum_match_arm,
                            reason = "AWS SDK enum is #[non_exhaustive]"
                        )]
                        match reasoning {
                            aws_sdk_bedrockruntime::types::ReasoningContentBlockDelta::Text(
                                text,
                            ) => vec![Ok(StreamEvent::Delta(StreamDelta::ThinkingDelta {
                                thinking: text.clone(),
                            }))],
                            aws_sdk_bedrockruntime::types::ReasoningContentBlockDelta::Signature(
                                signature,
                            ) => vec![Ok(StreamEvent::Delta(StreamDelta::SignatureDelta {
                                signature: signature.clone(),
                            }))],
                            _ => vec![],
                        }
                    }
                    _ => vec![],
                }
            } else {
                vec![]
            }
        }
        ConverseStreamOutput::ContentBlockStop(stop) => {
            let index = usize::try_from(stop.content_block_index()).unwrap_or(0);
            state.tool_call_ids.remove(&index);
            vec![Ok(StreamEvent::ContentBlockStop { index })]
        }
        ConverseStreamOutput::MessageStop(stop) => {
            // Store stop_reason in state; emit Completed only from Metadata
            // to avoid duplicate Completed events.
            // Bedrock StopReason is #[non_exhaustive] in the AWS SDK
            #[allow(
                clippy::wildcard_enum_match_arm,
                reason = "AWS SDK enum is #[non_exhaustive]"
            )]
            let reason = match stop.stop_reason() {
                aws_sdk_bedrockruntime::types::StopReason::EndTurn => Some(StopReason::EndTurn),
                aws_sdk_bedrockruntime::types::StopReason::MaxTokens => Some(StopReason::MaxTokens),
                aws_sdk_bedrockruntime::types::StopReason::ToolUse => Some(StopReason::ToolUse),
                aws_sdk_bedrockruntime::types::StopReason::StopSequence => {
                    Some(StopReason::StopSequence)
                }
                other => {
                    // Match string representation for ContentFiltered and
                    // GuardrailIntervened variants.
                    let s = other.as_str();
                    if s == "content_filtered" || s == "guardrail_intervened" {
                        Some(StopReason::ContentFilter)
                    } else {
                        Some(StopReason::EndTurn)
                    }
                }
            };
            state.stop_reason = reason;
            vec![]
        }
        ConverseStreamOutput::Metadata(meta) => {
            let usage = meta.usage().map(|u| TokenUsage {
                input_tokens: u32::try_from(u.input_tokens()).unwrap_or(0),
                output_tokens: u32::try_from(u.output_tokens()).unwrap_or(0),
            });

            vec![Ok(StreamEvent::Completed {
                metadata: CompletionMetadata {
                    model: None,
                    stop_reason: state.stop_reason.take(),
                    usage,
                },
            })]
        }
        _ => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Message, Role, ToolResultContent};
    use aws_sdk_bedrockruntime::types::{
        ContentBlockDelta as BedrockContentBlockDelta,
        ContentBlockDeltaEvent as BedrockContentBlockDeltaEvent,
        ContentBlockStart as BedrockContentBlockStart,
        ContentBlockStartEvent as BedrockContentBlockStartEvent,
        ConverseStreamOutput as BedrockConverseStreamOutput,
        ToolUseBlockDelta as BedrockToolUseBlockDelta,
        ToolUseBlockStart as BedrockToolUseBlockStart,
    };

    /// Build a simple user text message.
    fn user_msg(text: &str) -> Message {
        Message::new(Role::User, vec![ContentBlock::Text { text: text.into() }])
    }

    /// Build an assistant message with N tool_use blocks.
    fn assistant_tool_use(calls: &[(&str, &str)]) -> Message {
        let blocks = calls
            .iter()
            .map(|(id, name)| ContentBlock::ToolUse {
                id: ToolCallId::new(*id),
                name: ToolName::new(*name).unwrap(),
                input: serde_json::json!({}),
            })
            .collect();
        Message::new(Role::Assistant, blocks)
    }

    /// Build a tool-result message.
    fn tool_result(tool_use_id: &str, output: &str) -> Message {
        Message::new(
            Role::Tool,
            vec![ContentBlock::ToolResult {
                tool_use_id: ToolCallId::new(tool_use_id),
                content: ToolResultContent::Text(output.into()),
                is_error: false,
            }],
        )
    }

    fn bedrock_tool_start(index: i32, id: &str, name: &str) -> BedrockConverseStreamOutput {
        BedrockConverseStreamOutput::ContentBlockStart(
            BedrockContentBlockStartEvent::builder()
                .content_block_index(index)
                .start(BedrockContentBlockStart::ToolUse(
                    BedrockToolUseBlockStart::builder()
                        .tool_use_id(id)
                        .name(name)
                        .build()
                        .unwrap(),
                ))
                .build()
                .unwrap(),
        )
    }

    fn bedrock_tool_delta(index: i32, input: &str) -> BedrockConverseStreamOutput {
        BedrockConverseStreamOutput::ContentBlockDelta(
            BedrockContentBlockDeltaEvent::builder()
                .content_block_index(index)
                .delta(BedrockContentBlockDelta::ToolUse(
                    BedrockToolUseBlockDelta::builder()
                        .input(input)
                        .build()
                        .unwrap(),
                ))
                .build()
                .unwrap(),
        )
    }

    #[test]
    fn multi_tool_results_merged_into_single_user_message() {
        // Simulate: user asks, assistant calls 2 tools, 2 separate tool results
        let messages = vec![
            user_msg("List files in /tmp"),
            assistant_tool_use(&[
                ("tu1", "list_allowed_directories"),
                ("tu2", "list_directory"),
            ]),
            tool_result("tu1", "[/tmp]"),
            tool_result("tu2", "file1.txt\nfile2.txt"),
        ];

        let bedrock = convert_messages(&messages, false).unwrap();

        // Should be 3 messages: User, Assistant, User (merged tool results)
        assert_eq!(
            bedrock.len(),
            3,
            "expected 3 Bedrock messages, got {}",
            bedrock.len()
        );

        assert_eq!(bedrock[0].role(), &ConversationRole::User);
        assert_eq!(bedrock[1].role(), &ConversationRole::Assistant);
        assert_eq!(bedrock[2].role(), &ConversationRole::User);

        // The merged User message should contain both tool results
        let merged_content = bedrock[2].content();
        assert_eq!(
            merged_content.len(),
            2,
            "merged message should have 2 content blocks"
        );
        assert!(merged_content.iter().all(is_tool_result));
    }

    #[test]
    fn single_tool_result_not_merged_with_prior_user() {
        // Single tool call: should still produce alternating roles
        let messages = vec![
            user_msg("What time is it?"),
            assistant_tool_use(&[("tu1", "get_time")]),
            tool_result("tu1", "12:00"),
        ];

        let bedrock = convert_messages(&messages, false).unwrap();

        assert_eq!(bedrock.len(), 3);
        assert_eq!(bedrock[0].role(), &ConversationRole::User);
        assert_eq!(bedrock[1].role(), &ConversationRole::Assistant);
        assert_eq!(bedrock[2].role(), &ConversationRole::User);

        // Single tool result, no merging needed
        assert_eq!(bedrock[2].content().len(), 1);
    }

    #[test]
    fn system_messages_skipped() {
        let messages = vec![
            Message::new(
                Role::System,
                vec![ContentBlock::Text {
                    text: "You are helpful.".into(),
                }],
            ),
            user_msg("Hello"),
        ];

        let bedrock = convert_messages(&messages, false).unwrap();

        assert_eq!(bedrock.len(), 1);
        assert_eq!(bedrock[0].role(), &ConversationRole::User);
    }

    #[test]
    fn json_to_document_all_types() {
        use aws_smithy_types::Document;

        // Null
        assert!(matches!(
            json_to_document(&serde_json::json!(null)),
            Document::Null
        ));

        // Bool
        assert!(matches!(
            json_to_document(&serde_json::json!(true)),
            Document::Bool(true)
        ));
        assert!(matches!(
            json_to_document(&serde_json::json!(false)),
            Document::Bool(false)
        ));

        // Positive integer
        match json_to_document(&serde_json::json!(42)) {
            Document::Number(n) => assert!((n.to_f64_lossy() - 42.0).abs() < f64::EPSILON),
            other @ (Document::Object(_)
            | Document::Array(_)
            | Document::String(_)
            | Document::Bool(_)
            | Document::Null) => panic!("expected Number, got {other:?}"),
        }

        // Negative integer
        match json_to_document(&serde_json::json!(-7)) {
            Document::Number(n) => assert!((n.to_f64_lossy() - -7.0).abs() < f64::EPSILON),
            other @ (Document::Object(_)
            | Document::Array(_)
            | Document::String(_)
            | Document::Bool(_)
            | Document::Null) => panic!("expected Number, got {other:?}"),
        }

        // Float
        match json_to_document(&serde_json::json!(2.72)) {
            Document::Number(n) => assert!((n.to_f64_lossy() - 2.72).abs() < f64::EPSILON),
            other @ (Document::Object(_)
            | Document::Array(_)
            | Document::String(_)
            | Document::Bool(_)
            | Document::Null) => panic!("expected Number, got {other:?}"),
        }

        // String
        match json_to_document(&serde_json::json!("hello")) {
            Document::String(s) => assert_eq!(s, "hello"),
            other @ (Document::Object(_)
            | Document::Array(_)
            | Document::Number(_)
            | Document::Bool(_)
            | Document::Null) => panic!("expected String, got {other:?}"),
        }

        // Array
        match json_to_document(&serde_json::json!([1, "two", null])) {
            Document::Array(arr) => assert_eq!(arr.len(), 3),
            other @ (Document::Object(_)
            | Document::Number(_)
            | Document::String(_)
            | Document::Bool(_)
            | Document::Null) => panic!("expected Array, got {other:?}"),
        }

        // Nested object
        match json_to_document(&serde_json::json!({"key": "value", "nested": {"a": 1}})) {
            Document::Object(map) => {
                assert!(map.contains_key("key"));
                assert!(map.contains_key("nested"));
            }
            other @ (Document::Array(_)
            | Document::Number(_)
            | Document::String(_)
            | Document::Bool(_)
            | Document::Null) => panic!("expected Object, got {other:?}"),
        }
    }

    #[test]
    fn text_and_tool_result_not_merged() {
        // If a User text message is adjacent to a Tool result (shouldn't happen
        // in practice, but guard against it), they must NOT be merged because
        // Bedrock rejects messages mixing conversation and tool result blocks.
        let messages = vec![user_msg("Hello"), tool_result("tu1", "result")];
        let bedrock = convert_messages(&messages, false).unwrap();
        // Should be 2 separate messages, not merged
        assert_eq!(bedrock.len(), 2);
        assert_eq!(bedrock[0].role(), &ConversationRole::User);
        assert_eq!(bedrock[1].role(), &ConversationRole::User);
    }

    #[test]
    fn parser_emits_tool_start_and_tracks_index() {
        let mut state = StreamState::default();

        let events = parse_bedrock_event(bedrock_tool_start(2, "toolu_1", "get_time"), &mut state);

        assert_eq!(events.len(), 2);
        assert!(matches!(
            events[0].as_ref().unwrap(),
            StreamEvent::ContentBlockStart {
                index: 2,
                block_type: ContentBlockType::ToolUse,
            }
        ));
        assert!(matches!(
            events[1].as_ref().unwrap(),
            StreamEvent::Delta(StreamDelta::ToolUseStart { id, name })
                if id.as_str() == "toolu_1" && name.as_str() == "get_time"
        ));
        assert_eq!(
            state.tool_call_ids.get(&2).map(String::as_str),
            Some("toolu_1")
        );
    }

    #[test]
    fn parser_routes_tool_input_delta_by_block_index() {
        let mut state = StreamState::default();
        parse_bedrock_event(bedrock_tool_start(0, "toolu_1", "first_tool"), &mut state);
        parse_bedrock_event(bedrock_tool_start(1, "toolu_2", "second_tool"), &mut state);

        let events = parse_bedrock_event(bedrock_tool_delta(1, "{\"city\":"), &mut state);

        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0].as_ref().unwrap(),
            StreamEvent::Delta(StreamDelta::ToolInputDelta { id, partial_json })
                if id.as_str() == "toolu_2" && partial_json == "{\"city\":"
        ));
    }

    #[test]
    fn parser_rejects_tool_input_delta_for_unknown_index() {
        let mut state = StreamState::default();

        let events = parse_bedrock_event(bedrock_tool_delta(7, "{}"), &mut state);

        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0].as_ref().unwrap_err(),
            StreamError::Deserialize { message, raw_data: None }
                if message.contains("unknown block index 7")
        ));
    }

    // -------------------------------------------------------------------
    // Reasoning capture, replay shape, and the thinking request field
    // -------------------------------------------------------------------

    use aws_sdk_bedrockruntime::types::ReasoningContentBlockDelta as BedrockReasoningContentBlockDelta;

    fn bedrock_reasoning_delta(
        delta: BedrockReasoningContentBlockDelta,
    ) -> BedrockConverseStreamOutput {
        BedrockConverseStreamOutput::ContentBlockDelta(
            BedrockContentBlockDeltaEvent::builder()
                .content_block_index(0)
                .delta(BedrockContentBlockDelta::ReasoningContent(delta))
                .build()
                .unwrap(),
        )
    }

    fn reasoning_text_delta(text: &str) -> BedrockReasoningContentBlockDelta {
        BedrockReasoningContentBlockDelta::Text(text.to_owned())
    }

    fn reasoning_signature_delta(signature: &str) -> BedrockReasoningContentBlockDelta {
        BedrockReasoningContentBlockDelta::Signature(signature.to_owned())
    }

    /// A reasoning text delta surfaces as ThinkingDelta.
    #[test]
    fn reasoning_text_delta_surfaces_as_thinking_delta() {
        let mut state = StreamState::default();
        let events = parse_bedrock_event(
            bedrock_reasoning_delta(reasoning_text_delta("91 is not prime")),
            &mut state,
        );
        assert_eq!(events.len(), 1);
        match events.into_iter().next().unwrap().unwrap() {
            StreamEvent::Delta(StreamDelta::ThinkingDelta { thinking }) => {
                assert_eq!(thinking, "91 is not prime");
            }
            other => panic!("expected ThinkingDelta, got {other:?}"),
        }
    }

    /// A reasoning signature delta surfaces as SignatureDelta: the stream
    /// vocabulary carries it so downstream accumulation can retain it.
    #[test]
    fn reasoning_signature_delta_surfaces_as_signature_delta() {
        let mut state = StreamState::default();
        let events = parse_bedrock_event(
            bedrock_reasoning_delta(reasoning_signature_delta("sig-1")),
            &mut state,
        );
        assert_eq!(events.len(), 1);
        match events.into_iter().next().unwrap().unwrap() {
            StreamEvent::Delta(StreamDelta::SignatureDelta { signature }) => {
                assert_eq!(signature, "sig-1");
            }
            other => panic!("expected SignatureDelta, got {other:?}"),
        }
    }

    /// Redacted reasoning produces no event: the library has no
    /// redacted-thinking vocabulary yet.
    #[test]
    fn redacted_reasoning_delta_produces_no_event() {
        let mut state = StreamState::default();
        let events = parse_bedrock_event(
            bedrock_reasoning_delta(BedrockReasoningContentBlockDelta::RedactedContent(
                aws_smithy_types::Blob::new(b"redacted"),
            )),
            &mut state,
        );
        assert!(
            events.is_empty(),
            "redacted reasoning is skipped, got {events:?}"
        );
    }

    /// With thinking configured, a replayed Thinking block becomes a
    /// ReasoningContent block carrying the text.
    #[test]
    fn thinking_replays_as_reasoning_content_when_configured() {
        let messages = vec![Message::new(
            Role::Assistant,
            vec![ContentBlock::Thinking {
                text: "7 times 13".to_owned(),
            }],
        )];

        let bedrock = convert_messages(&messages, true).unwrap();
        let block = &bedrock[0].content()[0];
        match block {
            BedrockContentBlock::ReasoningContent(reasoning) => {
                let text = reasoning.as_reasoning_text().unwrap();
                assert_eq!(text.text(), "7 times 13");
                assert_eq!(
                    text.signature(),
                    None,
                    "signature is not stored on Thinking content blocks"
                );
            }
            other => panic!("expected ReasoningContent, got {other:?}"),
        }
    }

    /// Without thinking configured, a replayed Thinking block keeps the
    /// historical text flattening.
    #[test]
    fn thinking_replays_as_text_flattening_when_not_configured() {
        let messages = vec![Message::new(
            Role::Assistant,
            vec![ContentBlock::Thinking {
                text: "7 times 13".to_owned(),
            }],
        )];

        let bedrock = convert_messages(&messages, false).unwrap();
        let block = &bedrock[0].content()[0];
        match block {
            BedrockContentBlock::Text(text) => {
                assert_eq!(text, "<thinking>7 times 13</thinking>");
            }
            other => panic!("expected flattened text, got {other:?}"),
        }
    }

    /// The thinking request document carries the enabled type and budget.
    #[test]
    fn thinking_request_fields_shape() {
        let Document::Object(fields) = thinking_request_fields(2048) else {
            panic!("expected an object document");
        };
        let Document::Object(thinking) = fields.get("thinking").unwrap() else {
            panic!("expected a thinking object");
        };
        assert_eq!(
            thinking.get("type"),
            Some(&Document::String("enabled".to_owned()))
        );
        assert_eq!(
            thinking.get("budget_tokens"),
            Some(&Document::Number(aws_smithy_types::Number::PosInt(2048)))
        );
    }
}

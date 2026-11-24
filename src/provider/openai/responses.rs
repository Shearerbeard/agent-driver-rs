//! Responses API surface of the OpenAI provider.
//!
//! Runs the Responses API statelessly (`store: false`): the driver's
//! `Session` owns the conversation history and replays it as input
//! items each turn. Reasoning summaries stream as `ThinkingDelta`;
//! function calls map to the driver's tool-use vocabulary.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use async_openai::types::responses::{
    CreateResponseArgs, EasyInputContent, EasyInputMessage, FunctionCallOutput,
    FunctionCallOutputItemParam, FunctionTool, FunctionToolCall, IncludeEnum, InputItem,
    InputParam, Item, MessageType, OutputItem, OutputMessageContent, Reasoning,
    ReasoningEffort as SdkReasoningEffort, ReasoningItem, ReasoningItemContent,
    ReasoningSummary as SdkReasoningSummary, ResponseStreamEvent, ResponseUsage, Role, Status,
    SummaryPart, Tool,
};

use crate::config::{ReasoningEffort, ReasoningSummary};
use crate::error::{ProviderError, StreamError, StreamErrorKind, is_content_policy_message};
use crate::provider::{CompletionRequest, ProviderContext};
use crate::streaming::{
    CompletionMetadata, ContentBlockType, StopReason, StreamDelta, StreamEvent, StreamHandle,
    TokenUsage,
};
use crate::types::{ContentBlock, ModelId, Role as DriverRole, ToolCallId, ToolName};

use super::{OpenAiProvider, map_request_error};

#[allow(
    clippy::multiple_inherent_impl,
    reason = "the OpenAI impl is split per API surface (mod, chat, responses); the lint targets accidental fragmentation"
)]
impl OpenAiProvider {
    /// Run one Responses API streaming request (the `Provider` trait
    /// dispatches here when the API surface is responses).
    pub(super) fn complete_stream_responses(
        &self,
        request: CompletionRequest,
        ctx: ProviderContext,
    ) -> Pin<Box<dyn Future<Output = Result<StreamHandle, ProviderError>> + Send + '_>> {
        Box::pin(async move {
            let profile = &self.profile;

            // The provider pin is authoritative; a request naming a
            // different model must fail rather than be silently ignored.
            if request.model.as_str() != profile.model_id() {
                return Err(ProviderError::InvalidRequest {
                    provider: crate::provider::ProviderKind::OpenAi,
                    message: format!(
                        "request model {:?} does not match configured model {:?}",
                        request.model.as_str(),
                        profile.model_id()
                    ),
                });
            }

            // Streaming capability is per-surface: o3/o3-mini stream on
            // Responses even though the Chat Completions restriction
            // remains.
            if !profile.supports_streaming() {
                return Err(ProviderError::StreamingNotSupported {
                    provider: crate::provider::ProviderKind::OpenAi,
                    model: profile.model_id().to_owned(),
                });
            }

            let model = profile.model_id();
            let openai_request = self.build_responses_request(&request)?;

            let stream = self
                .client
                .responses()
                .create_stream(openai_request)
                .await
                .map_err(|e| map_request_error(e.to_string(), model))?;

            let event_stream = super::super::stream_adapter::buffered_sdk_stream(
                stream,
                ctx.cancellation.clone(),
                ResponsesStreamState::default(),
                |item, state| match item {
                    Ok(event) => parse_responses_event(event, state),
                    Err(e) => {
                        let mut events = Vec::new();
                        fail_stream(
                            StreamError::ConnectionLost {
                                kind: StreamErrorKind::TransportError,
                                message: e.to_string(),
                            },
                            state,
                            &mut events,
                        );
                        events
                    }
                },
                // A successful `Completed` is emitted only when the
                // parser actually saw `response.completed` or a
                // supported `response.incomplete`. A failed response,
                // mid-stream error, SDK transport error, or bare EOF
                // must not be followed by a successful terminal event,
                // so both collectors (which accept bare EOF) see an
                // error instead. Failure is reported exactly once.
                responses_terminal_event,
            );

            Ok(StreamHandle::new(
                Box::pin(event_stream),
                ctx.cancellation,
                ctx.correlation_id,
            ))
        })
    }

    /// Build the Responses API request body (separate so tests can
    /// pin the wire shape).
    fn build_responses_request(
        &self,
        request: &CompletionRequest,
    ) -> Result<async_openai::types::responses::CreateResponse, ProviderError> {
        if !request.config.stop_sequences.is_empty() {
            return Err(ProviderError::InvalidRequest {
                provider: crate::provider::ProviderKind::OpenAi,
                message: "Responses does not support stop sequences".into(),
            });
        }
        let mut builder = CreateResponseArgs::default();
        builder
            .model(self.profile.model_id())
            .input(InputParam::Items(build_input_items(request)))
            .max_output_tokens(request.config.max_tokens.get())
            // Stateless by design: the Session owns history, so the
            // response must not be stored server-side.
            .store(false)
            .stream(true);

        if let Some(system) = system_text(request) {
            builder.instructions(system);
        }

        if let Some(temp) = request.config.temperature {
            if !self.profile.supports_temperature() {
                return Err(ProviderError::InvalidRequest {
                    provider: crate::provider::ProviderKind::OpenAi,
                    message: format!(
                        "model {:?} on the Responses surface does not support temperature",
                        self.profile.model_id()
                    ),
                });
            }
            builder.temperature(temp.get());
        }

        if !request.tools.is_empty() {
            builder.tools(convert_tools(&request.tools));
        }

        if let Some(reasoning) = self.profile.reasoning() {
            builder.reasoning(to_sdk_reasoning(reasoning));
            // Ask for the encrypted reasoning payload so multi-turn
            // replay works statelessly (store: false).
            builder.include(vec![IncludeEnum::ReasoningEncryptedContent]);
        }

        builder.build().map_err(|e| ProviderError::InvalidRequest {
            provider: crate::provider::ProviderKind::OpenAi,
            message: format!("Failed to build responses request: {e}"),
        })
    }
}

/// System prompt text: `request.system` first, else the first
/// `Role::System` message (mirrors the chat-completions fallback).
fn system_text(request: &CompletionRequest) -> Option<String> {
    request
        .system
        .as_ref()
        .filter(|s| !s.as_str().is_empty())
        .map(|s| s.as_str().to_owned())
        .or_else(|| {
            request
                .messages
                .iter()
                .find(|m| m.role == DriverRole::System)
                .map(|m| {
                    m.content
                        .iter()
                        .filter_map(ContentBlock::as_text)
                        .collect::<Vec<_>>()
                        .join("\n")
                })
                .filter(|text| !text.is_empty())
        })
}

/// Convert the driver's message history to Responses input items.
///
/// Assistant history replays as an easy assistant message (text only —
/// thinking summaries are not replayed; the reasoning items that carry
/// them round-trip natively) plus one function-call item per
/// tool use, and tool results replay as function-call outputs.
fn build_input_items(request: &CompletionRequest) -> Vec<InputItem> {
    let mut items = Vec::new();

    for msg in request
        .messages
        .iter()
        .filter(|m| m.role != DriverRole::System)
    {
        match msg.role {
            DriverRole::System => unreachable!("filtered above"),
            DriverRole::User => {
                let text = msg
                    .content
                    .iter()
                    .filter_map(ContentBlock::as_text)
                    .collect::<Vec<_>>()
                    .join("\n");
                if !text.is_empty() {
                    items.push(InputItem::EasyMessage(easy_message(Role::User, text)));
                }
            }
            DriverRole::Assistant => {
                // Single ordered pass: the aggregated assistant text
                // flushes as an easy message before the first non-text
                // item that follows it, so items replay in content
                // order (reasoning items precede the text they
                // produced, matching how the API emitted them).
                let mut text = String::new();
                let flush_text = |items: &mut Vec<InputItem>, text: &mut String| {
                    if !text.is_empty() {
                        items.push(InputItem::EasyMessage(easy_message(
                            Role::Assistant,
                            std::mem::take(text),
                        )));
                    }
                };
                for block in &msg.content {
                    match block {
                        ContentBlock::Text { text: block_text } => {
                            if !text.is_empty() {
                                text.push('\n');
                            }
                            text.push_str(block_text);
                        }
                        // Replayable reasoning state: round-trips as the
                        // reasoning item it came from (summaries are not
                        // resent; the encrypted payload carries them).
                        ContentBlock::EncryptedReasoning { id, encrypted } => {
                            flush_text(&mut items, &mut text);
                            items.push(InputItem::Item(Item::Reasoning(ReasoningItem {
                                id: Some(id.clone()),
                                summary: Vec::new(),
                                content: None,
                                encrypted_content: Some(encrypted.clone()),
                                status: None,
                            })));
                        }
                        ContentBlock::ToolUse { id, name, input } => {
                            flush_text(&mut items, &mut text);
                            items.push(InputItem::Item(Item::FunctionCall(FunctionToolCall {
                                arguments: serde_json::to_string(input).unwrap_or_default(),
                                call_id: id.as_str().to_owned(),
                                name: name.as_str().to_owned(),
                                namespace: None,
                                id: None,
                                status: None,
                            })));
                        }
                        ContentBlock::Thinking { .. } | ContentBlock::ToolResult { .. } => {}
                    }
                }
                flush_text(&mut items, &mut text);
            }
            DriverRole::Tool => {
                for block in &msg.content {
                    if let ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        ..
                    } = block
                    {
                        let output = match content {
                            crate::types::ToolResultContent::Text(s) => s.clone(),
                        };
                        items.push(InputItem::Item(Item::FunctionCallOutput(
                            FunctionCallOutputItemParam {
                                call_id: tool_use_id.as_str().to_owned(),
                                output: FunctionCallOutput::Text(output),
                                id: None,
                                status: None,
                            },
                        )));
                    }
                }
            }
        }
    }

    items
}

fn easy_message(role: Role, text: String) -> EasyInputMessage {
    EasyInputMessage {
        role,
        content: EasyInputContent::Text(text),
        phase: None,
        r#type: MessageType::default(),
    }
}

/// Convert the driver's tools to flattened Responses function tools.
fn convert_tools(tools: &[crate::tool::ToolDefinition]) -> Vec<Tool> {
    tools
        .iter()
        .map(|t| {
            Tool::Function(FunctionTool {
                name: t.name.as_str().to_owned(),
                description: Some(t.description.clone()),
                parameters: Some(t.input_schema.to_value()),
                strict: None,
                defer_loading: None,
            })
        })
        .collect()
}

/// Map the driver's reasoning config onto the Responses `reasoning`
/// parameter (the two vocabularies are one-to-one).
fn to_sdk_reasoning(config: &crate::config::ReasoningConfig) -> Reasoning {
    Reasoning {
        effort: Some(match config.effort {
            ReasoningEffort::None => SdkReasoningEffort::None,
            ReasoningEffort::Minimal => SdkReasoningEffort::Minimal,
            ReasoningEffort::Low => SdkReasoningEffort::Low,
            ReasoningEffort::Medium => SdkReasoningEffort::Medium,
            ReasoningEffort::High => SdkReasoningEffort::High,
            ReasoningEffort::XHigh => SdkReasoningEffort::Xhigh,
        }),
        summary: Some(match config.summary {
            ReasoningSummary::Auto => SdkReasoningSummary::Auto,
            ReasoningSummary::Concise => SdkReasoningSummary::Concise,
            ReasoningSummary::Detailed => SdkReasoningSummary::Detailed,
        }),
    }
}

/// Reported reason why a Responses stream ended incomplete.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IncompleteReason {
    /// `max_output_tokens` bound was reached.
    MaxTokens,
    /// `content_filter` stopped generation.
    ContentFilter,
}

impl IncompleteReason {
    /// Parse the provider's incomplete-reason string.
    ///
    /// Returns `None` for unknown or unsupported reasons so the caller
    /// can turn the stream into a typed error rather than defaulting to
    /// token exhaustion.
    fn from_reason(reason: &str) -> Option<Self> {
        match reason {
            "max_output_tokens" => Some(Self::MaxTokens),
            "content_filter" => Some(Self::ContentFilter),
            _ => None,
        }
    }
}

/// Terminal status of a Responses stream.
///
/// All terminal variants are absorbing: once the parser leaves
/// [`Active`], no later event may change the status. This includes
/// successful terminals (`SawCompleted`, `SawIncomplete`) and
/// [`Cancelled`]; a failure after any terminal is dropped rather than
/// overwriting it.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum TerminalStatus {
    /// No terminal event seen yet.
    #[default]
    Active,
    /// `response.completed` was observed.
    SawCompleted,
    /// A supported `response.incomplete` was observed.
    SawIncomplete,
    /// A failure was recorded; a later success cannot override it.
    Failed,
    /// The response was cancelled by the provider.
    Cancelled,
}

impl TerminalStatus {
    /// Record a successful terminal. Only transitions from [`Active`].
    fn succeed(&mut self, status: Self) {
        if *self == Self::Active {
            *self = status;
        }
    }

    /// Record a failure. Only transitions from [`Active`].
    fn fail(&mut self) {
        if *self == Self::Active {
            *self = Self::Failed;
        }
    }
}

/// State for tracking Responses stream parsing.
#[derive(Default)]
struct ResponsesStreamState {
    model: Option<ModelId>,
    stop_reason: Option<StopReason>,
    usage: Option<TokenUsage>,
    /// call_id per function-call item, keyed by the item id the
    /// delta events reference.
    tool_items: HashMap<String, ToolCallId>,
    started: bool,
    /// Whether the parser saw a successful terminal (`completed` or a
    /// supported `incomplete`) or recorded a failure.
    terminal: TerminalStatus,
    /// Whether a terminal stream error was already pushed as an event.
    /// When set, the EOF callback stays silent so a failure or SDK
    /// transport error yields exactly one error on the wire.
    terminal_error_surfaced: bool,
    /// Whether any live content delta was forwarded. An empty terminal
    /// `response.output` is only valid when no content appeared earlier.
    saw_live_content: bool,
}

impl ResponsesStreamState {
    /// Mark that live content has appeared on the stream.
    fn record_live_content(&mut self) {
        self.saw_live_content = true;
    }
}

/// Parse one Responses stream event into driver stream events.
///
/// Every terminal is absorbing: once the parser leaves
/// [`TerminalStatus::Active`] (a successful `SawCompleted`/`SawIncomplete`,
/// a `Cancelled`, or a `Failed`), all later events are dropped. A failure
/// cannot be followed by content or a successful terminal, and a success
/// or cancellation cannot be overwritten by a later error.
fn parse_responses_event(
    event: ResponseStreamEvent,
    state: &mut ResponsesStreamState,
) -> Vec<Result<StreamEvent, StreamError>> {
    if state.terminal != TerminalStatus::Active {
        return Vec::new();
    }

    let mut events = Vec::new();

    match event {
        ResponseStreamEvent::ResponseCreated(created) => {
            state.model = ModelId::new(&created.response.model).ok();
            if !state.started {
                state.started = true;
                events.push(Ok(StreamEvent::Started {
                    metadata: CompletionMetadata {
                        model: state.model.clone(),
                        stop_reason: None,
                        usage: None,
                    },
                }));
            }
        }
        ResponseStreamEvent::ResponseOutputItemAdded(added) => match &added.item {
            OutputItem::Message(_) => {
                events.push(Ok(StreamEvent::ContentBlockStart {
                    index: added.output_index as usize,
                    block_type: ContentBlockType::Text,
                }));
            }
            OutputItem::FunctionCall(call) => {
                let name = match ToolName::new(&call.name) {
                    Ok(name) => name,
                    Err(e) => {
                        fail_stream(
                            StreamError::Deserialize {
                                message: format!("invalid OpenAI tool name: {e}"),
                                raw_data: None,
                            },
                            state,
                            &mut events,
                        );
                        return events;
                    }
                };
                let id = ToolCallId::new(&call.call_id);
                // Argument deltas reference the item id; without it the
                // stream cannot be correlated, so fail loud instead of
                // guessing at the call-id domain.
                let Some(item_key) = call.id.clone() else {
                    fail_stream(
                        StreamError::Deserialize {
                            message: format!(
                                "function call item for {} carries no item id",
                                call.call_id
                            ),
                            raw_data: None,
                        },
                        state,
                        &mut events,
                    );
                    return events;
                };
                state.tool_items.insert(item_key, id.clone());
                events.push(Ok(StreamEvent::ContentBlockStart {
                    index: added.output_index as usize,
                    block_type: ContentBlockType::ToolUse,
                }));
                state.record_live_content();
                events.push(Ok(StreamEvent::Delta(StreamDelta::ToolUseStart {
                    id,
                    name,
                })));
            }
            // Reasoning items stream their summaries through the
            // dedicated delta events. Every other item type is a hosted
            // tool the driver cannot replay, so fail loud rather than
            // silently dropping content the terminal array will carry.
            OutputItem::Reasoning(_) => {}
            OutputItem::FileSearchCall(_)
            | OutputItem::FunctionCallOutput(_)
            | OutputItem::WebSearchCall(_)
            | OutputItem::ComputerCall(_)
            | OutputItem::ComputerCallOutput(_)
            | OutputItem::Compaction(_)
            | OutputItem::ImageGenerationCall(_)
            | OutputItem::CodeInterpreterCall(_)
            | OutputItem::LocalShellCall(_)
            | OutputItem::ShellCall(_)
            | OutputItem::ShellCallOutput(_)
            | OutputItem::ApplyPatchCall(_)
            | OutputItem::ApplyPatchCallOutput(_)
            | OutputItem::McpCall(_)
            | OutputItem::McpListTools(_)
            | OutputItem::McpApprovalRequest(_)
            | OutputItem::CustomToolCall(_)
            | OutputItem::CustomToolCallOutput(_)
            | OutputItem::ToolSearchCall(_)
            | OutputItem::ToolSearchOutput(_) => {
                fail_stream(
                    StreamError::Deserialize {
                        message: format!(
                            "unsupported Responses output item: {}",
                            output_item_kind(&added.item)
                        ),
                        raw_data: None,
                    },
                    state,
                    &mut events,
                );
                return events;
            }
        },
        ResponseStreamEvent::ResponseOutputTextDelta(delta) => {
            if !delta.delta.is_empty() {
                state.record_live_content();
                events.push(Ok(StreamEvent::Delta(StreamDelta::TextDelta {
                    text: delta.delta,
                })));
            }
        }
        // Summary deltas (summarized reasoning) and reasoning-text
        // deltas (raw chain of thought, newer reasoning models) both
        // map to the driver's ThinkingDelta channel.
        ResponseStreamEvent::ResponseReasoningSummaryTextDelta(delta) => {
            if !delta.delta.is_empty() {
                state.record_live_content();
            }
            push_thinking_delta(delta.delta, &mut events);
        }
        ResponseStreamEvent::ResponseReasoningTextDelta(delta) => {
            if !delta.delta.is_empty() {
                state.record_live_content();
            }
            push_thinking_delta(delta.delta, &mut events);
        }
        ResponseStreamEvent::ResponseFunctionCallArgumentsDelta(delta) => {
            if !delta.delta.is_empty() {
                state.record_live_content();
                let Some(id) = state.tool_items.get(&delta.item_id) else {
                    fail_stream(
                        StreamError::Deserialize {
                            message: format!(
                                "function-call argument delta references unknown item_id {}",
                                delta.item_id
                            ),
                            raw_data: None,
                        },
                        state,
                        &mut events,
                    );
                    return events;
                };
                events.push(Ok(StreamEvent::Delta(StreamDelta::ToolInputDelta {
                    id: ToolCallId::new(id.as_str()),
                    partial_json: delta.delta,
                })));
            }
        }
        // Refusals are answer-channel content: surface them as text so
        // a safety refusal is visible instead of an empty turn.
        ResponseStreamEvent::ResponseRefusalDelta(delta) => {
            if !delta.delta.is_empty() {
                state.record_live_content();
                events.push(Ok(StreamEvent::Delta(StreamDelta::TextDelta {
                    text: delta.delta,
                })));
            }
        }
        // The done event duplicates the streamed refusal text.
        ResponseStreamEvent::ResponseRefusalDone(_) => {}
        ResponseStreamEvent::ResponseCompleted(completed) => {
            state.usage = completed.response.usage.map(to_token_usage);
            match &completed.response.status {
                // A cancelled completion is a terminal cancellation, not a
                // successful turn, and emits no terminal blocks.
                Status::Cancelled => state.terminal.succeed(TerminalStatus::Cancelled),
                // `response.completed` must carry `completed`. Any other
                // status is a protocol contradiction and fails loud before
                // any terminal block is emitted, so a failed or unfinished
                // response can never be reported as a successful turn.
                Status::Completed => {
                    if completed.response.output.is_empty() && state.saw_live_content {
                        fail_stream(
                            StreamError::Deserialize {
                                message:
                                    "response.completed reported empty output after live content deltas"
                                        .to_owned(),
                                raw_data: None,
                            },
                            state,
                            &mut events,
                        );
                    } else {
                        state.stop_reason = Some(completed_stop_reason(&completed.response.output));
                        emit_terminal_blocks(&completed.response.output, state, &mut events);
                        state.terminal.succeed(TerminalStatus::SawCompleted);
                    }
                }
                Status::Failed | Status::InProgress | Status::Queued | Status::Incomplete => {
                    fail_stream(
                        StreamError::Deserialize {
                            message: format!(
                                "response.completed carried unexpected status {:?}",
                                completed.response.status
                            ),
                            raw_data: None,
                        },
                        state,
                        &mut events,
                    );
                }
            }
        }
        ResponseStreamEvent::ResponseIncomplete(incomplete) => {
            state.usage = incomplete.response.usage.map(to_token_usage);
            if incomplete.response.status != Status::Incomplete {
                fail_stream(
                    StreamError::Deserialize {
                        message: format!(
                            "response.incomplete carried unexpected status {:?}",
                            incomplete.response.status
                        ),
                        raw_data: None,
                    },
                    state,
                    &mut events,
                );
            } else if incomplete.response.output.is_empty() && state.saw_live_content {
                fail_stream(
                    StreamError::Deserialize {
                        message:
                            "response.incomplete reported empty output after live content deltas"
                                .to_owned(),
                        raw_data: None,
                    },
                    state,
                    &mut events,
                );
            } else {
                let raw_reason = incomplete
                    .response
                    .incomplete_details
                    .as_ref()
                    .map(|d| d.reason.as_str());
                match raw_reason.and_then(IncompleteReason::from_reason) {
                    Some(r) => {
                        state.stop_reason = Some(match r {
                            IncompleteReason::MaxTokens => StopReason::MaxTokens,
                            IncompleteReason::ContentFilter => StopReason::ContentFilter,
                        });
                        emit_terminal_blocks(&incomplete.response.output, state, &mut events);
                        state.terminal.succeed(TerminalStatus::SawIncomplete);
                    }
                    None => fail_stream(
                        StreamError::Deserialize {
                            message: format!(
                                "unsupported or missing response.incomplete reason: {raw_reason:?}"
                            ),
                            raw_data: None,
                        },
                        state,
                        &mut events,
                    ),
                }
            }
        }
        ResponseStreamEvent::ResponseFailed(failed) => {
            let message = failed
                .response
                .error
                .as_ref()
                .map(|e| e.message.clone())
                .unwrap_or_else(|| "responses stream failed".to_owned());
            record_terminal_failure(message, state, &mut events);
        }
        // A completed reasoning item carries the encrypted payload the
        // request asked for; without id or encrypted content (include
        // not requested) there is nothing replayable to emit.
        ResponseStreamEvent::ResponseOutputItemDone(done) => match &done.item {
            OutputItem::Reasoning(item) => {
                if let (Some(id), Some(encrypted)) =
                    (item.id.clone(), item.encrypted_content.clone())
                {
                    state.record_live_content();
                    events.push(Ok(StreamEvent::Delta(StreamDelta::ReasoningItemDelta {
                        id,
                        encrypted,
                    })));
                }
            }
            // Text and tool items opened a `ContentBlockStart`; close it
            // at the matching item boundary so the live lifecycle is
            // balanced even before the terminal complete blocks arrive.
            OutputItem::Message(_) | OutputItem::FunctionCall(_) => {
                events.push(Ok(StreamEvent::ContentBlockStop {
                    index: done.output_index as usize,
                }));
            }
            // Hosted items fail at `response.output_item.added`, so their
            // done events are unreachable; the arm keeps the match
            // exhaustive without a wildcard.
            OutputItem::FileSearchCall(_)
            | OutputItem::FunctionCallOutput(_)
            | OutputItem::WebSearchCall(_)
            | OutputItem::ComputerCall(_)
            | OutputItem::ComputerCallOutput(_)
            | OutputItem::Compaction(_)
            | OutputItem::ImageGenerationCall(_)
            | OutputItem::CodeInterpreterCall(_)
            | OutputItem::LocalShellCall(_)
            | OutputItem::ShellCall(_)
            | OutputItem::ShellCallOutput(_)
            | OutputItem::ApplyPatchCall(_)
            | OutputItem::ApplyPatchCallOutput(_)
            | OutputItem::McpCall(_)
            | OutputItem::McpListTools(_)
            | OutputItem::McpApprovalRequest(_)
            | OutputItem::CustomToolCall(_)
            | OutputItem::CustomToolCallOutput(_)
            | OutputItem::ToolSearchCall(_)
            | OutputItem::ToolSearchOutput(_) => {}
        },
        // Reasoning summary parts bracket the live thinking deltas, so
        // the direct stream sees a balanced thinking lifecycle.
        ResponseStreamEvent::ResponseReasoningSummaryPartAdded(added) => {
            events.push(Ok(StreamEvent::ContentBlockStart {
                index: added.output_index as usize,
                block_type: ContentBlockType::Thinking,
            }));
        }
        ResponseStreamEvent::ResponseReasoningSummaryPartDone(done) => {
            events.push(Ok(StreamEvent::ContentBlockStop {
                index: done.output_index as usize,
            }));
        }
        // Terminal bookkeeping events without driver-facing payloads.
        ResponseStreamEvent::ResponseContentPartAdded(_)
        | ResponseStreamEvent::ResponseContentPartDone(_)
        | ResponseStreamEvent::ResponseOutputTextDone(_)
        | ResponseStreamEvent::ResponseFunctionCallArgumentsDone(_)
        | ResponseStreamEvent::ResponseReasoningSummaryTextDone(_)
        | ResponseStreamEvent::ResponseReasoningTextDone(_) => {}
        // Lifecycle noise that carries no content for this surface.
        ResponseStreamEvent::ResponseInProgress(_) | ResponseStreamEvent::ResponseQueued(_) => {}
        // Hosted tool-call progress events (web/file search, code
        // interpreter, MCP, image gen, computer use, shell, custom tools)
        // carry no payload for this surface. The hosted output items
        // themselves are rejected loudly at item add and in the terminal
        // walk, so these progress events are inert here.
        ResponseStreamEvent::ResponseFileSearchCallInProgress(_)
        | ResponseStreamEvent::ResponseFileSearchCallSearching(_)
        | ResponseStreamEvent::ResponseFileSearchCallCompleted(_)
        | ResponseStreamEvent::ResponseWebSearchCallInProgress(_)
        | ResponseStreamEvent::ResponseWebSearchCallSearching(_)
        | ResponseStreamEvent::ResponseWebSearchCallCompleted(_)
        | ResponseStreamEvent::ResponseImageGenerationCallCompleted(_)
        | ResponseStreamEvent::ResponseImageGenerationCallGenerating(_)
        | ResponseStreamEvent::ResponseImageGenerationCallInProgress(_)
        | ResponseStreamEvent::ResponseImageGenerationCallPartialImage(_)
        | ResponseStreamEvent::ResponseMCPCallArgumentsDelta(_)
        | ResponseStreamEvent::ResponseMCPCallArgumentsDone(_)
        | ResponseStreamEvent::ResponseMCPCallCompleted(_)
        | ResponseStreamEvent::ResponseMCPCallFailed(_)
        | ResponseStreamEvent::ResponseMCPCallInProgress(_)
        | ResponseStreamEvent::ResponseMCPListToolsCompleted(_)
        | ResponseStreamEvent::ResponseMCPListToolsFailed(_)
        | ResponseStreamEvent::ResponseMCPListToolsInProgress(_)
        | ResponseStreamEvent::ResponseCodeInterpreterCallInProgress(_)
        | ResponseStreamEvent::ResponseCodeInterpreterCallInterpreting(_)
        | ResponseStreamEvent::ResponseCodeInterpreterCallCompleted(_)
        | ResponseStreamEvent::ResponseCodeInterpreterCallCodeDelta(_)
        | ResponseStreamEvent::ResponseCodeInterpreterCallCodeDone(_)
        | ResponseStreamEvent::ResponseOutputTextAnnotationAdded(_)
        | ResponseStreamEvent::ResponseCustomToolCallInputDelta(_)
        | ResponseStreamEvent::ResponseCustomToolCallInputDone(_) => {}
        // A mid-stream error event fails the stream loudly.
        ResponseStreamEvent::ResponseError(error) => {
            record_terminal_failure(error.message, state, &mut events);
        }
    }

    events
}

/// Push a non-empty reasoning delta onto the thinking channel.
fn push_thinking_delta(thinking: String, events: &mut Vec<Result<StreamEvent, StreamError>>) {
    if !thinking.is_empty() {
        events.push(Ok(StreamEvent::Delta(StreamDelta::ThinkingDelta {
            thinking,
        })));
    }
}

/// Provider-reported type tag for an output item, used in error messages.
const fn output_item_kind(item: &OutputItem) -> &'static str {
    match item {
        OutputItem::Message(_) => "message",
        OutputItem::FunctionCall(_) => "function_call",
        OutputItem::Reasoning(_) => "reasoning",
        OutputItem::FileSearchCall(_) => "file_search_call",
        OutputItem::FunctionCallOutput(_) => "function_call_output",
        OutputItem::WebSearchCall(_) => "web_search_call",
        OutputItem::ComputerCall(_) => "computer_call",
        OutputItem::ComputerCallOutput(_) => "computer_call_output",
        OutputItem::Compaction(_) => "compaction",
        OutputItem::ImageGenerationCall(_) => "image_generation_call",
        OutputItem::CodeInterpreterCall(_) => "code_interpreter_call",
        OutputItem::LocalShellCall(_) => "local_shell_call",
        OutputItem::ShellCall(_) => "shell_call",
        OutputItem::ShellCallOutput(_) => "shell_call_output",
        OutputItem::ApplyPatchCall(_) => "apply_patch_call",
        OutputItem::ApplyPatchCallOutput(_) => "apply_patch_call_output",
        OutputItem::McpCall(_) => "mcp_call",
        OutputItem::McpListTools(_) => "mcp_list_tools",
        OutputItem::McpApprovalRequest(_) => "mcp_approval_request",
        OutputItem::CustomToolCall(_) => "custom_tool_call",
        OutputItem::CustomToolCallOutput(_) => "custom_tool_call_output",
        OutputItem::ToolSearchCall(_) => "tool_search_call",
        OutputItem::ToolSearchOutput(_) => "tool_search_output",
    }
}

/// Convert one terminal `response.output` item into the ordered complete
/// blocks the driver can represent.
///
/// Summary and raw reasoning text both map to thinking blocks, mirroring
/// the live `ThinkingDelta` channel; the encrypted payload is the
/// replayable reasoning block. Returns an error message for unsupported
/// hosted items or malformed function-call arguments so the caller fails
/// the stream instead of dropping or empty-filling content.
fn output_item_blocks(item: &OutputItem) -> Result<Vec<ContentBlock>, String> {
    let mut blocks = Vec::new();
    match item {
        OutputItem::Message(message) => {
            for part in &message.content {
                match part {
                    OutputMessageContent::OutputText(text) if !text.text.is_empty() => {
                        blocks.push(ContentBlock::Text {
                            text: text.text.clone(),
                        });
                    }
                    OutputMessageContent::Refusal(refusal) if !refusal.refusal.is_empty() => {
                        blocks.push(ContentBlock::Text {
                            text: refusal.refusal.clone(),
                        });
                    }
                    OutputMessageContent::OutputText(_) | OutputMessageContent::Refusal(_) => {}
                }
            }
        }
        OutputItem::Reasoning(reasoning) => {
            for part in &reasoning.summary {
                let SummaryPart::SummaryText(text) = part;
                if !text.text.is_empty() {
                    blocks.push(ContentBlock::Thinking {
                        text: text.text.clone(),
                    });
                }
            }
            if let Some(content) = &reasoning.content {
                for part in content {
                    let ReasoningItemContent::ReasoningText(text) = part;
                    if !text.text.is_empty() {
                        blocks.push(ContentBlock::Thinking {
                            text: text.text.clone(),
                        });
                    }
                }
            }
            match (&reasoning.id, &reasoning.encrypted_content) {
                // Replayable reasoning must carry both a usable id and the
                // encrypted payload.
                (Some(id), Some(encrypted)) if !id.is_empty() && !encrypted.is_empty() => {
                    blocks.push(ContentBlock::EncryptedReasoning {
                        id: id.clone(),
                        encrypted: encrypted.clone(),
                    });
                }
                // Encrypted content without a usable id would be dropped,
                // silently breaking multi-turn replay; fail loud instead.
                (_, Some(encrypted)) if !encrypted.is_empty() => {
                    return Err(
                        "reasoning item carried encrypted content without a usable id".to_owned(),
                    );
                }
                // No encrypted payload present (include not requested):
                // absence is allowed.
                _ => {}
            }
        }
        OutputItem::FunctionCall(call) => {
            let name = ToolName::new(&call.name)
                .map_err(|e| format!("invalid OpenAI tool name {:?}: {e}", call.name))?;
            let input: serde_json::Value = serde_json::from_str(&call.arguments).map_err(|e| {
                format!(
                    "function call {} carried invalid JSON arguments: {e}",
                    call.call_id
                )
            })?;
            blocks.push(ContentBlock::ToolUse {
                id: ToolCallId::new(&call.call_id),
                name,
                input,
            });
        }
        OutputItem::FileSearchCall(_)
        | OutputItem::FunctionCallOutput(_)
        | OutputItem::WebSearchCall(_)
        | OutputItem::ComputerCall(_)
        | OutputItem::ComputerCallOutput(_)
        | OutputItem::Compaction(_)
        | OutputItem::ImageGenerationCall(_)
        | OutputItem::CodeInterpreterCall(_)
        | OutputItem::LocalShellCall(_)
        | OutputItem::ShellCall(_)
        | OutputItem::ShellCallOutput(_)
        | OutputItem::ApplyPatchCall(_)
        | OutputItem::ApplyPatchCallOutput(_)
        | OutputItem::McpCall(_)
        | OutputItem::McpListTools(_)
        | OutputItem::McpApprovalRequest(_)
        | OutputItem::CustomToolCall(_)
        | OutputItem::CustomToolCallOutput(_)
        | OutputItem::ToolSearchCall(_)
        | OutputItem::ToolSearchOutput(_) => {
            return Err(format!(
                "unsupported Responses output item: {}",
                output_item_kind(item)
            ));
        }
    }
    Ok(blocks)
}

/// Validate and flatten a whole terminal `response.output` array into
/// complete blocks in provider order.
///
/// Validation is atomic: an unsupported or malformed item fails the whole
/// walk before any `BlockComplete` is emitted, so a partial authoritative
/// array never reaches a collector.
fn terminal_blocks(output: &[OutputItem]) -> Result<Vec<ContentBlock>, String> {
    let mut blocks = Vec::new();
    for item in output {
        blocks.extend(output_item_blocks(item)?);
    }
    Ok(blocks)
}

/// Emit one `BlockComplete` per representable terminal block, with
/// consecutive zero-based ordinals.
///
/// The walk is atomic: [`terminal_blocks`] validates and flattens the whole
/// array before any `BlockComplete` is emitted, so a partial authoritative
/// sequence never reaches a collector.
///
/// A terminal array yielding no representable blocks is a contradiction
/// when live content already streamed: the collector would otherwise keep
/// its provisional buffers and report a successful completion whose content
/// the provider never confirmed. That fails the stream through
/// [`fail_stream`] instead. With no live content it is a genuine empty
/// completion and stays silent (the spec forbids a dummy block).
fn emit_terminal_blocks(
    output: &[OutputItem],
    state: &mut ResponsesStreamState,
    events: &mut Vec<Result<StreamEvent, StreamError>>,
) {
    match terminal_blocks(output) {
        Ok(blocks) if blocks.is_empty() && state.saw_live_content => fail_stream(
            StreamError::Deserialize {
                message:
                    "response.output produced no representable blocks after live content deltas"
                        .to_owned(),
                raw_data: None,
            },
            state,
            events,
        ),
        Ok(blocks) => {
            for (index, block) in blocks.into_iter().enumerate() {
                events.push(Ok(StreamEvent::BlockComplete { index, block }));
            }
        }
        Err(message) => fail_stream(
            StreamError::Deserialize {
                message,
                raw_data: None,
            },
            state,
            events,
        ),
    }
}

/// Mark the stream failed and surface at most one terminal error.
///
/// Failure is absorbing: the status becomes [`TerminalStatus::Failed`]
/// and stays there. The first terminal error is pushed to `events`; any
/// later terminal error is dropped so the consumer observes exactly one.
fn fail_stream(
    error: StreamError,
    state: &mut ResponsesStreamState,
    events: &mut Vec<Result<StreamEvent, StreamError>>,
) {
    if state.terminal != TerminalStatus::Active {
        return;
    }
    state.terminal.fail();
    state.terminal_error_surfaced = true;
    events.push(Err(error));
}

/// Record a `response.failed`/`response.error` terminal failure.
///
/// A content-policy message still records the `ContentFilter` stop
/// reason, but every failure marks the stream failed so the EOF callback
/// cannot later emit a successful `Completed`.
fn record_terminal_failure(
    message: String,
    state: &mut ResponsesStreamState,
    events: &mut Vec<Result<StreamEvent, StreamError>>,
) {
    if is_content_policy_message(&message) {
        state.stop_reason = Some(StopReason::ContentFilter);
    }
    fail_stream(
        StreamError::ConnectionLost {
            kind: StreamErrorKind::ProviderError,
            message,
        },
        state,
        events,
    );
}

/// Terminal event for the Responses stream.
///
/// Emits a successful `Completed` only when the parser saw an actual
/// `response.completed` or a supported `response.incomplete`. A failed
/// response, mid-stream error, SDK transport error, or bare EOF yields a
/// typed stream error instead, so collectors that accept bare EOF still
/// fail on a truncated or failed response.
///
/// Failure is absorbing and reported exactly once: if the parser already
/// pushed a terminal error (a `response.failed`/`response.error` event
/// or an SDK transport error), the EOF callback returns `None` so the
/// consumer observes a single error rather than a duplicate.
fn responses_terminal_event(
    state: &ResponsesStreamState,
) -> Option<Result<StreamEvent, StreamError>> {
    match state.terminal {
        TerminalStatus::SawCompleted | TerminalStatus::SawIncomplete => {
            Some(Ok(StreamEvent::Completed {
                metadata: CompletionMetadata {
                    model: state.model.clone(),
                    stop_reason: state.stop_reason,
                    usage: state.usage,
                },
            }))
        }
        TerminalStatus::Cancelled => Some(Err(StreamError::Cancelled)),
        TerminalStatus::Failed => {
            if state.terminal_error_surfaced {
                None
            } else {
                Some(Err(StreamError::ConnectionLost {
                    kind: StreamErrorKind::ProviderError,
                    message: "Responses stream failed".to_owned(),
                }))
            }
        }
        TerminalStatus::Active => Some(Err(StreamError::ConnectionLost {
            kind: StreamErrorKind::ConnectionDropped,
            message: "Responses stream ended without response.completed or response.incomplete"
                .to_owned(),
        })),
    }
}

/// Stop reason for a validated `completed` response: tool use when the
/// terminal output carries a function call, end of turn otherwise.
///
/// Incomplete responses derive their stop reason from the reported
/// incomplete reason, and every other status fails loud before reaching
/// here, so no status fallback is needed.
fn completed_stop_reason(output: &[OutputItem]) -> StopReason {
    if output
        .iter()
        .any(|item| matches!(item, OutputItem::FunctionCall(_)))
    {
        StopReason::ToolUse
    } else {
        StopReason::EndTurn
    }
}

fn to_token_usage(usage: ResponseUsage) -> TokenUsage {
    TokenUsage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Message;

    /// Recording observer that captures only the live text and thinking
    /// deltas the agent collector forwards.
    #[derive(Default)]
    struct RecordingObserver {
        observed: std::sync::Mutex<Vec<String>>,
    }

    #[async_trait::async_trait]
    impl crate::agent::AgentObserver for RecordingObserver {
        async fn on_event(&self, event: &crate::agent::AgentEvent) {
            use crate::agent::AgentEvent;
            let tag = match event {
                AgentEvent::TextDelta { text } => format!("text:{text}"),
                AgentEvent::ThinkingDelta { thinking } => format!("thinking:{thinking}"),
                AgentEvent::IterationStart { .. }
                | AgentEvent::ToolCallStart { .. }
                | AgentEvent::ToolCallComplete { .. }
                | AgentEvent::IterationComplete { .. }
                | AgentEvent::LoopComplete { .. } => return,
            };
            self.observed
                .lock()
                .expect("observer mutex poisoned")
                .push(tag);
        }
    }

    /// Build a `StreamHandle` from Responses wire events using the real
    /// adapter, parser, and terminal callback.
    fn responses_handle(
        inner_items: Vec<Result<ResponseStreamEvent, StreamError>>,
    ) -> StreamHandle {
        let inner = futures::stream::iter(inner_items);
        let cancel = tokio_util::sync::CancellationToken::new();
        let stream = crate::provider::stream_adapter::buffered_sdk_stream(
            inner,
            cancel.clone(),
            ResponsesStreamState::default(),
            |item, state| match item {
                Ok(event) => parse_responses_event(event, state),
                Err(e) => {
                    let mut events = Vec::new();
                    fail_stream(e, state, &mut events);
                    events
                }
            },
            responses_terminal_event,
        );
        StreamHandle::new(
            Box::pin(stream),
            cancel,
            crate::types::CorrelationId::generate(),
        )
    }

    fn delta_signature(delta: &StreamDelta) -> String {
        match delta {
            StreamDelta::TextDelta { text } => format!("text:{text:?}"),
            StreamDelta::ThinkingDelta { thinking } => format!("thinking:{thinking:?}"),
            StreamDelta::SignatureDelta { signature } => format!("signature:{signature:?}"),
            StreamDelta::ReasoningItemDelta { id, encrypted } => {
                format!("reasoning:{id}:{encrypted}")
            }
            StreamDelta::ToolUseStart { id, name } => {
                format!("tool_start:{}:{}", id.as_str(), name.as_str())
            }
            StreamDelta::ToolInputDelta { id, partial_json } => {
                format!("tool_input:{}:{partial_json}", id.as_str())
            }
        }
    }

    fn block_signature(block: &ContentBlock) -> String {
        match block {
            ContentBlock::Text { text } => format!("text:{text:?}"),
            ContentBlock::Thinking { text } => format!("thinking:{text:?}"),
            ContentBlock::EncryptedReasoning { id, encrypted } => {
                format!("encrypted:{id}:{encrypted}")
            }
            ContentBlock::ToolUse { id, name, input } => {
                format!("tool:{}:{}:{input}", id.as_str(), name.as_str())
            }
            ContentBlock::ToolResult { .. } => "tool_result".to_owned(),
        }
    }

    fn event_signature(event: &Result<StreamEvent, StreamError>) -> String {
        match event {
            Err(StreamError::Cancelled) => "error:cancelled".to_owned(),
            Err(StreamError::ConnectionLost { message, .. }) => {
                format!("error:connection:{message}")
            }
            Err(StreamError::Deserialize { message, .. }) => format!("error:deserialize:{message}"),
            Ok(StreamEvent::Started { metadata }) => {
                format!("started:{:?}", metadata.model.as_ref().map(ModelId::as_str))
            }
            Ok(StreamEvent::ContentBlockStart { index, block_type }) => {
                format!("start:{index}:{block_type:?}")
            }
            Ok(StreamEvent::Delta(delta)) => format!("delta:{}", delta_signature(delta)),
            Ok(StreamEvent::ContentBlockStop { index }) => format!("stop:{index}"),
            Ok(StreamEvent::BlockComplete { index, block }) => {
                format!("block:{index}:{}", block_signature(block))
            }
            Ok(StreamEvent::Completed { metadata }) => format!(
                "completed:{:?}:{:?}",
                metadata.stop_reason,
                metadata.usage.map(|u| (u.input_tokens, u.output_tokens))
            ),
            Ok(StreamEvent::Error { error }) => format!("error:{error:?}"),
        }
    }

    fn reasoning_item_json() -> serde_json::Value {
        serde_json::json!({
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{ "type": "summary_text", "text": "Weigh options." }],
            "encrypted_content": "ENC",
            "status": "completed"
        })
    }

    fn message_item_json() -> serde_json::Value {
        serde_json::json!({
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [
                { "type": "output_text", "text": "Answer: done.", "annotations": [] }
            ]
        })
    }

    fn function_call_item_json(
        call_id: &str,
        id: &str,
        name: &str,
        arguments: &str,
    ) -> serde_json::Value {
        serde_json::json!({
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
            "id": id,
            "status": "completed"
        })
    }

    fn item_added_event(
        sequence: u64,
        output_index: u32,
        item: serde_json::Value,
    ) -> ResponseStreamEvent {
        stream_event(serde_json::json!({
            "type": "response.output_item.added",
            "sequence_number": sequence,
            "output_index": output_index,
            "item": item
        }))
    }

    fn item_done_event(
        sequence: u64,
        output_index: u32,
        item: serde_json::Value,
    ) -> ResponseStreamEvent {
        stream_event(serde_json::json!({
            "type": "response.output_item.done",
            "sequence_number": sequence,
            "output_index": output_index,
            "item": item
        }))
    }

    /// The complete spec fixture: encrypted reasoning, then answer text,
    /// then two function calls whose argument deltas interleave.
    fn complete_response_fixture() -> Vec<ResponseStreamEvent> {
        vec![
            created_event(),
            item_added_event(2, 0, reasoning_item_json()),
            stream_event(serde_json::json!({
                "type": "response.reasoning_summary_part.added",
                "sequence_number": 3,
                "item_id": "rs_1",
                "output_index": 0,
                "summary_index": 0,
                "part": { "type": "summary_text", "text": "" }
            })),
            stream_event(serde_json::json!({
                "type": "response.reasoning_summary_text.delta",
                "sequence_number": 4,
                "item_id": "rs_1",
                "output_index": 0,
                "summary_index": 0,
                "delta": "Weigh options."
            })),
            stream_event(serde_json::json!({
                "type": "response.reasoning_summary_part.done",
                "sequence_number": 5,
                "item_id": "rs_1",
                "output_index": 0,
                "summary_index": 0,
                "part": { "type": "summary_text", "text": "Weigh options." }
            })),
            item_done_event(6, 0, reasoning_item_json()),
            item_added_event(7, 1, message_item_json()),
            stream_event(serde_json::json!({
                "type": "response.output_text.delta",
                "sequence_number": 8,
                "item_id": "msg_1",
                "output_index": 1,
                "content_index": 0,
                "delta": "Answer: done."
            })),
            item_done_event(9, 1, message_item_json()),
            item_added_event(
                10,
                2,
                function_call_item_json("call_1", "fc_1", "read_file", "{\"path\": \"/tmp/a\"}"),
            ),
            item_added_event(
                11,
                3,
                function_call_item_json("call_2", "fc_2", "list_dir", "{\"dir\": \"/tmp/b\"}"),
            ),
            stream_event(serde_json::json!({
                "type": "response.function_call_arguments.delta",
                "sequence_number": 12,
                "item_id": "fc_1",
                "output_index": 2,
                "delta": "{\"path\": \""
            })),
            stream_event(serde_json::json!({
                "type": "response.function_call_arguments.delta",
                "sequence_number": 13,
                "item_id": "fc_2",
                "output_index": 3,
                "delta": "{\"dir\": \""
            })),
            stream_event(serde_json::json!({
                "type": "response.function_call_arguments.delta",
                "sequence_number": 14,
                "item_id": "fc_1",
                "output_index": 2,
                "delta": "/tmp/a\"}"
            })),
            stream_event(serde_json::json!({
                "type": "response.function_call_arguments.delta",
                "sequence_number": 15,
                "item_id": "fc_2",
                "output_index": 3,
                "delta": "/tmp/b\"}"
            })),
            item_done_event(
                16,
                2,
                function_call_item_json("call_1", "fc_1", "read_file", "{\"path\": \"/tmp/a\"}"),
            ),
            item_done_event(
                17,
                3,
                function_call_item_json("call_2", "fc_2", "list_dir", "{\"dir\": \"/tmp/b\"}"),
            ),
            stream_event(serde_json::json!({
                "type": "response.completed",
                "sequence_number": 18,
                "response": {
                    "id": "resp_1",
                    "object": "response",
                    "created_at": 1,
                    "model": "gpt-5.2",
                    "status": "completed",
                    "output": [
                        reasoning_item_json(),
                        message_item_json(),
                        function_call_item_json("call_1", "fc_1", "read_file", "{\"path\": \"/tmp/a\"}"),
                        function_call_item_json("call_2", "fc_2", "list_dir", "{\"dir\": \"/tmp/b\"}")
                    ],
                    "usage": {
                        "input_tokens": 12,
                        "input_tokens_details": { "cached_tokens": 0 },
                        "output_tokens": 7,
                        "output_tokens_details": { "reasoning_tokens": 3 },
                        "total_tokens": 19
                    }
                }
            })),
        ]
    }

    fn fixture_as_items(
        events: Vec<ResponseStreamEvent>,
    ) -> Vec<Result<ResponseStreamEvent, StreamError>> {
        events.into_iter().map(Ok).collect()
    }

    fn stream_event(json: serde_json::Value) -> ResponseStreamEvent {
        serde_json::from_value(json).expect("valid stream event JSON")
    }

    fn created_event() -> ResponseStreamEvent {
        stream_event(serde_json::json!({
            "type": "response.created",
            "sequence_number": 0,
            "response": {
                "id": "resp_test",
                "object": "response",
                "created_at": 1,
                "model": "gpt-5.2",
                "status": "in_progress",
                "output": []
            }
        }))
    }

    #[test]
    fn created_event_emits_started_with_model() {
        let mut state = ResponsesStreamState::default();
        let events = parse_responses_event(created_event(), &mut state);

        assert!(state.started);
        assert_eq!(state.model.as_ref().map(ModelId::as_str), Some("gpt-5.2"));
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Ok(StreamEvent::Started { .. })))
        );
    }

    #[test]
    fn text_delta_maps_to_text_delta() {
        let mut state = ResponsesStreamState {
            started: true,
            ..Default::default()
        };
        let event = stream_event(serde_json::json!({
            "type": "response.output_text.delta",
            "sequence_number": 3,
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hello"
        }));

        let events = parse_responses_event(event, &mut state);
        assert!(events.iter().any(|e| matches!(
            e,
            Ok(StreamEvent::Delta(StreamDelta::TextDelta { text })) if text == "Hello"
        )));
    }

    #[test]
    fn reasoning_summary_delta_maps_to_thinking_delta() {
        let mut state = ResponsesStreamState {
            started: true,
            ..Default::default()
        };
        let event = stream_event(serde_json::json!({
            "type": "response.reasoning_summary_text.delta",
            "sequence_number": 2,
            "item_id": "rs_1",
            "output_index": 0,
            "summary_index": 0,
            "delta": "Step one"
        }));

        let events = parse_responses_event(event, &mut state);
        assert!(events.iter().any(|e| matches!(
            e,
            Ok(StreamEvent::Delta(StreamDelta::ThinkingDelta { thinking })) if thinking == "Step one"
        )));
        assert!(
            events
                .iter()
                .all(|e| !matches!(e, Ok(StreamEvent::Delta(StreamDelta::TextDelta { .. }))))
        );
    }

    #[test]
    fn function_call_item_and_arguments_map_to_tool_events() {
        let mut state = ResponsesStreamState {
            started: true,
            ..Default::default()
        };

        let added = stream_event(serde_json::json!({
            "type": "response.output_item.added",
            "sequence_number": 4,
            "output_index": 1,
            "item": {
                "type": "function_call",
                "call_id": "call_abc",
                "name": "read_file",
                "arguments": "",
                "id": "fc_1",
                "status": "in_progress"
            }
        }));
        let events = parse_responses_event(added, &mut state);
        assert!(events.iter().any(|e| matches!(
            e,
            Ok(StreamEvent::ContentBlockStart {
                index: 1,
                block_type: ContentBlockType::ToolUse,
            })
        )));
        assert!(events.iter().any(|e| matches!(
            e,
            Ok(StreamEvent::Delta(StreamDelta::ToolUseStart { name, .. }))
                if name.as_str() == "read_file"
        )));

        let args = stream_event(serde_json::json!({
            "type": "response.function_call_arguments.delta",
            "sequence_number": 5,
            "item_id": "fc_1",
            "output_index": 1,
            "delta": "{\"path\":"
        }));
        let events = parse_responses_event(args, &mut state);
        assert!(events.iter().any(|e| matches!(
            e,
            Ok(StreamEvent::Delta(StreamDelta::ToolInputDelta { partial_json, .. }))
                if partial_json == "{\"path\":"
        )));
    }

    #[test]
    fn completed_event_maps_usage_and_end_turn() {
        let mut state = ResponsesStreamState {
            started: true,
            ..Default::default()
        };
        let event = stream_event(serde_json::json!({
            "type": "response.completed",
            "sequence_number": 9,
            "response": {
                "id": "resp_test",
                "object": "response",
                "created_at": 1,
                "model": "gpt-5.2",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            { "type": "output_text", "text": "Hi", "annotations": [] }
                        ]
                    }
                ],
                "usage": {
                    "input_tokens": 12,
                    "input_tokens_details": { "cached_tokens": 0 },
                    "output_tokens": 7,
                    "output_tokens_details": { "reasoning_tokens": 3 },
                    "total_tokens": 19
                }
            }
        }));

        parse_responses_event(event, &mut state);
        assert_eq!(state.stop_reason, Some(StopReason::EndTurn));
        assert_eq!(
            state.usage.map(|u| (u.input_tokens, u.output_tokens)),
            Some((12, 7))
        );
    }

    #[test]
    fn completed_event_with_function_call_maps_tool_use() {
        let mut state = ResponsesStreamState {
            started: true,
            ..Default::default()
        };
        let event = stream_event(serde_json::json!({
            "type": "response.completed",
            "sequence_number": 9,
            "response": {
                "id": "resp_test",
                "object": "response",
                "created_at": 1,
                "model": "gpt-5.2",
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_abc",
                        "name": "read_file",
                        "arguments": "{}",
                        "id": "fc_1",
                        "status": "completed"
                    }
                ]
            }
        }));

        parse_responses_event(event, &mut state);
        assert_eq!(state.stop_reason, Some(StopReason::ToolUse));
    }

    #[test]
    fn incomplete_event_maps_max_tokens() {
        let mut state = ResponsesStreamState {
            started: true,
            ..Default::default()
        };
        let event = stream_event(serde_json::json!({
            "type": "response.incomplete",
            "sequence_number": 9,
            "response": {
                "id": "resp_test",
                "object": "response",
                "created_at": 1,
                "model": "gpt-5.2",
                "status": "incomplete",
                "output": [],
                "incomplete_details": { "reason": "max_output_tokens" }
            }
        }));

        parse_responses_event(event, &mut state);
        assert_eq!(state.stop_reason, Some(StopReason::MaxTokens));
    }

    /// `response.completed` must carry the `completed` status. A failed,
    /// incomplete, queued, or in-progress status is a protocol
    /// contradiction and fails loud before any terminal block is emitted.
    #[test]
    fn completed_event_with_unexpected_status_fails_loud() {
        for status in ["failed", "incomplete", "queued", "in_progress"] {
            let mut state = ResponsesStreamState {
                started: true,
                ..Default::default()
            };
            let event = stream_event(serde_json::json!({
                "type": "response.completed",
                "sequence_number": 9,
                "response": {
                    "id": "resp_test",
                    "object": "response",
                    "created_at": 1,
                    "model": "gpt-5.2",
                    "status": status,
                    "output": [{
                        "type": "message",
                        "id": "msg_1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{ "type": "output_text", "text": "hi", "annotations": [] }]
                    }]
                }
            }));

            let events = parse_responses_event(event, &mut state);
            assert!(
                events
                    .iter()
                    .any(|e| matches!(e, Err(StreamError::Deserialize { .. }))),
                "status {status} must fail loud: {events:?}"
            );
            assert!(
                !events
                    .iter()
                    .any(|e| matches!(e, Ok(StreamEvent::BlockComplete { .. }))),
                "status {status} must not emit a terminal block: {events:?}"
            );
            assert!(
                !events
                    .iter()
                    .any(|e| matches!(e, Ok(StreamEvent::Completed { .. }))),
                "status {status} must not report success: {events:?}"
            );
            assert_eq!(state.terminal, TerminalStatus::Failed);
        }
    }

    /// `response.incomplete` must carry the `incomplete` status; a
    /// `completed` status on that event is a contradiction.
    #[test]
    fn incomplete_event_with_completed_status_fails_loud() {
        let mut state = ResponsesStreamState {
            started: true,
            ..Default::default()
        };
        let event = stream_event(serde_json::json!({
            "type": "response.incomplete",
            "sequence_number": 9,
            "response": {
                "id": "resp_test",
                "object": "response",
                "created_at": 1,
                "model": "gpt-5.2",
                "status": "completed",
                "output": [],
                "incomplete_details": { "reason": "max_output_tokens" }
            }
        }));

        let events = parse_responses_event(event, &mut state);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Err(StreamError::Deserialize { .. }))),
            "completed status on response.incomplete must fail loud: {events:?}"
        );
        assert_eq!(state.terminal, TerminalStatus::Failed);
    }

    /// An encrypted reasoning payload without an id cannot be replayed;
    /// fail loud instead of silently dropping the reasoning item.
    #[test]
    fn completed_encrypted_reasoning_without_id_fails_loud() {
        let mut state = ResponsesStreamState {
            started: true,
            ..Default::default()
        };
        let event = stream_event(serde_json::json!({
            "type": "response.completed",
            "sequence_number": 9,
            "response": {
                "id": "resp_test",
                "object": "response",
                "created_at": 1,
                "model": "gpt-5.2",
                "status": "completed",
                "output": [{
                    "type": "reasoning",
                    "summary": [],
                    "encrypted_content": "ENC",
                    "status": "completed"
                }]
            }
        }));

        let events = parse_responses_event(event, &mut state);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Err(StreamError::Deserialize { .. }))),
            "encrypted reasoning without an id must fail loud: {events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Ok(StreamEvent::BlockComplete { .. }))),
            "no block may be emitted for the malformed reasoning item: {events:?}"
        );
        assert_eq!(state.terminal, TerminalStatus::Failed);
    }

    /// An empty reasoning id alongside a non-empty encrypted payload is
    /// equally unusable and fails loud.
    #[test]
    fn completed_encrypted_reasoning_with_empty_id_fails_loud() {
        let mut state = ResponsesStreamState {
            started: true,
            ..Default::default()
        };
        let event = stream_event(serde_json::json!({
            "type": "response.completed",
            "sequence_number": 9,
            "response": {
                "id": "resp_test",
                "object": "response",
                "created_at": 1,
                "model": "gpt-5.2",
                "status": "completed",
                "output": [{
                    "type": "reasoning",
                    "id": "",
                    "summary": [],
                    "encrypted_content": "ENC",
                    "status": "completed"
                }]
            }
        }));

        let events = parse_responses_event(event, &mut state);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Err(StreamError::Deserialize { .. }))),
            "empty encrypted reasoning id must fail loud: {events:?}"
        );
        assert_eq!(state.terminal, TerminalStatus::Failed);
    }

    /// Absence of both id and encrypted content is allowed when the
    /// encrypted payload was not requested; the summary still collects.
    #[test]
    fn completed_reasoning_without_id_or_encrypted_emits_thinking_only() {
        let mut state = ResponsesStreamState {
            started: true,
            ..Default::default()
        };
        let event = stream_event(serde_json::json!({
            "type": "response.completed",
            "sequence_number": 9,
            "response": {
                "id": "resp_test",
                "object": "response",
                "created_at": 1,
                "model": "gpt-5.2",
                "status": "completed",
                "output": [{
                    "type": "reasoning",
                    "summary": [{ "type": "summary_text", "text": "consider" }],
                    "status": "completed"
                }]
            }
        }));

        let events = parse_responses_event(event, &mut state);
        assert!(
            events.iter().any(|e| matches!(
                e,
                Ok(StreamEvent::BlockComplete { index: 0, block: ContentBlock::Thinking { text } })
                    if text == "consider"
            )),
            "summary must still collect without an encrypted payload: {events:?}"
        );
        assert!(
            events.iter().all(Result::is_ok),
            "absent encrypted payload must not error: {events:?}"
        );
        assert_eq!(state.terminal, TerminalStatus::SawCompleted);
    }

    /// A real `response.completed` is the only way to a successful
    /// terminal event.
    #[test]
    fn terminal_after_completed_is_success() {
        let mut state = ResponsesStreamState::default();
        parse_responses_event(
            stream_event(serde_json::json!({
                "type": "response.completed",
                "sequence_number": 9,
                "response": {
                    "id": "resp_test",
                    "object": "response",
                    "created_at": 1,
                    "model": "gpt-5.2",
                    "status": "completed",
                    "output": []
                }
            })),
            &mut state,
        );
        assert!(matches!(
            responses_terminal_event(&state),
            Some(Ok(StreamEvent::Completed { .. }))
        ));
    }

    /// Bare EOF (no terminal event) must not be reported as success.
    #[test]
    fn terminal_after_bare_eof_is_error() {
        let state = ResponsesStreamState {
            started: true,
            ..Default::default()
        };
        assert!(matches!(responses_terminal_event(&state), Some(Err(_))));
    }

    /// A `response.failed` marks failure; the terminal event is an
    /// error, never `Completed`.
    #[test]
    fn terminal_after_failed_is_error() {
        let mut state = ResponsesStreamState::default();
        let events = parse_responses_event(
            stream_event(serde_json::json!({
                "type": "response.failed",
                "sequence_number": 4,
                "response": {
                    "id": "resp_test",
                    "object": "response",
                    "created_at": 1,
                    "model": "gpt-5.2",
                    "status": "failed",
                    "output": [],
                    "error": { "code": "server_error", "message": "boom" }
                }
            })),
            &mut state,
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Err(StreamError::ConnectionLost { .. }))),
            "failed response must surface a stream error: {events:?}"
        );
        assert!(
            responses_terminal_event(&state).is_none(),
            "the error was already surfaced; EOF must not duplicate it"
        );
    }

    /// Failure is absorbing: a `response.completed` after a
    /// `response.failed` cannot turn the stream into a success.
    #[test]
    fn terminal_failed_then_completed_stays_failed() {
        let mut state = ResponsesStreamState::default();
        parse_responses_event(
            stream_event(serde_json::json!({
                "type": "response.failed",
                "sequence_number": 4,
                "response": {
                    "id": "resp_test",
                    "object": "response",
                    "created_at": 1,
                    "model": "gpt-5.2",
                    "status": "failed",
                    "output": [],
                    "error": { "code": "server_error", "message": "boom" }
                }
            })),
            &mut state,
        );
        let events = parse_responses_event(
            stream_event(serde_json::json!({
                "type": "response.completed",
                "sequence_number": 5,
                "response": {
                    "id": "resp_test",
                    "object": "response",
                    "created_at": 1,
                    "model": "gpt-5.2",
                    "status": "completed",
                    "output": []
                }
            })),
            &mut state,
        );
        assert_eq!(
            state.terminal,
            TerminalStatus::Failed,
            "completed after failed must stay failed"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Ok(StreamEvent::Completed { .. }))),
            "failed stream must not emit a Completed event"
        );
        assert!(
            responses_terminal_event(&state).is_none(),
            "the failure was already reported once"
        );
    }

    /// Failure is also absorbing against `response.incomplete`: a
    /// truncated-success terminal after a failure cannot win.
    #[test]
    fn terminal_failed_then_incomplete_stays_failed() {
        let mut state = ResponsesStreamState::default();
        parse_responses_event(failed_response_event(4), &mut state);
        let events = parse_responses_event(
            stream_event(serde_json::json!({
                "type": "response.incomplete",
                "sequence_number": 5,
                "response": {
                    "id": "resp_test",
                    "object": "response",
                    "created_at": 1,
                    "model": "gpt-5.2",
                    "status": "incomplete",
                    "output": [],
                    "incomplete_details": { "reason": "max_output_tokens" }
                }
            })),
            &mut state,
        );
        assert_eq!(state.terminal, TerminalStatus::Failed);
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Ok(StreamEvent::Completed { .. }))),
            "failed stream must not emit a Completed event"
        );
        assert!(responses_terminal_event(&state).is_none());
    }

    /// A mid-stream `response.error` marks failure; a later bare EOF can
    /// never emit a successful `Completed`, and the error is not
    /// duplicated at EOF.
    #[test]
    fn terminal_after_error_event_is_error() {
        let mut state = ResponsesStreamState::default();
        parse_responses_event(
            stream_event(serde_json::json!({
                "type": "error",
                "sequence_number": 3,
                "code": "server_error",
                "message": "upstream failed"
            })),
            &mut state,
        );
        assert!(responses_terminal_event(&state).is_none());
    }

    /// A content-policy failure records the filter stop reason but still
    /// terminates as a failure (no successful `Completed`).
    #[test]
    fn terminal_after_content_policy_failure_is_error() {
        let mut state = ResponsesStreamState::default();
        parse_responses_event(
            stream_event(serde_json::json!({
                "type": "response.failed",
                "sequence_number": 4,
                "response": {
                    "id": "resp_test",
                    "object": "response",
                    "created_at": 1,
                    "model": "gpt-5.2",
                    "status": "failed",
                    "output": [],
                    "error": {
                        "code": "content_policy_violation",
                        "message": "Rejected: this request violates our content policy."
                    }
                }
            })),
            &mut state,
        );
        assert_eq!(state.stop_reason, Some(StopReason::ContentFilter));
        assert!(responses_terminal_event(&state).is_none());
    }

    #[test]
    fn input_items_round_trip_conversation() {
        let request = CompletionRequest::new(
            ModelId::new("gpt-5.2").unwrap(),
            vec![
                Message::new(
                    DriverRole::User,
                    vec![ContentBlock::Text {
                        text: "list the files".into(),
                    }],
                ),
                Message::new(
                    DriverRole::Assistant,
                    vec![
                        ContentBlock::Text {
                            text: "Reading.".into(),
                        },
                        ContentBlock::ToolUse {
                            id: ToolCallId::new("call_abc"),
                            name: ToolName::new("list_dir").unwrap(),
                            input: serde_json::json!({"path": "/tmp"}),
                        },
                    ],
                ),
                Message::new(
                    DriverRole::Tool,
                    vec![ContentBlock::ToolResult {
                        tool_use_id: ToolCallId::new("call_abc"),
                        content: crate::types::ToolResultContent::Text("a b".into()),
                        is_error: false,
                    }],
                ),
            ],
        );

        let items = build_input_items(&request);
        assert_eq!(
            items.len(),
            4,
            "user, assistant text, function call, output"
        );

        let wire = serde_json::to_value(&items).unwrap();
        assert_eq!(wire[0]["role"], "user");
        assert_eq!(wire[0]["content"], "list the files");
        assert_eq!(wire[1]["role"], "assistant");
        assert_eq!(wire[1]["content"], "Reading.");
        assert_eq!(wire[2]["type"], "function_call");
        assert_eq!(wire[2]["call_id"], "call_abc");
        assert_eq!(wire[2]["name"], "list_dir");
        assert_eq!(wire[2]["arguments"], "{\"path\":\"/tmp\"}");
        assert_eq!(wire[3]["type"], "function_call_output");
        assert_eq!(wire[3]["call_id"], "call_abc");
        assert_eq!(wire[3]["output"], "a b");
    }

    /// Assistant thinking blocks are not replayed as assistant text on
    /// the responses surface; the reasoning items that carry them
    /// round-trip natively instead.
    #[test]
    fn assistant_thinking_is_not_replayed_as_text() {
        let request = CompletionRequest::new(
            ModelId::new("gpt-5.2").unwrap(),
            vec![Message::new(
                DriverRole::Assistant,
                vec![ContentBlock::Thinking {
                    text: "internal chain of thought".into(),
                }],
            )],
        );

        let items = build_input_items(&request);
        assert!(
            items.is_empty(),
            "a thinking-only assistant message replays nothing"
        );
    }

    #[test]
    fn request_rejects_unsupported_stop_sequences() {
        let provider = OpenAiProvider::new(crate::config::OpenAiConfig {
            api_key: crate::config::ApiKey::new("test-key").unwrap(),
            api: crate::config::OpenAiApi::Responses,
            base_url: None,
            model: crate::config::OpenAiModel::Gpt4o,
            max_tokens: crate::types::MaxTokens::new(32).unwrap(),
            temperature: None,
            reasoning: None,
            structured_output: false,
            chat_reasoning_output: crate::config::OpenAiChatReasoningOutput::None,
            custom_capabilities: None,
        })
        .unwrap();
        let mut request = CompletionRequest::new(
            ModelId::new("gpt-4o").unwrap(),
            vec![Message::user("hello")],
        );
        drop(provider.build_responses_request(&request).unwrap());
        request.config.stop_sequences.push("STOP".into());
        assert!(matches!(
            provider.build_responses_request(&request),
            Err(ProviderError::InvalidRequest { .. })
        ));
    }

    /// The wire body must be stateless streaming: store=false,
    /// stream=true, model and instructions present, reasoning effort
    /// mapped one-to-one.
    #[test]
    fn request_body_is_stateless_streaming() {
        let config = crate::config::OpenAiConfig {
            api_key: crate::config::ApiKey::new("test-key").unwrap(),
            api: crate::config::OpenAiApi::Responses,
            base_url: None,
            model: crate::config::OpenAiModel::Gpt5,
            max_tokens: crate::types::MaxTokens::new(512).unwrap(),
            temperature: None,
            reasoning: Some(crate::config::ReasoningConfig {
                effort: ReasoningEffort::High,
                summary: ReasoningSummary::Auto,
            }),
            structured_output: false,
            chat_reasoning_output: crate::config::OpenAiChatReasoningOutput::None,
            custom_capabilities: None,
        };
        let provider = OpenAiProvider::new(config).unwrap();

        let request = CompletionRequest::new(
            ModelId::new("gpt-5").unwrap(),
            vec![Message::new(
                DriverRole::User,
                vec![ContentBlock::Text { text: "hi".into() }],
            )],
        )
        .with_system(crate::types::SystemPrompt::new("be brief"))
        .with_config(crate::provider::CompletionConfig::new(
            crate::types::MaxTokens::new(512).unwrap(),
        ));

        let wire = serde_json::to_value(provider.build_responses_request(&request).unwrap())
            .expect("request serializes");

        assert_eq!(
            wire,
            serde_json::json!({
                "model": "gpt-5",
                "input": [{ "type": "message", "role": "user", "content": "hi" }],
                "max_output_tokens": 512,
                "store": false,
                "stream": true,
                "instructions": "be brief",
                "reasoning": { "effort": "high", "summary": "auto" },
                "include": ["reasoning.encrypted_content"]
            })
        );
    }

    /// Encrypted reasoning items replay verbatim as reasoning input
    /// items in content order (reasoning before the text it produced);
    /// thinking summaries do not replay.
    #[test]
    fn encrypted_reasoning_round_trips_as_reasoning_item() {
        let request = CompletionRequest::new(
            ModelId::new("gpt-5.2").unwrap(),
            vec![Message::new(
                DriverRole::Assistant,
                vec![
                    ContentBlock::EncryptedReasoning {
                        id: "rs_abc".into(),
                        encrypted: "gAAAAA...".into(),
                    },
                    ContentBlock::Text {
                        text: "the answer".into(),
                    },
                ],
            )],
        );

        let wire = serde_json::to_value(build_input_items(&request)).unwrap();
        let items = wire.as_array().expect("input items serialize to an array");
        assert_eq!(items.len(), 2, "reasoning item then assistant text: {wire}");
        assert_eq!(items[0]["type"], "reasoning");
        assert_eq!(items[0]["id"], "rs_abc");
        assert_eq!(items[0]["encrypted_content"], "gAAAAA...");
        assert_eq!(items[1]["role"], "assistant");
        assert_eq!(items[1]["content"], "the answer");
    }

    /// A completed reasoning item carrying encrypted content emits a
    /// ReasoningItemDelta; without the encrypted payload there is
    /// nothing replayable and the event is silent.
    #[test]
    fn reasoning_item_done_emits_reasoning_item_delta() {
        let mut state = ResponsesStreamState {
            started: true,
            ..Default::default()
        };
        let event = stream_event(serde_json::json!({
            "type": "response.output_item.done",
            "sequence_number": 6,
            "output_index": 0,
            "item": {
                "type": "reasoning",
                "id": "rs_abc",
                "summary": [],
                "encrypted_content": "gAAAAA..."
            }
        }));

        let events = parse_responses_event(event, &mut state);
        assert!(events.iter().any(|e| matches!(
            e,
            Ok(StreamEvent::Delta(StreamDelta::ReasoningItemDelta { id, encrypted }))
                if id == "rs_abc" && encrypted == "gAAAAA..."
        )));

        let mut state = ResponsesStreamState {
            started: true,
            ..Default::default()
        };
        let plain = stream_event(serde_json::json!({
            "type": "response.output_item.done",
            "sequence_number": 7,
            "output_index": 0,
            "item": {
                "type": "reasoning",
                "id": "rs_def",
                "summary": []
            }
        }));
        let events = parse_responses_event(plain, &mut state);
        assert!(events.iter().all(|e| !matches!(
            e,
            Ok(StreamEvent::Delta(StreamDelta::ReasoningItemDelta { .. }))
        )));
    }

    /// Collected responses keep the replayable item and the flush
    /// order is deterministic: reasoning lands ahead of the answer
    /// text it precedes, regardless of delta timing.
    #[test]
    fn reasoning_item_delta_accumulates_into_content() {
        let mut response = crate::streaming::CollectedResponse::new();
        response.apply_delta(StreamDelta::TextDelta {
            text: "the answer".into(),
        });
        response.apply_delta(StreamDelta::ReasoningItemDelta {
            id: "rs_abc".into(),
            encrypted: "gAAAAA...".into(),
        });
        response.flush_pending();

        assert_eq!(response.content.len(), 2, "{:?}", response.content);
        assert!(matches!(
            response.content[0],
            ContentBlock::EncryptedReasoning { ref id, .. } if id == "rs_abc"
        ));
        assert!(matches!(
            response.content[1],
            ContentBlock::Text { ref text } if text == "the answer"
        ));
    }

    /// A custom model that declares reasoning support on Responses gets
    /// the native reasoning parameter and encrypted-content request; the
    /// profile, not a model-ID heuristic, supplies it. Asserted as a
    /// full body.
    #[test]
    fn custom_capability_reasoning_reaches_responses_wire() {
        let config = crate::config::OpenAiConfig {
            api_key: crate::config::ApiKey::new("test-key").unwrap(),
            api: crate::config::OpenAiApi::Responses,
            base_url: None,
            model: crate::config::OpenAiModel::Custom("org/reasoner".into()),
            max_tokens: crate::types::MaxTokens::new(512).unwrap(),
            temperature: None,
            reasoning: Some(crate::config::ReasoningConfig {
                effort: ReasoningEffort::Low,
                summary: ReasoningSummary::Concise,
            }),
            structured_output: false,
            chat_reasoning_output: crate::config::OpenAiChatReasoningOutput::None,
            custom_capabilities: Some(crate::config::CustomModelCapabilities::new(
                false, true, true,
            )),
        };
        let provider = OpenAiProvider::new(config).unwrap();
        let request = CompletionRequest::new(
            ModelId::new("org/reasoner").unwrap(),
            vec![Message::new(
                DriverRole::User,
                vec![ContentBlock::Text { text: "hi".into() }],
            )],
        );

        let wire =
            serde_json::to_value(provider.build_responses_request(&request).unwrap()).unwrap();
        assert_eq!(
            wire,
            serde_json::json!({
                "model": "org/reasoner",
                "input": [{ "type": "message", "role": "user", "content": "hi" }],
                "max_output_tokens": 4096,
                "store": false,
                "stream": true,
                "reasoning": { "effort": "low", "summary": "concise" },
                "include": ["reasoning.encrypted_content"]
            })
        );
    }

    /// Raw reasoning-text deltas (newer reasoning models) map to the
    /// same ThinkingDelta channel as summaries.
    #[test]
    fn reasoning_text_delta_maps_to_thinking_delta() {
        let mut state = ResponsesStreamState {
            started: true,
            ..Default::default()
        };
        let event = stream_event(serde_json::json!({
            "type": "response.reasoning_text.delta",
            "sequence_number": 2,
            "item_id": "rs_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "raw chain of thought"
        }));

        let events = parse_responses_event(event, &mut state);
        assert!(events.iter().any(|e| matches!(
            e,
            Ok(StreamEvent::Delta(StreamDelta::ThinkingDelta { thinking }))
                if thinking == "raw chain of thought"
        )));
    }

    /// Refusal deltas surface as answer text, not silence.
    #[test]
    fn refusal_delta_maps_to_text_delta() {
        let mut state = ResponsesStreamState {
            started: true,
            ..Default::default()
        };
        let event = stream_event(serde_json::json!({
            "type": "response.refusal.delta",
            "sequence_number": 4,
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "I can't help with that."
        }));

        let events = parse_responses_event(event, &mut state);
        assert!(events.iter().any(|e| matches!(
            e,
            Ok(StreamEvent::Delta(StreamDelta::TextDelta { text }))
                if text == "I can't help with that."
        )));
    }

    /// A function-call item without an item id cannot be correlated
    /// with its argument deltas; the stream fails loud instead of
    /// guessing.
    #[test]
    fn function_call_item_without_id_fails_loud() {
        let mut state = ResponsesStreamState {
            started: true,
            ..Default::default()
        };
        let event = stream_event(serde_json::json!({
            "type": "response.output_item.added",
            "sequence_number": 4,
            "output_index": 1,
            "item": {
                "type": "function_call",
                "call_id": "call_abc",
                "name": "read_file",
                "arguments": ""
            }
        }));

        let events = parse_responses_event(event, &mut state);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Err(StreamError::Deserialize { .. }))),
            "missing item id must fail loud, got: {events:?}"
        );
    }

    /// Drive the Responses events through the real stream adapter
    /// (`buffered_sdk_stream`) with the production parser and terminal
    /// callback, and return the full wire sequence. This exercises the
    /// adapter's EOF handling, not just the state helper.
    async fn run_responses_adapter(
        inner_items: Vec<Result<ResponseStreamEvent, StreamError>>,
    ) -> Vec<Result<StreamEvent, StreamError>> {
        use crate::provider::stream_adapter::buffered_sdk_stream;
        use futures::StreamExt as _;

        let inner = futures::stream::iter(inner_items);
        let cancel = tokio_util::sync::CancellationToken::new();
        let stream = buffered_sdk_stream(
            inner,
            cancel,
            ResponsesStreamState::default(),
            |item, state| match item {
                Ok(event) => parse_responses_event(event, state),
                Err(e) => {
                    let mut events = Vec::new();
                    fail_stream(e, state, &mut events);
                    events
                }
            },
            responses_terminal_event,
        );
        stream.collect().await
    }

    fn completed_response_event(sequence: u64) -> ResponseStreamEvent {
        stream_event(serde_json::json!({
            "type": "response.completed",
            "sequence_number": sequence,
            "response": {
                "id": "resp_test",
                "object": "response",
                "created_at": 1,
                "model": "gpt-5.2",
                "status": "completed",
                "output": []
            }
        }))
    }

    fn failed_response_event(sequence: u64) -> ResponseStreamEvent {
        stream_event(serde_json::json!({
            "type": "response.failed",
            "sequence_number": sequence,
            "response": {
                "id": "resp_test",
                "object": "response",
                "created_at": 1,
                "model": "gpt-5.2",
                "status": "failed",
                "output": [],
                "error": { "code": "server_error", "message": "boom" }
            }
        }))
    }

    /// Adapter-level: `failed` then `completed` then EOF yields exactly
    /// one error and never a `Completed`.
    #[tokio::test]
    async fn adapter_failed_then_completed_emits_one_error() {
        let events = run_responses_adapter(vec![
            Ok(failed_response_event(4)),
            Ok(completed_response_event(5)),
        ])
        .await;

        let errors = events.iter().filter(|e| e.is_err()).count();
        assert_eq!(
            errors, 1,
            "failure must be reported exactly once, got: {events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Ok(StreamEvent::Completed { .. }))),
            "a failed stream must never emit Completed: {events:?}"
        );
    }

    /// Adapter-level: `failed` then `incomplete` then EOF also yields
    /// exactly one error and never a `Completed`.
    #[tokio::test]
    async fn adapter_failed_then_incomplete_emits_one_error() {
        let incomplete = stream_event(serde_json::json!({
            "type": "response.incomplete",
            "sequence_number": 5,
            "response": {
                "id": "resp_test",
                "object": "response",
                "created_at": 1,
                "model": "gpt-5.2",
                "status": "incomplete",
                "output": [],
                "incomplete_details": { "reason": "max_output_tokens" }
            }
        }));
        let events =
            run_responses_adapter(vec![Ok(failed_response_event(4)), Ok(incomplete)]).await;

        let errors = events.iter().filter(|e| e.is_err()).count();
        assert_eq!(errors, 1, "got: {events:?}");
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Ok(StreamEvent::Completed { .. }))),
            "a failed stream must never emit Completed: {events:?}"
        );
    }

    /// Adapter-level: a `response.error` event followed by EOF emits
    /// exactly one error and no `Completed`.
    #[tokio::test]
    async fn adapter_error_event_then_eof_emits_one_error() {
        let error_event = stream_event(serde_json::json!({
            "type": "error",
            "sequence_number": 3,
            "code": "server_error",
            "message": "upstream failed"
        }));
        let events = run_responses_adapter(vec![Ok(error_event)]).await;

        let errors = events.iter().filter(|e| e.is_err()).count();
        assert_eq!(errors, 1, "expected one error, got: {events:?}");
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Ok(StreamEvent::Completed { .. }))),
            "error stream must never emit Completed: {events:?}"
        );
    }

    /// Adapter-level: an SDK transport error followed by EOF emits
    /// exactly one error and no `Completed`.
    #[tokio::test]
    async fn adapter_transport_error_then_eof_emits_one_error() {
        let transport = StreamError::ConnectionLost {
            kind: StreamErrorKind::TransportError,
            message: "connection reset".to_owned(),
        };
        let events = run_responses_adapter(vec![Err(transport)]).await;

        let errors = events.iter().filter(|e| e.is_err()).count();
        assert_eq!(errors, 1, "expected one error, got: {events:?}");
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Ok(StreamEvent::Completed { .. }))),
            "transport failure must never emit Completed: {events:?}"
        );
    }

    /// Adapter-level: an abrupt EOF with no terminal event is a single
    /// error.
    #[tokio::test]
    async fn adapter_bare_eof_is_single_error() {
        let events = run_responses_adapter(vec![Ok(created_event())]).await;

        let errors = events.iter().filter(|e| e.is_err()).count();
        assert_eq!(errors, 1, "bare EOF must yield one error, got: {events:?}");
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Ok(StreamEvent::Completed { .. }))),
            "bare EOF must never emit Completed: {events:?}"
        );
    }

    fn error_response_event(sequence: u64) -> ResponseStreamEvent {
        stream_event(serde_json::json!({
            "type": "error",
            "sequence_number": sequence,
            "code": "server_error",
            "message": "upstream failed"
        }))
    }

    fn text_delta_event(sequence: u64, text: &str) -> ResponseStreamEvent {
        stream_event(serde_json::json!({
            "type": "response.output_text.delta",
            "sequence_number": sequence,
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": text
        }))
    }

    /// Adapter-level: `failed` then a second terminal error event then
    /// EOF still yields exactly one error.
    #[tokio::test]
    async fn adapter_failed_then_error_then_eof_emits_one_error() {
        let events = run_responses_adapter(vec![
            Ok(failed_response_event(4)),
            Ok(error_response_event(5)),
        ])
        .await;

        let errors = events.iter().filter(|e| e.is_err()).count();
        assert_eq!(errors, 1, "got: {events:?}");
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Ok(StreamEvent::Completed { .. }))),
            "failed stream must never emit Completed: {events:?}"
        );
    }

    /// Adapter-level: a parser error (invalid function-call item id)
    /// followed by EOF yields exactly one error.
    #[tokio::test]
    async fn adapter_parser_error_then_eof_emits_one_error() {
        let bad_item = stream_event(serde_json::json!({
            "type": "response.output_item.added",
            "sequence_number": 4,
            "output_index": 1,
            "item": {
                "type": "function_call",
                "call_id": "call_abc",
                "name": "read_file",
                "arguments": ""
            }
        }));
        let events = run_responses_adapter(vec![Ok(bad_item)]).await;

        let errors = events.iter().filter(|e| e.is_err()).count();
        assert_eq!(errors, 1, "got: {events:?}");
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Ok(StreamEvent::Completed { .. }))),
            "parser failure must never emit Completed: {events:?}"
        );
    }

    /// Adapter-level: an SDK transport error then a `completed` then EOF
    /// yields exactly one error and never a `Completed`.
    #[tokio::test]
    async fn adapter_transport_error_then_completed_emits_one_error() {
        let transport = StreamError::ConnectionLost {
            kind: StreamErrorKind::TransportError,
            message: "connection reset".to_owned(),
        };
        let events =
            run_responses_adapter(vec![Err(transport), Ok(completed_response_event(9))]).await;

        let errors = events.iter().filter(|e| e.is_err()).count();
        assert_eq!(errors, 1, "got: {events:?}");
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Ok(StreamEvent::Completed { .. }))),
            "transport failure must never emit Completed: {events:?}"
        );
    }

    /// Adapter-level: content deltas after a failure are suppressed; a
    /// post-failure `completed` cannot win either.
    #[tokio::test]
    async fn adapter_post_failure_events_are_suppressed() {
        let events = run_responses_adapter(vec![
            Ok(failed_response_event(4)),
            Ok(text_delta_event(5, "late text")),
            Ok(completed_response_event(6)),
        ])
        .await;

        let errors = events.iter().filter(|e| e.is_err()).count();
        assert_eq!(errors, 1, "got: {events:?}");
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Ok(StreamEvent::Delta(StreamDelta::TextDelta { .. })))),
            "post-failure text must be suppressed: {events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Ok(StreamEvent::Completed { .. }))),
            "post-failure Completed must be suppressed: {events:?}"
        );
    }

    /// Adapter-level: a real `completed` still yields a successful
    /// `Completed` terminal.
    #[tokio::test]
    async fn adapter_completed_emits_completed() {
        let events =
            run_responses_adapter(vec![Ok(created_event()), Ok(completed_response_event(9))]).await;
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Ok(StreamEvent::Completed { .. }))),
            "completed stream must emit Completed: {events:?}"
        );
        assert!(
            events.iter().all(Result::is_ok),
            "completed stream must have no errors: {events:?}"
        );
    }

    /// `response.incomplete` with reason `content_filter` maps to the
    /// `ContentFilter` stop reason.
    #[test]
    fn incomplete_event_content_filter_reason_maps_to_stop_reason() {
        let mut state = ResponsesStreamState {
            started: true,
            ..Default::default()
        };
        let event = stream_event(serde_json::json!({
            "type": "response.incomplete",
            "sequence_number": 9,
            "response": {
                "id": "resp_test",
                "object": "response",
                "created_at": 1,
                "model": "gpt-5.2",
                "status": "incomplete",
                "output": [],
                "incomplete_details": { "reason": "content_filter" }
            }
        }));

        parse_responses_event(event, &mut state);
        assert_eq!(state.stop_reason, Some(StopReason::ContentFilter));
        assert!(matches!(state.terminal, TerminalStatus::SawIncomplete));
    }

    /// An unknown or missing `response.incomplete` reason is a stream
    /// error; it must not default to token exhaustion.
    #[test]
    fn incomplete_event_unknown_reason_fails_loud() {
        let mut state = ResponsesStreamState {
            started: true,
            ..Default::default()
        };
        let event = stream_event(serde_json::json!({
            "type": "response.incomplete",
            "sequence_number": 9,
            "response": {
                "id": "resp_test",
                "object": "response",
                "created_at": 1,
                "model": "gpt-5.2",
                "status": "incomplete",
                "output": [],
                "incomplete_details": { "reason": "unknown_reason" }
            }
        }));

        let events = parse_responses_event(event, &mut state);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Err(StreamError::Deserialize { .. }))),
            "unknown incomplete reason must error: {events:?}"
        );
        assert_eq!(state.terminal, TerminalStatus::Failed);
    }

    /// A `response.completed` with empty `output` after live content
    /// deltas is an error: the terminal array cannot contradict the
    /// observed stream.
    #[test]
    fn completed_empty_output_after_live_content_fails() {
        let mut state = ResponsesStreamState {
            started: true,
            ..Default::default()
        };
        parse_responses_event(text_delta_event(2, "live text"), &mut state);
        let completed = stream_event(serde_json::json!({
            "type": "response.completed",
            "sequence_number": 9,
            "response": {
                "id": "resp_test",
                "object": "response",
                "created_at": 1,
                "model": "gpt-5.2",
                "status": "completed",
                "output": []
            }
        }));

        let events = parse_responses_event(completed, &mut state);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Err(StreamError::Deserialize { .. }))),
            "empty output after live content must error: {events:?}"
        );
        assert_eq!(state.terminal, TerminalStatus::Failed);
    }

    /// All successful terminals are absorbing: a `response.completed`
    /// after a supported `response.incomplete` keeps the incomplete
    /// terminal and emits no second success event.
    #[test]
    fn terminal_incomplete_then_completed_stays_incomplete() {
        let mut state = ResponsesStreamState::default();
        parse_responses_event(
            stream_event(serde_json::json!({
                "type": "response.incomplete",
                "sequence_number": 4,
                "response": {
                    "id": "resp_test",
                    "object": "response",
                    "created_at": 1,
                    "model": "gpt-5.2",
                    "status": "incomplete",
                    "output": [],
                    "incomplete_details": { "reason": "max_output_tokens" }
                }
            })),
            &mut state,
        );
        let events = parse_responses_event(completed_response_event(5), &mut state);

        assert_eq!(state.terminal, TerminalStatus::SawIncomplete);
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Ok(StreamEvent::Completed { .. }))),
            "post-terminal completed must be suppressed: {events:?}"
        );
    }

    /// A successful terminal is absorbing against later failures: a
    /// `response.failed` after `response.completed` cannot overwrite it.
    #[test]
    fn terminal_completed_then_failed_stays_completed() {
        let mut state = ResponsesStreamState::default();
        parse_responses_event(completed_response_event(4), &mut state);
        let events = parse_responses_event(failed_response_event(5), &mut state);

        assert_eq!(state.terminal, TerminalStatus::SawCompleted);
        assert!(
            events.iter().all(Result::is_ok),
            "post-terminal failure must not emit an error: {events:?}"
        );
    }

    /// A cancelled terminal is absorbing: a later `response.completed`
    /// cannot turn it into a success, and no delta can follow it.
    #[test]
    fn terminal_cancelled_is_absorbing() {
        let mut state = ResponsesStreamState {
            started: true,
            ..Default::default()
        };
        let cancelled = stream_event(serde_json::json!({
            "type": "response.completed",
            "sequence_number": 4,
            "response": {
                "id": "resp_test",
                "object": "response",
                "created_at": 1,
                "model": "gpt-5.2",
                "status": "cancelled",
                "output": []
            }
        }));
        parse_responses_event(cancelled, &mut state);
        let post_delta = text_delta_event(5, "late");
        let post_completed = completed_response_event(6);

        let delta_events = parse_responses_event(post_delta, &mut state);
        let completed_events = parse_responses_event(post_completed, &mut state);

        assert_eq!(state.terminal, TerminalStatus::Cancelled);
        assert!(
            delta_events.is_empty(),
            "post-terminal delta must be suppressed: {delta_events:?}"
        );
        assert!(
            completed_events.is_empty(),
            "post-terminal completed must be suppressed: {completed_events:?}"
        );
    }

    /// A function-call argument delta whose `item_id` is unknown is a
    /// stream error; it must not be silently dropped.
    #[test]
    fn function_call_arguments_delta_unknown_item_id_fails_loud() {
        let mut state = ResponsesStreamState {
            started: true,
            ..Default::default()
        };
        let event = stream_event(serde_json::json!({
            "type": "response.function_call_arguments.delta",
            "sequence_number": 5,
            "item_id": "unknown_fc",
            "output_index": 0,
            "delta": "{\"path\": \"/tmp\"}"
        }));

        let events = parse_responses_event(event, &mut state);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Err(StreamError::Deserialize { .. }))),
            "unknown function-call item_id must error: {events:?}"
        );
        assert_eq!(state.terminal, TerminalStatus::Failed);
    }

    // -----------------------------------------------------------------------
    // Terminal complete blocks in provider order
    // -----------------------------------------------------------------------

    /// The complete fixture must produce the full driver event sequence:
    /// live deltas and balanced lifecycle for every item, then the
    /// authoritative `BlockComplete` array in provider order, then
    /// `Completed` last.
    #[tokio::test]
    async fn complete_fixture_matches_full_event_sequence() {
        let events = run_responses_adapter(fixture_as_items(complete_response_fixture())).await;
        let signatures: Vec<String> = events.iter().map(event_signature).collect();

        let expected = [
            "started:Some(\"gpt-5.2\")",
            "start:0:Thinking",
            "delta:thinking:\"Weigh options.\"",
            "stop:0",
            "delta:reasoning:rs_1:ENC",
            "start:1:Text",
            "delta:text:\"Answer: done.\"",
            "stop:1",
            "start:2:ToolUse",
            "delta:tool_start:call_1:read_file",
            "start:3:ToolUse",
            "delta:tool_start:call_2:list_dir",
            "delta:tool_input:call_1:{\"path\": \"",
            "delta:tool_input:call_2:{\"dir\": \"",
            "delta:tool_input:call_1:/tmp/a\"}",
            "delta:tool_input:call_2:/tmp/b\"}",
            "stop:2",
            "stop:3",
            "block:0:thinking:\"Weigh options.\"",
            "block:1:encrypted:rs_1:ENC",
            "block:2:text:\"Answer: done.\"",
            "block:3:tool:call_1:read_file:{\"path\":\"/tmp/a\"}",
            "block:4:tool:call_2:list_dir:{\"dir\":\"/tmp/b\"}",
            "completed:Some(ToolUse):Some((12, 7))",
        ];

        assert_eq!(signatures, expected, "full event sequence drifted");
    }

    /// Both collection paths (`StreamHandle::collect` and the agent's
    /// observer-driven collector) must agree on the authoritative blocks,
    /// and the observer must still receive the live thinking/text deltas.
    /// The collected assistant turn then replays into the next Responses
    /// request in wire order.
    #[tokio::test]
    async fn complete_fixture_both_collectors_agree_and_replay_in_order() {
        let expected_content = vec![
            ContentBlock::Thinking {
                text: "Weigh options.".into(),
            },
            ContentBlock::EncryptedReasoning {
                id: "rs_1".into(),
                encrypted: "ENC".into(),
            },
            ContentBlock::Text {
                text: "Answer: done.".into(),
            },
            ContentBlock::ToolUse {
                id: ToolCallId::new("call_1"),
                name: ToolName::new("read_file").unwrap(),
                input: serde_json::json!({ "path": "/tmp/a" }),
            },
            ContentBlock::ToolUse {
                id: ToolCallId::new("call_2"),
                name: ToolName::new("list_dir").unwrap(),
                input: serde_json::json!({ "dir": "/tmp/b" }),
            },
        ];

        let collected = responses_handle(fixture_as_items(complete_response_fixture()))
            .collect()
            .await
            .expect("direct collect succeeds");

        let observer = RecordingObserver::default();
        let collected_observer = crate::agent::collect_with_observer(
            responses_handle(fixture_as_items(complete_response_fixture())),
            &tokio_util::sync::CancellationToken::new(),
            &observer,
        )
        .await
        .expect("observer collect succeeds");

        assert_eq!(collected.content, expected_content);
        assert_eq!(collected_observer.content, expected_content);
        assert_eq!(
            collected.metadata.stop_reason,
            Some(StopReason::ToolUse),
            "tool response keeps ToolUse stop reason"
        );
        assert_eq!(
            *observer.observed.lock().unwrap(),
            vec![
                "thinking:Weigh options.".to_owned(),
                "text:Answer: done.".to_owned(),
            ],
            "observer must forward live thinking and text, never complete blocks"
        );

        // Replay the collected turn plus its tool outputs into the next
        // stateless Responses request and pin the wire order.
        let request = CompletionRequest::new(
            ModelId::new("gpt-5.2").unwrap(),
            vec![
                Message::new(
                    DriverRole::User,
                    vec![ContentBlock::Text { text: "go".into() }],
                ),
                Message::new(DriverRole::Assistant, collected.content.clone()),
                Message::new(
                    DriverRole::Tool,
                    vec![
                        ContentBlock::ToolResult {
                            tool_use_id: ToolCallId::new("call_1"),
                            content: crate::types::ToolResultContent::Text("a".into()),
                            is_error: false,
                        },
                        ContentBlock::ToolResult {
                            tool_use_id: ToolCallId::new("call_2"),
                            content: crate::types::ToolResultContent::Text("b".into()),
                            is_error: false,
                        },
                    ],
                ),
            ],
        );

        let wire = serde_json::to_value(build_input_items(&request)).unwrap();
        let items = wire.as_array().expect("input items serialize to an array");
        assert_eq!(
            items.len(),
            7,
            "user, reasoning, answer, 2 calls, 2 outputs"
        );
        assert_eq!(items[0]["role"], "user");
        assert_eq!(items[1]["type"], "reasoning");
        assert_eq!(items[1]["id"], "rs_1");
        assert_eq!(items[1]["encrypted_content"], "ENC");
        assert_eq!(items[2]["role"], "assistant");
        assert_eq!(items[2]["content"], "Answer: done.");
        assert_eq!(items[3]["type"], "function_call");
        assert_eq!(items[3]["call_id"], "call_1");
        assert_eq!(items[3]["name"], "read_file");
        assert_eq!(items[3]["arguments"], "{\"path\":\"/tmp/a\"}");
        assert_eq!(items[4]["type"], "function_call");
        assert_eq!(items[4]["call_id"], "call_2");
        assert_eq!(items[4]["name"], "list_dir");
        assert_eq!(items[4]["arguments"], "{\"dir\":\"/tmp/b\"}");
        assert_eq!(items[5]["type"], "function_call_output");
        assert_eq!(items[5]["call_id"], "call_1");
        assert_eq!(items[6]["type"], "function_call_output");
        assert_eq!(items[6]["call_id"], "call_2");
    }

    /// A completed message whose only content is a refusal still yields a
    /// text block on the answer channel.
    #[test]
    fn completed_refusal_emits_text_block() {
        let mut state = ResponsesStreamState {
            started: true,
            ..Default::default()
        };
        let event = stream_event(serde_json::json!({
            "type": "response.completed",
            "sequence_number": 9,
            "response": {
                "id": "resp_test",
                "object": "response",
                "created_at": 1,
                "model": "gpt-5.2",
                "status": "completed",
                "output": [{
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{ "type": "refusal", "refusal": "I can't do that." }]
                }]
            }
        }));

        let events = parse_responses_event(event, &mut state);
        assert!(
            events.iter().any(|e| matches!(
                e,
                Ok(StreamEvent::BlockComplete { index: 0, block: ContentBlock::Text { text } })
                    if text == "I can't do that."
            )),
            "refusal must become an answer text block: {events:?}"
        );
        assert_eq!(state.terminal, TerminalStatus::SawCompleted);
    }

    /// A reasoning item with a summary but no encrypted payload emits a
    /// thinking block and is not an error (include was not requested).
    #[test]
    fn completed_reasoning_without_encrypted_emits_thinking_only() {
        let mut state = ResponsesStreamState {
            started: true,
            ..Default::default()
        };
        let event = stream_event(serde_json::json!({
            "type": "response.completed",
            "sequence_number": 9,
            "response": {
                "id": "resp_test",
                "object": "response",
                "created_at": 1,
                "model": "gpt-5.2",
                "status": "completed",
                "output": [{
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{ "type": "summary_text", "text": "consider" }],
                    "status": "completed"
                }]
            }
        }));

        let events = parse_responses_event(event, &mut state);
        assert!(
            events.iter().any(|e| matches!(
                e,
                Ok(StreamEvent::BlockComplete { index: 0, block: ContentBlock::Thinking { text } })
                    if text == "consider"
            )),
            "reasoning summary must become a thinking block: {events:?}"
        );
        assert!(
            !events.iter().any(|e| matches!(
                e,
                Ok(StreamEvent::BlockComplete {
                    block: ContentBlock::EncryptedReasoning { .. },
                    ..
                })
            )),
            "absent encrypted content must not fabricate a replayable block: {events:?}"
        );
        assert_eq!(state.terminal, TerminalStatus::SawCompleted);
    }

    /// A supported incomplete response preserves its partial output and
    /// emits terminal blocks before the EOF success terminal.
    #[test]
    fn incomplete_supported_emits_terminal_blocks() {
        let mut state = ResponsesStreamState {
            started: true,
            ..Default::default()
        };
        let event = stream_event(serde_json::json!({
            "type": "response.incomplete",
            "sequence_number": 9,
            "response": {
                "id": "resp_test",
                "object": "response",
                "created_at": 1,
                "model": "gpt-5.2",
                "status": "incomplete",
                "output": [{
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "incomplete",
                    "content": [{ "type": "output_text", "text": "partial", "annotations": [] }]
                }],
                "incomplete_details": { "reason": "content_filter" }
            }
        }));

        let events = parse_responses_event(event, &mut state);
        assert!(
            events.iter().any(|e| matches!(
                e,
                Ok(StreamEvent::BlockComplete { index: 0, block: ContentBlock::Text { text } })
                    if text == "partial"
            )),
            "supported incomplete must emit its partial blocks: {events:?}"
        );
        assert_eq!(state.stop_reason, Some(StopReason::ContentFilter));
        assert!(matches!(state.terminal, TerminalStatus::SawIncomplete));
    }

    /// A completed response with an empty output array and no earlier live
    /// content collects to an empty response: no block, no error.
    #[test]
    fn completed_empty_output_without_live_content_has_no_blocks() {
        let mut state = ResponsesStreamState {
            started: true,
            ..Default::default()
        };
        let events = parse_responses_event(completed_response_event(9), &mut state);

        assert!(
            events.iter().all(Result::is_ok),
            "empty valid completion must not error: {events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Ok(StreamEvent::BlockComplete { .. }))),
            "empty output must not fabricate a block: {events:?}"
        );
        assert_eq!(state.terminal, TerminalStatus::SawCompleted);
    }

    /// A non-empty terminal array whose items carry no representable
    /// content contradicts earlier live text: fail loud, emit no block,
    /// and never report success.
    #[test]
    fn completed_nonempty_output_with_no_blocks_after_live_text_fails() {
        let mut state = ResponsesStreamState {
            started: true,
            ..Default::default()
        };
        parse_responses_event(text_delta_event(2, "live text"), &mut state);
        let event = stream_event(serde_json::json!({
            "type": "response.completed",
            "sequence_number": 9,
            "response": {
                "id": "resp_test",
                "object": "response",
                "created_at": 1,
                "model": "gpt-5.2",
                "status": "completed",
                "output": [{
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{ "type": "output_text", "text": "", "annotations": [] }]
                }]
            }
        }));

        let events = parse_responses_event(event, &mut state);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Err(StreamError::Deserialize { .. }))),
            "empty terminal content after live text must fail loud: {events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Ok(StreamEvent::BlockComplete { .. }))),
            "no block may be emitted for the contradiction: {events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Ok(StreamEvent::Completed { .. }))),
            "the contradiction must not report success: {events:?}"
        );
        assert_eq!(state.terminal, TerminalStatus::Failed);
    }

    /// The same contradiction holds for live tool deltas against an empty
    /// reasoning item (non-empty output, zero representable blocks).
    #[test]
    fn completed_nonempty_output_with_no_blocks_after_live_tool_delta_fails() {
        let mut state = ResponsesStreamState {
            started: true,
            ..Default::default()
        };
        parse_responses_event(
            item_added_event(
                2,
                0,
                function_call_item_json("call_1", "fc_1", "read_file", "{}"),
            ),
            &mut state,
        );
        parse_responses_event(
            stream_event(serde_json::json!({
                "type": "response.function_call_arguments.delta",
                "sequence_number": 3,
                "item_id": "fc_1",
                "output_index": 0,
                "delta": "{}"
            })),
            &mut state,
        );
        let event = stream_event(serde_json::json!({
            "type": "response.completed",
            "sequence_number": 9,
            "response": {
                "id": "resp_test",
                "object": "response",
                "created_at": 1,
                "model": "gpt-5.2",
                "status": "completed",
                "output": [{
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [],
                    "status": "completed"
                }]
            }
        }));

        let events = parse_responses_event(event, &mut state);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Err(StreamError::Deserialize { .. }))),
            "empty terminal content after live tool deltas must fail loud: {events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Ok(StreamEvent::BlockComplete { .. }))),
            "no block may be emitted for the contradiction: {events:?}"
        );
        assert_eq!(state.terminal, TerminalStatus::Failed);
    }

    /// A supported incomplete with the same contradiction fails loud
    /// instead of preserving provisional live content.
    #[test]
    fn incomplete_nonempty_output_with_no_blocks_after_live_text_fails() {
        let mut state = ResponsesStreamState {
            started: true,
            ..Default::default()
        };
        parse_responses_event(text_delta_event(2, "live text"), &mut state);
        let event = stream_event(serde_json::json!({
            "type": "response.incomplete",
            "sequence_number": 9,
            "response": {
                "id": "resp_test",
                "object": "response",
                "created_at": 1,
                "model": "gpt-5.2",
                "status": "incomplete",
                "output": [{
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "incomplete",
                    "content": [{ "type": "output_text", "text": "", "annotations": [] }]
                }],
                "incomplete_details": { "reason": "max_output_tokens" }
            }
        }));

        let events = parse_responses_event(event, &mut state);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Err(StreamError::Deserialize { .. }))),
            "empty terminal content after live text must fail loud: {events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Ok(StreamEvent::BlockComplete { .. }))),
            "no block may be emitted for the contradiction: {events:?}"
        );
        assert_eq!(state.terminal, TerminalStatus::Failed);
    }

    /// A non-empty terminal array of empty items with no live content is a
    /// genuine empty completion: no error and no fabricated block.
    #[test]
    fn completed_nonempty_empty_items_without_live_content_is_empty_completion() {
        let mut state = ResponsesStreamState {
            started: true,
            ..Default::default()
        };
        let event = stream_event(serde_json::json!({
            "type": "response.completed",
            "sequence_number": 9,
            "response": {
                "id": "resp_test",
                "object": "response",
                "created_at": 1,
                "model": "gpt-5.2",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{ "type": "output_text", "text": "", "annotations": [] }]
                    },
                    {
                        "type": "reasoning",
                        "id": "rs_1",
                        "summary": [],
                        "status": "completed"
                    }
                ]
            }
        }));

        let events = parse_responses_event(event, &mut state);
        assert!(
            events.iter().all(Result::is_ok),
            "a genuinely empty completion must not error: {events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Ok(StreamEvent::BlockComplete { .. }))),
            "empty items must not fabricate a block: {events:?}"
        );
        assert_eq!(state.terminal, TerminalStatus::SawCompleted);
    }

    /// Terminal traversal validates every output item before emitting any
    /// block, so a malformed later item yields one error and zero blocks.
    #[test]
    fn terminal_invalid_tool_arguments_fail_loud_without_partial_blocks() {
        let mut state = ResponsesStreamState {
            started: true,
            ..Default::default()
        };
        let event = stream_event(serde_json::json!({
            "type": "response.completed",
            "sequence_number": 9,
            "response": {
                "id": "resp_test",
                "object": "response",
                "created_at": 1,
                "model": "gpt-5.2",
                "status": "completed",
                "output": [
                    message_item_json(),
                    function_call_item_json("call_1", "fc_1", "read_file", "{ not json")
                ]
            }
        }));

        let events = parse_responses_event(event, &mut state);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Err(StreamError::Deserialize { .. }))),
            "malformed tool arguments must fail loud: {events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Ok(StreamEvent::BlockComplete { .. }))),
            "validation must be atomic: {events:?}"
        );
        assert_eq!(state.terminal, TerminalStatus::Failed);
    }

    /// An unsupported hosted output item in the terminal array is a loud
    /// error, not a silent ignore.
    #[test]
    fn terminal_unsupported_output_item_fails_loud() {
        let mut state = ResponsesStreamState {
            started: true,
            ..Default::default()
        };
        let event = stream_event(serde_json::json!({
            "type": "response.completed",
            "sequence_number": 9,
            "response": {
                "id": "resp_test",
                "object": "response",
                "created_at": 1,
                "model": "gpt-5.2",
                "status": "completed",
                "output": [{
                    "type": "file_search_call",
                    "id": "fs_1",
                    "queries": [],
                    "status": "completed"
                }]
            }
        }));

        let events = parse_responses_event(event, &mut state);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Err(StreamError::Deserialize { .. }))),
            "hosted output item must fail loud: {events:?}"
        );
        assert_eq!(state.terminal, TerminalStatus::Failed);
    }

    /// A hosted output item announced live fails the stream immediately
    /// rather than streaming past it and failing only at EOF.
    #[tokio::test]
    async fn hosted_output_item_added_fails_loud() {
        let added = stream_event(serde_json::json!({
            "type": "response.output_item.added",
            "sequence_number": 4,
            "output_index": 0,
            "item": {
                "type": "file_search_call",
                "id": "fs_1",
                "queries": [],
                "status": "in_progress"
            }
        }));
        let events = run_responses_adapter(vec![Ok(created_event()), Ok(added)]).await;

        let errors = events.iter().filter(|e| e.is_err()).count();
        assert_eq!(errors, 1, "hosted item must yield one error: {events:?}");
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Err(StreamError::Deserialize { .. }))),
            "hosted item must fail as an unsupported output item: {events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Ok(StreamEvent::Completed { .. }))),
            "hosted item must never emit Completed: {events:?}"
        );
    }
}

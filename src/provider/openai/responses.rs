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
    InputParam, Item, MessageType, OutputItem, Reasoning, ReasoningEffort as SdkReasoningEffort,
    ReasoningItem, ReasoningSummary as SdkReasoningSummary, ResponseStreamEvent, ResponseUsage,
    Role, Status, Tool,
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

/// Terminal status of a Responses stream.
///
/// PR 1 only distinguishes "saw a real success terminal" from
/// everything else; reason classification (incomplete causes) belongs to
/// PR 2. Failure is absorbing: once a failure is recorded, a later
/// `response.completed`/`response.incomplete` cannot turn it into a
/// success.
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
}

impl TerminalStatus {
    /// Record a successful terminal, unless a failure already stuck.
    fn succeed(&mut self, status: Self) {
        if *self != Self::Failed {
            *self = status;
        }
    }

    /// Record a failure; failure is absorbing.
    fn fail(&mut self) {
        *self = Self::Failed;
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
}

/// Parse one Responses stream event into driver stream events.
///
/// Once the stream has failed, every later event is dropped: a failure
/// is absorbing, so no content, no successful terminal, and no second
/// error can follow it.
fn parse_responses_event(
    event: ResponseStreamEvent,
    state: &mut ResponsesStreamState,
) -> Vec<Result<StreamEvent, StreamError>> {
    if state.terminal == TerminalStatus::Failed {
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
        ResponseStreamEvent::ResponseOutputItemAdded(added) => match added.item {
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
                events.push(Ok(StreamEvent::Delta(StreamDelta::ToolUseStart {
                    id,
                    name,
                })));
            }
            // Reasoning items stream their summaries through the
            // dedicated delta events; hosted tool calls (web/file
            // search, code interpreter, MCP, image gen, computer use,
            // shell, compaction) are out of scope: the driver only
            // registers client-side function tools today. Ignoring
            // their items keeps unknown hosted-tool streams from
            // erroring the whole response.
            OutputItem::Reasoning(_)
            | OutputItem::FileSearchCall(_)
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
        ResponseStreamEvent::ResponseOutputTextDelta(delta) => {
            if !delta.delta.is_empty() {
                events.push(Ok(StreamEvent::Delta(StreamDelta::TextDelta {
                    text: delta.delta,
                })));
            }
        }
        // Summary deltas (summarized reasoning) and reasoning-text
        // deltas (raw chain of thought, newer reasoning models) both
        // map to the driver's ThinkingDelta channel.
        ResponseStreamEvent::ResponseReasoningSummaryTextDelta(delta) => {
            push_thinking_delta(delta.delta, &mut events);
        }
        ResponseStreamEvent::ResponseReasoningTextDelta(delta) => {
            push_thinking_delta(delta.delta, &mut events);
        }
        ResponseStreamEvent::ResponseFunctionCallArgumentsDelta(delta) => {
            if !delta.delta.is_empty() {
                if let Some(id) = state.tool_items.get(&delta.item_id) {
                    events.push(Ok(StreamEvent::Delta(StreamDelta::ToolInputDelta {
                        id: ToolCallId::new(id.as_str()),
                        partial_json: delta.delta,
                    })));
                }
            }
        }
        // Refusals are answer-channel content: surface them as text so
        // a safety refusal is visible instead of an empty turn.
        ResponseStreamEvent::ResponseRefusalDelta(delta) => {
            if !delta.delta.is_empty() {
                events.push(Ok(StreamEvent::Delta(StreamDelta::TextDelta {
                    text: delta.delta,
                })));
            }
        }
        // The done event duplicates the streamed refusal text.
        ResponseStreamEvent::ResponseRefusalDone(_) => {}
        ResponseStreamEvent::ResponseCompleted(completed) => {
            state.usage = completed.response.usage.map(to_token_usage);
            state.stop_reason = Some(stop_reason_for(
                &completed.response.output,
                completed.response.status,
            ));
            state.terminal.succeed(TerminalStatus::SawCompleted);
        }
        ResponseStreamEvent::ResponseIncomplete(incomplete) => {
            state.usage = incomplete.response.usage.map(to_token_usage);
            state.stop_reason = Some(StopReason::MaxTokens);
            // PR 1 treats a valid incomplete as a success terminal; PR 2
            // classifies the reported reason and rejects unknown ones.
            state.terminal.succeed(TerminalStatus::SawIncomplete);
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
        ResponseStreamEvent::ResponseOutputItemDone(done) => {
            if let OutputItem::Reasoning(item) = done.item {
                if let (Some(id), Some(encrypted)) = (item.id, item.encrypted_content) {
                    events.push(Ok(StreamEvent::Delta(StreamDelta::ReasoningItemDelta {
                        id,
                        encrypted,
                    })));
                }
            }
        }
        // Terminal bookkeeping events without driver-facing payloads.
        ResponseStreamEvent::ResponseContentPartAdded(_)
        | ResponseStreamEvent::ResponseContentPartDone(_)
        | ResponseStreamEvent::ResponseOutputTextDone(_)
        | ResponseStreamEvent::ResponseFunctionCallArgumentsDone(_)
        | ResponseStreamEvent::ResponseReasoningSummaryPartAdded(_)
        | ResponseStreamEvent::ResponseReasoningSummaryPartDone(_)
        | ResponseStreamEvent::ResponseReasoningSummaryTextDone(_)
        | ResponseStreamEvent::ResponseReasoningTextDone(_) => {}
        // Lifecycle noise that carries no content for this surface.
        ResponseStreamEvent::ResponseInProgress(_) | ResponseStreamEvent::ResponseQueued(_) => {}
        // Hosted tool-call streams (web/file search, code interpreter,
        // MCP, image gen, computer use, shell, custom tools) are out of
        // scope for the driver's client-side function tools; ignoring
        // their events keeps an unexpected hosted-tool response from
        // erroring the whole stream.
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
    state.terminal.fail();
    if !state.terminal_error_surfaced {
        state.terminal_error_surfaced = true;
        events.push(Err(error));
    }
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

fn stop_reason_for(output: &[OutputItem], status: Status) -> StopReason {
    match status {
        Status::Completed => {
            if output
                .iter()
                .any(|item| matches!(item, OutputItem::FunctionCall(_)))
            {
                StopReason::ToolUse
            } else {
                StopReason::EndTurn
            }
        }
        // Incomplete always means a bound (today: max_output_tokens)
        // cut the response short.
        Status::Incomplete => StopReason::MaxTokens,
        Status::Failed | Status::Cancelled | Status::Queued | Status::InProgress => {
            StopReason::EndTurn
        }
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
}

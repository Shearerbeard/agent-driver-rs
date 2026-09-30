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
            // The o3/o3-mini streaming caveat is recorded against the
            // chat-completions surface; it is carried over unchanged
            // until a live run proves the Responses behavior differs.
            if !self.config.model.supports_streaming() {
                return Err(ProviderError::StreamingNotSupported {
                    provider: crate::provider::ProviderKind::OpenAi,
                    model: self.config.model.as_str().to_owned(),
                });
            }

            let model = self.config.model.as_str();
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
                    Err(e) => vec![Err(StreamError::ConnectionLost {
                        kind: StreamErrorKind::TransportError,
                        message: e.to_string(),
                    })],
                },
                // The Responses terminal events (completed/incomplete/
                // failed) carry usage and stop reason into the state;
                // the stream ends right after them, and Completed is
                // emitted here — the same end-of-stream shape the chat
                // surface uses.
                |state| {
                    Some(Ok(StreamEvent::Completed {
                        metadata: CompletionMetadata {
                            model: state.model.clone(),
                            stop_reason: state.stop_reason,
                            usage: state.usage,
                        },
                    }))
                },
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
            .model(self.config.model.as_str())
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
            if self.config.model.supports_temperature() {
                builder.temperature(temp.get());
            }
        }

        if !request.tools.is_empty() {
            builder.tools(convert_tools(&request.tools));
        }

        if let Some(reasoning) = &self.config.reasoning {
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
                        .filter_map(|b| match b {
                            ContentBlock::Text { text } => Some(text.as_str()),
                            ContentBlock::Thinking { .. }
                            | ContentBlock::EncryptedReasoning { .. }
                            | ContentBlock::ToolUse { .. }
                            | ContentBlock::ToolResult { .. } => None,
                        })
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
/// them round-trip natively in Stage 4) plus one function-call item per
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
                    .filter_map(|b| match b {
                        ContentBlock::Text { text } => Some(text.as_str()),
                        ContentBlock::Thinking { .. }
                        | ContentBlock::EncryptedReasoning { .. }
                        | ContentBlock::ToolUse { .. }
                        | ContentBlock::ToolResult { .. } => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                if !text.is_empty() {
                    items.push(InputItem::EasyMessage(easy_message(Role::User, text)));
                }
            }
            DriverRole::Assistant => {
                let text = msg
                    .content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::Text { text } => Some(text.as_str()),
                        ContentBlock::Thinking { .. }
                        | ContentBlock::EncryptedReasoning { .. }
                        | ContentBlock::ToolUse { .. }
                        | ContentBlock::ToolResult { .. } => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                if !text.is_empty() {
                    items.push(InputItem::EasyMessage(easy_message(Role::Assistant, text)));
                }
                for block in &msg.content {
                    match block {
                        // Replayable reasoning state: round-trips as the
                        // reasoning item it came from (summaries are not
                        // resent; the encrypted payload carries them).
                        ContentBlock::EncryptedReasoning { id, encrypted } => {
                            items.push(InputItem::Item(Item::Reasoning(ReasoningItem {
                                id: Some(id.clone()),
                                summary: Vec::new(),
                                content: None,
                                encrypted_content: Some(encrypted.clone()),
                                status: None,
                            })));
                        }
                        ContentBlock::ToolUse { id, name, input } => {
                            items.push(InputItem::Item(Item::FunctionCall(FunctionToolCall {
                                arguments: serde_json::to_string(input).unwrap_or_default(),
                                call_id: id.as_str().to_owned(),
                                name: name.as_str().to_owned(),
                                namespace: None,
                                id: None,
                                status: None,
                            })));
                        }
                        ContentBlock::Text { .. }
                        | ContentBlock::Thinking { .. }
                        | ContentBlock::ToolResult { .. } => {}
                    }
                }
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
}

/// Parse one Responses stream event into driver stream events.
fn parse_responses_event(
    event: ResponseStreamEvent,
    state: &mut ResponsesStreamState,
) -> Vec<Result<StreamEvent, StreamError>> {
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
                        events.push(Err(StreamError::Deserialize {
                            message: format!("invalid OpenAI tool name: {e}"),
                            raw_data: None,
                        }));
                        return events;
                    }
                };
                let id = ToolCallId::new(&call.call_id);
                // Argument deltas reference the item id; without it the
                // stream cannot be correlated, so fail loud instead of
                // guessing at the call-id domain.
                let Some(item_key) = call.id.clone() else {
                    events.push(Err(StreamError::Deserialize {
                        message: format!(
                            "function call item for {} carries no item id",
                            call.call_id
                        ),
                        raw_data: None,
                    }));
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
        ResponseStreamEvent::ResponseReasoningSummaryTextDelta(delta) => {
            if !delta.delta.is_empty() {
                events.push(Ok(StreamEvent::Delta(StreamDelta::ThinkingDelta {
                    thinking: delta.delta,
                })));
            }
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
        }
        ResponseStreamEvent::ResponseIncomplete(incomplete) => {
            state.usage = incomplete.response.usage.map(to_token_usage);
            state.stop_reason = Some(StopReason::MaxTokens);
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
        | ResponseStreamEvent::ResponseReasoningTextDelta(_)
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

/// Record a terminal failure (failed response or mid-stream error):
/// content-policy messages end the turn with `ContentFilter`; anything
/// else fails the stream loudly.
fn record_terminal_failure(
    message: String,
    state: &mut ResponsesStreamState,
    events: &mut Vec<Result<StreamEvent, StreamError>>,
) {
    if is_content_policy_message(&message) {
        state.stop_reason = Some(StopReason::ContentFilter);
    } else {
        events.push(Err(StreamError::ConnectionLost {
            kind: StreamErrorKind::ProviderError,
            message,
        }));
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
        assert_eq!(state.model.as_ref().map(|m| m.as_str()), Some("gpt-5.2"));
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

        assert_eq!(wire["store"], false, "stateless: store must be false");
        assert_eq!(wire["stream"], true);
        assert_eq!(wire["model"], "gpt-5");
        assert_eq!(wire["max_output_tokens"], 512);
        assert_eq!(wire["instructions"], "be brief");
        assert_eq!(wire["reasoning"]["effort"], "high");
        assert_eq!(wire["reasoning"]["summary"], "auto");
        assert!(
            wire.get("temperature").is_none(),
            "reasoning models take no temperature"
        );
    }

    /// Encrypted reasoning items replay verbatim as reasoning input
    /// items; thinking summaries do not replay.
    #[test]
    fn encrypted_reasoning_round_trips_as_reasoning_item() {
        let request = CompletionRequest::new(
            ModelId::new("gpt-5.2").unwrap(),
            vec![Message::new(
                DriverRole::Assistant,
                vec![
                    ContentBlock::Thinking {
                        text: "summary text".into(),
                    },
                    ContentBlock::EncryptedReasoning {
                        id: "rs_abc".into(),
                        encrypted: "gAAAAA...".into(),
                    },
                ],
            )],
        );

        let wire = serde_json::to_value(build_input_items(&request)).unwrap();
        let items = wire.as_array().expect("input items serialize to an array");
        assert_eq!(items.len(), 1, "only the reasoning item replays: {wire}");
        assert_eq!(items[0]["type"], "reasoning");
        assert_eq!(items[0]["id"], "rs_abc");
        assert_eq!(items[0]["encrypted_content"], "gAAAAA...");
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

    /// Collected responses keep the replayable item: ReasoningItemDelta
    /// accumulates into an EncryptedReasoning content block.
    #[test]
    fn reasoning_item_delta_accumulates_into_content() {
        let mut response = crate::streaming::CollectedResponse::new();
        response.apply_delta(StreamDelta::ReasoningItemDelta {
            id: "rs_abc".into(),
            encrypted: "gAAAAA...".into(),
        });
        assert!(response.content.iter().any(|b| matches!(
            b,
            ContentBlock::EncryptedReasoning { id, encrypted }
                if id == "rs_abc" && encrypted == "gAAAAA..."
        )));
    }

    /// The reasoning request asks for the encrypted payload so replay
    /// works statelessly.
    #[test]
    fn reasoning_request_asks_for_encrypted_content() {
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
        };
        let provider = OpenAiProvider::new(config).unwrap();
        let request = CompletionRequest::new(
            ModelId::new("gpt-5").unwrap(),
            vec![Message::new(
                DriverRole::User,
                vec![ContentBlock::Text { text: "hi".into() }],
            )],
        );

        let wire =
            serde_json::to_value(provider.build_responses_request(&request).unwrap()).unwrap();
        assert_eq!(
            wire["include"],
            serde_json::json!(["reasoning.encrypted_content"])
        );
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
}

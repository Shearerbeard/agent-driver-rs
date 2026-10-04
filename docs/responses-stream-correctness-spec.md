# Responses stream correctness: PR 2 specification

**Parent:** The reviewed connection-profile PR based on PR #4.
**Decision:** [ADR-0008](adr/0008-compose-openai-compatible-connections.md).

## Purpose and scope

Make the OpenAI Responses stream report one honest outcome and preserve the
provider's content order through collection and tool continuation. Direct
stream consumers still receive incremental text and thinking. The library's
existing `ContentBlock::EncryptedReasoning` and `StreamEvent::BlockComplete`
are the only public carriers needed. The encrypted variant comes from PR #4;
this PR must not duplicate it.

This PR does not add hosted tools, generalize other providers, or change the
Chat Completions EOF policy. PR 1 already makes Responses EOF/failure terminal
errors; this PR supplies reason classification and content-order semantics.

## Terminal state and errors

Keep the terminal state private to the Responses parser: active, completed,
incomplete with the reported reason, failed, or cancelled. A completed response
ends with `StreamEvent::Completed` carrying the stop reason and usage. A valid
incomplete response also emits `Completed` with `MaxTokens` for
`max_output_tokens` or `ContentFilter` for `content_filter`; preserve its
partial output. Unknown or missing incomplete reasons are stream errors,
never default token exhaustion. A failed response, SDK transport error,
unsupported output type, or cancelled request cannot be followed by a
successful `Completed`. A visible refusal is text on the answer channel; a
normal completed refusal has `EndTurn` unless the API reports another terminal
reason. All end states are exercised by deterministic tests.

## Live events and collected blocks

- While the SDK stream is active, forward `TextDelta`, `ThinkingDelta`, and
  tool deltas promptly. Pair actual text, thinking summary, and function-call
  `ContentBlockStart`/`ContentBlockStop` events where the API supplies a block
  lifecycle. Encrypted reasoning is atomic and gets no fictitious start/stop.
- On `response.completed` or a supported `response.incomplete`, walk the
  terminal `response.output` array **in provider order** and emit one
  `StreamEvent::BlockComplete` for each representable Text, Thinking summary,
  EncryptedReasoning, and ToolUse block. Use consecutive zero-based ordinals
  for `BlockComplete.index`. These ordinals are a separate namespace from the
  live `ContentBlockStart/Stop.index` positions; callers must not correlate
  them. Tool arguments must be valid JSON; do not turn malformed data into
  an empty tool input. Unsupported hosted output item types fail loudly.
- `BlockComplete` is the authoritative collection carrier for Responses. Live
  deltas remain the observer/direct-stream channel, but collectors do not
  append their provisional buffers when complete blocks arrive. Keep the
  complete blocks in ordinal order, rejecting duplicate or missing ordinals
  through a fallible accumulation method returning `StreamError`.
- Factor existing shared block-type index tracking and delta/finalization
  logic into `CollectedResponse::apply_event` (or an equivalent private
  accumulator). Both `StreamHandle::collect` and the agent's observer-driven
  collection call it. Non-Responses providers emit no `BlockComplete` today;
  preserve their existing accumulation, EOF, and error behavior. The agent
  continues to forward live text/thinking deltas, never replaying complete
  blocks into observer deltas.
- If the terminal `response.output` is empty, require that no content delta
  appeared earlier. Then there are no complete blocks, and legacy empty
  buffers collect to the same empty response. Do not emit a dummy block or a
  new mode event. Any mismatched nonempty live content is an error.
- Complete reasoning and tool items retain their original ordering when
  replayed in the next Responses request. Thinking summaries are display
  content; encrypted payloads, not summaries, are replayed as reasoning input
  items. The final output array, not `HashMap` drain or `item.done` timing,
  determines tool execution and continuation order.

## Test surfaces and checks

Write expected whole-event sequences and complete serialized conversation
frames from this specification **before** implementing the parser changes;
record that they initially fail. The primary fixture runs from a reasoning item
through text and two interleaved function calls to the tool-result continuation,
with the encrypted item in wire order. It covers both collectors and direct
streaming. Separate fixtures cover empty output and malformed tool arguments;
unsupported output items, refusal, incomplete reasons, usage, duplicate or
missing ordinals, abrupt EOF, failed/error events, and cancellation also need
coverage.
Use expected event arrays or full wire fixtures if the repo has no snapshot
test dependency; do not add snapshot tooling solely for this PR.

Run `make check`, Rust 1.91.1 locked and isolated fresh-resolution
no-default/default/all-feature checks and tests, strict Clippy on tests,
doctests, examples, the standalone `bench` crate, cargo-deny, gitleaks, and
Vale for touched public prose. Keep live-key OpenAI text/reasoning/tool
continuation testing separate from offline CI. Present the diff, independent
Rust review, tests, and remaining risks before committing. PR 2 targets PR 1.

## Known limits

The public stream has two useful views: live deltas and final complete blocks.
Direct consumers may receive interleaved deltas before a block closes; only
the complete-block sequence claims final replay order. Third-party Chat
Completions reasoning fields are not replayable encrypted Responses items.

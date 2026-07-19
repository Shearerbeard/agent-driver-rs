# Rust Quality & Design Audit Checklist

<!-- vale-reason: This file uses Markdown horizontal rules and table separators that contain --- sequences; these are not prose em-dashes. -->
<!-- vale local.EmDashUsage = NO -->

Canonical checklist for the `agent-driver-rs` codebase. Use this during reviews, refactors, and before adding new providers/modules.

**Audit date:** 2026-06-21  
**Audited modules:** `types`, `error`, `session`, `streaming`, `task`, `provider` (facade + implementations), `config`, `agent`, `tool`, `otel`.  
**Validation baseline:** `cargo fmt --check`, `cargo check --all-features`, `cargo test --all-features`, and `cargo clippy --all-features --lib --tests -- -D warnings` pass (182 lib tests + 55 integration/doc checks).

---

## How to use this checklist

1. **Before a change:** Read the relevant section(s) for the modules you are touching.
2. **During review:** For each finding category, ask: *does this diff introduce, remove, or worsen one of these patterns?*
3. **After the change:** Run the validation commands at the end. No new clippy failures should be introduced.
4. **Triage:** P0 items are blockers; P1 items must be resolved in the same module; P2 items can be batched as cleanup debt.

Severity legend:
- **P0** — correctness, safety, or silent data-loss risk. Fix before merge.
- **P1** — maintainability, API stability, or semantic drift. Fix in the current workstream.
- **P2** — style, lint, or minor ergonomics. Batch-fix in a dedicated cleanup pass.

---

## Stage 1 resolution log

The following P0 items from the initial audit were fixed in Stage 1:

| # | Check | Resolution |
|---|-------|------------|
| 1.1 | Tool-call deltas accumulate by the correct ID. | `src/streaming.rs` now uses `HashMap<ToolCallId, PendingToolUse>`; regression tests added for parallel and duplicate-id streams. |
| 1.2 | No duplicate system messages in OpenAI conversion. | `src/provider/openai.rs` extracts a single system text from `request.system` or the first `Role::System` message, not both. |
| 1.3 | Retry sleep honors cancellation. | `src/provider/retry.rs` now races `sleep` against `cancellation.cancelled()` in a `biased` select. |
| 1.4 | No `.unwrap()` / `.expect()` in non-test code. | `src/tool/mcp.rs` validates `McpServerName` at the API boundary and returns a typed error instead of panicking. |
| 1.5 | Structured tool-result payloads are not silently dropped. | Removed the unused `structured` field and `ToolResult::json` constructor from `src/tool/executor.rs`. |
| 1.6 | MCP connection names are validated before use. | `McpConnection` stores `McpServerName` and validates in `connect_stdio`/`connect_http`. |
| 1.7 | Provider context defaults must not silently break cancellation. | `ProviderContext::default()` was removed. |
| 1.8 | Span setup/teardown flushes and does not leak providers. | `src/otel/mod.rs` registers the provider globally, sets the W3C propagator, and calls `force_flush()` before `shutdown()`. |
| 1.9 | Child contexts preserve correlation IDs. | `ProviderContext::child()` now inherits `self.correlation_id` and creates a child cancellation token. |

---

## Stage 2 resolution log

The following P1 items from the initial audit were fixed in Stage 2:

| # | Check | Resolution |
|---|---|------------|
| 2.1 | Public enums that may grow are `#[non_exhaustive]`. | `AgentDriverError`, `ConfigError`, `TaskPoolError`, `SessionError`, `AgentLoopError`, `OtelError`, and `ProviderKind` are now `#[non_exhaustive]`; `ProviderConfig` and all public model/reasoning/keepalive enums are also `#[non_exhaustive]`. |
| 2.2 | Newtypes validate at construction. | `ApiKey::new` and `AwsRegion::new` reject empty strings; `ToolName` enforces a 128-character cap and uses `TryFrom`/`FromStr`. |
| 2.3 | Env-loaded enum values use the same parser as JSON. | `AnthropicModel`, `OpenAiModel`, `OllamaModel`, `OpenRouterModel`, and `BedrockModel` implement `FromStr`; `from_env` routes through `parse()`. |
| 2.5 | Illegal states are unrepresentable. | `KeepAlive::Minutes(NonZeroU32)` prevents the `0` collision with `Unload`. |
| 2.7 | Error enums are typed, not `String`. | `ProviderKind::FromStr::Err` is now `ParseProviderKindError`; `OtelError` variants replace `String` errors in the OTel module. |
| 4.3 | No duplicate method aliases. | `Message::with_content` was removed; all callers use `Message::new`. |
| 5.9 | Bedrock inference-profile requirement is validated. | `BedrockConfig::validate()` rejects cross-region models without `inference_profile`. |
| 8.1 | Silent env-parse failures. | Added `env_parse_opt` and `env_parse_or_default` in `src/config/common.rs`; all provider configs use them and surface `ConfigError::InvalidValue` on parse failure. |

---

## Stage 3 resolution log

The following P1 items from the initial audit were fixed in Stage 3:

| # | Check | Resolution |
|---|---|------------|
| 3.2 | Tool execution can be interrupted by cancellation. | `execute_tools` in `src/agent/driver.rs` now races `futures::future::join_all` against `cancellation.cancelled()` and returns `AgentLoopError::Cancelled` when the token fires. |
| 3.3 | `Arc` clones use `Arc::clone(&x)`. | `src/provider/retry.rs` tests and `src/tool/registry.rs` tests now use `Arc::clone(&x)`. |
| 3.5 | Async locks are only used when guards cross `.await`. | `src/tool/registry.rs` switched from `tokio::sync::RwLock` to `parking_lot::RwLock` (no guard crosses `.await`). |
| 3.7 | Observer lifecycle is complete on all error/cancel paths. | `AgentLoop::run` now emits `LoopComplete` before returning on `send_streaming`, `continue_streaming`, `collect_with_observer`, and `execute_tools` errors; added `LoopStopReason::LoopFailed`. |
| 3.8 | `Default` for a pooled type matches its constructor shape. | Removed `Default` for `TaskPool` (its constructor returns `Arc<Self>`). |
| 3.9 | No orphan `CancellationToken` via `Default`. | `ToolContext` still has `Default` but is no longer used by the agent loop; `execute_tools` creates the context from the loop's cancellation token. |
| 4.2 | No single-use extension traits. | Removed `TrackedSpawn` trait and replaced it with the free function `spawn_tracked` in `src/task/spawn.rs`. |
| 4.3 | No duplicate state between config and resource. | Removed `otel_tracer` from `SessionConfig`; only `Session` holds it. |
| 8.1 | Silent env-parse failures. | `env_parse_opt` now includes the original parser error in the `ConfigError::InvalidValue` reason. |

---

## Stage 4 resolution log

The following P1 provider conversion, parser, and tool-schema items were fixed in Stage 4:

| # | Check | Resolution |
|---|---|------------|
| 3.9 | No orphan `CancellationToken` via `Default`. | `ToolContext::default()` was removed; callers must provide a context with an explicit cancellation token. |
| 5.2 | No speculative fallbacks for impossible failures. | Anthropic no longer falls back to temperature `1.0` or tool name `"unknown"`; OpenAI/OpenRouter/Anthropic invalid streamed tool names now return deserialization errors. |
| 5.3 | Finish reasons are mapped explicitly, not wildcarded. | OpenRouter logs unknown finish reasons instead of silently treating them as ordinary end-turn completions. |
| 5.7 | Parallel tool calls are keyed by index, not last-seen. | OpenAI and Bedrock parser state now tracks streamed tool calls by per-block/index maps instead of single current IDs. |
| 5.8 | Dead state fields are removed or assigned. | Ollama now marks final chunks completed and emits a single completed event; OpenRouter's unused tool-name state was removed. |
| 6.3 | Bedrock parser has unit tests. | Added `parse_bedrock_event` tests for tool-start tracking, input-delta routing by block index, and unknown-index errors. |
| 6.5 | Tautological assertions are replaced. | `sanitize_openai` now asserts the concrete strict empty-object schema for empty schemas. |
| Review follow-up | Missing OpenRouter tool-call indices fail loudly. | `parse_openrouter_event` now returns `StreamError::Deserialize` when `tool_call.index` is absent; a regression test covers the malformed chunk. |

Deferred items remain in the checklist for later stages: Bedrock still uses its custom AWS SDK stream loop, and OpenAI/OpenRouter parser sharing was not abstracted because the behavior fixes were smaller and easier to review.

---

## 1. Correctness & safety (P0)

These are validated by the agent audits and clippy/test runs.

| # | Check | Evidence in codebase | Suggested pattern |
|---|-------|---------------------|-------------------|
| 1.1 | **Tool-call deltas accumulate by the correct ID.** `ToolInputDelta.id` must be used to look up the pending tool, not written into a single shared slot. | `src/streaming.rs:194-198` drops `id` with `..` while `pending_tool_use` is a single `Option`. Parallel tool calls will mis-accumulate. | Store pending tool uses in `HashMap<ToolCallId, PendingToolUse>` and look up by `id`. |
| 1.2 | **No duplicate system messages in OpenAI conversion.** If both `request.system` and `Role::System` messages exist, only one system message should be sent. | `src/provider/openai.rs:80-106` and `:278-291` both push a system message. | Route `Role::System` exclusively through one path; skip it in `convert_messages` when `request.system` is set. |
| 1.3 | **Retry sleep honors cancellation.** A retry delay must not block cancellation. | `src/provider/retry.rs:113` uses `tokio::time::sleep(delay)` with no `select!`. | `tokio::select! { biased; _ = cancellation.cancelled() => return Err(ProviderError::Cancelled), _ = sleep(delay) => {} }`. |
| 1.4 | **No `.unwrap()` or `.expect()` in non-test code.** The crate denies `unwrap_used` and `expect_used` via `Cargo.toml`. | `src/tool/mcp.rs:147-150` uses `.expect(...)` on a user-controlled `McpServerName` value, turning a validation error into a panic. | Validate at the API boundary and return a typed `Result`. |
| 1.5 | **Structured tool-result payloads are not silently dropped.** If a public API offers structured success, it must survive serialization. | `src/tool/executor.rs:156-185` defines `ToolResult::Success { structured: Option<JsonValue> }`, but `src/tool/serializer.rs:76-95` destructures with `..` and discards it. | Pick a wire strategy (embed JSON envelope, drop the field, or document it as metadata-only) and test it. |
| 1.6 | **MCP connection names are validated before use.** | `src/tool/mcp.rs:49-52` stores `name: String` and later `McpServerName::new(self.name.clone()).expect(...)`. | Store `name: McpServerName` and validate in `connect_stdio`/`connect_http` with `?`. |
| 1.7 | **Provider context defaults must not silently break cancellation.** | `src/provider.rs:162-170` implements `Default` for `ProviderContext` with an orphan `CancellationToken`. | Delete `Default` or make it fail loudly; construct only via `new(...)` with a parent token. |
| 1.8 | **Span setup/s teardown flushes and does not leak providers.** | `src/otel/mod.rs:88-93` calls `shutdown()` without `force_flush()`; `init_tracer_provider` discards the previous provider. | `force_flush()` before `shutdown()`; replace providers safely with shutdown. |
| 1.9 | **Child contexts preserve correlation IDs.** | `src/provider.rs:148-154` regenerates `correlation_id` in `child()`. | Inherit `self.correlation_id`; add an explicit `new_subop()` if a fresh ID is needed. |
| 1.10 | **Avoid `assert!` on `Result` states in tests.** | ~60 occurrences across tests (e.g., `src/types/model.rs:163-191`). | Use `.unwrap()` / `.unwrap_err()` per `clippy::assertions_on_result_states`. |

---

## 2. Type modeling & domain boundaries (P1)

| # | Check | Evidence | Suggested pattern |
|---|-------|----------|-------------------|
| 2.1 | **Public enums that may grow are `#[non_exhaustive]`.** | `src/error.rs:81,98,305,318,331`; `src/config/provider.rs:20`; all `*Model` enums; `ProviderKind` (`src/provider.rs:192`). | Add `#[non_exhaustive]` and document the variant-growth policy. |
| 2.2 | **Newtypes validate at construction (parse-don't-validate).** | `ApiKey` (`src/config/common.rs:11-21`) accepts empty strings; `ToolName` (`src/types/message.rs:35-51`) has no length cap; `AwsRegion` accepts empty strings. | Add validation in `new()` and surface typed errors. |
| 2.3 | **Env-loaded enum values use the same parser as JSON.** | `AnthropicModel`/`OllamaModel`/`OpenRouterModel` become `Custom` in `from_env` even for well-known IDs (`src/config/anthropic.rs:89`, `ollama.rs:174-176`, `openrouter.rs:96-99`). | Implement `FromStr` and route `from_env` through it. |
| 2.4 | **No duplicate state between config and resource.** | `SessionConfig.otel_tracer` and `Session.otel_tracer` both hold the same value (`src/session.rs:50,102-103`). | Keep one source of truth. |
| 2.5 | **Illegal states are unrepresentable.** | `KeepAlive::Minutes(0)` collides with `Unload` (`src/config/ollama.rs:143-150`). | Use `NonZeroU32` or split variants. |
| 2.6 | **Correlated IDs are stable and not regenerated.** | `src/provider.rs:148-154` (see 1.9); `src/tool/mcp.rs` (see 1.6). | Validate and propagate, don't regenerate. |
| 2.7 | **Error enums are typed, not `String`.** | `ProviderKind::FromStr::Err = String` (`src/provider.rs:213`); `otel` functions return `Result<_, String>`. | Define small typed errors with `thiserror`. |
| 2.8 | **Custom-model validation is not duplicated 5×.** | `src/config/anthropic.rs:97-105`, `openai.rs:210-217`, `bedrock.rs:121-129`, `ollama.rs:228-236`, `openrouter.rs:123-138`. | Extract `validate_custom_id` in `common.rs`. |
| 2.9 | **Result/Option constructors match the domain reality.** | OTel span constructors return `Result<T, String>`/`Option<T>` despite the underlying API being infallible (`src/otel/instrumentation.rs`). | Return `Self` directly; switch to `Result` only when the API actually fails. |
| 2.10 | **Result-returning public functions are `#[must_use]`.** | `Session::send_streaming`, `continue_streaming`, `send`, `complete`, `execute_tool`, `process_tool_calls` (`src/session.rs:196,213,259,276,288,319`). | Add `#[must_use]` to fallible constructors and async methods whose results matter. |

---

## 3. Async, cancellation & ownership (P1)

| # | Check | Evidence | Suggested pattern |
|---|-------|----------|-------------------|
| 3.1 | **Cancellation is checked in stream loops.** Required by CLAUDE.md §5. | All providers implement this; `retry.rs` does not (see 1.3). | `tokio::select! { biased; _ = ctx.cancellation.cancelled() => return None, ... }`. |
| 3.2 | **Tool execution can be interrupted by cancellation.** | `src/agent/driver.rs:494` uses `join_all` with no `select!`. | Race `join_all` against `cancellation.cancelled()`. |
| 3.3 | **Clone of `Arc`/`Rc` uses `Arc::clone(&x)`.** | `src/provider/retry.rs:140,143,169,172,191,194`; `src/tool/registry.rs:203`; `src/agent/driver.rs:623`. | Replace `x.clone()` with `Arc::clone(&x)` for ref-counted types. |
| 3.4 | **No clone-to-satisfy-borrow-checker without comment.** | `src/agent/driver.rs:499-518` clones tool result content three times; `src/session.rs:305-309` clones content twice. | Restructure to move once or borrow; document unavoidable clones. |
| 3.5 | **Async locks are only used when guards cross `.await`.** | `src/tool/registry.rs:54-153` uses `tokio::sync::RwLock` but no guard crosses `.await`. | Prefer `std::sync::RwLock` or document why async is needed. |
| 3.6 | **Pin-projection rationale is documented.** | `src/streaming.rs:639-651` and `src/task/handle.rs:64-71` have correct but fragile comments. | Document *why* `Pin<Box<T>>: Unpin` / `JoinHandle<T>: Unpin` makes re-pinning sound. |
| 3.7 | **Observer lifecycle is complete on all paths.** | `src/agent/driver.rs:153-157` and `:267-269` use `?` before emitting `LoopComplete`. | Emit `LoopComplete` (or `LoopFailed`) on error/cancel paths. |
| 3.8 | **`Default` for a pooled type matches its constructor shape.** | `TaskPool::new()` returns `Arc<Self>` but `Default` returns `Self` (`src/task/pool.rs:198-207`). | Drop `Default` or align it with `Arc`. |
| 3.9 | **No orphan `CancellationToken` via `Default`.** | `ToolContext::default()` (`src/tool/executor.rs:49-55`) and `ProviderContext::default()` (see 1.7). | Remove `Default` or document that the token is unparented. |
| 3.10 | **Streams don't silently drop correlation IDs in `collect`.** | `src/streaming.rs:567-575` drops `correlation_id` in `collect()`. | Log it or thread it into a tracing span. |

---

## 4. API surface & module design (P1)

| # | Check | Evidence | Suggested pattern |
|---|-------|----------|-------------------|
| 4.1 | **God modules are split at ~500 non-test lines.** | `src/streaming.rs` is 843 lines (3 domains); `src/error.rs` is 721 lines; `src/session.rs` is 727 lines. | Split into `events.rs`, `response.rs`, `handle.rs`, `error/*.rs`, etc. |
| 4.2 | **No single-use extension traits.** | `src/task/spawn.rs:12-38` defines `TrackedSpawn` for exactly one type (`CorrelationId`). | Inline or replace with a free function. |
| 4.3 | **No duplicate method aliases.** | `Message::new` and `Message::with_content` are identical (`src/types/message.rs:258-260` vs. `:279-281`). | Keep one; migrate call sites. |
| 4.4 | **Trait objects are the minimum needed.** | `src/provider.rs:323-326` exports both `BoxedProvider` and `SharedProvider`; only `Arc<dyn Provider>` is used. | Drop `BoxedProvider`. |
| 4.5 | **Extension trait mirrors the base trait only when needed.** | `Provider` + `ProviderExt` (`src/provider.rs:292-303`) duplicate `complete` with `#[async_trait]`. | Keep object-safe base trait; implement extension as a native `async fn`. |
| 4.6 | **Lifetime parameters on builders are justified.** | `AgentLoop<'s>` holds `&'s Session` (`src/agent/driver.rs:71-72`), making it single-borrow and non-storable. | Use `Arc<Session>` or pass `&Session` per `run()`. |
| 4.7 | **Public statics are encapsulated.** | `PHOENIX_TRACER_PROVIDER` is `pub` (`src/otel/mod.rs:36-46`). | Make it `pub(crate)` or private. |
| 4.8 | **`Display` is lossless or explicitly lossy.** | `CorrelationId::Display` truncates to 8 chars (`src/types/correlation.rs:32-38`). | Add `to_short_string()`; make `Display` lossless. |
| 4.9 | **Public structs have protected fields where appropriate.** | `CorrelationContext` has `pub` fields (`src/types/correlation.rs:41-49`). | Use `pub(crate)` accessors. |
| 4.10 | **Feature-gated code is isolated, not scattered.** | `src/otel/mod.rs` and `instrumentation.rs` have 24+ individual `#[cfg(feature = "phoenix")]` attributes. | Wrap gated code in a single inner module or file. |

---

## 5. Provider conversion & error mapping (P1)

| # | Check | Evidence | Suggested pattern |
|---|-------|----------|-------------------|
| 5.1 | **No string-based provider error classification.** | `src/provider/openai.rs:338-374` matches `msg.contains("401")`; `src/provider/bedrock.rs:318-338` matches service strings. | Pattern-match on typed SDK variants. |
| 5.2 | **No speculative fallbacks for impossible failures.** | Temperature `Number::from_f64` fallback to `1.0` (`src/provider/anthropic.rs:82-84`, `openrouter.rs:73-76`); tool name fallback to `"unknown"` (`src/provider/anthropic.rs:348-353`, `openai.rs:542-544`, `openrouter.rs:485-487`). | Use `.expect()` with rationale or return typed errors. |
| 5.3 | **Finish reasons are mapped explicitly, not wildcarded.** | `src/provider/openai.rs:497-505` and `openrouter.rs:440-447` use `_ => StopReason::EndTurn`. | Match each variant explicitly; log/warn on unknowns. |
| 5.4 | **Rate-limit retry hints are preserved.** | `src/provider/bedrock.rs:322-323` always sets `retry_after: None`. | Extract `retry_after` from typed SDK errors. |
| 5.5 | **Bedrock uses the shared SDK stream adapter.** | `src/provider/bedrock.rs:354-400` duplicates `buffered_sdk_stream` logic. | Route through `stream_adapter::buffered_sdk_stream`. |
| 5.6 | **OpenAI and OpenRouter share the OpenAI-compatible chunk parser.** | `src/provider/openrouter.rs:385-522` and `openai.rs:463-574` are shape-identical. | Factor `parse_openai_compatible_chunk`. |
| 5.7 | **Parallel tool calls are keyed by index, not last-seen.** | `src/provider/openai.rs:449-456` uses single `current_tool_call_id`/`current_tool_name`. | Use `HashMap<usize, ToolCallId>` / `HashMap<usize, ToolName>`. |
| 5.8 | **Dead state fields are removed or assigned.** | `state.completed` is never set in `openai.rs` (`src/provider/openai.rs:388-399`) and `ollama.rs` (`src/provider/ollama.rs:301,380,400`). | Assign on finish or delete the field. |
| 5.9 | **Bedrock inference-profile requirement is validated.** | `src/config/bedrock.rs:54,121-131` — modern Claude models need `BEDROCK_INFERENCE_PROFILE`. | Enforce in `validate()` or split model variants. |
| 5.10 | **Tool result merging is symmetric across providers.** | Anthropic does not merge consecutive `Role::Tool` messages; Bedrock does (`src/provider/bedrock.rs:blocks_compatible`). | Document or apply shared helper. |

---

## 6. Testing & observability (P1/P2)

| # | Check | Evidence | Suggested pattern |
|---|-------|----------|-------------------|
| 6.1 | **Integration tests live in `#[cfg(test)]` modules.** | `tests/agent_loop.rs`, `tests/phoenix_openinference.rs`, `tests/session.rs` have `#[test]` functions at file root. | Wrap in `mod tests { ... }` with `#[cfg(test)]`. |
| 6.2 | **No `test_` prefix in test names.** | `tests/phoenix_openinference.rs` has `test_agent_span_is_agent_kind`, etc. | Rename to `agent_span_is_agent_kind`. |
| 6.3 | **Bedrock parser has unit tests.** | `src/provider/bedrock.rs` has 0 tests for `parse_bedrock_event` per ADR-0003. | Add per-event-variant tests. |
| 6.4 | **Tests do not use `.unwrap()` when the crate denies it.** | `tests/phoenix_openinference.rs:71,88`. | Use `?` propagation or `expect` with a rationale if truly infallible. |
| 6.5 | **Tautological assertions are replaced.** | `src/tool/types.rs:286-291` asserts `is_empty() || !is_empty()`. | Assert the expected concrete value. |
| 6.6 | **Phoenix `init_phoenix` registers the global provider.** | `src/otel/mod.rs:65-85` does not call `set_tracer_provider`. | Add global registration + W3C propagator. |
| 6.7 | **Resource / service name is configured.** | `src/otel/mod.rs:77-79` builds no resource. | Add `Resource::builder().with_service_name(...).build()`. |
| 6.8 | **OTel errors are typed, not `String`.** | `src/otel/mod.rs:54-58,65-85`; `instrumentation.rs:81,181-209`. | Define `OtelError` with `thiserror`. |
| 6.9 | **Observer events carry iteration context.** | `src/agent/observer.rs:65-101` — `TextDelta`/`ToolCallStart` lack `iteration`. | Add iteration counter or sequence ID. |
| 6.10 | **Tool spans are created per tool, not just the first one.** | `src/agent/driver.rs:222-229` traces only `response.tool_uses().first()`. | Create one span per tool in `execute_tools`. |

---

## 7. Style & lints (P2)

These are fully validated by the failing clippy run.

| # | Check | Evidence | Fix |
|---|-------|----------|-----|
| 7.1 | `format!("{x}")` instead of `format!("{}", x)`. | ~15 occurrences (`src/agent/driver.rs:634-649`, `src/session.rs:681`, `src/types/correlation.rs:110`, `src/tool/registry.rs:192`, `src/provider/bedrock.rs:762-795`, `tests/agent_loop.rs:477-637`). | Inline variable. |
| 7.2 | `&str.to_owned()` instead of `&str.to_string()`. | `src/agent/driver.rs:635`; `src/provider/stream_adapter.rs:234,269,299,331,355`; `src/provider/anthropic.rs:590,661`; `src/provider/ollama.rs:605,618,632,652,690,736,751,825,853`. | Use `.to_owned()`. |
| 7.3 | Anonymous trait imports as `_`. | `src/agent/driver.rs:674,716`; `src/provider/stream_adapter.rs:220`; `src/provider/mock.rs:342`; `src/session.rs:563`; `src/tool/executor.rs:361`. | `use futures::FutureExt as _;`. |
| 7.4 | `Arc::clone(&x)` for ref-counted pointers. | See 3.3. | Replace `.clone()`. |
| 7.5 | No wildcard match arms on owned enums. | `src/provider/bedrock.rs:762,768,774,780,786,795` (tests). | Exhaustive arms. |
| 7.6 | No strict float equality in tests. | `src/provider/bedrock.rs:761,767`. | Use epsilon comparison. |
| 7.7 | No needless `collect()` before `len()`/`count()`. | `src/provider/ollama.rs:713`. | Use `.count()` on the iterator. |
| 7.8 | `Default` constructors are not redundant with `new`. | `ToolAnnotations::new()` mirrors `Default::default()` (`src/tool/definition.rs:53-98`). | Drop `new()` or derive `Default` only. |
| 7.9 | Avoid verbose `match` chains for simple option building. | `src/provider/anthropic.rs:437-447` builds an error string with 4 arms. | Use combinators or string-builder. |
| 7.10 | Avoid redundant `test_` prefix. | `tests/phoenix_openinference.rs` and `tests/agent_loop.rs` (some). | Strip prefix. |

---

## 8. Cross-cutting themes

These patterns appear in multiple modules and should be fixed with shared helpers.

1. **Silent env-parse failures.** `config/*::from_env` uses `.ok().and_then(...).unwrap_or_default()` which swallows typos. Introduce `env_parse_opt`/`env_parse_or_default` in `src/config/common.rs` that return `ConfigError::InvalidValue` on parse failure.
2. **Speculative fallbacks.** Tool-name `"unknown"`, temperature `1.0`, empty JSON object on parse failure, etc. Replace with `.expect()` + rationale or typed errors.
3. **String-based classification.** Provider error mapping and agent-loop stop-reason detection rely on string matching. Move to typed variants or a central, tested helper.
4. **Dead state fields.** `state.completed`, `current_tool_name`, `tool_call_names` are written but never read. Remove or use them.
5. **Duplicate constructors/helpers.** `Message::new`/`with_content`, `send_streaming`/`continue_streaming`, `OpenAI`/`OpenRouter` chunk parsers, `connect_stdio`/`connect_http` serve blocks. Extract helpers.
6. **Lifetime-coupled builders.** `AgentLoop<'s>` and `ProviderContext` default patterns constrain composition. Prefer `Arc`/per-call references.
7. **Scattered `#[cfg]` blocks.** OTel and provider feature gates are item-level. Use module-level gating instead.

---

## 9. Validation commands

Run these after any quality/design change:

```bash
# Formatting
cargo fmt --check

# Type check with all public features
cargo check --all-features

# Library + unit tests
cargo test --lib --all-features

# Stricter lint gate (currently failing; target is clean)
cargo clippy --all-features --lib --tests -- -D warnings

# No-feature build smoke test
cargo check --no-default-features

# Phoenix-only build
cargo check --no-default-features --features phoenix
```

The goal is to make `cargo clippy --all-features --lib --tests -- -D warnings` pass while keeping the other commands green.

---

## 10. Per-module quick reference

| Module | Highest-severity finding | Category |
|--------|---------------------------|----------|
| `src/streaming.rs` | Parallel tool deltas mis-accumulate (P0) | correctness |
| `src/provider/openai.rs` | Duplicate system messages (P0) | correctness |
| `src/provider/retry.rs` | Sleep ignores cancellation (P0) | cancellation |
| `src/tool/mcp.rs` | Empty MCP name panics in `discover_tools` (P0) | error-handling |
| `src/tool/serializer.rs` | Structured tool result dropped (P0) | serialization |
| `src/otel/mod.rs` | Global provider not registered; shutdown doesn't flush (P0) | otel |
| `src/provider.rs` | `ProviderContext::default()` orphan token (P0) | cancellation |
| `src/error.rs` | Missing `#[non_exhaustive]` on public error enums (P1) | api-surface |
| `src/config/*` | Silent env-parse failures; `#[non_exhaustive]` missing (P1) | validation/api-surface |
| `src/session.rs` | Duplicate `otel_tracer` state; duplicated streaming methods (P1) | state-modeling |
| `src/agent/driver.rs` | `join_all` ignores cancellation; observer lifecycle incomplete (P1) | async/observer |
| `src/task/*` | `Default`/`new` asymmetry; single-use trait (P1) | api-surface |
| `src/tool/registry.rs` | `tokio::sync::RwLock` where `std` suffices (P1) | locking |
| `src/tool/types.rs` | `sanitize_openai` silently resets to empty schema (P1) | error-handling |
| All test files | `#[test]` outside `#[cfg(test)]` modules; `test_` prefixes (P2) | style/tests |

---

## 11. How to keep this checklist current

- After each ADR or major refactor, review whether new categories belong here.
- When a clippy lint is promoted from `allow` to `deny` in `Cargo.toml`, add a corresponding row.
- When a finding is fixed, mark it as resolved in the fix plan and remove the row from the active checklist (or move it to a "resolved" appendix).

<!-- vale local.EmDashUsage = YES -->

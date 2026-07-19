# Rust Quality & Design Fix Plan

Plan for fixing the findings documented in `docs/rust-quality-audit-checklist.md`.

---

## Scope

- **Core problem:** The `agent-driver-rs` codebase has accumulated quality and design debt across all modules, including P0 correctness bugs, P1 type/API modeling issues, and P2 style/test debt. The plan sequences fixes so each stage is reviewable and testable without destabilizing the next.
- **Audience:** Maintainers and future agents working on `agent-driver-rs`.
- **Success criteria:**
  1. All P0 findings are fixed and covered by regression tests.
  2. `cargo test --lib --all-features` remains green.
  3. `cargo clippy --all-features --lib --tests -- -D warnings` passes at the end of the final stage.
  4. Public API changes are documented and reviewed.
- **Non-goals:** This plan fixes existing code only. It excludes expanding the feature set and adding new providers; a full rewrite is also out of scope.

---

## Verification

- **Smoke test:** After each stage, `cargo check --all-features` and `cargo test --lib --all-features` must pass.
- **Deterministic checks:**
  - `cargo fmt --check`
  - `cargo clippy --all-features --lib --tests -- -D warnings` (target: clean by end of Stage 5)
- **Regression tests:** Each P0 fix must include at least one new unit or integration test that fails before the fix and passes after.
- **Manual verification:** For OTel/Phoenix changes, run `cargo test --features phoenix --test phoenix_openinference` and inspect span output.

---

## Blast radius

- **Files to change:** ~25 source files across `src/types/`, `src/config/`, `src/error.rs`, `src/session.rs`, `src/streaming.rs`, `src/task/`, `src/provider/`, `src/agent/`, `src/tool/`, `src/otel/`, plus `tests/agent_loop.rs`, `tests/phoenix_openinference.rs`, `tests/session.rs`.
- **Existing building blocks:** `src/config/common.rs` (env helpers), `src/provider/stream_adapter.rs` (shared stream adapters), `src/error.rs` (thiserror enums), `src/tool/serializer.rs` (wire serialization).
- **Test coverage gaps:** Bedrock parser has 0 tests; integration tests are outside `#[cfg(test)]` modules; several P0 paths lack regression tests.
- **Public API changes:** `#[non_exhaustive]` additions, `ToolResult` structured field behavior, `AgentLoop` lifetime, `ProviderContext` default removal, OTel error types, `ApiKey`/`AwsRegion` validation.

---

## Implementation stages

### Stage 1 — P0 correctness fixes

Fix the correctness and safety bugs first. These are the highest-risk changes and must be reviewed carefully.

- **Changes:**
  - `src/streaming.rs`: Accumulate `ToolInputDelta` by `ToolCallId` using `HashMap<ToolCallId, PendingToolUse>`; add regression test for interleaved tool deltas.
  - `src/provider/openai.rs`: Route `Role::System` through `request.system` only; prevent duplicate system messages; add regression test.
  - `src/provider/retry.rs`: Make retry sleep cancellation-aware via `tokio::select!`; add regression test.
  - `src/tool/mcp.rs`: Validate `McpServerName` in `connect_stdio`/`connect_http`; remove `.expect()` in `discover_tools`; add regression test for empty name.
  - `src/tool/executor.rs`: Remove the unused `ToolResult::Success.structured` field and `ToolResult::json` constructor so the API does not silently drop structured data; keep text-only serialization.
  - `src/otel/mod.rs`: Register global tracer provider with `set_tracer_provider`; call `force_flush()` before `shutdown()`; handle provider replacement safely.
  - `src/provider.rs`: Remove `Default` for `ProviderContext` or make it fail loudly; ensure `child()` preserves `correlation_id`.
- **Gates:** S → A → U 🛑
  - **Gate S:** `cargo fmt --check`, `cargo check --all-features`, `cargo test --lib --all-features`, new regression tests pass.
  - **Gate A:** Dispatch a fresh rust-review agent to review the P0 diff in isolation; it must find no new issues or sign off.
  - **Gate U:** Present the P0 diff and regression tests, summarize the risk, and await user approval before Stage 2.

---

### Stage 2 — Type modeling and public API hardening

Add `#[non_exhaustive]`, tighten validation, and unify env/JSON parsing paths. These are API-affecting changes but not behavior-affecting for existing callers.

- **Changes:**
  - `src/error.rs`: Add `#[non_exhaustive]` to `AgentDriverError`, `ConfigError`, `TaskPoolError`, `SessionError`, `AgentLoopError`; centralize `ProviderError` variant coverage.
  - `src/config/provider.rs`: Add `#[non_exhaustive]` to `ProviderConfig` and `ProviderKind`; introduce `ProviderConfigLike` trait to replace manual matches.
  - `src/config/{anthropic,openai,bedrock,ollama,openrouter}.rs`: Add `#[non_exhaustive]` to all public model/reasoning/keepalive enums; implement `FromStr` for model enums and route `from_env` through it.
  - `src/config/common.rs`: Validate `ApiKey` (non-empty), `AwsRegion` (non-empty); add `env_parse_opt`/`env_parse_or_default` helpers.
  - `src/config/{anthropic,openai,bedrock,ollama,openrouter}.rs`: Replace silent `.ok().unwrap_or_default()` env parsing with the new helpers.
  - `src/provider.rs`: Replace `ProviderKind::FromStr::Err = String` with a typed `ParseProviderKindError`.
  - `src/types/message.rs`: Remove duplicate `Message::new` or `Message::with_content`; add `ToolName` length cap and `TryFrom` impls.
  - `src/config/ollama.rs`: Constrain `KeepAlive::Minutes(0)` using `NonZeroU32`.
  - `src/config/bedrock.rs`: Validate `inference_profile` for cross-region models.
- **Gates:** S → A → U 🛑
  - **Gate S:** Same deterministic checks; verify no downstream exhaustive matches inside the crate break.
  - **Gate A:** Agent review focused on API compatibility and variant-growth policy.
  - **Gate U:** Present the API diff, highlight the new `#[non_exhaustive]` attributes, error types, and validation changes, and await approval.

---

### Stage 3 — Async, cancellation, and ownership cleanup

Address the async/cancellation races and ownership hotspots without changing the public API surface (except where Stage 2 already changed it).

- **Changes:**
  - `src/agent/driver.rs`: Race `join_all` in `execute_tools` against `cancellation.cancelled()`; emit `LoopComplete` on error/cancel paths; reduce tool-result content clones.
  - `src/session.rs`: Remove duplicate `SessionConfig.otel_tracer`; extract shared helper for `send_streaming`/`continue_streaming`; add `#[must_use]` to result-returning methods.
  - `src/task/pool.rs`: Remove `Default` or align it with `Arc<TaskPool>`; remove `accepting` redundancy or document it; decide on `abort_handle` field.
  - `src/task/spawn.rs`: Remove `TrackedSpawn` trait or replace with a free function.
  - `src/tool/registry.rs`: Switch to `std::sync::RwLock` if no guard crosses `.await`; use `Arc::clone(&tool)`.
  - `src/provider/retry.rs`: Use `Arc::clone(&attempts)` consistently.
  - `src/provider/stream_adapter.rs`, `src/provider/mock.rs`, `src/session.rs`, `src/tool/executor.rs`, `src/agent/driver.rs`: Use `FutureExt as _`, `StreamExt as _`, and `Arc::clone(&x)` where clippy flags.
  - `src/streaming.rs`: Thread `correlation_id` through `collect()` logging.
- **Gates:** S → A
  - **Gate S:** `cargo fmt --check`, `cargo check --all-features`, `cargo test --lib --all-features`, cancellation regression tests pass.
  - **Gate A:** Agent review focused on async soundness and cancellation semantics.

---

### Stage 4 — Provider conversion consolidation

Consolidate duplicated provider logic and fix conversion bugs. This stage is mostly internal refactoring with targeted behavior fixes.

- **Changes:**
  - `src/provider/bedrock.rs`: Route streaming through `stream_adapter::buffered_sdk_stream`; replace string-based AWS error classification with typed SDK variants; extract `retry_after` from `ThrottlingException`; remove dead `current_tool_name` field; add parser tests.
  - `src/provider/openai.rs` + `src/provider/openrouter.rs`: Factor a shared `parse_openai_compatible_chunk` helper; use `HashMap<usize, ToolCallId>` for parallel tool calls; map finish reasons explicitly.
  - `src/provider/anthropic.rs`: Remove speculative `"unknown"` tool-name fallback and `1.0` temperature fallback; merge consecutive tool results symmetrically with Bedrock.
  - `src/provider/ollama.rs`: Assign `state.completed` on finish; stabilize synthetic tool-call IDs; remove duplicate thinking suppression.
  - `src/tool/types.rs`: Fix `sanitize_openai` empty-schema tautology; add `ToolSchema::as_value`; consider `Hash` impl documentation.
  - `src/tool/executor.rs`: Add `ToolInput::get_f64`; fix `ToolContext::default()` orphan token; simplify `FnTool` bounds with a type alias.
- **Gates:** S → A → M
  - **Gate S:** Deterministic checks; all provider unit tests pass; new Bedrock parser tests pass.
  - **Gate A:** Agent review of the shared adapter/parser refactor.
  - **Gate M:** Run `cargo test --lib --all-features` and the provider-specific test suites; manually verify no regressions in mock-provider behavior.

---

### Stage 5 — Style, lints, and test structure

Clean up the ~130 clippy findings and normalize test layout. This stage should be mechanical and produce the cleanest diff.

- **Changes:**
  - All `tests/*.rs`: Wrap `#[test]` functions in `#[cfg(test)] mod tests { ... }`.
  - All tests: Remove `test_` prefixes; replace `assert!(result.is_ok())` with `.unwrap()` and `assert!(result.is_err())` with `.unwrap_err()`; remove `.unwrap()` in tests where `?` is possible (or use `expect` with rationale).
  - All sources: Replace `format!("{}", x)` with `format!("{x}")`; `&str.to_string()` with `&str.to_owned()`; anonymous trait imports with `as _`; `Arc::clone(&x)` where needed.
  - `src/provider/bedrock.rs` tests: Replace wildcard match arms and float equality.
  - `src/provider/ollama.rs` tests: Remove needless `collect()` and `.to_string()` on model strings.
  - `src/types/model.rs`, `src/config/anthropic.rs`, `src/config/ollama.rs`, `src/tool/types.rs`, `src/tool/executor.rs`, `src/types/message.rs`: Fix `assertions_on_result_states`.
- **Gates:** S → A
  - **Gate S:** `cargo fmt --check`, `cargo check --all-features`, `cargo test --lib --all-features`, and `cargo clippy --all-features --lib --tests -- -D warnings` all pass.
  - **Gate A:** Agent review to confirm the cleanup is purely mechanical and contains no logic changes.

---

### Stage 6 — Module splits and documentation

Split god modules and isolate feature-gated code. This is the lowest-risk structural cleanup.

- **Changes:**
  - `src/streaming.rs`: Split into `streaming/events.rs`, `streaming/response.rs`, `streaming/fallback.rs`, `streaming/handle.rs`; re-export from `streaming.rs`.
  - `src/otel/`: Wrap Phoenix-gated code in a single `otel::phoenix` module; reduce `#[cfg(feature = "phoenix")]` attributes.
  - `src/error.rs`: Consider splitting into `error/{provider,stream,tool,session,config,agent}.rs` if size justifies it.
  - `src/agent/driver.rs`: Extract Phoenix span recording into a single helper; reduce `#[cfg]` duplication.
  - Update `docs/ARCHITECTURE.md` and `docs/adr/README.md` if module structure changes.
- **Gates:** S → A → U 🛑
  - **Gate S:** All deterministic checks pass; `cargo build` with and without `--features phoenix` succeeds.
  - **Gate A:** Agent review of module boundaries and re-export surface.
  - **Gate U:** Present the structural diff; await approval because it affects navigation and imports.

---

## Rollback

- If a stage fails Gate S: fix the stage in place; do not proceed.
- If a stage fails Gate A: address the agent's findings or split the stage further.
- If a stage fails Gate U or M: revert the stage's commits and return to the previous stage's baseline. The stages are ordered so that each later stage depends only on the prior stages' approved state.
- Keep each stage's changes in a separate branch or commit group so rollback is a clean `git revert`.

---

## Dependencies between stages

```text
Stage 1 (P0 correctness)
  │
  ▼
Stage 2 (API hardening) ── depends on Stage 1 for error-type stability
  │
  ▼
Stage 3 (async/ownership) ── depends on Stage 2 for `#[must_use]` and `#[non_exhaustive]` boundaries
  │
  ▼
Stage 4 (provider consolidation) ── depends on Stage 3 for cancellation/adapter contracts
  │
  ▼
Stage 5 (style/tests) ── depends on Stages 1-4 being merged so clippy fixes don't conflict with logic changes
  │
  ▼
Stage 6 (module splits) ── depends on Stage 5 so the final module shape is lint-clean
```

---

## Prioritization rationale

- **P0 first:** Silent data loss, panics, and cancellation bugs are user-facing and cannot wait.
- **API hardening second:** `#[non_exhaustive]` and validation changes are breaking-adjacent; doing them early avoids shipping more public API surface that later needs the same treatment.
- **Async/ownership third:** These changes touch runtime behavior but build on the stabilized error and API types.
- **Provider consolidation fourth:** Internal refactoring is safest once the core P0/P1 fixes are in place and tested.
- **Style/test cleanup fifth:** Mechanical changes are done after all logic changes to avoid rebase churn and to finally make clippy clean.
- **Module splits sixth:** Structural changes are the lowest risk and are easiest to verify once everything else is clean.

---

## Review checkpoints for the user

- **After Stage 1:** Review P0 fixes and regression tests.
- **After Stage 2:** Review public API changes (`#[non_exhaustive]`, new error types, validation).
- **After Stage 6:** Review module structure and documentation updates.

## Stage 1 status

- **Completed:** All P0 correctness fixes are implemented and tested.
- **Gate S:** Passed. `cargo fmt --check`, `cargo check --all-features`, and `cargo test --all-features` pass (165 lib tests + 35 integration/doc tests). `cargo clippy --all-features --lib --tests -- -D warnings` still fails with pre-existing lint/style findings; no new failures were introduced by Stage 1 changes.
- **Gate A:** Passed. A rust-reviewer agent reviewed the async/tokio changes and found no must-fix issues. Two nits from the agent review were addressed: duplicate `#[cfg]` blocks in `src/agent/driver.rs` were collapsed, and a regression test for the defensive duplicate-tool-use-start path was added in `src/streaming.rs`.
- **Gate U:** Approved. User signed off on the Stage 1 P0 diff and regression tests.

## Stage 2 status

- **Completed:** All planned type modeling and public API hardening changes are implemented and tested.
- **Gate S:** Passed. `cargo fmt --check`, `cargo check --all-features`, and `cargo test --all-features` pass (178 lib tests + 35 integration/doc tests). `cargo clippy --all-features --lib --tests -- -D warnings` still fails with pre-existing lint/style findings; no new failures were introduced by Stage 2 changes.
- **Gate A:** Passed. A rust-reviewer agent reviewed the API changes and identified one Stage 2-introduced clippy deny (`redundant_closure_for_method_calls` in `src/config/common.rs`). That was fixed, and the reviewer nits for typed error propagation were also addressed: `ParseProviderKindError` now converts to `ConfigError`, `env_parse_opt` preserves the parser error message, and a `ProviderKind` parse-error unit test was added.
- **Gate U:** Approved. User signed off on the Stage 2 public API changes.

## Stage 3 status

- **Completed:** All async, cancellation, and ownership cleanup changes are implemented and tested.
- **Gate S:** Passed. `cargo fmt --check`, `cargo check --all-features`, and `cargo test --all-features` pass (178 lib tests + 35 integration/doc tests). `cargo clippy --all-features --lib -- -D warnings` has no Stage 3-introduced errors; full `cargo clippy --all-features --lib --tests -- -D warnings` still fails on pre-existing lint/style debt scheduled for Stage 5.
- **Gate A:** Passed. A rust-reviewer agent found three clippy/doc issues caused by Stage 3: redundant `#[must_use]` on `Result`-returning methods, `unused_async` on `ToolRegistry` after switching to `parking_lot::RwLock`, and a stale doc comment claiming tokio's `RwLock`. All three were fixed and `cargo clippy --all-features --lib -- -D warnings` now passes.
- **Gate U:** Not required for Stage 3 per the original plan.

## Stage 4 status

- **Completed:** Provider conversion cleanup is implemented and tested. The stage kept the changes surgical: OpenAI/OpenRouter and Bedrock parser state now track parallel tool calls by stream block/index, speculative tool-name and temperature fallbacks were removed, Ollama completion/thinking handling was tightened, and the tool schema/executor cleanup is complete.
- **Deferred from the original Stage 4 sketch:** Bedrock still uses its custom AWS SDK `.recv()` loop because the generated Bedrock event stream does not directly match the shared SDK stream adapter shape. OpenAI/OpenRouter parser sharing was limited to behavior cleanup instead of introducing a new shared parser abstraction.
- **Gate S:** Passed. `cargo fmt --check`, `cargo check --all-features`, `cargo test --all-features`, and `cargo clippy --all-features --lib -- -D warnings` pass (182 lib tests + 55 integration/doc checks). New Bedrock parser tests cover tool-start tracking, input-delta routing by block index, and unknown-index errors.
- **Gate A:** Passed. A rust-reviewer agent found no blocking issues. One low-risk OpenRouter parser finding was addressed: missing `tool_call.index` now returns a deserialization error instead of silently collapsing to index `0`.
- **Gate M:** Passed. Manual smoke verification from `docs/manual-testing.md` passed: Anthropic, OpenRouter, and Bedrock basic chat streamed responses successfully; Bedrock MCP single-tool and multi-tool calls completed with `loop done: end_turn` and no `ValidationException`.

## Stage 5 status

- **Completed:** Style, lint, and test-structure cleanup is implemented. Integration tests now live under `#[cfg(test)] mod tests`; redundant `test_` prefixes, uninlined format strings, `&str.to_string()` calls, result-state assertions, ref-count clone style, unreadable numeric literals, and Bedrock test wildcard/float comparisons were cleaned up.
- **Gate S:** Passed. `cargo fmt --check`, `cargo check --all-features`, `cargo test --all-features`, and `cargo clippy --all-features --lib --tests -- -D warnings` all pass.
- **Gate A:** Passed. A rust-reviewer agent found no blocking issues. Its actionable consistency notes were fixed: one remaining `&str.to_string()` in `src/streaming.rs` was changed to `.to_owned()`, and two redundant Anthropic parser-test `Option::is_some()` assertions were removed.

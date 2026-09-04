# Changelog

Notable project changes are recorded here for humans and cold agents. This file
is intentionally high level; implementation detail stays in ADRs, PRs, and
commit history.

## Unreleased

- Bedrock extended thinking support: `BedrockConfig.thinking`
  (`BEDROCK_THINKING_BUDGET`) sends the `additionalModelRequestFields`
  thinking block for Claude-family models; `reasoningContent` deltas from
  `converse_stream` surface as `ThinkingDelta` (text) and `SignatureDelta`
  (signature) instead of being dropped; replayed `Thinking` blocks take the
  `ReasoningContent` wire shape when thinking is configured and keep the
  `<thinking>` text flattening otherwise. The reasoning signature is
  emitted on the stream but not yet retained on the message type.
- Toolchain modernization: MSRV 1.91.1 declared, `rust-toolchain.toml` pins
  stable, migrated to edition 2024.
- Dependency modernization: `async-openai` 0.28→0.41 (removes transitive
  `backoff`), AWS SDK 1.124→1.135 (drops legacy rustls 0.21 stack), `thiserror`
  1→2, `backoff` direct dependency replaced with hand-rolled retry logic.
- OpenTelemetry 0.27→0.32 (`SdkTracerProvider` rename, `with_batch_exporter` no
  longer takes runtime, `SimpleSpanProcessor` now generic, `InMemorySpanExporter`
  moved to `trace` module). Clears the last cargo-deny RUSTSEC exception
  (async-std). cargo-deny exceptions now at 0.
- cargo-deny RUSTSEC exceptions reduced from 6 to 0.
- `ollama-rs` bumped 0.3.3→0.3.5 (fixed `Ollama::new` deprecation).
- Added ADR-0007 for codex-style lint and tooling adoption.
- Clarified the documented first-run path: Anthropic is the `.env.example`
  default; Bedrock is a supported production smoke path that requires AWS
  credentials and `BEDROCK_INFERENCE_PROFILE`.
- Added a README run/test cheat sheet and marked `TODO.md` as the canonical
  active roadmap.
- Added a minimal CI compile workflow for the public feature set.
- Replaced the private `mcp-openai-bridge` git dependency behind `schema-sanitize`
  with a local port of the two sanitizer functions from `mezmo/aura` (Apache-2.0,
  attribution in `src/tool/schema_sanitize.rs` and `NOTICE`); every `cargo`
  command now works without SSH access, and CI checks `schema-sanitize`.
- Prepared the repository for public release: licence files, environment-
  specific values moved to env vars, board state moved out of the repo, and
  the git history rewritten to drop the board and internal identifiers.

## 2026-06-20

- Ran a documentation bus-test audit (16/24 using the `docs-bustest` skill's
  24-item checklist) and a rust-design review.
- Compared the existing rust-seed lint baseline with openai/codex `codex-rs`
  lint tooling (`clippy.toml`, `deny.toml`, and cargo-shear metadata).

## 2026-02-09

- Added a standalone benchmark harness in `bench/` covering cold start,
  time-to-first-token, tool round trip, MCP discovery, and MCP round trip.
  Run it yourself; published figures are not carried here, since results
  depend heavily on provider, network, and machine.

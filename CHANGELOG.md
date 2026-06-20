# Changelog

Notable project changes are recorded here for humans and cold agents. This file
is intentionally high level; implementation detail stays in ADRs, PRs, and
commit history.

## Unreleased

- Added ADR-0007 for codex-style lint and tooling adoption.
- Clarified the documented first-run path: Anthropic is the `.env.example`
  default; Bedrock is a supported production smoke path that requires AWS
  credentials and `BEDROCK_INFERENCE_PROFILE`.
- Added a README run/test cheat sheet and marked `TODO.md` as the canonical
  active roadmap.
- Added a minimal CI compile workflow for the public-dependency feature set. It
  excludes `schema-sanitize` until the private git dependency is pinned and
  CI-accessible.

## 2026-06-20

- Ran a documentation bus-test audit (16/24 using the `docs-bustest` skill's
  24-item checklist) and a rust-design review.
- Compared the existing rust-seed lint baseline with openai/codex `codex-rs`
  lint tooling (`clippy.toml`, `deny.toml`, and cargo-shear metadata).

## 2026-02-09

- Recorded benchmark results comparing agent-driver-rs against rig.rs for
  cold start, time-to-first-token, tool round trip, MCP discovery, and MCP
  round trip. See `TODO.md` for current investigation items.

# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for agent-driver-rs.

## What is an ADR?

An ADR captures a significant architectural decision along with its context and consequences. They serve as a decision log so future contributors (including future-you) can understand *why* things are the way they are.

## Format

Each ADR follows this structure:

```
# ADR-NNNN: Title

**Status:** Proposed | Accepted | Superseded by ADR-NNNN | Deprecated
**Date:** YYYY-MM-DD
**Context tags:** [tool-system] [mcp] [provider] [streaming] etc.

## Context
What is the issue? What forces are at play?

## Decision
What is the change being proposed or enacted?

## Consequences
What becomes easier or harder as a result?
```

## Numbering

- `0000` — Reserved for this README
- `0001-0099` — Foundational/architectural decisions
- `0100-0199` — Provider-related decisions
- `0200-0299` — Tool system decisions
- `0300-0399` — Session/lifecycle decisions

## Index

| ADR | Title | Status | Tags | Impl Order |
|-----|-------|--------|------|------------|
| [0001](0001-tool-system-and-mcp-integration.md) | Tool System & MCP Integration | Accepted | tool-system, mcp | — (done) |
| [0002](0002-thinking-reasoning-support.md) | Thinking & Reasoning Support Across Providers | Proposed | thinking, reasoning, provider, streaming | **Wave 1** (correctness bug) |
| [0003](0003-unit-test-coverage.md) | Unit Test Coverage Across Provider Branches | Proposed | testing, provider, streaming | **Wave 2** (test foundation) |
| [0004](0004-live-provider-integration-tests.md) | Live Provider Integration Tests | Proposed | testing, provider, integration, ci | **Wave 3** (live validation) |
| [0005](0005-prompt-caching-support.md) | Prompt Caching Support Across Providers | Proposed | caching, provider, performance, telemetry | **Wave 4** (performance) |
| [0006](0006-multi-agent-trace-composition.md) | Multi-Agent Trace Composition with OpenInference | Proposed | otel, phoenix, multi-agent, tracing, architecture | **Wave 5** (observability) |
| [0007](0007-codex-style-lint-tooling-adoption.md) | Codex-Style Lint & Tooling Adoption | Proposed | linting, tooling, quality, ci | **Parallel track** (Tier 1 iterative) |

## Implementation Order

ADRs should be implemented in wave order. Each wave depends on the previous:

```
Wave 1: ADR-0002 (Thinking)     → fixes correctness bug, touches ContentBlock enum
  ↓
Wave 2: ADR-0003 (Unit Tests)   → tests the thinking fixes, must precede live tests
  ↓
Wave 3: ADR-0004 (Live Tests)   → validates thinking fix end-to-end across providers
  ↓
Wave 4: ADR-0005 (Caching)      → extends TokenUsage, needed before multi-agent traces
  ↓
Wave 5: ADR-0006 (Multi-Agent)  → depends on OTel infra + TokenUsage + thinking
```

ADR-0007 runs as a **parallel track** alongside the waves. It does not block
any wave, but its Layer B (`expect_used = deny`) is gated by the Tier 2
rust-design task T2.4 (const-constructible `MaxTokens`), which folds into
Wave 1's `ContentBlock` type work.

See [roadmap](../internal/agent-driver-roadmap.md) for phase mapping.

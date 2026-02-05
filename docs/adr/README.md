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

| ADR | Title | Status | Tags |
|-----|-------|--------|------|
| [0001](0001-tool-system-and-mcp-integration.md) | Tool System & MCP Integration | Proposed | tool-system, mcp |

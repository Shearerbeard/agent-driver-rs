# ADR-0009: Structured Output via Response Format with Forced-Tool Fallback

**Status:** Proposed
**Date:** 2026-10-05
**Context tags:** [provider] [structured-output] [tools] [streaming]

---

## Context

Consumers need typed, validated output from completions ("answer as JSON
matching this schema"). Provider support is uneven:

| Provider | Native mechanism |
|----------|------------------|
| OpenAI | `response_format: { type: "json_schema", ... }` with strict mode |
| OpenRouter | Pass-through to the underlying provider |
| Ollama | `format` parameter accepting a JSON schema |
| Anthropic | None; the established pattern is a forced tool call whose input is the payload |
| Bedrock | None (Converse API); same forced-tool pattern |

Current state:

- `CompletionConfig` has no response-format field (`src/provider.rs`).
- `OpenAiConfig.structured_output: bool` is dead config: written in
  constructors, never read when building requests. This is the same disease
  Wave 1 (ADR-0002) flags for `ReasoningConfig`.
- The `schema-sanitize` feature already rewrites JSON Schemas to OpenAI
  strict-mode shape (all properties required and nullable,
  `additionalProperties: false`), currently applied only to tool schemas.
- `tool_choice` is not configurable; OpenRouter hardcodes `"auto"`
  (`src/provider/openrouter.rs`). The forced-tool fallback depends on it.

### Decision drivers

- The API MUST be provider-agnostic: one `ResponseFormat` type on
  `CompletionConfig`.
- Streaming and collection types MUST NOT change; structured output arrives as
  text and is parsed after collection.
- Native mechanisms SHOULD be preferred where they exist: better adherence,
  and no tool slot consumed.
- Anthropic and Bedrock MUST have a working path: an "unsupported" error is
  not acceptable for primary providers.
- Fallback parse failures SHOULD feed back to the model as tool error results
  so it can self-correct under `continue_on_tool_error`.
- The dead `structured_output` config MUST be wired or removed as part of this
  work.

## Decision

### Considered options

1. **Native-only; error on unsupported providers** - rejected: Anthropic and
   Bedrock are primary providers, and the feature would be roulette.
2. **Forced tool call everywhere** - rejected: strictly worse adherence than
   native JSON-schema mode on OpenAI, and burns a tool slot. Kept as the
   fallback leg.
3. **Native where available, forced-tool fallback elsewhere** - chosen.

### Chosen design

- Add `ResponseFormat` (`Json` or `JsonSchema { name, schema, strict }`) to
  `CompletionConfig`.
- Add a structured-output capability to `ProviderCapabilities` (native /
  forced-tool / none).
- Provider mapping: OpenAI `response_format`; OpenRouter pass-through; Ollama
  `format`; Anthropic and Bedrock synthesize a reserved output tool, force it
  via `tool_choice`, and parse the tool input as the payload.
- Configurable `tool_choice` (auto / none / required / named tool) is a
  prerequisite mini-feature that the fallback shares with general tool-use
  control.
- In fallback mode, a payload parse failure becomes a tool error result,
  letting the model retry under `continue_on_tool_error`. In native mode it
  surfaces as a typed error variant.

**This ADR records a constraint, not scheduled work.** The drivers bind
adjacent changes and are tracked in `TODO.md` under "Future Flexibility".

## Consequences

### What becomes easier

- Typed extraction over any provider; parity with rust-genai's tool-choice and
  structured-output support.
- `schema-sanitize` gains a second customer, which justifies maintaining it.

### What becomes harder

- `tool_choice` plumbing across five providers.
- Two behavioral paths per provider to test (native and fallback).
- The fallback pollutes the tool namespace: the reserved output tool needs
  collision handling against user-registered tools.
- Strict-schema edge cases (`const`, nullable patterns, unsupported keywords)
  differ per provider; rust-genai had to normalize these for Gemini, and
  Bedrock will need the same here.

### Open questions

- Should typed parsing (`response.parse::<T>()`) live in this crate or stay
  the caller's job? Keeping it caller-side preserves the "text is text"
  invariant.
- How does the forced-tool fallback interact with `fallback_tool_parsing`?

## Links

- ADR-0002 (thinking) owns the analogous dead-config cleanup for
  `ReasoningConfig`.
- `TODO.md` section "Future Flexibility" tracks the binding constraints.

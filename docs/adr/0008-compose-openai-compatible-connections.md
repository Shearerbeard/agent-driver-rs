# ADR-0008: Compose OpenAI-compatible connection profiles

**Status:** Proposed
**Date:** 2026-10-03
**Decider:** Repository owner
**Context tags:** [provider] [streaming] [reasoning] [config]

## Context

The library exposes a small streaming provider interface. Its OpenAI provider
already supports Chat Completions and, on PR #4, native Responses. Compatible
endpoints may use the Chat Completions transport while placing reasoning text
in different fields. The current custom-model path guesses capabilities from
an ID and can accept a temperature that a reasoning model rejects. A model's
streaming support can also differ by API surface.

[ADR-0002](0002-thinking-reasoning-support.md) describes the provider mechanics and left the OpenAI Responses mode
question open. PR #4 chose a mode on the existing provider. The remaining
decision is how to configure compatible endpoints without turning each vendor
or model into a separate provider implementation.

## Decision drivers

- The public provider interface MUST stay streaming-first and usable without
  vendor-specific model enums.
- Invalid combinations MUST fail before an API request; custom model
  capabilities MUST NOT be inferred from the model ID.
- Configurable reasoning output fields SHOULD share the Chat Completions
  connection; revisit this if a server needs a distinct transport contract.
- A provider-key test MUST NOT be required by offline CI.

## Considered options

1. **A provider per compatible vendor.** Each would repeat the Chat Completions
   transport and message conversion for a different endpoint.
2. **Infer capabilities and reasoning fields from model ID prefixes.** This
   needs less configuration at first, but aliases and local model names can
   invalidate the inference silently.
3. **Compose a small connection profile.** Select Chat Completions or Responses
   and, where needed, a declared reasoning field and model capabilities. A
   compatible host supplies its endpoint and explicit optional settings.

## Decision

Choose option 3. The OpenAI provider owns a validated connection profile;
`ProviderConfig` remains the provider selector. Chat's reasoning field is an
output convention, not a promise that a server understands a particular
reasoning-effort request or accepts replayable reasoning history. Native
Responses retains its encrypted reasoning-item replay. The provider's model
pin is authoritative, and a mismatched request model fails rather than being
ignored. [PR 1's spec](../openai-connection-profile-spec.md) owns config
and request validation. [PR 2's spec](../responses-stream-correctness-spec.md)
owns ordered Responses collection and terminal reason classification.

## Consequences

Compatible endpoints can share a small interface without pretending that
their optional parameters are identical. Custom-model users must declare
capabilities, adding configuration work. The existing public `OpenAiConfig`
struct still permits invalid field combinations until provider construction;
that seam validates once, while request-specific options are checked at the
request seam. Changing that struct's fields can break downstream literals
before 0.1.0. Unknown request-side reasoning dialects and third-party
reasoning replay remain unsupported; the specs state the tests required
before either is claimed.

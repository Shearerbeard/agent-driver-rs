# OpenAI-compatible connection profiles: PR 1 specification

**Base:** PR #4 (`card/A23`) at `55acdd50ccdadb83c18933249799db403679c693`.
**Follow-up:** [Responses stream correctness](responses-stream-correctness-spec.md).
**Decision:** [ADR-0008](adr/0008-compose-openai-compatible-connections.md).

## Purpose and scope

Expose a small set of configurable inference connection styles: Chat Completions
and Responses. A custom endpoint can select optional reasoning conventions
without acquiring a new provider variant or a vendor-name heuristic. The
library remains streaming-first; the provider pin determines the connection
and model, while the session controls message history.

This PR changes OpenAI configuration, request validation, and a minimal
Responses terminal rule. It does not add hosted tools, claim cross-provider
reasoning replay, implement every compatible endpoint's request dialect, or
rework shared stream collection. OpenRouter remains a separate provider; the
OpenAI-compatible base URL may still point at an OpenRouter-compatible server.

## Type and seam contract

- Keep `ProviderConfig` and the `Provider` trait unchanged. `OpenAiApi` selects
  Chat Completions or Responses. Do not introduce a variant per vendor or a bag
  of untyped JSON options.
- Add one narrow OpenAI connection profile with private fields and a fallible
  constructor, composed once in `OpenAiProvider::new`. It contains the surface,
  the selected model's capabilities, and Chat's selected reasoning output
  field where applicable. Request builders consume the validated profile.
  Keep `OpenAiConfig`'s existing public fields for this release; reject invalid
  combinations at provider construction and request construction. Document
  that public struct literals are not proof of validation.
- A well-known model gets explicit per-surface streaming, temperature, and
  reasoning capabilities. A `Custom` model needs an explicit capability
  declaration; do not classify `gpt-5*` or `o*` strings by prefix. Capability
  declarations are properties of a selected endpoint/model, not brand IDs.
  Expose a small public constructor/configuration route for programmatic custom
  models, with serde parsing and validation that cannot silently invent a
  profile. Adding a field to `OpenAiConfig` breaks existing struct literals:
  update repository call sites and document the pre-0.1.0 source break.
- `config.model` is authoritative for the provider pin. `request.model` must
  equal it; a mismatch is `ProviderError::InvalidRequest`, not a silent
  override. Existing `SessionBuilder` constructs matching requests. Request
  `max_tokens` and `temperature` remain request-level; reject a supplied
  temperature when the selected model and surface do not allow it. Do not
  quietly discard a configured or requested option.
- Chat's output convention can be `none`, `reasoning_content`, or `reasoning`.
  This selects the preferred output field, not an unverified request payload.
  Preserve the tolerant BYOT parser for compatible servers: an unselected
  spelling still appears as thinking; if both are populated, prefer the
  selected spelling (use the existing `reasoning_content` preference for
  `none`). Unknown extension fields remain tolerated. Native Responses uses
  its existing `ReasoningConfig` and encrypted item replay; no new encrypted
  content type is needed because PR #4 already introduces it.
- Only send request-side reasoning effort where its wire parameter has been
  verified for the selected surface. Native Responses already sends effort.
  Chat's present `ReasoningConfig` is not sent; do not pretend a third-party
  server implements OpenAI's effort parameter. For custom Chat models, reject
  an explicitly requested effort until a verified request dialect is supplied.
  Preserve default Chat behavior for known models until that dialect is
  verified. Reconcile `from_env` auto-created reasoning config with this rule
  rather than making the default Chat path fail at startup.
- Allow `o3` and `o3-mini` to stream on Responses, per current OpenAI model
  documentation. Retain the existing Chat Completions restriction pending
  separate verification. Calculate `ProviderInfo.capabilities.streaming`
  from the selected surface, not a global model flag.
- `OpenAiProvider::new` validates the profile even when called without the
  CLI. SDK environment defaults must not override an explicitly configured
  connection. Document that a compatible base URL needs the endpoint path
  prefix (usually `/v1`); reject malformed URLs without guessing a server's
  path convention.

| Type | Rule it owns | Invalid state it forbids |
|---|---|---|
| Connection profile | One selected wire surface with its options | Chat-only output field on Responses |
| Custom capabilities | Explicit claims for an endpoint/model pair | Capability inferred from an ID prefix |
| Chat reasoning output | One preferred output field | Conflicting output-field preferences |

The public-facing custom-capability type must hold validated values. The
profile keeps the cross-field invariant private; `OpenAiConfig` remains a
deserializable input that `OpenAiProvider::new` parses into that profile.

## Environment and public documentation

`OPENAI_API` remains `chat-completions` (default) or `responses`.
`OPENAI_CHAT_REASONING_OUTPUT` is `none` (default), `reasoning_content`, or
`reasoning`. A custom model must set all three explicit capability variables:
`OPENAI_CUSTOM_TEMPERATURE`, `OPENAI_CUSTOM_REASONING`, and
`OPENAI_CUSTOM_STREAMING`, each strictly `true` or `false`. Programmatic callers
provide the equivalent typed declaration. No vendor name selects values
implicitly. Reject unknown values and incomplete custom declarations.

Update `.env.example`, `docs/PROVIDERS.md`, relevant README examples,
`AGENTS.md`/`CLAUDE.md`'s o3 caveat, and ADR-0002's unresolved OpenAI mode
question. Public prose must explain what the interface supports without
describing this as Aura compatibility.

## Terminal rule owned by this PR

On the Responses path, `buffered_sdk_stream`'s end callback emits a stream
error unless an actual `response.completed` or `response.incomplete` was seen.
An SDK transport error, `response.failed`, or `response.error` marks failure
and cannot later emit `Completed`. Return a typed stream error before EOF so
both existing collectors (which accept bare EOF) fail. Keep Chat Completions'
existing EOF policy unchanged. PR 2 classifies incomplete reasons and fixes
block ordering; PR 1 must not label a failed or truncated response successful.

## Verification and acceptance

- Before coding, map each new public type to its invariant and review the
  complete type surface. `clippy::todo` is denied here, so implement types and
  bodies in an uncommitted diff rather than landing a `todo!()` skeleton.
- Write full request-body fixtures for Responses, Chat with both reasoning
  output spellings, and ordinary Chat. Exercise both-spelling preference,
  unselected spelling tolerance, unknown extensions, usage-only chunks, and
  zero reasoning. Test model match/mismatch; custom capability omissions and
  contradictions; o3/o3-mini surface differences; temperature and effort
  rejection; malformed base URL; and truncated/failed Responses streams.
- Build the standalone `bench` manifest: root `make check` does not build it.
  Run default, no-default, and all-feature checks; Clippy with test targets;
  doctests and examples. Run an isolated Rust 1.91.1 fresh resolution plus
  locked builds. Run cargo-deny, gitleaks, and Vale on changed public prose.
  Provider-key smoke tests run separately, never as offline CI requirements.
- A live authorized OpenAI Responses smoke covers text, reasoning, and tool
  continuation when an accessible model is available; record the model and
  observed event/result sequence without copying keys or encrypted payloads.
- Present the complete diff, deterministic checks, live result, and an
  independent Rust review before any commit. The PR targets `card/A23`, not
  `master`. A second PR branches from this reviewed change.

## Review disposition and remaining risk

The frontier spec review passed in round three. Round-one blockers required
an explicit construction path and model precedence; round two required a
concrete collector seam in PR 2. No model-ID heuristic or vendor preset has
been validated for all compatible hosts. Chat Completions EOF semantics and
opaque third-party reasoning replay remain outside these two PRs.

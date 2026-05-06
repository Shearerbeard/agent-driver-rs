# ADR-0002: Structured Output / Response Format

**Status:** Proposed
**Date:** 2026-05-02
**Context tags:** [provider] [structured-output] [agent-loop] [aura]

---

## Context

agent-driver-rs needs to support API-level structured output — the ability to tell a provider "respond conforming to this JSON schema" and get a guaranteed (or near-guaranteed) parse on the other side.

**Motivation:** Aura's coordinator tells worker agents to output in a specific schema that coordinator code deserializes. Today Aura implements this at the orchestration layer by hand-rolling a tool-constrained-output pattern (define schema as tool, force `tool_choice`, unwrap args). Moving this into agent-driver-rs:

1. Eliminates duplicated extraction logic in every coordinator
2. Automatically upgrades to provider-native constrained decoding when available (zero parse failures)
3. Centralizes retry/validation logic

### Provider Landscape (May 2026)

We test on 8 providers. They split into two mechanisms:

**Native `response_format` — constrained decoding at the token level:**

| Provider | API | Param | Schema support |
|---|---|---|---|
| OpenAI | OpenAI | `response_format: { type: "json_schema", schema }` | Strict mode, full JSON Schema |
| Gemini | Google | `response_mime_type` + `response_schema` | Subset of OpenAPI 3.0 |
| Ollama | Ollama | `format: "json"` or `format: { schema }` | Full JSON Schema (0.5+, GBNF) |
| llama-cpp | OpenAI-compat | `response_format: { type: "json_schema", schema }` | GBNF grammar conversion |
| OpenRouter | OpenAI passthrough | `response_format` passthrough | Depends on upstream model |
| opencode.zen | Mixed | Per-model (OAI/Anthropic/Google protocols) | For OAI/Google-routed models |

**Tool-use extraction — synthetic tool + forced `tool_choice`:**

| Provider | API | Mechanism |
|---|---|---|
| Anthropic | Anthropic Messages | `tool_choice: { type: "tool", name }` + schema as tool input |
| Bedrock | Converse | `toolChoice: { tool: { name } }` + schema as tool input |
| opencode.zen | Mixed | For Anthropic-routed models |

Key difference: native constrained decoding **guarantees** schema conformance (model cannot emit non-conforming tokens). Tool-use extraction is highly reliable but **not grammatically constrained** — malformed output is possible.

---

## Decision

### 1. `ResponseFormat` enum on completion requests

```rust
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ResponseFormat {
    /// Guarantee valid JSON output, no schema enforcement
    Json,
    /// Constrained output against a JSON schema.
    /// Native providers use response_format; extraction providers
    /// use synthetic tool + forced tool_choice.
    JsonSchema {
        name: String,
        schema: serde_json::Value,
        /// OpenAI strict mode. Ignored by providers that don't support it.
        strict: Option<bool>,
    },
}
```

This lives on the completion request, not the session — per-call flexibility (structured for one turn, freeform for the next).

### 2. Provider mapping

Each provider maps `ResponseFormat` to its native mechanism:

- **OpenAI / llama-cpp / OpenRouter**: Wire to `response_format` parameter directly
- **Gemini**: Map to `response_mime_type: "application/json"` + `response_schema`
- **Ollama**: Map to `format: "json"` (Json variant) or `format: { schema }` (JsonSchema variant)
- **Anthropic**: Inject synthetic tool with the schema as its input schema, set `tool_choice: { type: "tool", name: "__structured_output" }`, unwrap tool call args as the response
- **Bedrock**: Same as Anthropic but via Converse API `toolChoice`

### 3. Synthetic tool disambiguation

For the extraction path (Anthropic/Bedrock), the provider injects a tool with a reserved name prefix `__structured_output`. The provider layer handles this internally:

- On request: inject the synthetic tool definition + forced tool_choice
- On response: detect `__structured_output` tool call, extract args, surface as `structured_output` field on the response — **not** as a tool call for the agent loop to execute

The agent loop never sees this tool. It's fully encapsulated in the provider.

### 4. Response surface

```rust
pub struct CompletionResponse {
    pub content: Vec<ContentBlock>,
    pub metadata: CompletionMetadata,
    /// Populated when ResponseFormat was set and the provider returned
    /// conforming output. None if no structured output was requested.
    pub structured_output: Option<serde_json::Value>,
}
```

For native providers, `structured_output` is parsed from the text content. For extraction providers, it's the unwrapped tool call arguments. Callers (including aura's coordinator) deserialize from this field uniformly.

### 5. Optional validation

Behind a feature flag (`structured-output-validate` or similar), validate `structured_output` against the provided schema using the `jsonschema` crate. This catches extraction-path failures before they reach the caller. Off by default — native constrained decoding doesn't need it.

### 6. Streaming considerations

Structured output streams normally — `TextDelta` events for native providers, tool call deltas for extraction providers. The `structured_output` field is populated on `collect()` / stream completion. No mid-stream schema validation (would defeat the point of streaming).

---

## Consequences

### Easier

- **Aura simplifies**: Coordinator goes from manual tool-constrained-output to `request.response_format(schema)` → `response.structured_output`. Eventually deprecates hand-rolled extraction code.
- **Provider-native upgrades are free**: Workers on OpenAI/Gemini/Ollama get guaranteed parse success without coordinator changes.
- **Unified retry**: One place to handle extraction failures (retry with same schema) instead of each coordinator reimplementing it.
- **New providers get structured output by default**: Any OpenAI-compatible provider (opencode.zen OAI models, llama-cpp, etc.) inherits it from the OpenAI provider path.

### Harder

- **Provider trait surface grows**: `complete_stream` needs to handle `ResponseFormat`, or we add a pre-processing step that modifies the request before dispatch.
- **Synthetic tool leaks into token counts**: Anthropic/Bedrock extraction path adds a tool definition to every request, consuming context window. For large schemas this is non-trivial.
- **Schema compatibility subset**: Gemini supports a subset of JSON Schema (OpenAPI 3.0). Complex schemas that work on OpenAI may fail on Gemini. We need to document the safe subset or add a schema simplification pass.
- **Testing matrix**: Need to test both paths (native + extraction) with MockProvider variants. Need live tests on at least one native (OpenAI) and one extraction (Anthropic) provider.

### Migration path for Aura

1. Ship `ResponseFormat` in agent-driver-rs
2. Aura workers adopt `response_format(schema)` one at a time
3. Aura coordinator switches from parsing tool call args to reading `structured_output`
4. Hand-rolled extraction code in Aura becomes dead → delete

---

## Alternatives Considered

### A. Session-level response format (sticky)
Set once on the session, applies to all completions. Rejected: too inflexible. Aura workers need structured output for result turns but freeform for reasoning turns.

### B. Expose mechanism choice to callers
Let callers pick "use native" vs "use extraction". Rejected: callers shouldn't care. The provider knows which mechanism it supports. Forcing mechanism choice leaks provider details into orchestration code.

### C. Always use tool-use extraction (even on native providers)
Simplest implementation — one codepath. Rejected: throws away the guaranteed-parse benefit of native constrained decoding, which is the main value proposition for 6 of 8 providers.

### D. Do nothing, let Aura keep handling it
Rejected: every new coordinator would reimplement the same pattern. The extraction logic is provider-specific (Anthropic tool_choice format differs from Bedrock's), so it belongs in the provider layer.

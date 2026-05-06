# ADR-0002: Thinking & Reasoning Support Across Providers

**Status:** Proposed
**Date:** 2026-05-03
**Context tags:** [thinking] [reasoning] [provider] [streaming]

---

## Context

Modern LLMs offer "thinking" or "reasoning" modes where the model allocates compute to an internal chain-of-thought before producing a final answer. This improves performance on complex tasks — planning, multi-step logic, code generation, and agentic tool use.

Every provider we support (or plan to support) exposes thinking differently. This ADR documents the actual API mechanics of each, identifies gaps in our current implementation, and proposes what to fix.

---

## Part 1: Provider Specifications

### 1.1 Anthropic (Direct API)

Anthropic has the most mature thinking implementation, with three activation modes and several recent additions.

#### Activation Modes

**Manual mode** (original, deprecated on Claude 4.6+):
```json
{
  "model": "claude-sonnet-4-20250514",
  "max_tokens": 16384,
  "thinking": {
    "type": "enabled",
    "budget_tokens": 10240
  }
}
```

**Adaptive mode** (recommended for Claude 4.6+, required on Opus 4.7):
```json
{
  "model": "claude-opus-4-7",
  "max_tokens": 16384,
  "thinking": { "type": "adaptive" }
}
```

**Disabled:**
```json
{ "thinking": { "type": "disabled" } }
```
Or omit `thinking` entirely (except on Opus 4.7 / Mythos Preview where adaptive is the default).

#### Effort Parameter (adaptive mode)

Lives at `output_config.effort` on the request body, NOT inside the `thinking` object:
```json
{
  "thinking": { "type": "adaptive" },
  "output_config": { "effort": "medium" }
}
```

Values: `"low"`, `"medium"`, `"high"` (default), `"xhigh"` (Opus 4.7 only), `"max"`.

#### Display Field

Controls whether you get full/summarized thinking text or an empty thinking field:
```json
{
  "thinking": {
    "type": "enabled",
    "budget_tokens": 10240,
    "display": "summarized"
  }
}
```

Values: `"summarized"` (default on Opus 4.6, Sonnet 4.6) or `"omitted"` (default on Opus 4.7, Mythos Preview). Claude 3.7 Sonnet returns full thinking (no summarization).

#### Budget Semantics (manual mode)

- Minimum: 1,024 tokens
- Must be strictly less than `max_tokens` (except with interleaved thinking, where it CAN exceed `max_tokens`)
- `budget_tokens` is a ceiling — the model may use fewer

#### Streaming Events

```
content_block_start  → { type: "thinking", thinking: "", signature: "" }
content_block_delta  → { type: "thinking_delta", thinking: "Let me work through..." }
content_block_delta  → { type: "thinking_delta", thinking: "First I need to..." }
content_block_delta  → { type: "signature_delta", signature: "EqQBCgIYAhIM1gbcDa9G..." }
content_block_stop   → {}
content_block_start  → { type: "text", text: "" }
content_block_delta  → { type: "text_delta", text: "Here's my answer..." }
```

With `display: "omitted"`: no `thinking_delta` events — only `signature_delta` before `content_block_stop`. Faster time-to-first-text-token.

#### Content Block Types

**Thinking block** (response and round-trip):
```json
{
  "type": "thinking",
  "thinking": "Let me analyze this step by step...",
  "signature": "WaUjzkypQ2mUEVM36O2TxuC06KN..."
}
```

**Redacted thinking block** (safety-system flagged reasoning):
```json
{
  "type": "redacted_thinking",
  "data": "EmwKAhgBEgy3va3pzix/LafPsn4..."
}
```

Both `thinking` AND `redacted_thinking` blocks must be preserved in multi-turn. Code that filters only on `type == "thinking"` will silently drop redacted blocks and break conversations.

#### Multi-Turn Requirements

- **Tool use cycles**: thinking + signature blocks from the last assistant turn MUST be sent back. Omitting them causes a 400 error.
- **Plain multi-turn** (no tools): you CAN omit thinking blocks from prior turns, but the docs recommend passing everything back.
- **Signature validation**: the API verifies thinking text against the signature. Any modification — even stripping Unicode surrogates — invalidates it and returns a 400 error.
- **Ordering**: the entire sequence of consecutive thinking blocks must match the original model output exactly.
- **Cross-platform**: signatures generated on Anthropic API, Bedrock, or Vertex AI are interchangeable.

#### Interleaved Thinking

On Claude 4.6+ with adaptive mode, thinking blocks can appear interleaved with text and tool use blocks (think → text → think → tool_use → think → text). On older models or manual mode, requires the `interleaved-thinking-2025-05-14` beta header.

#### Constraints with Thinking Enabled

- `tool_choice` only supports `"auto"` or `"none"` — `"any"` and `{"type": "tool", "name": "..."}` are rejected.
- Streaming is required when `max_tokens` > 21,333 (per Anthropic API docs as of 2026-05; this ceiling may change).

#### Model Support

| Model | Manual (`enabled`) | Adaptive | Interleaved | Display default |
|-------|-------------------|----------|-------------|-----------------|
| Opus 4.7 | Rejected | Required (only mode) | Auto | `"omitted"` |
| Mythos Preview | Deprecated | Default | Auto | `"omitted"` |
| Opus 4.6 | Deprecated | Recommended | Auto w/ adaptive | `"summarized"` |
| Sonnet 4.6 | Deprecated | Recommended | Auto w/ adaptive | `"summarized"` |
| Opus 4.5 | Yes | No | Beta header | N/A |
| Sonnet 4.5 | Yes | No | Beta header | N/A |
| Sonnet 3.7 | Yes | No | No | Full (not summarized) |

---

### 1.2 OpenAI

OpenAI has two distinct APIs with different reasoning mechanics. This matters — they are not interchangeable.

#### Chat Completions API

**Activation** — top-level `reasoning_effort` parameter:
```json
{
  "model": "o4-mini",
  "reasoning_effort": "medium",
  "max_completion_tokens": 5000,
  "messages": [...]
}
```

Values: `"none"`, `"minimal"`, `"low"`, `"medium"`, `"high"`, `"xhigh"` (model-dependent).

**Reasoning content is NOT exposed.** The Chat Completions API does not return reasoning tokens as content. You see the final answer only. Reasoning token counts appear in `usage.completion_tokens_details.reasoning_tokens`.

**Multi-turn**: Reasoning tokens are discarded between every request. No way to preserve reasoning context. This is a fundamental limitation — the Chat Completions API is stateless with respect to reasoning.

**Streaming**: Most models stream. Exceptions: o1, o3-pro, gpt-5-pro do NOT support streaming. Reasoning tokens are hidden from the stream — only final text is delivered via `delta.content`.

**Note**: Uses `max_completion_tokens` (not `max_tokens`) for reasoning models.

#### Responses API

**Activation** — nested `reasoning` object:
```json
{
  "model": "o4-mini",
  "reasoning": {
    "effort": "high",
    "summary": "auto"
  },
  "input": [...]
}
```

**Reasoning appears as output items:**
```json
{
  "output": [
    {
      "type": "reasoning",
      "id": "rs_...",
      "summary": [
        { "type": "summary_text", "text": "The model considered..." }
      ],
      "encrypted_content": null
    },
    {
      "type": "message",
      "role": "assistant",
      "content": [{ "type": "output_text", "text": "The answer is..." }]
    }
  ]
}
```

**Reasoning summary** (`reasoning.summary`):
- Values: `"auto"`, `"concise"`, `"detailed"`
- `"concise"` is supported by computer-use models; NOT supported by GPT-5 series (per current API docs — verify against latest)
- Summaries are not guaranteed for every request — needs minimum material reasoning tokens

**Multi-turn preservation** (Responses API only):
1. `previous_response_id` — auto-retains prior state including reasoning
2. Manual — include `encrypted_content` from previous reasoning items:
```json
{
  "include": ["reasoning.encrypted_content"],
  "input": [
    {"type": "reasoning", "id": "rs_...", "encrypted_content": "..."},
    {"type": "message", "role": "assistant", "content": [...]},
    {"role": "user", "content": "Follow-up"}
  ]
}
```

Including reasoning items between tool calls provides ~3% improvement in benchmarks.

**Streaming events** (Responses API):
- Reasoning summary: `response.reasoning_summary_text.delta`
- GPT-5 reasoning text: `response.reasoning_text.delta` (interleaved thinking)

#### Key Model Support

| Model | Chat Completions | Responses API | Streaming | Reasoning Summary |
|-------|-----------------|---------------|-----------|-------------------|
| o1 | Yes | Yes | No | No |
| o3-mini | Yes | Yes | Yes | No |
| o3 | Yes | Yes | Limited | Yes |
| o4-mini | Yes | Yes | Yes | Yes |
| o3-pro | No | Yes | No | No |
| gpt-5 | Yes | Yes | Yes | Yes |
| gpt-5-pro | No | Yes | No | Yes |
| gpt-5.1+ | Yes | Yes | Yes | Yes |

---

### 1.3 AWS Bedrock (Converse API)

Bedrock supports thinking for Claude models AND Amazon Nova models, with different config shapes for each.

#### Claude on Bedrock

Config goes in `additionalModelRequestFields` (NOT `inferenceConfig`, NOT `performanceConfig`).

**SDK casing note:** `additionalModelRequestFields` accepts a `Document` in the Rust AWS SDK — a pass-through type whose inner keys are forwarded unmodified. Use the exact casing from Anthropic's API docs (`thinking`, `budget_tokens`). There is no SDK-level typed intermediary that would transform casing. The `reasoning_config` key mentioned in some legacy SDK docs refers to Amazon Nova models (see below), not Claude.

**Manual mode:**
```json
{
  "modelId": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
  "additionalModelRequestFields": {
    "thinking": {
      "type": "enabled",
      "budget_tokens": 4000
    }
  }
}
```

**Adaptive mode** (Claude 4.6+):
```json
{
  "additionalModelRequestFields": {
    "thinking": {
      "type": "adaptive",
      "effort": "high"
    }
  }
}
```

**Interleaved thinking** (beta):
```json
{
  "additionalModelRequestFields": {
    "anthropic_beta": ["interleaved-thinking-2025-05-14"],
    "thinking": { "type": "enabled", "budget_tokens": 4000 }
  }
}
```

Same budget semantics as direct Anthropic API. `thinking.type: "enabled"` is deprecated on Claude Opus 4.6.

#### Bedrock Streaming Types

The Converse API uses its own type system, distinct from the direct Anthropic API:

| Direct Anthropic API | Bedrock Converse API |
|---------------------|---------------------|
| `thinking_delta` SSE event | `ContentBlockDelta::ReasoningContent.text` |
| `signature_delta` SSE event | `ContentBlockDelta::ReasoningContent.signature` |
| `type: "thinking"` block | `ContentBlock::ReasoningContent(ReasoningTextBlock { text, signature })` |
| `type: "redacted_thinking"` block | `ReasoningContentBlock::RedactedContent(bytes)` |

Streaming delta flow: reasoning text deltas first, then signature as a final delta for that content block, then text content blocks follow.

#### Nova on Bedrock

Amazon Nova models use a different config key:
```json
{
  "additionalModelRequestFields": {
    "reasoningConfig": {
      "type": "enabled",
      "maxReasoningEffort": "high"
    }
  }
}
```

Effort values: `"low"`, `"medium"`, `"high"`. Nova does NOT use signatures. Uses the same `ReasoningContentBlock` response types as Claude on Bedrock.

#### Multi-Turn

Same rules as direct Anthropic API: thinking + signature blocks from the last assistant turn must be sent back during tool use cycles. The API auto-strips thinking from older turns.

---

### 1.4 Ollama

#### Activation

The `think` parameter is **polymorphic** — not just a boolean:

- **Boolean**: `true` / `false` — works for most models
- **String**: `"low"`, `"medium"`, `"high"`, `"max"` — for GPT-OSS/Harmony-based models

```json
{
  "model": "qwen3",
  "messages": [...],
  "think": true,
  "stream": true
}
```

`think` sits at the **top level** of `/api/chat` and `/api/generate` requests (not inside `options`).

**ollama-rs limitation**: The Rust crate (0.3.3) only supports `Option<bool>` for `think` — string values like `"high"` are not accessible without patching.

#### Streaming — Thinking IS Incremental

Contrary to what we previously documented, **Ollama DOES stream thinking tokens incrementally**, token by token. The streaming is two-phase:

1. **Thinking phase**: chunks arrive with `message.thinking` populated, `message.content` empty
2. **Content phase**: chunks arrive with `message.content` populated, `message.thinking` empty

```json
{"model":"qwen3","message":{"role":"assistant","content":"","thinking":"Let me"},"done":false}
{"model":"qwen3","message":{"role":"assistant","content":"","thinking":" think"},"done":false}
...
{"model":"qwen3","message":{"role":"assistant","content":"The","thinking":""},"done":false}
{"model":"qwen3","message":{"role":"assistant","content":" answer","thinking":""},"done":false}
```

Accumulate `message.thinking` across all chunks for the full reasoning trace, then `message.content` for the final answer.

#### Non-streaming response

```json
{
  "message": {
    "role": "assistant",
    "content": "The answer is 42.",
    "thinking": "Let me think step by step..."
  },
  "done": true
}
```

#### Multi-Turn

Ollama's `Message` struct accepts `thinking` as an input field on assistant messages. You CAN send thinking back in subsequent turns. The model's Jinja template decides whether to use it or strip it — most templates strip thinking tags from past messages automatically, so including it is safe.

For tool-calling flows, Ollama docs state thinking + tool call content from the model + executed tool result must all be passed back.

#### Template Dependency

Thinking support is **deeply template-dependent**. Ollama detects thinking capability through:
1. GGUF metadata flags
2. Template pattern scanning for `<think>`/`</think>` delimiters
3. Model family heuristics

If a model doesn't support thinking, Ollama returns an error suggesting you re-pull the model.

**Confidence note:** Because thinking behavior is template-mediated, the exact streaming behavior, multi-turn handling, and whether thinking is even honored can vary per model. The specs above reflect the common case (qwen3, deepseek-r1), but edge cases should be validated per model family. Ollama's behavior is less deterministic than Anthropic's or OpenAI's due to this template dependency.

#### No Interleaved Thinking

Ollama's parser is a two-phase state machine: thinking phase, then response phase. Once it transitions from thinking to content, it does not transition back. No interleaved thinking.

#### Models

qwen3/3.5/3.6, deepseek-r1/v3.1/v4, gpt-oss (requires string think values), gemma4, glm-4.7/5/5.1, kimi-k2.5/2.6, minimax-m2.5/2.7, nemotron3/cascade-2, mistral-medium-3.5, gemini-3-flash-preview, and more.

#### OpenAI-Compatible Endpoint (`/v1/chat/completions`)

Uses different field names: `reasoning_effort` (not `think`) in requests, `reasoning` (not `thinking`) in responses. The native `think: true` does NOT work on this endpoint.

---

### 1.5 llama-server (llama.cpp)

llama-server is NOT currently a provider in agent-driver-rs, but is a likely addition. It uses an OpenAI-compatible API with reasoning extensions that differ from Ollama.

#### Activation

**Server flags:**
- `--reasoning [on|off|auto]` — enable reasoning detection
- `--reasoning-format FORMAT` — `none`, `deepseek`, `deepseek-legacy`, `auto`
- `--reasoning-budget N` — max thinking tokens (-1 = unlimited)
- `--reasoning-budget-message MSG` — text injected when budget exhausts

**Per-request parameters** (in JSON body):
- `reasoning_format` — override server default
- `thinking_budget_tokens` — per-request budget (only works if `--reasoning-budget` was NOT set on CLI)
- `chat_template_kwargs` — e.g., `{"enable_thinking": false}`

#### Field Names

Uses `reasoning_content` (matching DeepSeek API convention), NOT Ollama's `thinking`:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "The answer is 3.",
      "reasoning_content": "Let me count the r's in strawberry..."
    }
  }]
}
```

#### Streaming

Streams `reasoning_content` incrementally via `delta.reasoning_content`:
```
data: {"choices":[{"delta":{"reasoning_content":"Let me"}}]}
data: {"choices":[{"delta":{"reasoning_content":" think about"}}]}
data: {"choices":[{"delta":{"content":"The answer"}}]}
```

**Caveat**: In streaming mode with `deepseek` reasoning_format, separation may not work correctly — clients may need to parse `<think>` tags themselves. This is an active area of development in llama.cpp.

#### Budget Control

Unlike Ollama, llama-server has token-budget control:
- `--reasoning-budget N` at server startup
- `thinking_budget_tokens` per request
- When budget exhausts, injects `--reasoning-budget-message` text

#### Multi-Turn

`reasoning_content` can be included on input messages. Templates that support reasoning history preservation handle it correctly.

#### No Interleaved Thinking

Same as Ollama — reasoning content comes first, then response content.

---

### 1.6 OpenRouter

OpenRouter provides a unified `reasoning` parameter that normalizes across providers.

#### Activation

```json
{
  "model": "anthropic/claude-sonnet-4.5",
  "reasoning": {
    "max_tokens": 2000,
    "effort": "high"
  }
}
```

Effort-to-budget conversion for Anthropic models: `budget_tokens = max(min(max_tokens * effort_ratio, 128000), 1024)`. Approximate ratios per OpenRouter docs: `"xhigh"` = 95%, `"high"` = 80%, `"medium"` = 50%, `"low"` = 20%, `"minimal"` = 10%. These are subject to change.

The `:thinking` model variant (e.g., `anthropic/claude-3.7-sonnet:thinking`) is deprecated. Use the `reasoning` parameter.

#### Response Format

```json
{
  "choices": [{
    "message": {
      "content": "Final answer...",
      "reasoning": "Step by step thinking...",
      "reasoning_details": [
        { "type": "reasoning.text", "text": "..." },
        { "type": "reasoning.summary", "summary": "..." },
        { "type": "reasoning.encrypted", "data": "..." }
      ]
    }
  }]
}
```

Streaming: `reasoning_details` appear in `choices[].delta.reasoning_details`.

#### Multi-Turn

Pass `reasoning_details` unmodified back in subsequent messages to maintain reasoning continuity during tool-use workflows. This mirrors the Anthropic signature requirement.

#### Provider Pass-Through

- Claude models: thinking blocks pass through (including signatures via `reasoning_details`)
- OpenAI o-series: reasoning tokens remain hidden (effort is configurable but content is not returned)
- Open-source models (DeepSeek R1, Qwen3+, etc.): thinking content is surfaced

---

## Part 2: Current Implementation vs. Reality

### What We Have

| Aspect | Current State | Correct? |
|--------|--------------|----------|
| `ContentBlock::Thinking { text }` | Stores thinking text | Partially — missing signature field |
| `ContentBlock::Signature` | Does not exist | **Missing** — signatures are discarded |
| `ContentBlock::RedactedThinking` | Does not exist | **Missing** — redacted blocks are lost |
| `StreamDelta::ThinkingDelta` | Emitted by Anthropic, Ollama | Correct |
| `StreamDelta::SignatureDelta` | Streamed but explicitly discarded in `apply_delta()` | **Bug** — must be accumulated |
| `ThinkingConfig { budget_tokens }` | Anthropic only, manual mode | **Incomplete** — missing adaptive mode, effort, display |
| `ReasoningConfig { effort, summary }` | Exists for OpenAI | **Dead code** — never applied to requests |
| Bedrock thinking | `extended_thinking: false` | **Missing entirely** |
| Ollama streaming | Assumed single-chunk delivery | **Wrong** — thinking streams incrementally |
| OpenRouter thinking | Not surfaced | **Missing** |
| llama-server | Not a provider | Future work |

### The Signature / Multi-Turn Bug

This is the most critical issue. The flow today:

1. `StreamDelta::SignatureDelta` arrives from Anthropic
2. `CollectedResponse::apply_delta()` explicitly ignores it: "Signatures are metadata, not content"
3. No `ContentBlock::Signature` variant exists
4. Signature never reaches session history
5. Next turn sends `ContentBlock::Thinking` WITHOUT its signature
6. Anthropic rejects the request (or silently degrades)

This means **multi-turn conversations with extended thinking are broken** for both Anthropic and Bedrock.

Additionally, `redacted_thinking` blocks are not handled at all. Any safety-flagged reasoning is silently dropped, which also breaks multi-turn.

---

## Decision

### 1. Fix multi-turn thinking (critical correctness bug)

Add `ContentBlock::Signature { signature: String }` and `ContentBlock::RedactedThinking { data: String }` variants. Accumulate signatures in `apply_delta()` instead of discarding them. Serialize both block types when sending message history back to Anthropic/Bedrock.

### 2. Keep provider-specific config types — match the native APIs

Each provider's thinking config should use the same field names and shapes as its API, so users familiar with that provider's docs can find what they expect:

- **Anthropic**: `ThinkingConfig` gains `type` (enabled/adaptive/disabled), `budget_tokens`, `display`, plus `output_config.effort` for adaptive mode
- **OpenAI**: `ReasoningConfig` with `effort` and `summary` — AND wire it up so it's actually sent in requests
- **Bedrock**: Thinking config mirrors Anthropic (same underlying model). Nova gets its own `NovaReasoningConfig { effort }`
- **Ollama**: `think` stays but type changes from `Option<bool>` to accommodate string values (or we stay boolean until ollama-rs adds support)
- **OpenRouter**: `ReasoningConfig { effort, max_tokens, exclude }` matching OpenRouter's `reasoning` parameter

### 3. Fix provider gaps

- **Bedrock**: Add thinking support via `additionalModelRequestFields.thinking`. Handle `ReasoningContentBlockDelta` in streaming (both `.text` and `.signature` members).
- **OpenAI**: Wire up `ReasoningConfig` to actually send `reasoning_effort` / `reasoning.summary`. Parse `reasoning_content` from Responses API if we add Responses API support.
- **OpenRouter**: Parse `reasoning` and `reasoning_details` from responses. Pass `reasoning_details` back in multi-turn.
- **Ollama**: Fix the streaming assumption — thinking tokens arrive incrementally, not all at once (our current code may already handle this correctly since we process each chunk's `message.thinking`, but verify).

### 4. Validate budget constraints at request-build time

`budget_tokens < max_tokens` checked when constructing the completion request (both values available at that point). For adaptive mode, no budget validation needed (the API handles it).

### 5. Capability reporting mirrors the provider API

Replace `extended_thinking: bool` with `ThinkingSupport` enum whose variants match each provider's control surface:

```rust
pub enum ThinkingSupport {
    None,
    BudgetTokens {
        min_budget: u32,
        requires_signature_roundtrip: bool,
    },
    ReasoningEffort {
        levels: Vec<ReasoningEffort>,
        summary_available: bool,
    },
    Toggle,
}
```

### 6. Document llama-server as a future provider

llama-server (llama.cpp) uses `reasoning_content` (not `thinking`), has budget control via `thinking_budget_tokens`, and uses an OpenAI-compatible API format. When we add it as a provider, it should be treated as a distinct provider (not routed through the OpenAI provider) due to the `reasoning_content` field and budget control differences.

## Consequences

### What becomes easier
- Multi-turn agentic loops with extended thinking actually work (currently broken)
- Adding thinking to Bedrock is mechanical — same config shape as Anthropic
- Each provider's config feels native — no unfamiliar abstraction to learn
- Future llama-server integration has a clear spec to build against

### What becomes harder
- `ContentBlock` gains two new variants — every match arm needs `Signature` and `RedactedThinking` cases
- Signature blocks are opaque blobs in session history (only meaningful for Anthropic/Bedrock)
- Anthropic's adaptive vs manual distinction and the deprecated-but-still-supported manual mode adds config complexity
- Two OpenAI APIs (Chat Completions vs Responses) with different reasoning semantics may need different provider implementations or a mode switch

### Open Questions
- Should we support the OpenAI Responses API as a separate provider, or extend the existing OpenAI provider with a mode flag?
- When Anthropic's `display: "omitted"` is used, should observers still receive a `ThinkingDelta` (empty) or should the event be suppressed?
- Should OTel/Phoenix traces include thinking token counts as span attributes?
- How should we handle Ollama's GPT-OSS string think values when ollama-rs only supports `Option<bool>`? Fork/patch ollama-rs, or use raw HTTP?
- Should llama-server be a standalone provider or share infrastructure with the OpenAI provider (since it's OpenAI-compatible with extensions)?

### Specification Confidence

This ADR was independently reviewed for factual accuracy. Confidence ratings per provider:

| Provider | Confidence | Notes |
|----------|------------|-------|
| Anthropic | High | Validated against Claude 3.7+ specs. Adaptive mode, effort location, signature requirements, redacted_thinking all confirmed. |
| OpenAI (Chat Completions) | High | Stateless reasoning, hidden tokens, `max_completion_tokens` requirement confirmed. |
| OpenAI (Responses API) | High | `encrypted_content`, reasoning output items, `previous_response_id` confirmed. |
| Bedrock (Claude) | High | `additionalModelRequestFields` path confirmed. SDK casing may vary — verify against target SDK version. |
| Bedrock (Nova) | High | `reasoningConfig` with effort levels, no signatures confirmed. |
| Ollama | Medium-High | Two-phase incremental streaming confirmed. Template dependency means exact behavior varies per model — less deterministic than other providers. |
| llama-server | High | `reasoning_content` field, `thinking_budget_tokens`, `--reasoning-format` flags confirmed. Streaming separation with `deepseek` format is an active area of llama.cpp development. |
| OpenRouter | High | `reasoning` parameter, effort-to-budget conversion, `reasoning_details` round-trip confirmed. |

### Implementation Priority
1. **`ContentBlock::Signature` + `RedactedThinking` + multi-turn fix** — correctness bug blocking real usage
2. **Bedrock thinking support** — low effort, high value (same API shape as Anthropic)
3. **Anthropic adaptive mode + effort + display** — current config only covers deprecated manual mode
4. **OpenAI `ReasoningConfig` wiring** — dead code → working code
5. **OpenRouter reasoning pass-through** — parse `reasoning` / `reasoning_details`
6. **Ollama streaming verification** — confirm incremental thinking works correctly
7. **Budget validation** — prevent confusing API errors
8. **`ThinkingSupport` capability enum** — quality-of-life
9. **llama-server provider** — future work, spec is documented

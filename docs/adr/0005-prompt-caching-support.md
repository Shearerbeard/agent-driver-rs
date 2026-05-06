# ADR-0005: Prompt Caching Support Across Providers

**Status:** Proposed
**Date:** 2026-05-03
**Context tags:** [caching] [provider] [performance] [telemetry]

---

## Context

LLM providers have begun offering **prompt caching** as a mechanism to reduce latency and cost for repeated or long prompts. When a prompt is cached, subsequent requests with the same prompt content can skip recomputation, returning results faster and often at reduced cost.

Key providers and their caching mechanisms:


| Provider | Mechanism | Cache Identifier | Cache Duration | Metrics |
|----------|-----------|------------------|----------------|---------|
| **OpenAI** | Automatic by default; `OpenAI-Behavior-Experimental: cache=include\|only` header for control | Implicit (provider-managed) | 5-10 min (automatic), tunable via header | `prompt_tokens_details.cached_tokens` |
| **Anthropic** | `cache_control: { type: "ephemeral", ttl: "5m"\|"1h" }` on content blocks; or top-level automatic caching | Implicit (provider-managed, prefix-based) | 5 min (default), 1 hr (with `ttl`) | `cache_creation_input_tokens`, `cache_read_input_tokens` |
| **Google Gemini** | `CachedContent` resource with configurable `ttl`; also implicit auto-caching | Explicit via `CachedContent` name, or implicit | Configurable (min 1hr, max 24hr) | `cachedContentTokenCount` |
| **OpenRouter** | Inherits from underlying provider | Inherited | Inherited | Inherited |
| **Bedrock** | `cachePoint` in Converse API; `cache_control` in InvokeModel for Claude; automatic for Nova | Implicit (prefix-based checkpoint) | 5 min (default), 1 hr (with `ttl` for supported models) | `CacheReadInputTokens`, `CacheWriteInputTokens` |
| **vLLM** | Automatic Prefix Caching (APC) via `--enable-prefix-caching` flag (server-side, not per-request API) | Implicit (prefix hash) | Session-lifetime (no explicit TTL) | Not exposed in API response |
| **llama.cpp** | KV cache reuse via `--cache-prompt` flag (session-level, not per-request API) | Implicit (exact context match) | Session-lifetime (no explicit TTL) | Not exposed in API response |
| **Ollama** | Automatic KV cache preservation across consecutive requests to same model | Implicit (exact prefix match) | Session-lifetime (until model evicted) | Not exposed in API response |


> **Note on vLLM, llama.cpp, and Ollama:** These are local inference engines with internal KV cache mechanisms. While they do cache prompt prefixes between requests, this is server-side behavior — not a per-request HTTP API like OpenAI/Anthropic offer. vLLM's Automatic Prefix Caching (APC) hashes prompt prefixes and reuses KV cache across requests, but does not expose cache hit counts in API responses. llama.cpp's `--cache-prompt` persists KV state between sessions but is session-level. Ollama similarly preserves KV cache between consecutive requests. None of these expose cache metrics in their OpenAI-compatible API responses, so they fall into the "no API-level caching control" category for this ADR.

> **Note on OpenRouter:** As an aggregator, OpenRouter passes caching headers to the underlying provider if that provider supports them. This means caching works for OpenAI-backed models on OpenRouter, but not for providers that don't support caching.

Prompt caching is especially valuable in agentic loops where:
1. System prompts are large and repeated every turn
2. Tool definitions are constant across many requests
3. Conversation history grows, making repeated context chunks expensive

We need a **provider-agnostic abstraction** that:
- Enables caching when available (OpenAI, Anthropic, Gemini, Bedrock)
- Gracefully no-ops for providers without API-level caching (vLLM, llama.cpp, Ollama)
- Tracks cache metrics if exposed by the provider
- Supports provider-specific TTL configuration where applicable

---

## Decision

### 1. Add a `PromptCacheConfig` struct with provider-agnostic options

```rust
/// Configuration for prompt caching behavior.
/// 
/// Not all providers support caching; this config is ignored for
/// providers that don't support it.
#[derive(Debug, Clone, Default)]
pub struct PromptCacheConfig {
    /// Whether to attempt caching of the system prompt.
    /// Default: false (opt-in to avoid unexpected behavior).
    pub cache_system_prompt: bool,
    
    /// Whether to attempt caching of tool definitions.
    /// Default: false (opt-in).
    pub cache_tools: bool,
    
    /// For providers that support it (OpenAI), control cache behavior.
    /// - `Include`: Cache prompt, use cache if available, bill cached tokens
    /// - `Only`: Only use cached prompt; fail if not cached
    /// - `Exclude`: Don't use or create cache (default for most)
    pub cache_policy: CachePolicy,
    
    /// Cache TTL (time-to-live). Provider-specific:
    /// - Anthropic: "5m" (default) or "1h" (supported models only)
    /// - Bedrock: "5m" (default) or "1h" (supported models only)
    /// - Gemini: Duration string e.g. "3600s", "1h", "24h" (min 1hr, max 24hr)
    /// - OpenAI/vLLM/llama.cpp/Ollama: Not configurable
    /// None = provider default.
    pub ttl: Option<String>,
}

/// Cache policy for providers that support it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CachePolicy {
    /// Use cached prompt if available; create cache for future requests.
    Include,
    /// Only use cached prompt; fail if not in cache.
    Only,
    /// Don't use or create cache (explicit opt-out).
    Exclude,
}
```

### 2. Add cache behavior to `CompletionRequest`

```rust
/// Per-request cache configuration, merged with provider defaults.
#[derive(Debug, Clone, Default)]
pub struct RequestCacheConfig {
    /// Whether this specific request should attempt to use/create a cache.
    /// Overrides `PromptCacheConfig::cache_system_prompt` if set.
    pub enabled: Option<bool>,
    
    /// Per-request cache policy override.
    pub policy: Option<CachePolicy>,
    
    /// Per-request TTL override.
    /// Supported by: Anthropic ("5m", "1h"), Bedrock ("5m", "1h"), Gemini ("3600s", "1h", "24h").
    pub ttl: Option<String>,
    
    /// For Anthropic/Bedrock: use automatic caching instead of explicit breakpoints.
    /// When true, places a single cache_control at the top level instead of per-block.
    /// Default: true (recommended for agentic loops).
    pub automatic: bool,
}
```

### 3. Implement provider-specific caching headers

#### OpenAI

OpenAI provides **automatic prompt caching by default** on GPT-4o, GPT-4o-mini, and o-series models. No header is needed for basic caching — the API automatically caches prompts and reports `cached_tokens` in the response.

For explicit control, the `OpenAI-Behavior-Experimental` header can be used:

```rust
// In OpenAI provider request builder:
match config.policy {
    CachePolicy::Include => {
        // Explicitly request caching; use cached if available
        req_builder.header(
            "OpenAI-Behavior-Experimental",
            "cache=include",
        );
    }
    CachePolicy::Only => {
        // Only use cached prompt; fail if not in cache
        req_builder.header(
            "OpenAI-Behavior-Experimental",
            "cache=only",
        );
    }
    CachePolicy::Exclude => {
        // No header, rely on default automatic caching
    }
}
```

Key details:
- Caching is **automatic** — no opt-in needed for basic behavior
- Cache TTL: approximately 5-10 minutes depending on model
- `cache=include` explicitly requests caching but doesn't change behavior beyond automatic
- `cache=only` is more restrictive: fails if no cache hit
- Cache hit is reported in `usage.prompt_tokens_details.cached_tokens`

#### Anthropic

Anthropic supports **two caching modes**:

1. **Automatic caching** (recommended for agentic loops): Add a top-level `cache_control` field to the request body. The system automatically places a cache breakpoint at the last cacheable block and moves it forward as conversations grow.

   ```json
   {
     "model": "claude-sonnet-4-20250514",
     "cache_control": { "type": "ephemeral" },
     ...
   }
   ```

2. **Explicit cache breakpoints**: Place `cache_control` on individual content blocks for fine-grained control. This is useful when you have sections that change at different frequencies.

   ```json
   {
     "type": "text",
     "text": "System instructions...",
     "cache_control": { "type": "ephemeral", "ttl": "1h" }
   }
   ```

Key details:
- Cache is **prefix-based**: it caches everything up to and including the marked block
- TTL options: `"5m"` (default) or `"1h"` (for Claude Opus 4.5, Haiku 4.5, Sonnet 4.5)
- Minimum token count: varies by model (typically 1,024+ tokens)
- Automatic caching moves the breakpoint to the last cacheable block, so you don't need to update positions as conversations grow

```rust
// In Anthropic provider, modify message serialization:
fn serialize_message(&self, msg: &Message, cache_config: &RequestCacheConfig) -> JsonValue {
    let content: Vec<JsonValue> = msg.content.iter().map(|block| {
        match block {
            ContentBlock::Text { text } if cache_config.enabled.unwrap_or(false) => {
                // Anthropic: add cache_control breakpoint on this block
                let mut obj = serde_json::json!({
                    "type": "text",
                    "text": text,
                });
                if let Some(ttl) = &cache_config.ttl {
                    obj["cache_control"] = serde_json::json!({
                        "type": "ephemeral",
                        "ttl": ttl
                    });
                } else {
                    obj["cache_control"] = serde_json::json!({
                        "type": "ephemeral"
                    });
                }
                obj
            }
            // ... other block types unchanged
        }
    }).collect();
    // ...
}
```

Note: Anthropic's caching is applied to message content blocks, not headers. The `cache_control` annotation tells the API to cache the full prefix up to and including that block.


#### Google Gemini

Gemini uses a **resource-based** approach with `CachedContent` objects:


```rust
// Gemini API supports two modes:

// 1. Implicit caching (automatic, like OpenAI/Anthropic):
//    Gemini automatically caches repeated prompts, no special config needed.
//    Check `usage_metadata.cached_content_tokens` in response.

// 2. Explicit caching with CachedContent resource:
//    Create a CachedContent object, then reference it in requests:

#[derive(Debug, Serialize)]
struct CreateCachedContentRequest {
    model: String,           // e.g., "gemini-1.5-flash"
    contents: Vec<Content>,   // The prompt to cache
    ttl: Option<String>,      // Duration: "3600s", "1h", "24h", etc.
    display_name: Option<String>,
}

// In generateContent request, reference the cached content:
#[derive(Debug, Serialize)]
struct GenerateContentRequest {
    model: String,
    cached_content: String,  // Name of CachedContent resource
    contents: Vec<Content>,  // Only non-cached parts here
    // ...
}
```

Gemini's explicit caching requires:
1. Creating a `CachedContent` resource first (separate API call)
2. Referencing it by name in subsequent requests
3. TTL configuration (1 hour to 24 hours)
4. Minimum token counts vary by model (typically 32K+ tokens)

**Feasibility note:** Gemini's explicit caching requires resource lifecycle management. For agentic loops, implicit caching is more practical — Gemini handles cache keying automatically.


#### OpenRouter

OpenRouter acts as a proxy and passes caching headers to the underlying provider:

```rust
// OpenRouter forwards these headers to the backend:
// - `OpenAI-Behavior-Experimental: cache=include` (for OpenAI models)
// - Anthropic-style cache_control (for Anthropic models)

// Implementation: same as the underlying provider.
// OpenRouter provider wraps the target provider's logic.
```

**Feasibility note:** OpenRouter doesn't have its own cache — it inherits the behavior of whichever provider handles the request. Cache metrics are likewise inherited and exposed in the response.

#### Bedrock

AWS Bedrock supports prompt caching via **`CachePointBlock`** in the Converse API (which our Bedrock provider uses exclusively via `converse_stream`).

The AWS SDK exposes `CachePointBlock` with `CachePointType::Default` and `TokenUsage` with `cache_read_input_tokens` / `cache_write_input_tokens` fields.

You can place `CachePoint` in `system`, `messages`, and `tools` fields. Up to 4 cache checkpoints per request for Claude models.

**Converse API** format:

```rust
// Using the AWS SDK types:
use aws_sdk_bedrockruntime::types::{CachePointBlock, CachePointType};

// In system prompt:
let system_block = SystemContentBlock::CachePoint(
    CachePointBlock::builder()
        .r#type(CachePointType::Default)
        .build()
);

// In message content:
let content_block = ContentBlock::CachePoint(
    CachePointBlock::builder()
        .r#type(CachePointType::Default)
        .build()
);

// In tool config:
let tool = Tool::CachePoint(
    CachePointBlock::builder()
        .r#type(CachePointType::Default)
        .build()
);
```

Key details:
- Minimum 1,024 tokens per cache checkpoint
- Default TTL: 5 minutes
- Extended TTL: 1 hour (Claude Opus 4.5, Haiku 4.5, Sonnet 4.5) — note: **not yet exposed as a field in the Rust SDK's `CachePointBlock`**. **Fallback:** skip TTL for Bedrock until the SDK adds the field; default 5-minute TTL applies automatically.
- Simplified cache management available for Claude models (auto-lookback up to ~20 content blocks)
- Cache metrics present in `TokenUsage`: `cache_read_input_tokens`, `cache_write_input_tokens`
- Amazon Nova models have automatic caching for all text prompts

**Note on InvokeModel API:** The Bedrock InvokeModel API (for Claude) uses `cache_control: { "type": "ephemeral", "ttl": "1h" }` format, identical to Anthropic's direct API. Since our Bedrock provider uses the Converse API exclusively, we only need to implement the `CachePointBlock` path. If a future provider uses InvokeModel, the Anthropic `cache_control` format would apply.

#### vLLM, llama.cpp, and Ollama (Local Inference)

**Excluded from caching API** — these are local inference engines with different caching semantics:

| Engine | Caching Type | API-Level Control |
|--------|-------------|------------------|
| **vLLM** | PagedAttention KV cache (internal) | None — server-side, per-request |
| **llama.cpp** | Context window with KV cache (internal) | `-c` flag at server start |

Both engines manage KV cache internally based on:
- Available GPU memory
- Context window size (`-c` for llama.cpp, `--max-model-len` for vLLM)
- Attention pattern during inference

There is **no per-request prompt caching API** equivalent to OpenAI/Anthropic. Agents using these engines:
- Configure context window size once at startup
- Cannot selectively cache specific prompts
- Cannot read cache hit/miss metrics via API

**Future consideration:** If vLLM or llama.cpp add an HTTP-based prompt caching API similar to OpenAI, this ADR should be updated to include them.

### 4. Track cache metrics via `CompletionMetadata`

Providers that support caching may expose cache hit/miss information in their responses:

#### OpenAI Cache Metrics

OpenAI returns cache info in the usage object:

```json
{
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 200,
    "completion_tokens_details": {
      "reasoning_tokens": 50
    },
    "prompt_tokens_details": {
      "cached_tokens": 80   // ← Cache hit indicator
    }
  }
}
```

#### Anthropic Cache Metrics

Anthropic returns cache breakpoints in the usage response:

```json
{
  "usage": {
    "input_tokens": 100,
    "output_tokens": 200,
    "cache_creation_input_tokens": 50,   // ← First-time computation
    "cache_read_input_tokens": 80       // ← Tokens served from cache
  }
}
```

#### Extend `TokenUsage` and `CompletionMetadata`

```rust
/// Extended token usage information including cache metrics.
///
/// Cache metrics are provider-specific with consistent naming conventions:
/// each provider-prefixed field (`openai_`, `anthropic_`, `bedrock_`, `gemini_`)
/// ensures users never have to guess which field maps to which provider.
/// Each provider populates only the fields relevant to its API:
/// - **OpenAI**: `openai_cached_tokens`
/// - **Anthropic**: `anthropic_cache_creation_tokens` + `anthropic_cache_read_tokens`
/// - **Gemini**: `gemini_cached_tokens`
/// - **Bedrock**: `bedrock_cache_read_tokens` + `bedrock_cache_write_tokens`
#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    /// For OpenAI: tokens served from cache.
    /// Reported in `usage.prompt_tokens_details.cached_tokens`.
    pub openai_cached_tokens: Option<u32>,
    /// For Anthropic: tokens in a newly created cache entry.
    /// Reported in `usage.cache_creation_input_tokens`.
    pub anthropic_cache_creation_tokens: Option<u32>,
    /// For Anthropic: tokens read from existing cache.
    /// Reported in `usage.cache_read_input_tokens`.
    pub anthropic_cache_read_tokens: Option<u32>,
    /// For Gemini: tokens served from cached content.
    /// Reported in `usageMetadata.cachedContentTokenCount`.
    pub gemini_cached_tokens: Option<u32>,
    /// For Bedrock: tokens read from cache.
    /// Reported via AWS SDK `TokenUsage.cache_read_input_tokens`.
    pub bedrock_cache_read_tokens: Option<u32>,
    /// For Bedrock: tokens written to cache.
    /// Reported via AWS SDK `TokenUsage.cache_write_input_tokens`.
    pub bedrock_cache_write_tokens: Option<u32>,
}

impl TokenUsage {
    /// Returns the number of non-cached input tokens.
    /// Useful for billing/cost calculations.
    ///
    /// Normalizes across providers. For Anthropic and Bedrock — which can report
    /// both creation and read cache tokens in the same response — both are summed.
    pub fn non_cached_input_tokens(&self) -> u32 {
        let cached = self.total_cached_tokens();
        self.input_tokens.saturating_sub(cached)
    }

    /// Returns the total number of cached input tokens across all providers.
    ///
    /// For providers that can report both cache creation and cache read tokens
    /// simultaneously (Anthropic, Bedrock), both fields are summed.
    fn total_cached_tokens(&self) -> u32 {
        let mut cached = 0u32;
        cached = cached.saturating_add(self.openai_cached_tokens.unwrap_or(0));
        cached = cached.saturating_add(self.anthropic_cache_creation_tokens.unwrap_or(0));
        cached = cached.saturating_add(self.anthropic_cache_read_tokens.unwrap_or(0));
        cached = cached.saturating_add(self.gemini_cached_tokens.unwrap_or(0));
        cached = cached.saturating_add(self.bedrock_cache_read_tokens.unwrap_or(0));
        cached = cached.saturating_add(self.bedrock_cache_write_tokens.unwrap_or(0));
        cached
    }

    /// Returns the cache hit ratio for input tokens (0.0 to 1.0).
    ///
    /// For Anthropic and Bedrock, includes both cache creation and cache read
    /// tokens in the numerator (since both represent tokens not recomputed).
    pub fn cache_hit_ratio(&self) -> f64 {
        let total = self.input_tokens.max(1);
        let cached = self.total_cached_tokens() as f64;
        cached / total as f64
    }
}
```

#### Provider Response Formats

**OpenAI:**
```json
{
  "usage": {
    "prompt_tokens": 100,
    "prompt_tokens_details": {
      "cached_tokens": 80
    },
    "completion_tokens": 200,
    "total_tokens": 300
  }
}
```

**Anthropic:**
```json
{
  "usage": {
    "input_tokens": 100,
    "output_tokens": 200,
    "cache_creation_input_tokens": 50,
    "cache_read_input_tokens": 80
  }
}
```

**Gemini:**
```json
{
  "usageMetadata": {
    "promptTokenCount": 100,
    "candidatesTokenCount": 200,
    "totalTokenCount": 300,
    "cachedContentTokenCount": 80
  }
}
```

**Bedrock (Converse API via AWS SDK):**
```rust
// TokenUsage fields from ConverseStream Metadata:
// meta.usage().cache_read_input_tokens()  → Option<i32>
// meta.usage().cache_write_input_tokens() → Option<i32>
// These map to our TokenUsage.bedrock_cache_read_tokens and
// TokenUsage.bedrock_cache_write_tokens respectively.
```

### 5. Add cache info to `StreamEvent::Completed`

```rust
/// Metadata about the completion response.
#[derive(Debug, Clone)]
pub struct CompletionMetadata {
    pub model: Option<ModelId>,
    pub stop_reason: Option<StopReason>,
    pub usage: Option<TokenUsage>,
    /// Cache-specific information if available.
    pub cache_info: Option<CacheInfo>,
}

/// Detailed cache information from the provider.
#[derive(Debug, Clone)]
pub struct CacheInfo {
    /// Whether this response was served from cache.
    pub cache_hit: bool,
    /// Provider-specific cache identifier (if exposed).
    pub cache_key: Option<String>,
    /// TTL information if provided by the provider.
    pub ttl_secs: Option<u64>,
}
```

### 6. Provider capability reporting

Update `ProviderCapabilities` to expose caching support:

```rust
/// Capabilities of a provider.
#[derive(Debug, Clone)]
pub struct ProviderCapabilities {
    pub streaming: bool,
    pub tools: bool,
    pub vision: bool,
    pub extended_thinking: bool,
    pub max_context_tokens: Option<u32>,
    /// Whether this provider supports prompt caching.
    pub prompt_caching: bool,
    /// Whether this provider exposes cache metrics in responses.
    pub cache_metrics: bool,
}
```

### 7. Integration with OTel/Phoenix tracing

Cache metrics should flow into the existing OTel instrumentation layer (`src/otel/`):

```rust
/// Add to AgentLoopSpan attributes:
impl AgentLoopSpan {
    pub fn record_cache_metrics(&self, usage: &TokenUsage) {
        let cached = usage.total_cached_tokens();
        if cached > 0 {
            self.span.set_attribute(
                otel::attribute!("cache.hit_ratio" => usage.cache_hit_ratio());
            );
            self.span.set_attribute(
                otel::attribute!("cache.tokens" => cached as i64);
            );
        }
    }
}
```

---

## Consequences

### What becomes easier
- **Cost optimization**: Agents can be configured to prefer caching for repeated system prompts
- **Latency reduction**: Cached responses return faster (especially valuable in tool loops)
- **Observability**: Cache hit ratios visible in traces and metrics
- **Provider migration**: Caching config is ignored for providers that don't support it

### What becomes harder
- **Testing**: Cache behavior depends on provider state, making tests non-deterministic
- **Debugging**: Cache hits can mask actual prompt behavior; need clear logging
- **Provider differences**: OpenAI uses headers, Anthropic/Bedrock use content markers, Gemini uses resource objects; abstraction leaks
- **TTL management**: Different providers support different TTL values (5m vs 1h vs 1d); configuration must be validated per-provider
- **Bedrock Converse API**: Uses `CachePointBlock` enum from the AWS SDK rather than `cache_control` JSON; different serialization path from Anthropic even though the semantics are similar

### Open Questions for Future ADRs
- **Cache invalidation strategy**: When to force-fresh vs. use cache (e.g., after tool definition changes)
- **Cost attribution**: How to split billing between cached/non-cached tokens in multi-turn sessions
- **Bedrock Converse API only**: Our Bedrock provider uses the Converse API exclusively, so we only need `CachePointBlock` — not the InvokeModel `cache_control` format. If a future provider adds InvokeModel support, that path would be needed separately.
- **Local provider caching**: Whether vLLM's prefix caching should be reported in `ProviderCapabilities` even without metrics exposure

---

## Implementation Checklist

- [ ] Add `PromptCacheConfig`, `CachePolicy`, `RequestCacheConfig` to `src/config/`
- [ ] Extend `TokenUsage` with cache fields (OpenAI, Anthropic, Gemini, Bedrock)
- [ ] Add `CacheInfo` to `CompletionMetadata`
- [ ] Update `ProviderCapabilities` with `prompt_caching` and `cache_metrics`
- [ ] Implement automatic caching header in OpenAI provider
- [ ] Implement `cache_control` (automatic + explicit breakpoints) in Anthropic provider
- [ ] Implement `CachePointBlock` in Bedrock Converse API provider (system, messages, tools)
- [ ] Verify AWS SDK `CachePointBlock` supports `ttl` field; if not, skip TTL for Bedrock (default 5-min TTL applies automatically)
- [ ] Implement CachedContent resource support in Gemini provider *(if Gemini exists)*
- [ ] Implement header forwarding in OpenRouter provider
- [ ] Update stream parsing to extract cache metrics from all provider responses
- [ ] Add cache metrics to OTel spans
- [ ] Add example: `examples/cache_metrics.rs`
- [ ] Update `TODO.md` to mark caching complete
- [ ] Add per-provider TTL validation (Anthropic: 5m/1h, Bedrock: 5m/1h, Gemini: 1h-24h)
- [ ] Test cache metrics extraction with live provider responses

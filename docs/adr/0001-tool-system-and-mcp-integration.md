# ADR-0001: Tool System & MCP Integration

**Status:** Proposed
**Date:** 2026-02-04
**Context tags:** [tool-system] [mcp] [prompting]

---

## Context

agent-driver-rs provides a unified abstraction over multiple LLM providers. All major providers now support "tool use" (also called "function calling") — a mechanism where the model can request structured actions from the host application. We need to understand how tools flow through the system at the **prompt level** before implementing MCP dynamic tool discovery.

This ADR focuses on **prompting mechanics**: how tools get described to models, how models request tool execution, and how results flow back. It is not a code implementation plan.

---

## Part 1: How Tool Use Works (The Prompt Mechanics)

### 1.1 The Tool Use Conversation Loop

Tool use is a multi-turn conversation pattern. The model never executes tools directly — it requests them, and the host application decides whether and how to execute.

```
┌─────────┐         ┌─────────┐         ┌──────────┐
│  Host    │         │  Model  │         │  Tool    │
│  App     │         │  (LLM)  │         │  Runtime │
└────┬─────┘         └────┬────┘         └────┬─────┘
     │                    │                   │
     │  1. Request        │                   │
     │  (system + msgs    │                   │
     │   + tool defs)     │                   │
     │───────────────────>│                   │
     │                    │                   │
     │  2. Response        │                   │
     │  (tool_use block)  │                   │
     │<───────────────────│                   │
     │                    │                   │
     │  3. Execute tool   │                   │
     │────────────────────────────────────────>│
     │                    │                   │
     │  4. Tool result    │                   │
     │<────────────────────────────────────────│
     │                    │                   │
     │  5. Continue       │                   │
     │  (msgs + result)   │                   │
     │───────────────────>│                   │
     │                    │                   │
     │  6. Final response │                   │
     │<───────────────────│                   │
```

**Key insight:** Steps 2-5 can repeat multiple times. The model may chain tool calls, call multiple tools in parallel, or call tools conditionally based on previous results. The host controls the loop.

### 1.2 What Gets Sent to the Model

Every completion request that involves tools has three parts in the prompt:

#### A. Tool Definitions (the "menu")

These tell the model what tools exist. Each definition includes:
- **name** — identifier the model uses to call it
- **description** — natural language explaining what it does and when to use it
- **input_schema** — JSON Schema describing the expected parameters

The description is the most important part for prompting. Models use it to decide *when* to call a tool. A bad description means the model won't call it, or will call it at the wrong time.

**Anthropic/Claude format:**
```json
{
  "tools": [
    {
      "name": "read_file",
      "description": "Read the contents of a file at the given path. Use this when you need to examine file contents.",
      "input_schema": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string",
            "description": "Absolute path to the file"
          }
        },
        "required": ["path"]
      }
    }
  ]
}
```

**OpenAI format (also used by OpenRouter):**
```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "read_file",
        "description": "Read the contents of a file at the given path.",
        "parameters": {
          "type": "object",
          "properties": {
            "path": {
              "type": "string",
              "description": "Absolute path to the file"
            }
          },
          "required": ["path"]
        }
      }
    }
  ]
}
```

The formats differ in nesting, but the semantic content is identical. Our `ToolFormat` serializer handles both.

#### B. Messages (the conversation so far)

The full message history, including any prior tool use/result turns. Models need to see the full chain to understand what's already been tried.

#### C. System Prompt

Can include guidance about tool use behavior, e.g., "Always confirm before using destructive tools" or "Prefer reading files before modifying them."

### 1.3 What the Model Returns

When the model decides to use a tool, it returns a **tool_use content block** instead of (or alongside) text:

**Claude format (streaming):**
```
content_block_start  → { type: "tool_use", id: "toolu_abc", name: "read_file" }
content_block_delta  → { type: "input_json_delta", partial_json: "{\"path\":" }
content_block_delta  → { type: "input_json_delta", partial_json: " \"/src/main.rs\"}" }
content_block_stop   → {}
```

**OpenAI format (streaming):**
```
delta: { tool_calls: [{ index: 0, id: "call_abc", function: { name: "read_file", arguments: "" } }] }
delta: { tool_calls: [{ index: 0, function: { arguments: "{\"path\":" } }] }
delta: { tool_calls: [{ index: 0, function: { arguments: " \"/src/main.rs\"}" } }] }
finish_reason: "tool_calls"
```

Both formats stream the tool name first, then incrementally stream the JSON arguments. The arguments arrive as partial strings that must be concatenated.

**Important:** The model also sets `stop_reason: "tool_use"` (Claude) or `finish_reason: "tool_calls"` (OpenAI) to signal the host should execute tools before continuing.

### 1.4 How Results Get Fed Back

After the host executes the tool, the result goes back as a new message in the conversation:

**Claude format:**
```json
{
  "role": "user",
  "content": [
    {
      "type": "tool_result",
      "tool_use_id": "toolu_abc",
      "content": "fn main() {\n    println!(\"Hello\");\n}",
      "is_error": false
    }
  ]
}
```

**OpenAI format:**
```json
{
  "role": "tool",
  "tool_call_id": "call_abc",
  "content": "fn main() {\n    println!(\"Hello\");\n}"
}
```

The model then continues generating, potentially calling more tools or producing a final text response.

### 1.5 How Our Codebase Maps to This

```
ToolDefinition          → Serialized into the request's "tools" array
  .name: ToolName       → The identifier the model calls
  .description: String  → Natural language for the model
  .input_schema         → JSON Schema for parameters

ToolFormat::serialize_tools()  → Converts definitions to Claude or OpenAI JSON format
ToolFormat::serialize_result() → Converts results back to the right format

StreamDelta::ToolUseStart { id, name }    → Model requesting a tool
StreamDelta::ToolInputDelta { partial }   → Streaming the arguments JSON

Session::process_tool_calls()  → Host-side: find tool, execute, add result to history
Session::execute_tool()        → Look up Tool in Registry, call execute(), push result
```

### 1.6 What Makes a Good Tool Description

The description field is a **prompt to the model**. It governs whether the model will use the tool correctly.

**Effective patterns:**
- State what the tool does: "Read the contents of a file at the given path"
- State when to use it: "Use this when you need to examine existing code before making changes"
- State constraints: "Only works with files under 10MB. Returns an error for binary files."
- Describe parameter semantics in the schema's `description` fields, not just names

**Anti-patterns:**
- Vague descriptions: "File utility" (model won't know when to use it)
- Missing parameter descriptions: Model will guess at expected formats
- Overly long descriptions: Models weight recency; put key info first

---

## Part 2: MCP Protocol — Dynamic Tool Negotiation

### 2.1 What MCP Solves

Static tools are compiled into the host application. MCP (Model Context Protocol) lets external processes expose tools at runtime. A file server, database, git tool, or any other capability can be a separate process that the host discovers dynamically.

The key difference from static tools: **the host doesn't know what tools exist until it asks the MCP server.**

### 2.2 MCP Architecture

```
┌──────────────┐     stdio/SSE      ┌──────────────┐
│  Host (us)   │ ←────────────────→ │  MCP Server  │
│  MCP Client  │     JSON-RPC       │  (external)  │
└──────────────┘                    └──────────────┘
```

MCP uses JSON-RPC 2.0 over one of two transports:
- **stdio** — MCP server is a child process, communicate over stdin/stdout
- **SSE** — MCP server is a remote HTTP endpoint (out of scope for v1)

### 2.3 The Negotiation Sequence

```
Host                                    MCP Server
  │                                        │
  │  1. initialize                         │
  │  { protocolVersion, capabilities,      │
  │    clientInfo }                         │
  │───────────────────────────────────────→│
  │                                        │
  │  2. initialize response                │
  │  { protocolVersion, capabilities,      │
  │    serverInfo }                         │
  │←───────────────────────────────────────│
  │                                        │
  │  3. initialized (notification)         │
  │  {}                                    │
  │───────────────────────────────────────→│
  │                                        │
  │  4. tools/list                         │
  │  {}                                    │
  │───────────────────────────────────────→│
  │                                        │
  │  5. tools/list response                │
  │  { tools: [...] }                      │
  │←───────────────────────────────────────│
  │                                        │
  │  ... (tools are now registered)        │
  │                                        │
  │  6. tools/call                         │
  │  { name, arguments }                   │
  │───────────────────────────────────────→│
  │                                        │
  │  7. tools/call response                │
  │  { content: [...], isError }           │
  │←───────────────────────────────────────│
```

### 2.4 Phase 1: Capability Negotiation

The `initialize` handshake establishes:
- **Protocol version** — both sides must agree
- **Capabilities** — what the server supports: tools, resources, prompts, logging
- **Server info** — name, version (useful for debugging)

The client declares its capabilities too (e.g., whether it supports `roots`, `sampling`). This is a bilateral negotiation — both sides advertise what they can do.

**After initialize, the client sends `initialized`** — a notification (no response expected) that signals "I'm ready, start accepting requests."

### 2.5 Phase 2: Tool Discovery

`tools/list` returns an array of tool definitions:

```json
{
  "tools": [
    {
      "name": "read_file",
      "description": "Read a file's contents",
      "inputSchema": {
        "type": "object",
        "properties": {
          "path": { "type": "string" }
        },
        "required": ["path"]
      }
    }
  ]
}
```

**This maps directly to our `ToolDefinition`:**
- `name` → `ToolName`
- `description` → `String`
- `inputSchema` → `ToolSchema`
- Source becomes `ToolSource::Mcp { server_name }`

MCP also supports optional **annotations** on tools:
- `destructive: bool` — does the tool modify state?
- `idempotent: bool` — safe to retry?
- `openWorldHint: bool` — does it interact with external systems?

These map to our existing `ToolAnnotations` struct.

### 2.6 Phase 3: Tool Execution

When the LLM requests a tool that came from an MCP server, the host routes the call:

1. LLM returns `tool_use { name: "read_file", input: {...} }`
2. Host finds `read_file` in `ToolRegistry`
3. Registry returns an `McpToolWrapper` (implements `Tool` trait)
4. `McpToolWrapper::execute()` sends `tools/call` to the MCP server
5. MCP server returns `{ content: [...], isError: false }`
6. Result is converted to `ToolResult` and fed back to the LLM

**The model doesn't know or care whether a tool is native or MCP.** The tool definitions look identical in the prompt. This is the key architectural win — MCP tools and native tools are interchangeable from the model's perspective.

### 2.7 Tool Lifecycle: Dynamic Registration and Updates

MCP servers can notify the host that tools have changed:

```
MCP Server → Host: notifications/tools/list_changed
Host → MCP Server: tools/list  (re-fetch)
```

This means the tool set can change between completion requests. The `ToolRegistry` already supports runtime add/remove, so this maps naturally. The flow:

1. Receive `tools/list_changed` notification
2. Call `tools/list` to get the new set
3. Diff against current registry
4. Unregister removed tools, register new ones
5. Next completion request automatically picks up the new set

### 2.8 Multiple MCP Servers

A host can connect to multiple MCP servers simultaneously. Tool names should be namespaced to avoid collisions:

```
Server "filesystem" → tools: read_file, write_file, list_dir
Server "database"   → tools: query, execute, list_tables
```

If two servers expose the same tool name, the host must decide: namespace them (e.g., `filesystem__read_file`), prefer one, or reject the collision. This is a design decision for ADR-0002.

---

## Part 3: How This Connects to Our Providers

### 3.1 Provider-Agnostic Tool Flow

```
                    ┌──────────────┐
                    │   Session    │
                    │              │
                    │  ToolRegistry│ ← contains both Native and MCP tools
                    └──────┬───────┘
                           │
                     .list() → Vec<ToolDefinition>
                           │
                    ┌──────▼───────┐
                    │ Completion   │
                    │ Request      │ ← tools: Vec<ToolDefinition>
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         Anthropic      OpenAI     Bedrock/etc.
         ToolFormat     ToolFormat
         ::claude()     ::openai()
              │            │            │
              ▼            ▼            ▼
         Provider-specific JSON in API request
```

Each provider serializes the same `Vec<ToolDefinition>` into its own format. The model sees tool definitions. It doesn't know whether they came from native Rust code or an MCP server.

### 3.2 Provider-Specific Tool Behavior

| Aspect | Claude/Anthropic | OpenAI | Bedrock | OpenRouter |
|--------|-----------------|--------|---------|------------|
| Format key | `input_schema` | `parameters` | `inputSchema` | `parameters` (OpenAI) |
| Streaming args | `input_json_delta` | `function.arguments` | `toolUse.input` | `function.arguments` |
| Stop signal | `stop_reason: "tool_use"` | `finish_reason: "tool_calls"` | `stopReason: "tool_use"` | `finish_reason: "tool_calls"` |
| Parallel calls | Yes (multiple blocks) | Yes (multiple indices) | Yes | Yes |
| Strict mode | No | Optional (`strict: true`) | No | Depends on model |

### 3.3 The Agentic Loop

For a fully autonomous agent, the session runs a loop:

```
loop {
    response = session.send(user_input)

    if response.has_tool_use() {
        // Execute all tool calls (native or MCP, doesn't matter)
        session.process_tool_calls(&response)

        // Continue the conversation with tool results
        response = session.send("")  // empty prompt, model continues
    } else {
        // Model is done, show response to user
        break
    }
}
```

This loop is the same regardless of whether tools are native or MCP. The abstraction boundary is at `ToolRegistry` — everything above it is provider-agnostic and source-agnostic.

---

## Decision

Before implementing MCP integration, this ADR establishes the mental model:

1. **Tools are prompt components** — they are descriptions serialized into API requests, not code the model executes
2. **Tool descriptions are prompts** — quality of description directly impacts model behavior
3. **MCP tools and native tools are identical at the prompt level** — the model never sees the difference
4. **MCP adds a negotiation layer** — initialize → discover → execute, over JSON-RPC
5. **The registry is the integration point** — MCP tools register/unregister via the same `ToolRegistry`

## Consequences

### What becomes easier
- MCP implementation can focus on the transport/protocol layer without touching provider code
- Tool descriptions can be iterated independently of tool implementations
- Multiple MCP servers compose naturally through the registry

### What becomes harder
- Tool name collisions across MCP servers need a strategy
- MCP server lifecycle management (startup, crash recovery, reconnection) adds complexity
- Tool set changes between turns could surprise models mid-conversation

### Open Questions for Future ADRs
- **ADR-0002:** Tool namespacing strategy for multiple MCP servers
- **ADR-0003:** MCP server lifecycle management (spawn, monitor, reconnect)
- **ADR-0004:** Agentic loop implementation (auto tool execution, max iterations, guardrails)

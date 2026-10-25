# ADR-0010: Interrupts for Human-in-the-Loop Parking and Durable Resume

**Status:** Proposed
**Date:** 2026-10-05
**Context tags:** [agent-loop] [session] [hitl] [lifecycle] [tools]

---

## Context

Agent runs need to pause for external decisions. The canonical case is human
approval of a destructive tool call; the same shape covers a review of the
output by another process or a value that an external system returns hours
later. Two flavors exist:

- **In-process parking** - the pause lives inside the same runtime (an approval
  prompt, a channel receive). This already works today: `Tool::execute` is
  async and receives a `ToolContext` carrying a child cancellation token, so a
  tool can await an external event indefinitely and remain cancellable.
- **Durable park/resume** - state is serialized and the run resumes later,
  possibly in another process (LangGraph-style checkpointing). This is not
  supported: `AgentLoop::run` always appends a new user message (there is no
  resume entry point), `LoopStopReason` has no `Interrupted` variant, and
  `SessionBuilder` cannot be seeded with prior history.

Gaps in what exists:

- `ToolAnnotations.requires_confirmation` exists but nothing reads it; there is
  no approval gate between `execute_tools` phase 1 (emit `ToolCallStart`) and
  phase 2 (execute via `join_all`).
- A parked tool cannot report "waiting" to observers; there is no
  mid-execution tool-to-observer channel.
- `Message` and `ContentBlock` already derive serde, so history export is
  nearly free; import (seeding a builder) is the missing half.

### Decision drivers

- Approval gates MUST fail closed: when a hook is configured, a hook error or
  the absence of a decision blocks execution.
- A parked tool MUST remain cancellable through the existing token hierarchy.
- The tool-execution boundary MUST remain the canonical park point, the same
  seam ADR-0008 drains nudges at.
- In-process HITL SHOULD NOT require persistence infrastructure.
- The durable-resume shape MUST be recorded now so later work attaches without
  reworking callers: `LoopStopReason` stays `#[non_exhaustive]`, and the
  resume entry point mirrors `continue_streaming` semantics.

## Decision

### Considered options

1. **Approval via observer veto** - rejected: the observer is
   notification-only, and giving it blocking or veto power conflates
   observation with control flow.
2. **Approval hook trait consulted in `execute_tools`** - chosen for
   in-process parking.
3. **Durable checkpoint/resume** - deferred, not rejected. The named gaps
   below own it.

### Chosen design (in-process)

- An `ApprovalHook` trait, consulted between phase 1 and phase 2 of
  `execute_tools` for tools annotated `requires_confirmation`.
- The default hook allows everything (backwards compatible). With a hook
  configured, errors and non-approval block the tool and surface as a tool
  error result, so the model can react under `continue_on_tool_error`.
- Parking is implemented by the hook awaiting whatever it needs (UI prompt,
  channel); cancellation flows through the existing child token.

### Deferred: durable resume, with owning gaps

- `AgentLoop::resume()` that continues from existing history instead of
  appending a user message.
- `LoopStopReason::Interrupted` and a parked-state record in `AgentOutcome`.
- History seeding on `SessionBuilder` plus `Session::export_history()`.
- Semantics for partial tool results: which results must be persisted before
  the park is acknowledged, and whether re-executed tools must be idempotent.

**This ADR records a constraint, not scheduled work.** The drivers bind
adjacent changes and are tracked in `TODO.md` under "Future Flexibility".

## Consequences

### What becomes easier

- HITL approval without leaving the typed loop; annotations become live
  instead of decorative.
- One canonical park/drain boundary shared with ADR-0008.

### What becomes harder

- The `join_all` barrier means one parked tool parks the whole round's history
  writes. That is semantically correct, but sibling tool results wait; this
  needs documentation.
- Approval UX wants mid-execution progress events; that channel does not exist
  (the same gap as the Aura-compat MCP progress item in TODO.md).
- Durable resume defers the hard questions (partial results, idempotency,
  mid-round persistence) rather than answering them.

### Failure modes

- Hook timeout or error: fail closed; the tool call becomes an error result.
- Cancellation while parked: the token cascades; the hook and tool observe
  cancellation.
- Process death while parked: state loss; the durable variant owns this case.

## Links

- ADR-0008 (nudges) shares the tool-boundary seam.
- `TODO.md` section "Future Flexibility" tracks the binding constraints.

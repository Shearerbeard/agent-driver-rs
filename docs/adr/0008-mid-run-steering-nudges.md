# ADR-0008: Mid-Run Steering via Queued User Nudges

**Status:** Proposed
**Date:** 2026-10-05
**Context tags:** [agent-loop] [session] [streaming] [concurrency]

---

## Context

Interactive agents need mid-run steering: the user watches streamed output and
types a correction ("focus on X", "stop using that tool") while the agent
works. Claude Code queues such input and delivers it at the next turn boundary.

Today the only mid-run input channel is cancellation
(`AgentLoop::with_cancellation`). `Session::add_message` takes `&self` and the
session uses split locks, so a UI task *can* push a user message while a stream
is in flight. Doing so is unsafe:

- **Causal inversion.** The agent loop appends the assistant message only after
  collecting the response (`add_assistant_to_history` in
  `src/agent/driver.rs`). A nudge pushed during collection serializes *before*
  the assistant message it chronologically follows.
- **Tool-call adjacency violation.** Anthropic rejects histories where a user
  message sits between an assistant `tool_use` block and its `tool_result`
  messages. The window between adding the assistant message and writing tool
  results spans the entire parallel tool execution, which is exactly when a
  user is most likely to nudge.
- **Snapshot freeze.** `Session::build_completion_request()` freezes state at
  send time, so an in-flight turn can never see a nudge regardless of timing.

### Decision drivers

- Injected messages MUST NOT land between a `tool_use` block and its
  `tool_result` messages; Anthropic returns 400 for that history shape.
- The messages lock MUST NOT be held across a provider call.
- History MUST preserve causal order.
- Delivery SHOULD reuse the existing per-turn re-snapshot
  (`continue_streaming`), not a side channel into the request builder.
- Nudge application SHOULD surface through the observer so UIs can render it.
- Mutating an in-flight completion is out of scope: no provider supports it.
  Cancel-and-redirect ("interrupt steering") is a separate, later decision.

## Decision

### Considered options

1. **Direct `add_message` from the caller** - possible today. Rejected:
   correctness depends on caller timing, and both hazards above bite in
   practice.
2. **Cancel-and-redirect only** - cancel the loop, append the nudge, re-run.
   Rejected as the whole story: it discards partial work, resends context, and
   is an abrupt user experience. Kept as a complementary mode for urgent
   stops.
3. **Queued nudges drained at the tool boundary** - chosen.

### Chosen design

- `Session` gains a nudge queue (`tokio::mpsc` or a locked `VecDeque<String>`)
  and a `nudge(text)` method callable from any task.
- The agent loop drains the queue at one point: after `execute_tools` phase 3
  writes all tool results and before `continue_streaming()` snapshots the next
  request. Adjacency and causal order hold by construction. The queue is also
  drained before the initial `send_streaming`.
- Drained nudges are appended as user message(s); the next request snapshot
  includes them. A new `AgentEvent::NudgesApplied { count }` notifies
  observers.
- Nudges queued when the loop ends remain queued; the next `run()` drains them
  before its initial send.

**This ADR records a constraint, not scheduled work.** No implementation is
planned now. The drivers above bind adjacent changes and are tracked in
`TODO.md` under "Future Flexibility".

## Consequences

### What becomes easier

- Steering UX without cancel-and-restart.
- One canonical injection point at the tool boundary, the same seam ADR-0010
  (interrupts) parks at, so both features share a single hook.

### What becomes harder

- A second stateful channel into the loop; race coverage needed in tests.
- Delivery latency is one turn: a nudge never affects the in-flight response.
- Queue lifecycle semantics need decisions at implementation time: what
  happens to queued nudges on cancel, and whether the drain runs before or
  after the cancellation check.

### Open questions

- Multiple nudges per round: one batched user message or several?
- Should `AgentOutcome` report nudges that were never delivered?

## Links

- ADR-0010 (interrupts) shares the tool-boundary seam.
- `TODO.md` section "Future Flexibility" tracks the binding constraints.

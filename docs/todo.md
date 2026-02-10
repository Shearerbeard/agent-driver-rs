# Benchmark Results & Investigation Items

## Run: 2026-02-09 (gpt-4o-mini, n=30, warmup=3, release build)

```
Scenario         Library                Min   Median     Mean   StdDev      Max       Alloc  vs
────────────────────────────────────────────────────────────────────────────────────────────────────────────
cold_start       agent-driver-rs       59us     89us     87us     16us    110us      4.6 KB
cold_start       rig.rs                60us     76us     81us     25us    169us      8.8 KB  ◀ 1.2x
────────────────────────────────────────────────────────────────────────────────────────────────────────────
ttft             agent-driver-rs    377.8ms  413.7ms  436.0ms   61.4ms  685.7ms     87.9 KB  ◀ 1.2x
ttft             rig.rs             391.5ms  509.2ms  616.7ms  346.8ms    1.93s    120.6 KB
────────────────────────────────────────────────────────────────────────────────────────────────────────────
tool_roundtrip   agent-driver-rs    990.8ms    1.16s    1.27s  383.9ms    2.95s    241.9 KB  ◀ 1.5x
tool_roundtrip   rig.rs               1.43s    1.78s    1.81s  267.0ms    2.55s    383.9 KB
────────────────────────────────────────────────────────────────────────────────────────────────────────────
mcp_discovery    agent-driver-rs      320us    415us    455us    118us    725us    213.2 KB
mcp_discovery    rig.rs               327us    393us    456us    131us    999us    210.3 KB  ◀ 1.1x
────────────────────────────────────────────────────────────────────────────────────────────────────────────
mcp_roundtrip    agent-driver-rs      1.31s    1.76s    1.83s  558.1ms    4.22s    675.5 KB  ◀ 1.3x
mcp_roundtrip    rig.rs               1.76s    2.20s    2.30s  507.2ms    4.25s    622.6 KB
```

Errors during run: 2x "Empty response from agent loop" (ours/mcp_rt), 1x "MaxTurnError" (rig/mcp_rt)

## Analysis

### Where we win
- **ttft (1.2x)**: Consistent advantage. Our median 414ms vs rig's 509ms. Notably, our stddev is 61ms vs rig's 347ms — we're not just faster, we're dramatically more consistent. Rig had a 1.93s max outlier.
- **tool_roundtrip (1.5x)**: Largest win. Our streaming agent loop completes a full tool call round in 1.16s median vs rig's 1.78s. Both now stream, so this is a genuine throughput advantage.
- **mcp_roundtrip (1.3x)**: Same pattern — our agent loop + MCP tool execution is faster end-to-end.

### Where rig wins
- **cold_start (1.2x rig)**: Rig's object construction is ~13us faster at median (76us vs 89us). Negligible in absolute terms but we allocate ~half the memory (4.6 KB vs 8.8 KB). Not a concern.
- **mcp_discovery (1.1x rig)**: Virtual tie at ~400us median. Both just call `list_all_tools()` over stdio now (after the fairness fix). The 22us difference is noise.

## Investigation Items for Future Sessions

### HIGH PRIORITY

- [ ] **cold_start: 89us vs 76us — profile our SessionBuilder overhead**
  - We lose despite allocating half the memory (4.6 KB vs 8.8 KB). The 13us gap suggests overhead in `SessionBuilder::build()` or `OpenAiProvider::new()` — possibly the split-lock `RwLock` initialization or `ToolRegistry::new()`. Worth profiling with `flamegraph` to identify if there's unnecessary work during construction. Low absolute impact but easy to fix if found.

- [ ] **ttft stddev: investigate rig's 347ms stddev vs our 61ms**
  - Our consistency advantage is striking. This is likely our stream adapter buffering + `biased` select providing more deterministic first-token delivery. Worth understanding *why* to ensure it's architectural and not accidental. Could inform documentation/marketing claims about reliability.

- [ ] **mcp_roundtrip: 2/30 "Empty response" errors from max_tool_depth=2**
  - When the model uses both allowed tool rounds for file listing, it sometimes doesn't produce a final text response before the depth cap. The harness skips these (correct behavior), but 6.7% error rate means we're measuring 28/30 iterations, not 30/30. Two options:
    1. Bump `max_tool_depth` to 3 (risk: reintroduces the asymmetry with rig's `max_turns(2)`)
    2. Use a simpler prompt that needs only 1 tool round (e.g., "What is the current directory?" which just needs `pwd`)
  - Rig also hit 1 MaxTurnError, so this is partly a model behavior issue with the prompt.

### MEDIUM PRIORITY

- [ ] **tool_roundtrip: our 384ms stddev — investigate outlier sources**
  - Min 991ms, max 2.95s. The 3x spread suggests occasional OpenAI API latency spikes hitting our streaming path harder than rig's. Could be worth adding retry/timeout instrumentation to understand if it's network or stream-processing overhead.

- [ ] **Allocation: our mcp_roundtrip uses 675 KB vs rig's 623 KB**
  - We're 8% higher on allocation despite being 1.3x faster on time. This is likely the `DynTool` wrapping + `ToolRegistry` overhead for MCP tools. Not urgent since we win on latency, but worth tracking — if allocation grows with tool count it could become a concern for large MCP deployments.

### LOW PRIORITY

- [ ] **Add `--json` output flag for CI integration**
  - Would enable tracking these numbers over time and catching regressions automatically.

- [ ] **Test with gpt-4o and claude-sonnet-4-5 via OpenRouter**
  - Current results are gpt-4o-mini only. Larger models have different streaming characteristics (bigger tokens, longer generation). Results may differ significantly.

- [ ] **StreamHandle drop doesn't cancel — consider adding CancellationToken propagation on drop**
  - The ttft scenario explicitly drops the stream but the underlying connection may linger. This was flagged in the original audit. Low impact for benchmarks (warmup absorbs it), but matters for production use where rapid stream abandonment is common.

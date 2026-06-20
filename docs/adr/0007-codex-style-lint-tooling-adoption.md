# ADR-0007: Codex-Style Lint & Tooling Adoption

**Status:** Proposed
**Date:** 2026-06-20
**Context tags:** [linting] [tooling] [quality] [ci]

## Context

The crate already adopted a progressive rust-seed lint baseline: clippy broad
groups at `warn` (`all`, `correctness`, `suspicious`, `complexity`, `perf`,
`style`, `pedantic`) plus seven waves of restriction lints at `deny`
(`LINT_WAVES.md`, all complete). That baseline catches agent-introduced slop
(`dbg`, `todo`, `panic_in_result_fn`, `clone_on_ref_ptr`, `wildcard_enum_match_arm`)
and enforces ownership clarity (`clone_on_ref_ptr`, `str_to_string`, `string_slice`,
`rc_buffer`, `rc_mutex`).

Three gaps remain versus mature OSS Rust codebases. The reference is
[openai/codex](https://github.com/openai/codex) `codex-rs/` — a 96% Rust
codebase with a disciplined lint setup. The comparison surfaced three missing
layers and a promotion gap.

### Decision drivers

- **Correctness MUST come before style.** The `await-holding-invalid-types`
  check catches a real deadlock-risk class (tokio guard held across `.await`
  in `Session`'s split locks); this alone justifies Layer A regardless of
  style preferences.
- **Reproducible builds MUST be enforceable.** Unpinned git deps
  (`mcp-openai-bridge`) drift silently; source pinning is non-negotiable for
  a library others depend on.
- **Each lint promotion SHOULD be independently committable.** The
  `LINT_WAVES.md` one-lint-per-commit loop has worked for 7 waves; Layer B
  follows the same discipline so review stays tractable.
- **Test code SHOULD retain `unwrap`/`expect` for ergonomics.** codex's
  `allow-unwrap-in-tests` pattern gates the deny to non-test code only; we
  adopt it rather than forcing `?` in every test assertion.
- **The async-trait ban MUST NOT be adopted yet.** We use `async_trait!` in
  three core traits (`Provider`, `Tool`, `AgentObserver`); removing it is a
  separate migration with its own ADR.

### Considered options

**Option A — Continue rust-seed baseline only (finish pedantic triage).**
The `LINT_WAVES.md` Future Work section lists six silenced pedantic lints
(`cast_lossless`, `map_unwrap_or`, `single_match_else`, `unnested_or_patterns`,
`needless_pass_by_value`, `redundant_closure_for_method_calls`) to triage.
This option promotes those six and stops. **Rejected because**: it leaves the
three structural gaps (no `clippy.toml`, no `deny.toml`, no
await-holding guard check) unaddressed. The await-holding gap is a
correctness issue, not a style preference — Option A does not fix it.

**Option B — Adopt codex lint set wholesale (including bans, edition, thiserror).**
Mirror codex's entire `codex-rs/` toolchain: `clippy.toml`, `deny.toml` with
the async-trait ban, edition 2024, `thiserror` 2. **Rejected because**: the
async-trait ban, edition migration, and `thiserror` 2 upgrade are each
large, independent migrations that touch every provider and trait
implementation. Bundling them into one ADR creates an all-or-nothing change
that blocks the high-value items (await-holding check, source pinning)
behind low-value-but-high-effort migrations. These belong in separate ADRs.

**Option C — Adopt codex lint layers incrementally (this ADR).**
Layer the codex tooling on top of the existing rust-seed baseline in four
independently-committable layers (`clippy.toml`, promote core set to `deny`,
`deny.toml`, `cargo-shear`). Defer edition 2024, `thiserror` 2, and
async-trait removal to separate ADRs. **Chosen because**: it delivers the
correctness-critical `await-holding-invalid-types` check and source pinning
immediately, while keeping each change reviewable via the proven
one-lint-per-commit loop. It does not require the large migrations Option B
bundles in.

**Option D — Add only cargo-deny (advisories + licenses), skip clippy changes.**
Add `deny.toml` for RUSTSEC/license scanning but leave the clippy
configuration unchanged. **Rejected because**: it misses the
`await-holding-invalid-types` correctness check, which is the highest-value
item in this ADR. Advisory scanning without the guard check is incomplete
protection.

## Decision

Layer openai/codex-style lint tooling on top of the existing rust-seed
baseline, in four independently-committable layers. Each layer is a separate
commit; no layer is merged until `cargo clippy --all-features -- -D warnings`
and `cargo test --all-features` are clean.

### Layer A — `clippy.toml` (conditional + anti-slop rules)

Add a `clippy.toml` at the crate root:

```toml
allow-unwrap-in-tests = true
allow-expect-in-tests = true
await-holding-invalid-types = [
    "tokio::sync::MutexGuard",
    "tokio::sync::RwLockReadGuard",
    "tokio::sync::RwLockWriteGuard",
]
large-error-threshold = 256
disallowed-methods = [
    { path = "std::env::var", reason = "env access belongs in config/ only" },
    { path = "serde_json::from_str", reason = "JSON parsing belongs in provider/ and streaming/ only" },
]
```

Add rustc anti-unicode lints to `[lints.rust]` in `Cargo.toml` (extends the
existing `non_ascii_idents = deny`):

```toml
mixed_script_confusables = { level = "deny", priority = 127 }
confusable_idents = { level = "warn", priority = 127 }
```

The `disallowed-methods` list is scoped to library code (`src/`, excluding
`src/bin/` and `examples/`) via the `disallowed-methods` path-prefix matching
or a `#[allow]` at the binary crate root. The `println!`/`eprintln!` ban is
deferred to Layer B (requires fixing the few diagnostic prints in
`src/tool/mcp.rs` first).

### Layer B — Promote codex core clippy set to `deny`

Adopt codex's `[workspace.lints.clippy]` set as individual `deny` entries in
`Cargo.toml` `[lints.clippy]`. Process **one lint per commit**: fix all
violations, commit, move to the next. This is the iterative linting loop the
`LINT_WAVES.md` "Instructions for the loop" already prescribes.

**Flip our current `allow` → `deny`** ( Cargo.toml currently silences these):

- `uninlined_format_args` (Cargo.toml notes ~3 violations)
- `redundant_closure_for_method_calls` (Cargo.toml notes ~2 violations)
- `needless_borrowed_reference` (already `allow`)

**Add new `deny` entries** (codex set, not currently in our config):

- `unwrap_used`, `expect_used` — requires Layer A's test allowances and
  Tier 2 item T2.4 (const-constructible `MaxTokens` / `Default` impl) to
  remove the 6 `MaxTokens::new(4096).expect(...)` sites across
  `provider.rs`, `config/{anthropic,openai,bedrock,openrouter}.rs`, and
  `bin/chat.rs`. A further ~35 `.expect()` calls exist in provider
  implementations for hardcoded valid `ModelId`/`ToolName`/URL/JSON — these
  need either `const` constructors on the newtypes, `#[allow]` with
  recorded justification, or refactoring to `?`. T2.4 must land first;
  the provider `.expect()` cleanup may be staged across multiple commits.
- `redundant_clone`, `unnecessary_to_owned`, `needless_collect`
- `manual_clamp`, `manual_filter`, `manual_find`, `manual_flatten`,
  `manual_map`, `manual_memcpy`, `manual_non_exhaustive`, `manual_ok_or`,
  `manual_range_contains`, `manual_retain`, `manual_strip`,
  `manual_try_fold`, `manual_unwrap_or`
- `needless_borrow`, `needless_late_init`, `needless_option_as_deref`,
  `needless_question_mark`, `needless_update`
- `redundant_closure`, `redundant_static_lifetimes`
- `trivially_copy_pass_by_ref`, `uninlined_format_args`
- `unnecessary_filter_map`, `unnecessary_lazy_evaluations`,
  `unnecessary_sort_by`
- `await_holding_invalid_type`, `await_holding_lock` (Layer A config makes
  these effective for tokio guards)

**Sequencing constraint**: `expect_used = deny` is the last lint promoted in
this layer, after T2.4 lands. Everything else can proceed in any order.

### Layer C — `deny.toml` (cargo-deny)

Add a `deny.toml` at the crate root, modeled on codex's:

- **`[advisories]`**: build a reviewed ignore list. Each ignored advisory must
  identify the dependency path and a removal condition (codex's format: `{ id =
  "RUSTSEC-XXXX-YYYY", reason = "..." }`). Initial pass will surface
  advisories from `opentelemetry 0.27`, `aws-sdk-*`, `rmcp`, and
  `mcp-openai-bridge` transitive deps; each gets a reviewed-with-reason
  ignore entry until upstreams update.
- **`[licenses]`**: SPDX allow-list matching our dependency set (`MIT`,
  `Apache-2.0`, `Apache-2.0 WITH LLVM-exception`, `BSD-2-Clause`,
  `BSD-3-Clause`, `ISC`, `Zlib`, `CC0-1.0`). `confidence-threshold = 0.8`.
- **`[bans]`**: `multiple-versions = "warn"`. The `async-trait` ban codex
  uses is **deferred** — we use `async_trait!` in three core traits; removal
  is a separate ADR.
- **`[sources]`**: `unknown-registry = "deny"`, `unknown-git = "deny"`,
  `required-git-spec = "rev"`. **Real fix required**: pin
  `mcp-openai-bridge` (`Cargo.toml:62`) to an immutable `rev` before this
  layer can pass.

Add `make deny` to the `Makefile`:

```makefile
deny:
	cargo deny check
```

### Layer D — `cargo-shear` (unused dependency detection)

Run `cargo shear`, remove unused deps from `Cargo.toml`, add a
`[metadata.cargo-shear] ignored` section for platform-specific false positives
(as codex does for `openssl-sys`). Add `make unused-deps` to the `Makefile`:

```makefile
unused-deps:
	cargo shear
```

## Consequences

### Easier

- **Await-holding tokio guards becomes a compile error** — protects the
  `Session` split-lock invariant. This is the highest-value item: a real
  correctness bug class, not just style.
- **RUSTSEC advisories surface before they ship**; licenses are enforced at
  build time.
- **Git deps are immutable-pinned** — reproducible builds, no surprise
  upstream drift.
- **Unused deps pruned** — marginally faster compile times, clearer
  dependency surface.
- **Anti-slop `disallowed-methods`** keeps `env::var` in `config/` and JSON
  parsing in `provider/`/`streaming/`, enforcing module boundaries the
  architecture diagram already implies.

### Harder

- **Layer B requires fixing every violation as each lint is promoted.**
  Estimated low-count for `uninlined_format_args` /
  `redundant_closure_for_method_calls` (Cargo.toml notes 2–3 each). Higher
  for `unwrap_used` / `expect_used` (requires auditing every `unwrap()` /
  `expect()` in non-test `src/`).
- **`expect_used = deny` requires T2.4 first** — the 6
  `MaxTokens::new(4096).expect(...)` sites (`provider.rs:123`,
  `config/{anthropic,openai,bedrock,openrouter}.rs`, `bin/chat.rs`) must
  become const-constructible `MaxTokens` or `Default` impls. A further ~35
  `.expect()` calls in provider code (hardcoded valid `ModelId`,
  `ToolName`, URL, JSON parse) need individual resolution. This is the
  highest-effort lint in Layer B and may span multiple commits. Tier 2
  rust-design task T2.4 gates it.
- **`deny.toml` advisories may surface unmaintained transitive crates**
  (`opentelemetry 0.27`, `aws-sdk-*`, `rmcp`) requiring
  ignored-with-reason entries until upstreams update. Each ignore is a
  documented tech-debt entry, not a silent suppression.
- **Source pinning forces a `rev` lookup for `mcp-openai-bridge`** before
  Layer C can pass — a one-time coordination step.

### Out of scope

These are larger migrations that warrant their own ADRs and are not part
of this decision:

- Edition 2024 upgrade (CLAUDE.md Gotcha #5; codex uses 2024)
- `thiserror` 1 → 2 migration (codex uses `thiserror = "2.0.17"`)
- `async-trait` removal (codex bans it via `deny.toml`; we use it in three
  core traits — `Provider`, `Tool`, `AgentObserver`)
- CI workflow (`.github/workflows/ci.yml`) — complementary but separable;
  the `Makefile` already provides `make check`

## Verification

Each layer must leave the tree green before the next begins:

- **Layer A**: `cargo clippy --all-features -- -D warnings` clean with the new
  `clippy.toml`. Confirm a deliberate await-holding-guard test case fires the
  lint (e.g., a `#[cfg(test)]` snippet holding a `RwLockReadGuard` across an
  `.await`).
- **Layer B**: after each lint promotion, `cargo clippy --all-features -- -D
  warnings` clean. After the final lint, `cargo clippy --all-features --tests
  -- -D warnings` clean (test code uses the `allow-unwrap/expect-in-tests`
  allowance).
- **Layer C**: `cargo deny check` clean (advisories, bans, sources, licenses).
  Confirm `mcp-openai-bridge` resolves to the pinned `rev`.
- **Layer D**: `cargo shear` clean (or only listed `[metadata]` ignores).
- **Throughout**: `cargo test --all-features` (215+ tests) remains green.

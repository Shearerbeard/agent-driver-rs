# Progressive Lint Adoption Plan

Each wave adds lints to `[lints.clippy]` in Cargo.toml at `deny` level, then fixes all violations.
The loop should process ONE wave per iteration, commit, then proceed to the next.

## Status

- [x] Wave 1: Broad groups at `warn` (all, correctness, suspicious, complexity, perf, style, pedantic)
- [ ] Wave 2: Agent-critical safety (no panics, no debug leftovers)
- [ ] Wave 3: Ownership clarity (Arc clones, string handling)
- [ ] Wave 4: Exhaustiveness & correctness
- [ ] Wave 5: Code hygiene & style enforcement
- [ ] Wave 6: Concurrency & unsafe discipline
- [ ] Wave 7: Final restriction lints

## Wave 2: Agent-critical safety

These catch mistakes coding agents commonly make — panics, debug code left in, unhandled results.

```toml
dbg_macro = { level = "deny", priority = 127 }
todo = { level = "deny", priority = 127 }
unimplemented = { level = "deny", priority = 127 }
panic_in_result_fn = { level = "deny", priority = 127 }
unused_result_ok = { level = "deny", priority = 127 }
exit = { level = "deny", priority = 127 }
missing_assert_message = { level = "deny", priority = 127 }
tests_outside_test_module = { level = "deny", priority = 127 }
```

## Wave 3: Ownership clarity

Forces explicit Arc/Rc cloning and prevents implicit string operations that hide allocations.

```toml
clone_on_ref_ptr = { level = "deny", priority = 127 }
str_to_string = { level = "deny", priority = 127 }
string_add = { level = "deny", priority = 127 }
string_slice = { level = "deny", priority = 127 }
rc_buffer = { level = "deny", priority = 127 }
rc_mutex = { level = "deny", priority = 127 }
```

## Wave 4: Exhaustiveness & correctness

Forces exhaustive match arms and explicit error handling patterns.

```toml
wildcard_enum_match_arm = { level = "deny", priority = 127 }
rest_pat_in_fully_bound_structs = { level = "deny", priority = 127 }
assertions_on_result_states = { level = "deny", priority = 127 }
try_err = { level = "deny", priority = 127 }
error_impl_error = { level = "deny", priority = 127 }
return_and_then = { level = "deny", priority = 127 }
```

## Wave 5: Code hygiene & style enforcement

Catches style issues that make code harder for agents to parse and modify.

```toml
needless_raw_strings = { level = "deny", priority = 127 }
redundant_test_prefix = { level = "deny", priority = 127 }
semicolon_inside_block = { level = "deny", priority = 127 }
multiple_inherent_impl = { level = "deny", priority = 127 }
deref_by_slicing = { level = "deny", priority = 127 }
lossy_float_literal = { level = "deny", priority = 127 }
verbose_file_reads = { level = "deny", priority = 127 }
unused_trait_names = { level = "deny", priority = 127 }
```

## Wave 6: Concurrency & unsafe discipline

Important for this async-heavy codebase — prevents subtle concurrency bugs.

```toml
mutex_atomic = { level = "deny", priority = 127 }
mutex_integer = { level = "deny", priority = 127 }
mem_forget = { level = "deny", priority = 127 }
multiple_unsafe_ops_per_block = { level = "deny", priority = 127 }
undocumented_unsafe_blocks = { level = "deny", priority = 127 }
unnecessary_safety_comment = { level = "deny", priority = 127 }
unnecessary_safety_doc = { level = "deny", priority = 127 }
let_underscore_must_use = { level = "deny", priority = 127 }
infinite_loop = { level = "deny", priority = 127 }
```

## Wave 7: Final restriction lints

The remaining high-value restriction lints that require more sweeping changes.

```toml
impl_trait_in_params = { level = "deny", priority = 127 }
partial_pub_fields = { level = "deny", priority = 127 }
mod_module_files = { level = "deny", priority = 127 }
renamed_function_params = { level = "deny", priority = 127 }
field_scoped_visibility_modifiers = { level = "deny", priority = 127 }
allow_attributes_without_reason = { level = "deny", priority = 127 }
mixed_read_write_in_expression = { level = "deny", priority = 127 }
cfg_not_test = { level = "deny", priority = 127 }
```

## Instructions for the loop

1. Read this file to find the next unchecked wave
2. Add the lint entries from that wave to `Cargo.toml` under `[lints.clippy]`
3. Run `cargo clippy --all-features 2>&1` to find violations
4. Fix all violations (prefer fixing the code over `#[allow]`; use `#[allow]` only if the fix would be a major refactor)
5. Run `cargo clippy --all-features 2>&1` again to confirm clean
6. Run `cargo test --all-features` to confirm nothing broke
7. Mark the wave as `[x]` in this file
8. Commit with message: `lint: apply wave N — <brief description>`
9. Move to the next wave

.PHONY: check build test clippy clippy-tests fmt fmt-fix bench clean

# Default target
check: fmt clippy test

# Build with all features
build:
	cargo build --all-features

# Run all tests
test:
	cargo test --all-features

# Clippy (main code, deny warnings)
clippy:
	cargo clippy --all-features -- -D warnings

# Clippy including test code
clippy-tests:
	cargo clippy --all-features --tests -- -D warnings

# Format check
fmt:
	cargo fmt --check

# Format fix
fmt-fix:
	cargo fmt

# Run benchmarks
bench:
	cd bench && cargo run --release -- -n 20 -w 3

# Clean build artifacts
clean:
	cargo clean

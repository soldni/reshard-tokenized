# Repository Guidelines

## Project Structure & Module Organization
`src/lib.rs` contains the merge engine (file discovery, sharding, `.npy` concatenation, and `.csv.gz` metadata remapping). `src/main.rs` is the CLI entrypoint (`clap`) and logging bootstrap (`tracing`).

Integration coverage lives in `tests/cli_integration.rs` and exercises the compiled binary end to end. Test fixtures are under `tests/data/`:
- `tests/data/tokenized_input/` for tokenized shard inputs
- `tests/data/source_jsonl/` for source documents used to regenerate fixtures
- `tests/data/README.md` for fixture regeneration steps

## Build, Test, and Development Commands
- `cargo build` compiles the crate.
- `cargo run -- --input-path tests/data/tokenized_input --num-files 2 --output-path /tmp/merged` runs the CLI locally.
- `cargo test` runs unit/integration tests (including CLI behavior checks).
- `cargo fmt -- --check` verifies standard formatting.
- `cargo clippy --all-targets --all-features -- -D warnings` enforces lint cleanliness before review.

## Coding Style & Naming Conventions
Use Rust 2024 idioms and rustfmt defaults (4-space indentation, trailing commas where appropriate). Follow standard naming:
- `snake_case` for functions/variables
- `PascalCase` for structs/enums
- `SCREAMING_SNAKE_CASE` for constants

Keep error handling explicit with `Result<_, MergeError>` and context-rich variants (`thiserror`). Prefer deterministic filesystem ordering (`sort`) and structured logs via `tracing`.

## Testing Guidelines
Write behavioral tests in `tests/cli_integration.rs` with names like `cli_<expected_behavior>`. Validate both output files and content semantics (byte concat for `.npy`, offset remap for CSV metadata).

When changing fixture data, regenerate with commands in `tests/data/README.md` and commit updated fixtures in the same PR.

## Commit & Pull Request Guidelines
Commit messages in this repo are short and imperative (for example: `Expand .gitignore entries`, `setup for release`). Keep subject lines focused and scoped to one logical change.

PRs should include:
- what changed and why
- commands run (`cargo test`, `cargo clippy`, etc.)
- fixture/version notes when applicable
- linked issue or release context if relevant

## Release Notes
The release workflow publishes on GitHub release tags and checks `Cargo.toml` version matches the tag (`vX.Y.Z`). After publish, CI bumps the next minor version on the default branch.

<p align="center">
  <img src="https://github.com/soldni/reshard-tokenized/blob/master/assets/logo.png?raw=true" alt="Library logo" width="384"/>
</p>

# reshard-tokenized

A Rust CLI that merges tokenized shard files from a directory tree into one or more output shards:
- `.npy` files are concatenated per shard
- `.csv.gz` metadata rows are merged with remapped offsets

## Prerequisites

- Rust toolchain (stable)
- Cargo (included with Rust)

## Build

```bash
cargo build
```

## Run

```bash
cargo run -- \
  --input-path tests/data/tokenized_input \
  --num-files 2 \
  --output-path /tmp/merged
```

CLI help:

```bash
cargo run -- --help
```

## Test

```bash
cargo test
```

Integration tests use fixtures in `tests/data/tokenized_input`.

## Output

- If `--num-files 1`, output is written as `<output-path>.npy` and `<output-path>.csv.gz`
- If `--num-files > 1`, output is written as:
  - `<output-path>/00000000.npy`, `<output-path>/00000000.csv.gz`
  - `<output-path>/00000001.npy`, `<output-path>/00000001.csv.gz`
  - ...

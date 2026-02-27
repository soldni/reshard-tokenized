# Integration Test Fixtures

`tests/data/tokenized_input` contains fixture files used by `tests/cli_integration.rs`:
- 8 `.npy` files
- 8 `.csv.gz` files
- nested directory layout to validate recursive discovery

The files were generated from `tests/data/source_jsonl/*.jsonl` using Dolma with the `openai-community/gpt2` tokenizer.

Regenerate fixtures:

```bash
work=/tmp/dolma-fixtures-8
rm -rf "$work"
mkdir -p "$work/out" "$work/work-in" "$work/work-out"
cp tests/data/source_jsonl/*.jsonl "$work/"

UV_CACHE_DIR=.uv-cache uv run --with=dolma dolma tokens \
  --documents "$work"/doc-*.jsonl \
  --destination "$work/out" \
  --tokenizer.name_or_path openai-community/gpt2 \
  --tokenizer.eos_token_id 50256 \
  --processes 1 \
  --files_per_process 1 \
  --batch_size 2 \
  --work_dir.input "$work/work-in" \
  --work_dir.output "$work/work-out"
```

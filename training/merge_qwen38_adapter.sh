#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 BASE_MODEL ADAPTER_CHECKPOINT OUTPUT_DIR" >&2
  exit 2
fi

BASE_MODEL=$(realpath "$1")
ADAPTER_DIR=$(realpath "$2")
OUTPUT_DIR=$(realpath -m "$3")

if [[ ! -s "$BASE_MODEL/config.json" ]]; then
  echo "Base model is incomplete: $BASE_MODEL" >&2
  exit 1
fi
if [[ ! -s "$ADAPTER_DIR/adapter_config.json" ]]; then
  echo "Adapter checkpoint is incomplete: $ADAPTER_DIR" >&2
  exit 1
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite output path: $OUTPUT_DIR" >&2
  exit 1
fi
if ! command -v swift >/dev/null 2>&1; then
  echo "MS-SWIFT is not installed or its CLI is not on PATH" >&2
  exit 1
fi

swift export \
  --model "$BASE_MODEL" \
  --adapters "$ADAPTER_DIR" \
  --merge_lora true \
  --output_dir "$OUTPUT_DIR"

for filename in \
  processor_config.json \
  preprocessor_config.json \
  video_preprocessor_config.json \
  chat_template.jinja \
  tokenizer_config.json \
  generation_config.json
do
  if [[ -f "$BASE_MODEL/$filename" ]]; then
    cp -f "$BASE_MODEL/$filename" "$OUTPUT_DIR/$filename"
  fi
done

test -s "$OUTPUT_DIR/config.json"
test -s "$OUTPUT_DIR/model.safetensors.index.json"
echo "Merged checkpoint ready: $OUTPUT_DIR"


#!/usr/bin/env bash
set -u

ROOT_DIR="MP-Bench"
OUT_ROOT="annotated"
SCRIPT="build_annotated_from_log_source.py"

if [[ ! -f "$SCRIPT" ]]; then
  echo "Error: $SCRIPT not found in current directory."
  exit 1
fi

if [[ ! -d "$ROOT_DIR" ]]; then
  echo "Error: $ROOT_DIR directory not found."
  exit 1
fi

total=0
success=0
failed=0

# Process all: MP-Bench/<group>/(automatic|manual)/*.json
for input_json in "$ROOT_DIR"/*/automatic/*.json "$ROOT_DIR"/*/manual/*.json; do
  if [[ ! -f "$input_json" ]]; then
    continue
  fi

  # path parts: MP-Bench/<group>/<split>/<file>.json
  group="$(echo "$input_json" | cut -d'/' -f2)"
  split="$(echo "$input_json" | cut -d'/' -f3)"
  file_name="$(basename "$input_json")"

  output_dir="$OUT_ROOT/$group/$split"
  output_json="$output_dir/$file_name"

  mkdir -p "$output_dir"
  total=$((total + 1))

  echo "[$total] Processing: $input_json"
  if python "$SCRIPT" --input_json "$input_json" --output "$output_json"; then
    success=$((success + 1))
  else
    failed=$((failed + 1))
    echo "  -> Failed: $input_json"
  fi
done

echo ""
echo "Done."
echo "Total:   $total"
echo "Success: $success"
echo "Failed:  $failed"

if [[ $failed -gt 0 ]]; then
  exit 1
fi
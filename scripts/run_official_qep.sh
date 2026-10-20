#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
if [[ -f "$PROJECT_ROOT/server.env" ]]; then
  source "$PROJECT_ROOT/server.env"
fi
PYTHON="${PYTHON:-python}"
BACKEND_ROOT="${SQ_QEP_ROOT:-$PROJECT_ROOT/third_party/qep}"
if [[ ! -f "$BACKEND_ROOT/src/llama.py" ]]; then
  echo "Compatible external backend missing. Set SQ_QEP_ROOT; see docs/backends.md." >&2
  exit 2
fi
export NAP_AWQ_EMBED_BATCH="${NAP_AWQ_EMBED_BATCH:-1}"
export NAP_AWQ_LAYER_BATCH="${NAP_AWQ_LAYER_BATCH:-32}"

MODEL_PATH="${1:?usage: run_official_qep.sh MODEL_PATH OUTPUT.pt METHOD BITS [SEED]}"
OUTPUT_PATH="${2:?missing output path}"
METHOD="${3:?method must be rtn, gptq, awq, qep_gptq, or qep_awq}"
BITS="${4:?bits must be 2, 3, or 4}"
SEED="${5:-0}"

mkdir -p "$(dirname -- "$OUTPUT_PATH")"
TEMP_PATH="${OUTPUT_PATH}.incomplete"
rm -f -- "$TEMP_PATH"
cd "$BACKEND_ROOT/src"

base_method="$METHOD"
qep_flag=()
gptq_flag=()
case "$METHOD" in
  qep_gptq) base_method="gptq"; qep_flag=(--qep) ;;
  qep_awq) base_method="awq"; qep_flag=(--qep) ;;
  rtn|gptq|awq) ;;
  *) echo "Unsupported QEP method: $METHOD" >&2; exit 2 ;;
esac

if [[ "$base_method" == "gptq" ]]; then
  gptq_flag=(--act-order)
fi

"$PYTHON" -u llama.py "$MODEL_PATH" c4 "$base_method" \
  --seed "$SEED" \
  --nsamples 128 \
  --wbits "$BITS" \
  --groupsize 128 \
  "${gptq_flag[@]}" \
  "${qep_flag[@]}" \
  --save-model "$TEMP_PATH"

mv -f -- "$TEMP_PATH" "$OUTPUT_PATH"

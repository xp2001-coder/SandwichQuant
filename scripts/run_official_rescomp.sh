#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
if [[ -f "$PROJECT_ROOT/server.env" ]]; then
  source "$PROJECT_ROOT/server.env"
fi
PYTHON="${PYTHON:-python}"
BACKEND_ROOT="${SQ_RESCOMP_ROOT:-$PROJECT_ROOT/third_party/ResComp}"
if [[ ! -f "$BACKEND_ROOT/fake_quant/main.py" ]]; then
  echo "Compatible external backend missing. Set SQ_RESCOMP_ROOT; see docs/backends.md." >&2
  exit 2
fi

MODEL_PATH="${1:?usage: run_official_rescomp.sh MODEL_PATH OUTPUT.pt METHOD BITS [SEED]}"
OUTPUT_PATH="${2:?missing output path}"
METHOD="${3:?method must be gptq, gptaq, or rescomp_gptaq}"
BITS="${4:?bits must be 2, 3, or 4}"
SEED="${5:-0}"

mkdir -p "$(dirname -- "$OUTPUT_PATH")"
TEMP_PATH="${OUTPUT_PATH}.incomplete"
rm -f -- "$TEMP_PATH"
cd "$BACKEND_ROOT/fake_quant"
export NAP_QUANTIZE_ONLY=1
export NAP_EXPERIMENTS_ROOT="${NAP_EXPERIMENTS_ROOT:-$(dirname -- "$OUTPUT_PATH")/author_logs}"

entry="main.py"
extra=()
case "$METHOD" in
  gptq) ;;
  gptaq) extra+=(--asym_calibrate) ;;
  rescomp|rescomp_gptaq) entry="main_r.py"; extra+=(--asym_calibrate) ;;
  *) echo "Unsupported ResComp method: $METHOD" >&2; exit 2 ;;
esac

"$PYTHON" -u "$entry" \
  --model "$MODEL_PATH" \
  --seed "$SEED" \
  --w_bits "$BITS" \
  --w_groupsize 128 \
  --w_clip \
  --a_bits 16 --v_bits 16 --k_bits 16 \
  --cal_dataset c4 \
  --nsamples 128 \
  --act_order \
  --bsz 1 \
  --save_qmodel_path "$TEMP_PATH" \
  "${extra[@]}"

mv -f -- "$TEMP_PATH" "$OUTPUT_PATH"

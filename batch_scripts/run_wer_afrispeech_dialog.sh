#!/bin/bash
#SBATCH --job-name=wer_dialog
#SBATCH --output=batch_scripts/wer_dialog_%j.out
#SBATCH --error=batch_scripts/wer_dialog_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --account=ufdatastudios
#SBATCH --partition=hpg-b200

set -euo pipefail

module load cuda/12.8.1 || true
module load intel/2020.0.166 ffmpeg/n7.2 || true
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ASR_VENV_PATH="${ASR_VENV_PATH:-/orange/ufdatastudios/c.okocha/afrispeech-entailment/.venvASR}"
if [ ! -f "$ASR_VENV_PATH/bin/activate" ]; then
  echo "FAIL: ASR virtual environment not found: $ASR_VENV_PATH"
  exit 1
fi

source "$ASR_VENV_PATH/bin/activate"
cd /orange/ufdatastudios/c.okocha/CodecEval-Africa

nvidia-smi || true

OUT_DIR="${WER_OUT_DIR:-results/WER_v2/metrics}"
mkdir -p "$OUT_DIR"

DATASET="afrispeech_dialog"
DATASET_OUTPUTS="outputs/${DATASET}"
TRANSCRIPT_CSV="data/${DATASET}/metadata.csv"

WER_LIMIT="${WER_LIMIT:--1}"
ASR_MODEL="${ASR_MODEL:-openai/whisper-large-v3}"
ASR_LANGUAGE="${ASR_LANGUAGE:-en}"

SUMMARY_FILE="$OUT_DIR/wer_${DATASET}_summary_$(date +%Y%m%d_%H%M%S).txt"

echo "==================================================" | tee "$SUMMARY_FILE"
echo "WER Evaluation - ${DATASET} only" | tee -a "$SUMMARY_FILE"
echo "Started: $(date)" | tee -a "$SUMMARY_FILE"
echo "Output dir: $OUT_DIR" | tee -a "$SUMMARY_FILE"
echo "Transcript CSV: $TRANSCRIPT_CSV" | tee -a "$SUMMARY_FILE"
echo "Per-variant file limit: $WER_LIMIT" | tee -a "$SUMMARY_FILE"
echo "ASR model: $ASR_MODEL" | tee -a "$SUMMARY_FILE"
echo "ASR language: $ASR_LANGUAGE" | tee -a "$SUMMARY_FILE"
echo "ASR venv: $ASR_VENV_PATH" | tee -a "$SUMMARY_FILE"
echo "==================================================" | tee -a "$SUMMARY_FILE"

TOTAL=0
SUCCESS=0
FAILED=0
SKIPPED=0

if [ ! -d "$DATASET_OUTPUTS" ]; then
  echo "FAIL: missing outputs dir: $DATASET_OUTPUTS" | tee -a "$SUMMARY_FILE"
  exit 1
fi

if [ ! -f "$TRANSCRIPT_CSV" ]; then
  echo "FAIL: missing transcript CSV: $TRANSCRIPT_CSV" | tee -a "$SUMMARY_FILE"
  exit 1
fi

for codec_dir in "$DATASET_OUTPUTS"/*_outputs; do
  [ -d "$codec_dir" ] || continue

  codec_name=$(basename "$codec_dir" | sed 's/_outputs$//')
  echo "Codec: $codec_name" | tee -a "$SUMMARY_FILE"

  for variant_dir in "$codec_dir"/*; do
    [ -d "$variant_dir" ] || continue

    variant=$(basename "$variant_dir")
    wav_count=$(find "$variant_dir" -maxdepth 1 -type f -name "*.wav" | wc -l)
    TOTAL=$((TOTAL + 1))

    if [ "$wav_count" -eq 0 ]; then
      echo "  [$variant] SKIP: no wav files" | tee -a "$SUMMARY_FILE"
      SKIPPED=$((SKIPPED + 1))
      continue
    fi

    out_name="${DATASET}_${codec_name}_${variant}_wer"
    json_path="$OUT_DIR/${out_name}.json"

    echo "  [$variant] Running WER on $wav_count files..." | tee -a "$SUMMARY_FILE"

    if python scripts/evaluate_quality.py \
      --metric wer \
      --audio-dir "$variant_dir" \
      --transcript-csv "$TRANSCRIPT_CSV" \
      --asr-model "$ASR_MODEL" \
      --asr-language "$ASR_LANGUAGE" \
      --num-samples "$WER_LIMIT" \
      --output-dir "$OUT_DIR" \
      --output-name "$out_name"; then

      if [ -s "$json_path" ]; then
        wer=$(python - <<PY
import json
with open('$json_path', 'r') as f:
    d = json.load(f)
v = d.get('wer', None)
print('None' if v is None else f"{v:.4f}")
PY
)
        nfiles=$(python - <<PY
import json
with open('$json_path', 'r') as f:
    d = json.load(f)
print(d.get('num_scored_files', 0))
PY
)
        echo "  [$variant] OK: WER=$wer files=$nfiles" | tee -a "$SUMMARY_FILE"
        SUCCESS=$((SUCCESS + 1))
      else
        echo "  [$variant] FAIL: JSON missing: $json_path" | tee -a "$SUMMARY_FILE"
        FAILED=$((FAILED + 1))
      fi
    else
      echo "  [$variant] FAIL: evaluate_quality.py wer failed" | tee -a "$SUMMARY_FILE"
      FAILED=$((FAILED + 1))
    fi
  done
done

echo "" | tee -a "$SUMMARY_FILE"
echo "==================================================" | tee -a "$SUMMARY_FILE"
echo "Completed: $(date)" | tee -a "$SUMMARY_FILE"
echo "Total: $TOTAL | Success: $SUCCESS | Failed: $FAILED | Skipped: $SKIPPED" | tee -a "$SUMMARY_FILE"
echo "Summary: $SUMMARY_FILE" | tee -a "$SUMMARY_FILE"
echo "==================================================" | tee -a "$SUMMARY_FILE"

#!/bin/bash
#SBATCH --job-name=utmos_all_codecs
#SBATCH --output=batch_scripts/utmos_all_codecs_%j.out
#SBATCH --error=batch_scripts/utmos_all_codecs_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --time=48:00:00
#SBATCH --partition=hpg-default
#SBATCH --account=ufdatastudios

set -euo pipefail

source /orange/ufdatastudios/c.okocha/CodecEval-Africa/.venv_wavtokenizer39/bin/activate
cd /orange/ufdatastudios/c.okocha/CodecEval-Africa

OUT_DIR="results/UTMOS/quality_metrics"
mkdir -p "$OUT_DIR"
UTMOS_LIMIT="${UTMOS_LIMIT:--1}"
UTMOS_MAX_SECONDS="${UTMOS_MAX_SECONDS:-10}"

SUMMARY_FILE="$OUT_DIR/utmos_all_codecs_summary_$(date +%Y%m%d_%H%M%S).txt"

echo "==================================================" | tee "$SUMMARY_FILE"
echo "UTMOS Evaluation - All Codecs / All Datasets" | tee -a "$SUMMARY_FILE"
echo "Started: $(date)" | tee -a "$SUMMARY_FILE"
echo "Output dir: $OUT_DIR" | tee -a "$SUMMARY_FILE"
echo "Per-variant file limit: $UTMOS_LIMIT" | tee -a "$SUMMARY_FILE"
echo "UTMOS max seconds per file: $UTMOS_MAX_SECONDS" | tee -a "$SUMMARY_FILE"
echo "==================================================" | tee -a "$SUMMARY_FILE"

TOTAL=0
SUCCESS=0
FAILED=0
SKIPPED=0

for dataset in afrispeech_dialog afrinames afrispeech_multilingual; do
  dataset_outputs="outputs/$dataset"

  echo "" | tee -a "$SUMMARY_FILE"
  echo "Dataset: $dataset" | tee -a "$SUMMARY_FILE"

  if [ ! -d "$dataset_outputs" ]; then
    echo "  SKIP: missing outputs dir: $dataset_outputs" | tee -a "$SUMMARY_FILE"
    continue
  fi

  for codec_dir in "$dataset_outputs"/*_outputs; do
    [ -d "$codec_dir" ] || continue

    codec_name=$(basename "$codec_dir" | sed 's/_outputs$//')
    echo "  Codec: $codec_name" | tee -a "$SUMMARY_FILE"

    for variant_dir in "$codec_dir"/*; do
      [ -d "$variant_dir" ] || continue

      variant=$(basename "$variant_dir")
      wav_count=$(find "$variant_dir" -maxdepth 1 -type f -name "*.wav" | wc -l)

      TOTAL=$((TOTAL + 1))

      if [ "$wav_count" -eq 0 ]; then
        echo "    [$variant] SKIP: no wav files" | tee -a "$SUMMARY_FILE"
        SKIPPED=$((SKIPPED + 1))
        continue
      fi

      out_name="${dataset}_${codec_name}_${variant}_utmos"
      json_path="$OUT_DIR/${out_name}.json"

      echo "    [$variant] Running UTMOS on $wav_count files..." | tee -a "$SUMMARY_FILE"

      if python scripts/evaluate_quality.py \
        --metric utmos \
        --audio-dir "$variant_dir" \
        --num-samples "$UTMOS_LIMIT" \
        --utmos-max-seconds "$UTMOS_MAX_SECONDS" \
        --output-dir "$OUT_DIR" \
        --output-name "$out_name"; then

        if [ -s "$json_path" ]; then
          utmos=$(python - <<PY
import json
with open('$json_path', 'r') as f:
    d = json.load(f)
v = d.get('utmos', None)
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
          echo "    [$variant] OK: UTMOS=$utmos files=$nfiles" | tee -a "$SUMMARY_FILE"
          SUCCESS=$((SUCCESS + 1))
        else
          echo "    [$variant] FAIL: JSON missing: $json_path" | tee -a "$SUMMARY_FILE"
          FAILED=$((FAILED + 1))
        fi
      else
        echo "    [$variant] FAIL: evaluate_quality.py utmos failed" | tee -a "$SUMMARY_FILE"
        FAILED=$((FAILED + 1))
      fi
    done
  done
done

echo "" | tee -a "$SUMMARY_FILE"
echo "==================================================" | tee -a "$SUMMARY_FILE"
echo "Completed: $(date)" | tee -a "$SUMMARY_FILE"
echo "Total: $TOTAL | Success: $SUCCESS | Failed: $FAILED | Skipped: $SKIPPED" | tee -a "$SUMMARY_FILE"
echo "Summary: $SUMMARY_FILE" | tee -a "$SUMMARY_FILE"
echo "==================================================" | tee -a "$SUMMARY_FILE"

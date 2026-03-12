#!/bin/bash
#SBATCH --job-name=nisqa_all_codecs
#SBATCH --output=batch_scripts/nisqa_all_codecs_%j.out
#SBATCH --error=batch_scripts/nisqa_all_codecs_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=48:00:00
#SBATCH --account=ufdatastudios
#SBATCH --partition=hpg-default

set -euo pipefail

source /orange/ufdatastudios/c.okocha/CodecEval-Africa/.venv/bin/activate
cd /orange/ufdatastudios/c.okocha/CodecEval-Africa

OUT_DIR="results/NISQA/quality_metrics"
mkdir -p "$OUT_DIR"
NISQA_LIMIT="${NISQA_LIMIT:--1}"
NISQA_MS_MAX_SEGMENTS="${NISQA_MS_MAX_SEGMENTS:-50000}"
export NISQA_MS_MAX_SEGMENTS

SUMMARY_FILE="$OUT_DIR/nisqa_all_codecs_summary_$(date +%Y%m%d_%H%M%S).txt"

echo "==================================================" | tee "$SUMMARY_FILE"
echo "NISQA Evaluation - All Codecs / All Datasets" | tee -a "$SUMMARY_FILE"
echo "Started: $(date)" | tee -a "$SUMMARY_FILE"
echo "Output dir: $OUT_DIR" | tee -a "$SUMMARY_FILE"
echo "Per-variant file limit: $NISQA_LIMIT" | tee -a "$SUMMARY_FILE"
echo "NISQA ms_max_segments: $NISQA_MS_MAX_SEGMENTS" | tee -a "$SUMMARY_FILE"
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

      out_name="${dataset}_${codec_name}_${variant}_nisqa"
      json_path="$OUT_DIR/${out_name}.json"

      echo "    [$variant] Running NISQA on $wav_count files..." | tee -a "$SUMMARY_FILE"

      if python scripts/evaluate_quality.py \
        --metric nisqa \
        --audio-dir "$variant_dir" \
        --num-samples "$NISQA_LIMIT" \
        --output-dir "$OUT_DIR" \
        --output-name "$out_name"; then

        if [ -s "$json_path" ]; then
          mos=$(python - <<PY
import json
from math import isnan
with open('$json_path', 'r') as f:
    d = json.load(f)
if isinstance(d, list) and len(d) > 0:
    vals=[x.get('mos') for x in d if isinstance(x.get('mos'), (int,float))]
    if vals:
        print(f"{sum(vals)/len(vals):.4f}")
    else:
        print('nan')
else:
    print('nan')
PY
)
          nfiles=$(python - <<PY
import json
with open('$json_path', 'r') as f:
    d = json.load(f)
print(len(d) if isinstance(d, list) else 0)
PY
)
          echo "    [$variant] OK: NISQA_MOS=$mos files=$nfiles" | tee -a "$SUMMARY_FILE"
          SUCCESS=$((SUCCESS + 1))
        else
          echo "    [$variant] FAIL: JSON missing: $json_path" | tee -a "$SUMMARY_FILE"
          FAILED=$((FAILED + 1))
        fi
      else
        echo "    [$variant] FAIL: evaluate_quality.py nisqa failed" | tee -a "$SUMMARY_FILE"
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

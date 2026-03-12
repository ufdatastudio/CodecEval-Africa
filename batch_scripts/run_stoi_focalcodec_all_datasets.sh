#!/bin/bash
#SBATCH --job-name=stoi_focal_all
#SBATCH --output=batch_scripts/stoi_focal_all_%j.out
#SBATCH --error=batch_scripts/stoi_focal_all_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=24:00:00
#SBATCH --partition=hpg-default
#SBATCH --account=ufdatastudios

set -euo pipefail

source /orange/ufdatastudios/c.okocha/CodecEval-Africa/.venv/bin/activate
cd /orange/ufdatastudios/c.okocha/CodecEval-Africa

OUT_DIR="results/STOI/quality_metrics"
mkdir -p "$OUT_DIR"

SUMMARY_FILE="$OUT_DIR/focalcodec_stoi_summary_$(date +%Y%m%d_%H%M%S).txt"

echo "==================================================" | tee "$SUMMARY_FILE"
echo "FocalCodec STOI Evaluation - All Datasets" | tee -a "$SUMMARY_FILE"
echo "Started: $(date)" | tee -a "$SUMMARY_FILE"
echo "Output dir: $OUT_DIR" | tee -a "$SUMMARY_FILE"
echo "==================================================" | tee -a "$SUMMARY_FILE"

declare -A REF_DIRS
REF_DIRS["afrispeech_dialog"]="data/afrispeech_dialog/data"
REF_DIRS["afrinames"]="data/afri_names_150_flat"
REF_DIRS["afrispeech_multilingual"]="/orange/ufdatastudios/c.okocha/Dataset/afrispeech_multilingual_wav"

VARIANTS=(
  "focalcodec_12_5hz"
  "focalcodec_25hz"
  "focalcodec_50hz"
  "focalcodec_50hz_2k_causal"
  "focalcodec_50hz_4k_causal"
  "focalcodec_50hz_65k_causal"
)

TOTAL=0
SUCCESS=0
FAILED=0
SKIPPED=0

for dataset in afrispeech_dialog afrinames afrispeech_multilingual; do
  ref_dir="${REF_DIRS[$dataset]}"
  deg_root="outputs/$dataset/FocalCodec_outputs"

  echo "" | tee -a "$SUMMARY_FILE"
  echo "Dataset: $dataset" | tee -a "$SUMMARY_FILE"

  if [ ! -d "$ref_dir" ]; then
    echo "  SKIP: missing reference dir: $ref_dir" | tee -a "$SUMMARY_FILE"
    SKIPPED=$((SKIPPED + ${#VARIANTS[@]}))
    continue
  fi

  if [ ! -d "$deg_root" ]; then
    echo "  SKIP: missing degraded root: $deg_root" | tee -a "$SUMMARY_FILE"
    SKIPPED=$((SKIPPED + ${#VARIANTS[@]}))
    continue
  fi

  ref_count=$(find "$ref_dir" -maxdepth 1 -type f -name "*.wav" | wc -l)
  echo "  Reference files: $ref_count" | tee -a "$SUMMARY_FILE"

  for variant in "${VARIANTS[@]}"; do
    deg_dir="$deg_root/$variant"
    out_name="${dataset}_FocalCodec_${variant}_stoi"
    json_path="$OUT_DIR/${out_name}.json"

    TOTAL=$((TOTAL + 1))

    if [ ! -d "$deg_dir" ]; then
      echo "  [$variant] SKIP: missing dir" | tee -a "$SUMMARY_FILE"
      SKIPPED=$((SKIPPED + 1))
      continue
    fi

    deg_count=$(find "$deg_dir" -maxdepth 1 -type f -name "*.wav" | wc -l)
    if [ "$deg_count" -eq 0 ]; then
      echo "  [$variant] SKIP: no wav files" | tee -a "$SUMMARY_FILE"
      SKIPPED=$((SKIPPED + 1))
      continue
    fi

    echo "  [$variant] Running STOI on $deg_count files..." | tee -a "$SUMMARY_FILE"

    if python scripts/evaluate_quality.py \
      --metric stoi \
      --ref-dir "$ref_dir" \
      --deg-dir "$deg_dir" \
      --output-dir "$OUT_DIR" \
      --output-name "$out_name"; then

      if [ -s "$json_path" ]; then
        stoi=$(python - <<PY
import json
with open('$json_path', 'r') as f:
    d = json.load(f)
print(f"{d.get('stoi', float('nan')):.4f}")
PY
)
        nfiles=$(python - <<PY
import json
with open('$json_path', 'r') as f:
    d = json.load(f)
print(d.get('num_files', 0))
PY
)
        echo "  [$variant] OK: STOI=$stoi files=$nfiles json=$json_path" | tee -a "$SUMMARY_FILE"
        SUCCESS=$((SUCCESS + 1))
      else
        echo "  [$variant] FAIL: run returned success but JSON missing: $json_path" | tee -a "$SUMMARY_FILE"
        FAILED=$((FAILED + 1))
      fi
    else
      echo "  [$variant] FAIL: evaluate_quality.py failed" | tee -a "$SUMMARY_FILE"
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

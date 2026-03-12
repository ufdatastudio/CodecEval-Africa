#!/bin/bash
#SBATCH --job-name=asv_all_codecs
#SBATCH --output=batch_scripts/asv_all_codecs_%j.out
#SBATCH --error=batch_scripts/asv_all_codecs_%j.err
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

OUT_DIR="${ASV_OUT_DIR:-results/ASV/metrics}"
mkdir -p "$OUT_DIR"

ASV_LIMIT="${ASV_LIMIT:--1}"
ASV_IMPOSTORS="${ASV_IMPOSTORS:-10}"
ASV_MODEL_TYPE="${ASV_MODEL_TYPE:-mfcc}"
ASV_IMPOSTOR_MODE="${ASV_IMPOSTOR_MODE:-all}"
ASV_BOOTSTRAP="${ASV_BOOTSTRAP:-200}"

SUMMARY_FILE="$OUT_DIR/asv_all_codecs_summary_$(date +%Y%m%d_%H%M%S).txt"

echo "==================================================" | tee "$SUMMARY_FILE"
echo "ASV Evaluation (EER/minDCF) - All Codecs / All Datasets" | tee -a "$SUMMARY_FILE"
echo "Started: $(date)" | tee -a "$SUMMARY_FILE"
echo "Output dir: $OUT_DIR" | tee -a "$SUMMARY_FILE"
echo "Per-variant file limit: $ASV_LIMIT" | tee -a "$SUMMARY_FILE"
echo "Impostors per file: $ASV_IMPOSTORS" | tee -a "$SUMMARY_FILE"
echo "Impostor mode: $ASV_IMPOSTOR_MODE" | tee -a "$SUMMARY_FILE"
echo "Bootstrap samples: $ASV_BOOTSTRAP" | tee -a "$SUMMARY_FILE"
echo "Model type: $ASV_MODEL_TYPE" | tee -a "$SUMMARY_FILE"
echo "==================================================" | tee -a "$SUMMARY_FILE"

declare -A REF_DIRS
REF_DIRS["afrispeech_dialog"]="data/afrispeech_dialog/data"
REF_DIRS["afrinames"]="data/afri_names_150_flat"
REF_DIRS["afrispeech_multilingual"]="/orange/ufdatastudios/c.okocha/Dataset/afrispeech_multilingual_wav"

TOTAL=0
SUCCESS=0
FAILED=0
SKIPPED=0

for dataset in afrispeech_dialog afrinames afrispeech_multilingual; do
  ref_dir="${REF_DIRS[$dataset]}"

  echo "" | tee -a "$SUMMARY_FILE"
  echo "Dataset: $dataset" | tee -a "$SUMMARY_FILE"

  if [ ! -d "$ref_dir" ]; then
    echo "  SKIP: missing reference dir: $ref_dir" | tee -a "$SUMMARY_FILE"
    continue
  fi

  dataset_outputs="outputs/$dataset"
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
      deg_count=$(find "$variant_dir" -maxdepth 1 -type f -name "*.wav" | wc -l)
      TOTAL=$((TOTAL + 1))

      if [ "$deg_count" -eq 0 ]; then
        echo "    [$variant] SKIP: no wav files" | tee -a "$SUMMARY_FILE"
        SKIPPED=$((SKIPPED + 1))
        continue
      fi

      out_name="${dataset}_${codec_name}_${variant}_asv"
      json_path="$OUT_DIR/${out_name}.json"

      echo "    [$variant] Running ASV on $deg_count files..." | tee -a "$SUMMARY_FILE"

      if python scripts/evaluate_asv.py \
        --ref-dir "$ref_dir" \
        --deg-dir "$variant_dir" \
        --output-dir "$OUT_DIR" \
        --output-name "$out_name" \
        --model-type "$ASV_MODEL_TYPE" \
        --impostors-per-file "$ASV_IMPOSTORS" \
        --impostor-mode "$ASV_IMPOSTOR_MODE" \
        --bootstrap-samples "$ASV_BOOTSTRAP" \
        --limit "$ASV_LIMIT"; then

        if [ -s "$json_path" ]; then
          eer=$(python - <<PY
import json
with open('$json_path', 'r') as f:
    d = json.load(f)
v = d.get('metrics', {}).get('eer_percent', None)
print('None' if v is None else f"{v:.2f}")
PY
)
          mindcf=$(python - <<PY
import json
with open('$json_path', 'r') as f:
    d = json.load(f)
v = d.get('metrics', {}).get('min_dcf', None)
print('None' if v is None else f"{v:.4f}")
PY
)
          ntrials=$(python - <<PY
import json
with open('$json_path', 'r') as f:
    d = json.load(f)
print(d.get('num_trials', 0))
PY
)
          echo "    [$variant] OK: EER(%)=$eer minDCF=$mindcf trials=$ntrials" | tee -a "$SUMMARY_FILE"
          SUCCESS=$((SUCCESS + 1))
        else
          echo "    [$variant] FAIL: JSON missing: $json_path" | tee -a "$SUMMARY_FILE"
          FAILED=$((FAILED + 1))
        fi
      else
        echo "    [$variant] FAIL: evaluate_asv.py failed" | tee -a "$SUMMARY_FILE"
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

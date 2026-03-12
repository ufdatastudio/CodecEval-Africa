#!/bin/bash
#SBATCH --job-name=pesq_all_complete
#SBATCH --output=batch_scripts/pesq_all_complete_%j.out
#SBATCH --error=batch_scripts/pesq_all_complete_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
#SBATCH --account=ufdatastudios
#SBATCH --partition=hpg-default
#SBATCH --mail-user=c.okocha@ufl.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -e

ulimit -c 0

echo "=================================================="
echo "PESQ Evaluation - ALL CODECS (Complete)"
echo "Started: $(date)"
echo "=================================================="

source /orange/ufdatastudios/c.okocha/CodecEval-Africa/.venv_wavtokenizer39/bin/activate

cd /orange/ufdatastudios/c.okocha/CodecEval-Africa

# Dataset-to-reference mapping
declare -A REF_DIRS
REF_DIRS["afrispeech_dialog"]="data/afrispeech_dialog/data"
REF_DIRS["afrinames"]="data/afri_names_150_flat"
REF_DIRS["afrispeech_multilingual"]="/orange/ufdatastudios/c.okocha/Dataset/afrispeech_multilingual_wav"

TOTAL_JOBS=0
SUCCESS_JOBS=0
FAILED_JOBS=0
SKIPPED=0

SUMMARY_FILE="results/quality_metrics/pesq_all_codecs_summary_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p "$(dirname "$SUMMARY_FILE")"

echo "PESQ Evaluation - ALL CODECS" > "$SUMMARY_FILE"
echo "Date: $(date)" >> "$SUMMARY_FILE"
echo "================================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

for dataset in afrispeech_dialog afrinames afrispeech_multilingual; do
    REF_DIR="${REF_DIRS[$dataset]}"

    if [ -z "$REF_DIR" ] || [ ! -d "$REF_DIR" ]; then
        echo "⚠ Skipping $dataset: no reference directory"
        continue
    fi

    REF_COUNT=$(find "$REF_DIR" -name "*.wav" -type f 2>/dev/null | wc -l)
    if [ "$REF_COUNT" -eq 0 ]; then
        echo "⚠ Skipping $dataset: no reference files"
        continue
    fi

    echo ""
    echo "=================================================="
    echo "Dataset: $dataset ($REF_COUNT reference files)"
    echo "=================================================="
    echo ""
    echo "Dataset: $dataset ($REF_COUNT reference files)" >> "$SUMMARY_FILE"
    echo "---" >> "$SUMMARY_FILE"

    for codec_dir in outputs/$dataset/*_outputs/; do
        [ -d "$codec_dir" ] || continue
        codec_name=$(basename "$codec_dir" | sed 's/_outputs$//')

        echo "Codec: $codec_name"

        for subdir in "$codec_dir"*/; do
            [ -d "$subdir" ] || continue
            variant=$(basename "$subdir")

            DEG_COUNT=$(find "$subdir" -maxdepth 1 -name "*.wav" -type f 2>/dev/null | wc -l)
            if [ "$DEG_COUNT" -eq 0 ]; then
                echo "  ⚠ Skipping $variant (no .wav files)"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi

            echo "  Processing: $variant ($DEG_COUNT files)"
            TOTAL_JOBS=$((TOTAL_JOBS + 1))

            OUT_NAME="${dataset}_${codec_name}_${variant}_pesq"
            if python scripts/evaluate_quality.py \
                --metric pesq \
                --ref-dir "$REF_DIR" \
                --deg-dir "$subdir" \
                --output-name "$OUT_NAME"; then

                RESULT_FILE="results/quality_metrics/${OUT_NAME}.json"
                if [ -f "$RESULT_FILE" ]; then
                    WB=$(python - <<PY
import json
r=json.load(open('$RESULT_FILE'))
print(f"{r.get('wb_pesq', float('nan')):.4f}")
PY
)
                    NB=$(python - <<PY
import json
r=json.load(open('$RESULT_FILE'))
print(f"{r.get('nb_pesq', float('nan')):.4f}")
PY
)
                    NF=$(python - <<PY
import json
r=json.load(open('$RESULT_FILE'))
print(r.get('num_files', 0))
PY
)
                    echo "  $codec_name @ $variant: WB=$WB NB=$NB ($NF files)" >> "$SUMMARY_FILE"
                    echo "  ✓ $variant: WB=$WB NB=$NB ($NF files)"
                else
                    echo "  $codec_name @ $variant: COMPLETED (result file missing)" >> "$SUMMARY_FILE"
                    echo "  ✓ $variant: completed"
                fi

                SUCCESS_JOBS=$((SUCCESS_JOBS + 1))
            else
                echo "  $codec_name @ $variant: FAILED" >> "$SUMMARY_FILE"
                echo "  ✗ $variant: failed"
                FAILED_JOBS=$((FAILED_JOBS + 1))
            fi
        done
    done

    echo "" >> "$SUMMARY_FILE"
done

echo "" >> "$SUMMARY_FILE"
echo "================================================" >> "$SUMMARY_FILE"
echo "Summary Statistics" >> "$SUMMARY_FILE"
echo "================================================" >> "$SUMMARY_FILE"
echo "Total evaluations: $TOTAL_JOBS" >> "$SUMMARY_FILE"
echo "Successful: $SUCCESS_JOBS" >> "$SUMMARY_FILE"
echo "Failed: $FAILED_JOBS" >> "$SUMMARY_FILE"
echo "Skipped (no files): $SKIPPED" >> "$SUMMARY_FILE"

echo ""
echo "=================================================="
echo "PESQ Evaluation Complete"
echo "=================================================="
echo "Completed: $(date)"
echo "Total: $TOTAL_JOBS | Success: $SUCCESS_JOBS | Failed: $FAILED_JOBS | Skipped: $SKIPPED"
echo "Summary saved to: $SUMMARY_FILE"
echo "=================================================="

cat "$SUMMARY_FILE"

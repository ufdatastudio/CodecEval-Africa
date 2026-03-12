#!/bin/bash
#SBATCH --job-name=stoi_all_complete
#SBATCH --output=batch_scripts/stoi_all_complete_%j.out
#SBATCH --error=batch_scripts/stoi_all_complete_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=24:00:00
#SBATCH --partition=hpg-default
#SBATCH --account=ufdatastudios

# Complete STOI evaluation for ALL codecs (including LanguageCodec, SemantiCodec, WavTokenizer)
# Handles different directory naming conventions

set -e

module load conda
source /orange/ufdatastudios/c.okocha/CodecEval-Africa/.venv/bin/activate

cd /orange/ufdatastudios/c.okocha/CodecEval-Africa

echo "=================================================="
echo "STOI Evaluation - ALL CODECS (Complete)"
echo "Started: $(date)"
echo "=================================================="

# Dataset-to-reference mapping
declare -A REF_DIRS
REF_DIRS["afrispeech_dialog"]="data/afrispeech_dialog/data"
REF_DIRS["afrinames"]="data/afri_names_150_flat"
REF_DIRS["afrispeech_multilingual"]="/orange/ufdatastudios/c.okocha/Dataset/afrispeech_multilingual_wav"

TOTAL_JOBS=0
SUCCESS_JOBS=0
FAILED_JOBS=0
SKIPPED=0

# Create results summary
SUMMARY_FILE="results/quality_metrics/stoi_all_codecs_summary_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p "$(dirname "$SUMMARY_FILE")"

echo "STOI Evaluation - ALL CODECS" > "$SUMMARY_FILE"
echo "Date: $(date)" >> "$SUMMARY_FILE"
echo "================================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Process each dataset
for dataset in afrispeech_dialog afrinames afrispeech_multilingual; do
    REF_DIR="${REF_DIRS[$dataset]}"
    
    # Check if reference directory exists
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
    
    # Process each codec directory in outputs
    for codec_dir in outputs/$dataset/*_outputs/; do
        if [ ! -d "$codec_dir" ]; then
            continue
        fi
        
        codec_name=$(basename "$codec_dir" | sed 's/_outputs$//')
        
        echo "Codec: $codec_name"
        
        # Find all subdirectories (different naming patterns)
        for subdir in "$codec_dir"*/; do
            if [ ! -d "$subdir" ]; then
                continue
            fi
            
            variant=$(basename "$subdir")
            
            # Count WAV files
            DEG_COUNT=$(find "$subdir" -maxdepth 1 -name "*.wav" -type f 2>/dev/null | wc -l)
            
            if [ "$DEG_COUNT" -eq 0 ]; then
                echo "  ⚠ Skipping $variant (no .wav files)"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi
            
            echo "  Processing: $variant ($DEG_COUNT files)"
            
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            
            # Run STOI evaluation (include dataset name to prevent overwriting)
            if python scripts/evaluate_quality.py \
                --metric stoi \
                --ref-dir "$REF_DIR" \
                --deg-dir "$subdir" \
                --output-name "${dataset}_${codec_name}_${variant}_stoi"; then
                
                # Extract STOI score
                RESULT_FILE=$(ls -t results/quality_metrics/${dataset}_${codec_name}_${variant}_stoi_*.json 2>/dev/null | head -1)
                if [ -f "$RESULT_FILE" ]; then
                    STOI_SCORE=$(python -c "import json; print(f\"{json.load(open('$RESULT_FILE'))['stoi']:.4f}\")" 2>/dev/null || echo "N/A")
                    NUM_FILES=$(python -c "import json; print(json.load(open('$RESULT_FILE'))['num_files'])" 2>/dev/null || echo "?")
                    echo "  $codec_name @ $variant: STOI = $STOI_SCORE ($NUM_FILES files)" >> "$SUMMARY_FILE"
                    echo "  ✓ $variant: STOI = $STOI_SCORE ($NUM_FILES files)"
                else
                    echo "  $codec_name @ $variant: COMPLETED (result file not found)" >> "$SUMMARY_FILE"
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
echo "STOI Evaluation Complete"
echo "=================================================="
echo "Completed: $(date)"
echo "Total: $TOTAL_JOBS | Success: $SUCCESS_JOBS | Failed: $FAILED_JOBS | Skipped: $SKIPPED"
echo ""
echo "Summary saved to: $SUMMARY_FILE"
echo "=================================================="

cat "$SUMMARY_FILE"

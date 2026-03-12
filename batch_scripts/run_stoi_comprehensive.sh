#!/bin/bash
#SBATCH --job-name=stoi_comprehensive
#SBATCH --output=batch_scripts/stoi_comp_%j.out
#SBATCH --error=batch_scripts/stoi_comp_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=24:00:00
#SBATCH --partition=hpg-default
#SBATCH --account=ufdatastudios

# Comprehensive STOI evaluation for all available codec outputs
# Automatically discovers datasets, codecs, and bitrates

set -e

module load conda
source /orange/ufdatastudios/c.okocha/CodecEval-Africa/.venv/bin/activate

cd /orange/ufdatastudios/c.okocha/CodecEval-Africa

echo "=================================================="
echo "STOI Comprehensive Evaluation"
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
SKIPPED_NOREF=0
SKIPPED_NOFILES=0

# Create results summary file
SUMMARY_FILE="results/quality_metrics/stoi_summary_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p "$(dirname "$SUMMARY_FILE")"

echo "STOI Evaluation Summary" > "$SUMMARY_FILE"
echo "Date: $(date)" >> "$SUMMARY_FILE"
echo "================================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Loop through all datasets in outputs/
for dataset_dir in outputs/*/; do
    dataset=$(basename "$dataset_dir")
    
    echo ""
    echo "=================================================="
    echo "Dataset: $dataset"
    echo "=================================================="
    
    # Get reference directory for this dataset
    REF_DIR="${REF_DIRS[$dataset]}"
    
    # Check if reference directory exists and has files
    if [ -z "$REF_DIR" ] || [ ! -d "$REF_DIR" ]; then
        echo "⚠ No reference directory configured for $dataset"
        SKIPPED_NOREF=$((SKIPPED_NOREF + 1))
        continue
    fi
    
    REF_COUNT=$(find "$REF_DIR" -name "*.wav" -type f 2>/dev/null | wc -l)
    if [ "$REF_COUNT" -eq 0 ]; then
        echo "⚠ No reference files found in $REF_DIR"
        SKIPPED_NOREF=$((SKIPPED_NOREF + 1))
        continue
    fi
    
    echo "✓ Reference: $REF_DIR ($REF_COUNT files)"
    echo ""
    echo "Dataset: $dataset ($REF_COUNT reference files)" >> "$SUMMARY_FILE"
    echo "---" >> "$SUMMARY_FILE"
    
    # Loop through all codec output directories
    for codec_dir in outputs/$dataset/*_outputs/; do
        if [ ! -d "$codec_dir" ]; then
            continue
        fi
        
        codec_name=$(basename "$codec_dir" | sed 's/_outputs$//')
        
        # Loop through all bitrate subdirectories
        for bitrate_dir in "$codec_dir"out_*/; do
            if [ ! -d "$bitrate_dir" ]; then
                continue
            fi
            
            bitrate=$(basename "$bitrate_dir" | sed 's/out_//')
            
            # Count files in degraded directory
            DEG_COUNT=$(find "$bitrate_dir" -name "*.wav" -type f 2>/dev/null | wc -l)
            
            if [ "$DEG_COUNT" -eq 0 ]; then
                echo "⚠ Skipping $codec_name @ $bitrate (no .wav files)"
                SKIPPED_NOFILES=$((SKIPPED_NOFILES + 1))
                continue
            fi
            
            echo "Processing: $codec_name @ $bitrate ($DEG_COUNT files)"
            
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            
            # Run STOI evaluation
            if python scripts/evaluate_quality.py \
                --metric stoi \
                --ref-dir "$REF_DIR" \
                --deg-dir "$bitrate_dir"; then
                
                # Extract STOI score from the result file
                RESULT_FILE=$(ls -t results/quality_metrics/*_stoi_*.json | head -1)
                if [ -f "$RESULT_FILE" ]; then
                    STOI_SCORE=$(python -c "import json; print(f\"{json.load(open('$RESULT_FILE'))['stoi']:.4f}\")" 2>/dev/null || echo "N/A")
                    echo "  $codec_name @ $bitrate: STOI = $STOI_SCORE" >> "$SUMMARY_FILE"
                    echo "✓ $codec_name @ $bitrate: STOI = $STOI_SCORE"
                else
                    echo "  $codec_name @ $bitrate: RESULT FILE NOT FOUND" >> "$SUMMARY_FILE"
                    echo "✓ $codec_name @ $bitrate: completed (result file not found)"
                fi
                
                SUCCESS_JOBS=$((SUCCESS_JOBS + 1))
            else
                echo "  $codec_name @ $bitrate: FAILED" >> "$SUMMARY_FILE"
                echo "✗ $codec_name @ $bitrate: failed"
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
echo "Skipped (no reference): $SKIPPED_NOREF" >> "$SUMMARY_FILE"
echo "Skipped (no files): $SKIPPED_NOFILES" >> "$SUMMARY_FILE"

echo ""
echo "=================================================="
echo "STOI Evaluation Complete"
echo "=================================================="
echo "Completed: $(date)"
echo "Total evaluations: $TOTAL_JOBS"
echo "Successful: $SUCCESS_JOBS"
echo "Failed: $FAILED_JOBS"
echo "Skipped (no reference): $SKIPPED_NOREF"
echo "Skipped (no files): $SKIPPED_NOFILES"
echo ""
echo "Summary saved to: $SUMMARY_FILE"
echo "=================================================="

cat "$SUMMARY_FILE"

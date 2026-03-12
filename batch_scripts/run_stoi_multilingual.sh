#!/bin/bash
#SBATCH --job-name=stoi_multilingual
#SBATCH --output=batch_scripts/stoi_multilingual_%j.out
#SBATCH --error=batch_scripts/stoi_multilingual_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=12:00:00
#SBATCH --partition=hpg-default
#SBATCH --account=ufdatastudios

# STOI evaluation for afrispeech_multilingual dataset

set -e

module load conda
source /orange/ufdatastudios/c.okocha/CodecEval-Africa/.venv/bin/activate

cd /orange/ufdatastudios/c.okocha/CodecEval-Africa

echo "=================================================="
echo "STOI Evaluation - Multilingual Dataset"
echo "Started: $(date)"
echo "=================================================="

DATASET="afrispeech_multilingual"
REF_DIR="/orange/ufdatastudios/c.okocha/Dataset/afrispeech_multilingual_wav"

# Verify reference directory
if [ ! -d "$REF_DIR" ]; then
    echo "ERROR: Reference directory not found: $REF_DIR"
    exit 1
fi

REF_COUNT=$(find "$REF_DIR" -name "*.wav" -type f 2>/dev/null | wc -l)
echo "Reference files: $REF_COUNT"
echo ""

TOTAL_JOBS=0
SUCCESS_JOBS=0
FAILED_JOBS=0

# Create results summary
SUMMARY_FILE="results/quality_metrics/stoi_multilingual_summary_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p "$(dirname "$SUMMARY_FILE")"

echo "STOI Evaluation - Multilingual Dataset" > "$SUMMARY_FILE"
echo "Date: $(date)" >> "$SUMMARY_FILE"
echo "Reference: $REF_DIR ($REF_COUNT files)" >> "$SUMMARY_FILE"
echo "================================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Process all codec outputs
for codec_dir in outputs/$DATASET/*_outputs/; do
    if [ ! -d "$codec_dir" ]; then
        continue
    fi
    
    codec_name=$(basename "$codec_dir" | sed 's/_outputs$//')
    
    for bitrate_dir in "$codec_dir"out_*/; do
        if [ ! -d "$bitrate_dir" ]; then
            continue
        fi
        
        bitrate=$(basename "$bitrate_dir" | sed 's/out_//')
        
        DEG_COUNT=$(find "$bitrate_dir" -name "*.wav" -type f 2>/dev/null | wc -l)
        
        if [ "$DEG_COUNT" -eq 0 ]; then
            echo "⚠ Skipping $codec_name @ $bitrate (no .wav files)"
            continue
        fi
        
        echo "Processing: $codec_name @ $bitrate ($DEG_COUNT files)"
        
        TOTAL_JOBS=$((TOTAL_JOBS + 1))
        
        # Run STOI evaluation
        if python scripts/evaluate_quality.py \
            --metric stoi \
            --ref-dir "$REF_DIR" \
            --deg-dir "$bitrate_dir"; then
            
            # Extract STOI score
            RESULT_FILE=$(ls -t results/quality_metrics/*_stoi_*.json | head -1)
            if [ -f "$RESULT_FILE" ]; then
                STOI_SCORE=$(python -c "import json; print(f\"{json.load(open('$RESULT_FILE'))['stoi']:.4f}\")" 2>/dev/null || echo "N/A")
                echo "  $codec_name @ $bitrate: STOI = $STOI_SCORE" >> "$SUMMARY_FILE"
                echo "✓ $codec_name @ $bitrate: STOI = $STOI_SCORE"
            else
                echo "  $codec_name @ $bitrate: RESULT FILE NOT FOUND" >> "$SUMMARY_FILE"
                echo "✓ $codec_name @ $bitrate: completed"
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
echo "================================================" >> "$SUMMARY_FILE"
echo "Summary" >> "$SUMMARY_FILE"
echo "================================================" >> "$SUMMARY_FILE"
echo "Total evaluations: $TOTAL_JOBS" >> "$SUMMARY_FILE"
echo "Successful: $SUCCESS_JOBS" >> "$SUMMARY_FILE"
echo "Failed: $FAILED_JOBS" >> "$SUMMARY_FILE"

echo ""
echo "=================================================="
echo "Completed: $(date)"
echo "Total: $TOTAL_JOBS | Success: $SUCCESS_JOBS | Failed: $FAILED_JOBS"
echo "Summary: $SUMMARY_FILE"
echo "=================================================="

cat "$SUMMARY_FILE"

#!/bin/bash
#SBATCH --job-name=stoi_all_codecs
#SBATCH --output=batch_scripts/stoi_all_%j.out
#SBATCH --error=batch_scripts/stoi_all_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=24:00:00
#SBATCH --partition=hpg-default
#SBATCH --account=ufdatastudios

# Comprehensive STOI evaluation for all codec outputs
# This script evaluates intelligibility using STOI metric across all datasets and codecs

set -e

module load conda
source /orange/ufdatastudios/c.okocha/CodecEval-Africa/.venv/bin/activate

cd /orange/ufdatastudios/c.okocha/CodecEval-Africa

echo "=================================================="
echo "STOI Evaluation - All Codecs"
echo "Started: $(date)"
echo "=================================================="

# Define datasets
DATASETS=("afrispeech_dialog" "afrinames" "afrispeech_multilingual")

# Define codecs and their bitrates
declare -A CODEC_BITRATES
CODEC_BITRATES["DAC"]="3kbps 6kbps 12kbps 24kbps"
CODEC_BITRATES["Encodec"]="3kbps 6kbps 12kbps 24kbps"
CODEC_BITRATES["LanguageCodec"]="3kbps 6kbps 12kbps 24kbps"
CODEC_BITRATES["SemantiCodec"]="3kbps 6kbps 12kbps 24kbps"
CODEC_BITRATES["UniCodec"]="3kbps 6kbps 12kbps 24kbps"
CODEC_BITRATES["WavTokenizer"]="3kbps 6kbps 12kbps 24kbps"

TOTAL_JOBS=0
SUCCESS_JOBS=0
FAILED_JOBS=0

# Loop through datasets and codecs
for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "=================================================="
    echo "Dataset: $dataset"
    echo "=================================================="
    
    REF_DIR="data/$dataset/data"
    
    # Check if reference directory exists
    if [ ! -d "$REF_DIR" ]; then
        echo "⚠ Reference directory not found: $REF_DIR"
        continue
    fi
    
    for codec in "${!CODEC_BITRATES[@]}"; do
        for bitrate in ${CODEC_BITRATES[$codec]}; do
            DEG_DIR="outputs/$dataset/${codec}_outputs/out_${bitrate}"
            
            # Check if degraded directory exists
            if [ ! -d "$DEG_DIR" ]; then
                echo "⚠ Skipping $codec @ $bitrate (directory not found)"
                continue
            fi
            
            # Count files
            NUM_FILES=$(find "$DEG_DIR" -name "*.wav" | wc -l)
            if [ "$NUM_FILES" -eq 0 ]; then
                echo "⚠ Skipping $codec @ $bitrate (no .wav files)"
                continue
            fi
            
            echo ""
            echo "Processing: $codec @ $bitrate ($NUM_FILES files)"
            
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            
            # Run STOI evaluation
            if python scripts/evaluate_quality.py \
                --metric stoi \
                --ref-dir "$REF_DIR" \
                --deg-dir "$DEG_DIR"; then
                echo "✓ $codec @ $bitrate completed"
                SUCCESS_JOBS=$((SUCCESS_JOBS + 1))
            else
                echo "✗ $codec @ $bitrate failed"
                FAILED_JOBS=$((FAILED_JOBS + 1))
            fi
        done
    done
done

echo ""
echo "=================================================="
echo "STOI Evaluation Complete"
echo "=================================================="
echo "Completed: $(date)"
echo "Total jobs: $TOTAL_JOBS"
echo "Successful: $SUCCESS_JOBS"
echo "Failed: $FAILED_JOBS"
echo "=================================================="

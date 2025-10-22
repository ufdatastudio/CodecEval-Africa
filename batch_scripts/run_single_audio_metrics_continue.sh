#!/bin/bash
#SBATCH --job-name=single_metrics_continue
#SBATCH --output=slurm-single-metrics-continue.out
#SBATCH --time=06:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --account=ufdatastudios
#SBATCH --mem=50GB
#SBATCH --mail-user=c.okocha@ufl.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/single_metrics_continue_%j.out


echo "=== SINGLE AUDIO METRICS CONTINUATION ==="
echo "Continuing from completed encode/decode stage"
echo "Computing metrics for: 1 audio × 6 codecs × 4 bitrates = 24 combinations"
echo "Metrics: NISQA, DNSMOS, ViSQOL, Speaker, Prosody, WER"
echo ""

# Activate environment
source .venv/bin/activate

# Load modules
module load sox/14.4.2 ffmpeg/4.3.1

# Verify we have the decoded files
echo "Step 1: Verifying decoded files exist..."
DECODED_DIR="results/test_single/decoded"
if [ -d "$DECODED_DIR" ]; then
    FILE_COUNT=$(find "$DECODED_DIR" -name "*.wav" | wc -l)
    echo "Found $FILE_COUNT decoded audio files in $DECODED_DIR"
    if [ $FILE_COUNT -eq 0 ]; then
        echo "ERROR: No decoded files found!"
        exit 1
    fi
else
    echo "ERROR: Decoded directory not found!"
    exit 1
fi

# Run metrics computation (includes all metrics + ASR WER)
echo ""
echo "Step 2: Computing all metrics (NISQA, DNSMOS, ViSQOL, Speaker, Prosody, WER)..."
python -m code.pipeline_fixed --config configs/single_audio_test.yml --stage metrics

echo ""
echo "Step 3: Analyzing results..."
python scripts/analyze_results.py --results_dir results/test_single --output_dir results/test_single/reports

echo ""
echo "=== SINGLE AUDIO METRICS COMPLETE ==="
echo "Results saved to: results/test_single/"
echo "Reports saved to: results/test_single/reports/"

#!/bin/bash
#SBATCH --job-name=test_stoi
#SBATCH --output=batch_scripts/test_stoi_%j.out
#SBATCH --error=batch_scripts/test_stoi_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=01:00:00
#SBATCH --partition=hpg-default
#SBATCH --account=ufdatastudios

echo "============================================"
echo "STOI Quality Metric Evaluation"
echo "============================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

# Activate virtual environment
source /orange/ufdatastudios/c.okocha/CodecEval-Africa/.venv/bin/activate

# Change to project directory
cd /orange/ufdatastudios/c.okocha/CodecEval-Africa

# Configuration
CODEC="${CODEC:-Encodec}"
BITRATE="${BITRATE:-out_24kbps}"
REF_DIR="data/afrispeech_dialog/data"
DEG_DIR="outputs/afrispeech_dialog/${CODEC}_outputs/${BITRATE}"

echo "Configuration:"
echo "  Codec: $CODEC"
echo "  Bitrate: $BITRATE"
echo "  Reference directory: $REF_DIR"
echo "  Degraded directory: $DEG_DIR"
echo ""

# Run STOI evaluation (pystoi is installed)
python3 scripts/evaluate_quality.py \
    --metric stoi \
    --ref-dir "$REF_DIR" \
    --deg-dir "$DEG_DIR" \
    --output-dir "results/quality_metrics" \
    --output-name "${CODEC}_${BITRATE}_stoi_$(date +%Y%m%d_%H%M%S)"

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "============================================"
    echo "✓ STOI evaluation completed!"
    echo "============================================"
else
    echo "============================================"
    echo "✗ Evaluation failed with exit code $exit_code"
    echo "============================================"
fi

echo "Job completed at: $(date)"
exit $exit_code

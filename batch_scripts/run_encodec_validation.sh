#!/bin/bash
#SBATCH --job-name=encodec_validation
#SBATCH --output=encodec_validation_%j.out
#SBATCH --error=encodec_validation_%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=hpg-turin
#SBATCH --mem=8G

echo "=== ENCODEC VALIDATION TEST ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "Working directory: $(pwd)"

# Activate virtual environment
source .venv/bin/activate

# Run comprehensive EnCodec validation
python validate_encodec.py

echo "=== ENCODEC VALIDATION COMPLETE ==="
echo "Date: $(date)"

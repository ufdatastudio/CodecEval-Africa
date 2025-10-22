#!/bin/bash
#SBATCH --job-name=encoding_diagnosis
#SBATCH --output=encoding_diagnosis_%j.out
#SBATCH --error=encoding_diagnosis_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --mem=16G

echo "=== ENCODING PIPELINE DIAGNOSIS JOB ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "Working directory: $(pwd)"

# Activate virtual environment
source .venv/bin/activate

# Run diagnosis
python test_encoding_diagnosis.py

echo "=== DIAGNOSIS COMPLETE ==="
echo "Date: $(date)"

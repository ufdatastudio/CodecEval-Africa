#!/bin/bash
#SBATCH --job-name=test_pipeline_fix
#SBATCH --output=test_pipeline_fix_%j.out
#SBATCH --error=test_pipeline_fix_%j.err
#SBATCH --time=00:15:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --mem=8G

echo "=== TESTING PIPELINE FIX ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "Working directory: $(pwd)"

# Activate virtual environment
source .venv/bin/activate

# Test the pipeline fix
python test_pipeline_fix.py

echo "=== PIPELINE FIX TEST COMPLETE ==="
echo "Date: $(date)"

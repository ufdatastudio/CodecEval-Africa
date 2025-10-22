#!/bin/bash
#SBATCH --job-name=cuda_fix_test
#SBATCH --output=cuda_fix_test_%j.out
#SBATCH --error=cuda_fix_test_%j.err
#SBATCH --time=00:15:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --mem=8G

echo "=== CUDA TENSOR TYPE MISMATCH FIX TEST ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "Working directory: $(pwd)"

# Activate virtual environment
source .venv/bin/activate

# Test CUDA fixes
python test_cuda_fix.py

echo "=== CUDA FIX TEST COMPLETE ==="
echo "Date: $(date)"

#!/bin/bash
#SBATCH --job-name=nisqa_test
#SBATCH --output=logs/nisqa_test_%j.out
#SBATCH --time=01:00:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50GB
#SBATCH --account=ufdatastudios
#SBATCH --mail-user=c.okocha@ufl.edu
#SBATCH --mail-type=BEGIN,END,FAIL

echo "=== NISQA v2.0 TEST PROCESSING ==="
echo "Testing NISQA v2.0 on a small subset of files"
echo "Using GPU partition for faster processing"
echo ""

# Activate environment
source .venv/bin/activate

# Load modules
module load sox/14.4.2 ffmpeg/4.3.1

# Create output directory
mkdir -p results/nisqa_test_results

# Run NISQA on a small subset (max 50 files for testing)
echo "Step 1: Running NISQA v2.0 on test files (max 50 files)..."
python scripts/run_nisqa_batch.py \
    --input_dir results/ \
    --output_dir results/nisqa_test_results \
    --batch_size 5 \
    --num_workers 2 \
    --max_files 50

echo ""
echo "Step 2: Analyzing test results..."
python scripts/analyze_nisqa_results.py \
    --results_dir results/nisqa_test_results \
    --output_dir results/nisqa_test_results/analysis

echo ""
echo "=== NISQA TEST COMPLETE ==="
echo "Test results saved to: results/nisqa_test_results/"
echo "Analysis saved to: results/nisqa_test_results/analysis/"

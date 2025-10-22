#!/bin/bash
#SBATCH --job-name=dnsmos_test
#SBATCH --output=logs/dnsmos_test_%j.out
#SBATCH --time=01:00:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50GB
#SBATCH --account=ufdatastudios
#SBATCH --mail-user=c.okocha@ufl.edu
#SBATCH --mail-type=BEGIN,END,FAIL

echo "=== DNSMOS TEST PROCESSING ==="
echo "Testing DNSMOS on a small subset of files"
echo "Using GPU partition for faster processing"
echo ""

# Activate environment
source .venv/bin/activate

# Load modules
module load sox/14.4.2 ffmpeg/4.3.1

# Create output directory
mkdir -p results/dnsmos_test_results

# Run DNSMOS on a small subset (max 20 files for testing)
echo "Step 1: Running DNSMOS on test files (max 20 files)..."
python scripts/run_dnsmos_batch.py \
    --input_dir results/ \
    --output_dir results/dnsmos_test_results \
    --batch_size 5 \
    --max_files 20

echo ""
echo "Step 2: Analyzing test results..."
python scripts/analyze_dnsmos_results.py \
    --results_dir results/dnsmos_test_results \
    --output_dir results/dnsmos_test_results/analysis

echo ""
echo "=== DNSMOS TEST COMPLETE ==="
echo "Test results saved to: results/dnsmos_test_results/"
echo "Analysis saved to: results/dnsmos_test_results/analysis/"



#!/bin/bash
#SBATCH --job-name=all_metrics_test
#SBATCH --output=logs/all_metrics_test_%j.out
#SBATCH --time=02:00:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50GB
#SBATCH --account=ufdatastudios
#SBATCH --mail-user=c.okocha@ufl.edu
#SBATCH --mail-type=BEGIN,END,FAIL

echo "=== ALL METRICS TEST PROCESSING ==="
echo "Testing all metrics on a small subset of files"
echo "Metrics: NISQA v2.0, DNSMOS, ViSQOL, Speaker Similarity, Prosody F0 RMSE"
echo ""

# Activate environment
source .venv/bin/activate

# Load modules
module load sox/14.4.2 ffmpeg/4.3.1

# Create output directory
mkdir -p results/all_metrics_test_results

# Run all metrics on a small subset (max 10 files for testing)
echo "Step 1: Running all metrics on test files (max 10 files)..."
python scripts/run_all_metrics_batch.py \
    --input_dir results/ \
    --output_dir results/all_metrics_test_results \
    --batch_size 2 \
    --max_files 10

echo ""
echo "Step 2: Analyzing test results..."
python scripts/analyze_all_metrics.py \
    --results_dir results/all_metrics_test_results \
    --output_dir results/all_metrics_test_results/analysis

echo ""
echo "=== ALL METRICS TEST COMPLETE ==="
echo "Test results saved to: results/all_metrics_test_results/"
echo "Analysis saved to: results/all_metrics_test_results/analysis/"



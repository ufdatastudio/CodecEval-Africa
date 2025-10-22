#!/bin/bash
#SBATCH --job-name=all_metrics
#SBATCH --output=logs/all_metrics_%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50GB
#SBATCH --account=ufdatastudios
#SBATCH --mail-user=c.okocha@ufl.edu
#SBATCH --mail-type=BEGIN,END,FAIL

echo "=== ALL METRICS BATCH PROCESSING ==="
echo "Processing all decoded audio files with comprehensive metrics"
echo "Metrics: NISQA v2.0, DNSMOS, ViSQOL, Speaker Similarity, Prosody F0 RMSE"
echo ""

# Activate environment
source .venv/bin/activate

# Load modules
module load sox/14.4.2 ffmpeg/4.3.1

# Create output directory
mkdir -p results/all_metrics_results

# Run all metrics on decoded files
echo "Step 1: Running all metrics on decoded files..."
python scripts/run_all_metrics_batch.py \
    --input_dir results/ \
    --output_dir results/all_metrics_results \
    --batch_size 5 \
    --max_files 50

echo ""
echo "Step 2: Analyzing comprehensive results..."
python scripts/analyze_all_metrics.py \
    --results_dir results/all_metrics_results \
    --output_dir results/all_metrics_results/analysis

echo ""
echo "=== ALL METRICS PROCESSING COMPLETE ==="
echo "Results saved to: results/all_metrics_results/"
echo "Analysis saved to: results/all_metrics_results/analysis/"



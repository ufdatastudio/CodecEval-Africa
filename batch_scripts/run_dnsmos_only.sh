#!/bin/bash
#SBATCH --job-name=dnsmos_only
#SBATCH --output=logs/dnsmos_only_%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50GB
#SBATCH --account=ufdatastudios
#SBATCH --mail-user=c.okocha@ufl.edu
#SBATCH --mail-type=BEGIN,END,FAIL

echo "=== DNSMOS BATCH PROCESSING ==="
echo "Processing all decoded audio files with DNSMOS"
echo "Using GPU partition for faster processing"
echo ""

# Activate environment
source .venv/bin/activate

# Load modules
module load sox/14.4.2 ffmpeg/4.3.1

# Create output directory
mkdir -p results/dnsmos_results

# Run DNSMOS on all decoded files
echo "Step 1: Running DNSMOS on all decoded files..."
python scripts/run_dnsmos_batch.py \
    --input_dir results/ \
    --output_dir results/dnsmos_results \
    --batch_size 10

echo ""
echo "Step 2: Analyzing DNSMOS results..."
python scripts/analyze_dnsmos_results.py \
    --results_dir results/dnsmos_results \
    --output_dir results/dnsmos_results/analysis

echo ""
echo "=== DNSMOS PROCESSING COMPLETE ==="
echo "Results saved to: results/dnsmos_results/"
echo "Analysis saved to: results/dnsmos_results/analysis/"



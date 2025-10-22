#!/bin/bash
#SBATCH --job-name=nisqa_only
#SBATCH --output=logs/nisqa_only_%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50GB
#SBATCH --account=ufdatastudios
#SBATCH --mail-user=c.okocha@ufl.edu
#SBATCH --mail-type=BEGIN,END,FAIL

echo "=== NISQA v2.0 BATCH PROCESSING ==="
echo "Processing all decoded audio files with NISQA v2.0"
echo "Using GPU partition for faster processing"
echo ""

# Activate environment
source .venv/bin/activate

# Load modules
module load sox/14.4.2 ffmpeg/4.3.1

# Create output directory
mkdir -p results/nisqa_results

# Run NISQA on all decoded files
echo "Step 1: Running NISQA v2.0 on all decoded files..."
python scripts/run_nisqa_batch.py \
    --input_dir results/ \
    --output_dir results/nisqa_results \
    --batch_size 10 \
    --num_workers 2

echo ""
echo "Step 2: Analyzing NISQA results..."
python scripts/analyze_nisqa_results.py \
    --results_dir results/nisqa_results \
    --output_dir results/nisqa_results/analysis

echo ""
echo "=== NISQA PROCESSING COMPLETE ==="
echo "Results saved to: results/nisqa_results/"
echo "Analysis saved to: results/nisqa_results/analysis/"

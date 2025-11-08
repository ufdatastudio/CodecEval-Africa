#!/bin/bash
#SBATCH --job-name=encodec_afrinames
#SBATCH --output=batch_scripts/encodec_afrinames_%j.out
#SBATCH --error=batch_scripts/encodec_afrinames_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=06:00:00
#SBATCH --account=ufdatastudios
#SBATCH --mail-user=c.okocha@ufl.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Activate virtual environment
source /orange/ufdatastudios/c.okocha/CodecEval-Africa/.venv/bin/activate

# Load required modules
module load sox/14.4.2 ffmpeg/4.3.1 

# Set cache directories for faster processing
export HF_HOME="/orange/ufdatastudios/c.okocha/.cache/huggingface"
export TRANSFORMERS_CACHE="/orange/ufdatastudios/c.okocha/.cache/transformers"
export TORCH_HOME="/orange/ufdatastudios/c.okocha/.cache/torch"

# Performance settings
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4

echo "Environment setup complete"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
echo "CUDA: $(nvcc --version | grep release)"

# Change to the script directory
cd /orange/ufdatastudios/c.okocha/CodecEval-Africa/code/codecs

# Run the encodec script
python encodec_runner.py

echo "Job completed at: $(date)"
echo "Encodec processing complete!"
echo "Output files saved to: /orange/ufdatastudios/c.okocha/CodecEval-Africa/outputs/afrinames/Encodec_outputs/"


#!/bin/bash
#SBATCH --job-name=unicodec_afrispeech
#SBATCH --output=unicodec_afrispeech_%j.out
#SBATCH --error=unicodec_afrispeech_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=04:00:00
#SBATCH --account=ufdatastudios
#SBATCH --mail-user=c.okocha@ufl.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=hpg-b200
#SBATCH --gpus=

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Activate virtual environment
source /orange/ufdatastudios/c.okocha/CodecEval-Africa/.venv/bin/activate

# Load required modules (CPU-only)
module load sox/14.4.2 ffmpeg/4.3.1 

# Set cache directories for faster processing
export HF_HOME="/orange/ufdatastudios/c.okocha/.cache/huggingface"
export TRANSFORMERS_CACHE="/orange/ufdatastudios/c.okocha/.cache/transformers"
export TORCH_HOME="/orange/ufdatastudios/c.okocha/.cache/torch"

# Performance settings (CPU-only)
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Environment setup complete"
# CPU-only run; no GPU/CUDA info

# Change to the script directory
cd /orange/ufdatastudios/c.okocha/CodecEval-Africa/code/codecs

# Run the UniCodec script
python unicodec_runner.py

echo "Job completed at: $(date)"
echo "UniCodec processing complete!"


#!/bin/bash
#SBATCH --job-name=convert_afrispeech_multilingual
#SBATCH --output=batch_scripts/convert_afrispeech_multilingual_%j.out
#SBATCH --error=batch_scripts/convert_afrispeech_multilingual_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --partition=hpg-default
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=c.okocha@ufl.edu
#SBATCH --account=ufdatastudios

echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"

# Load required modules
module load sox/14.4.2
module load ffmpeg/4.3.1

# Activate virtual environment
source /orange/ufdatastudios/c.okocha/CodecEval-Africa/.venv/bin/activate

# Set cache directories
export HF_HOME=/orange/ufdatastudios/c.okocha/CodecEval-Africa/.cache/huggingface
export TRANSFORMERS_CACHE=/orange/ufdatastudios/c.okocha/CodecEval-Africa/.cache/huggingface/transformers
export TORCH_HOME=/orange/ufdatastudios/c.okocha/CodecEval-Africa/.cache/torch

# Change to project directory
cd /orange/ufdatastudios/c.okocha/CodecEval-Africa

echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Run conversion script
echo "Starting audio conversion..."
python scripts/convert_afrispeech_multilingual_to_wav.py

echo "Job completed at: $(date)"
echo "Conversion complete!"
echo "Output directory: /orange/ufdatastudios/c.okocha/CodecEval-Africa/data/afrispeech_multilingual_wav"


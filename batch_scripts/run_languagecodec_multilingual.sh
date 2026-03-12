#!/bin/bash
#SBATCH --job-name=languagecodec_multilingual
#SBATCH --output=batch_scripts/languagecodec_multilingual_%j.out
#SBATCH --error=batch_scripts/languagecodec_multilingual_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --partition=hpg-default
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=c.okocha@ufl.edu

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Load required modules (CPU run — no CUDA)
module load sox/14.4.2
module load ffmpeg/4.3.1

# Activate virtual environment (LanguageCodec-specific)
source /orange/ufdatastudios/c.okocha/CodecEval-Africa/.venv_languagecodec/bin/activate

# Force CPU usage
export CUDA_VISIBLE_DEVICES=""

# Set cache directories
export HF_HOME=/orange/ufdatastudios/c.okocha/CodecEval-Africa/.cache/huggingface
export TRANSFORMERS_CACHE=/orange/ufdatastudios/c.okocha/CodecEval-Africa/.cache/huggingface/transformers
export HF_DATASETS_CACHE=/orange/ufdatastudios/c.okocha/CodecEval-Africa/.cache/huggingface/datasets
export TORCH_HOME=/orange/ufdatastudios/c.okocha/CodecEval-Africa/.cache/torch

# Print Python and CUDA info
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Change to project directory
cd /orange/ufdatastudios/c.okocha/CodecEval-Africa

# Run LanguageCodec for multilingual dataset
echo "Running LanguageCodec on multilingual dataset..."
python scripts/run_languagecodec_multilingual.py

echo "End Time: $(date)"
echo "Job completed!"
echo "Output files saved to: /orange/ufdatastudios/c.okocha/CodecEval-Africa/outputs/afrispeech_multilingual/LanguageCodec_outputs/"


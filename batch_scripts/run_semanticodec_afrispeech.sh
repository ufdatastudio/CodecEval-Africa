#!/bin/bash
#SBATCH --job-name=semanticodec_afrispeech
#SBATCH --output=batch_scripts/semanticodec_afrispeech_%j.out
#SBATCH --error=batch_scripts/semanticodec_afrispeech_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=15:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gres=
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=c.okocha@ufl.edu
#SBATCH --account=ufdatastudios

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Load required modules (CPU-only)
module load sox/14.4.2
module load ffmpeg/4.3.1

# Activate virtual environment
source /orange/ufdatastudios/c.okocha/CodecEval-Africa/.venv/bin/activate

# Set cache directories
export HF_HOME=/orange/ufdatastudios/c.okocha/CodecEval-Africa/.cache/huggingface
export TRANSFORMERS_CACHE=/orange/ufdatastudios/c.okocha/CodecEval-Africa/.cache/huggingface/transformers
export HF_DATASETS_CACHE=/orange/ufdatastudios/c.okocha/CodecEval-Africa/.cache/huggingface/datasets
export TORCH_HOME=/orange/ufdatastudios/c.okocha/CodecEval-Africa/.cache/torch

# Set FFmpeg library path for TorchCodec compatibility
export LD_LIBRARY_PATH="/apps/ffmpeg/4.3.1/lib:$LD_LIBRARY_PATH"

# Performance settings (CPU-only)
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print environment info
echo "Environment setup complete"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Change to the project directory
cd /orange/ufdatastudios/c.okocha/CodecEval-Africa

# Run SemantiCodec
echo "Running SemantiCodec on AfriSpeech dialog data..."
echo "This will process all audio files at 6 different bitrates (starting from 0.63 kbps if set):"
echo "- 0.31 kbps (token_rate=25, vocab_size=4096)"
echo "- 0.63 kbps (token_rate=50, vocab_size=4096)"
echo "- 1.25 kbps (token_rate=100, vocab_size=4096)"
echo "- 0.33 kbps (token_rate=25, vocab_size=8192)"
echo "- 0.68 kbps (token_rate=50, vocab_size=16384)"
echo "- 1.40 kbps (token_rate=100, vocab_size=32768)"
echo ""

# Resume from 1.40 kbps (incomplete - only 3/49 files done)
export SEMANTICODEC_START_LABEL="1.40kbps"
python code/codecs/sematicodec_runner.py

echo "End Time: $(date)"
echo "SemantiCodec processing complete!"
echo "Output files saved to: /orange/ufdatastudios/c.okocha/CodecEval-Africa/outputs/SemantiCodec_outputs/"

#!/bin/bash
#SBATCH --job-name=wavtokenizer_afrispeech
#SBATCH --output=batch_scripts/wavtokenizer_afrispeech_%j.out
#SBATCH --error=batch_scripts/wavtokenizer_afrispeech_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=c.okocha@ufl.edu
#SBATCH --account=ufdatastudios

# GPU settings
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# Environment setup
module load sox/14.4.2
module load ffmpeg/4.3.1
source /orange/ufdatastudios/c.okocha/CodecEval-Africa/.venv_wavtokenizer39/bin/activate

cd /orange/ufdatastudios/c.okocha/CodecEval-Africa

echo "Start Time: $(date)"
echo "="*80
echo "WavTokenizer AfriSpeech Compression Job"
echo "="*80
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
    echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)MB"
fi
echo "="*80

echo ""
echo "Running WavTokenizer folder compression (GPU)..."
python code/codecs/wavtokenizer_runner.py

echo ""
echo "End Time: $(date)"
echo "Done. Outputs in: /orange/ufdatastudios/c.okocha/CodecEval-Africa/outputs/WavTokenizer_outputs"


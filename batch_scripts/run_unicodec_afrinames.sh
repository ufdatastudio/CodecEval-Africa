#!/bin/bash
#SBATCH --job-name=unicodec_afrinames
#SBATCH --output=batch_scripts/unicodec_afrinames_%j.out
#SBATCH --error=batch_scripts/unicodec_afrinames_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=c.okocha@ufl.edu
#SBATCH --account=ufdatastudios

# GPU settings
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# Env and deps
module load sox/14.4.2
module load ffmpeg/4.3.1
source /orange/ufdatastudios/c.okocha/CodecEval-Africa/.venv_unicodec39/bin/activate

cd /orange/ufdatastudios/c.okocha/CodecEval-Africa

echo "Start Time: $(date)"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
fi

echo "Running UniCodec on Afri-Names dataset (GPU)..."
python code/codecs/unicodec_runner.py

echo "End Time: $(date)"
echo "Done. Outputs in: /orange/ufdatastudios/c.okocha/CodecEval-Africa/outputs/afrinames/UniCodec_outputs"


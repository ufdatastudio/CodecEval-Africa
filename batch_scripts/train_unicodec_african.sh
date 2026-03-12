#!/bin/bash
#SBATCH --account=ufdatastudios
#SBATCH --job-name=unicodec_african
#SBATCH --output=/orange/ufdatastudios/c.okocha/CodecEval-Africa/batch_scripts/logs/unicodec_african_%j.out
#SBATCH --error=/orange/ufdatastudios/c.okocha/CodecEval-Africa/batch_scripts/logs/unicodec_african_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --partition=hpg-turin
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=c.okocha@ufl.edu

echo "========================================"
echo "SLURM Job: UniCodec African Finetuning"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================"

# GPU Info
echo ""
echo "===== GPU Info ====="
nvidia-smi || true

# Load required modules
module load sox/14.4.2 ffmpeg/4.3.1 || true

# Add sox library to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/apps/sox/14.4.2/lib:${LD_LIBRARY_PATH:-}

export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

# Environment setup
cd /orange/ufdatastudios/c.okocha/CodecEval-Africa
source .venv_unicodec39/bin/activate

# WandB configuration
export WANDB_API_KEY="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"
export WANDB_PROJECT="african-codec-training"
export WANDB_NAME="unicodec-african-finetuning-${SLURM_JOB_ID}"
export WANDB_DIR="/orange/ufdatastudios/c.okocha/CodecEval-Africa/UniCodec/logs/african_finetuning"

# Print environment info
echo ""
echo "Python: $(which python3)"
echo "PyTorch version: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python3 -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# Create log directory
mkdir -p /orange/ufdatastudios/c.okocha/CodecEval-Africa/batch_scripts/logs
mkdir -p $WANDB_DIR

echo "Dataset info:"
echo "Training samples: $(wc -l < /orange/ufdatastudios/c.okocha/CodecEval-Africa/data/codec_training_data/african_speech_train_unicodec.txt)"
echo "Validation samples: $(wc -l < /orange/ufdatastudios/c.okocha/CodecEval-Africa/data/codec_training_data/african_speech_val_unicodec.txt)"
echo ""

echo "Starting training..."
echo "========================================"

# Run training
cd /orange/ufdatastudios/c.okocha/CodecEval-Africa/UniCodec
python3 train.py fit --config ./configs/african_finetuning.yaml

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Training completed with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "========================================"

if [ $EXIT_CODE -ne 0 ]; then
    echo "Training failed! Check logs for details."
fi

exit $EXIT_CODE

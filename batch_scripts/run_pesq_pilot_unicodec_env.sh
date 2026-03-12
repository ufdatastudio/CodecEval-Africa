#!/bin/bash
#SBATCH --job-name=pesq_pilot_uc
#SBATCH --output=batch_scripts/pesq_pilot_uc_%j.out
#SBATCH --error=batch_scripts/pesq_pilot_uc_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=02:00:00
#SBATCH --account=ufdatastudios
#SBATCH --partition=hpg-default

set -e

ulimit -c 0

source /orange/ufdatastudios/c.okocha/CodecEval-Africa/.venv_unicodec39/bin/activate
cd /orange/ufdatastudios/c.okocha/CodecEval-Africa

echo "PESQ pilot (unicodec env) started: $(date)"
python scripts/evaluate_quality.py \
  --metric pesq \
  --ref-dir data/afrispeech_dialog/data \
  --deg-dir outputs/afrispeech_dialog/Encodec_outputs/out_24kbps \
  --output-name afrispeech_dialog_Encodec_out_24kbps_pesq_ucenv

echo "PESQ pilot (unicodec env) completed: $(date)"

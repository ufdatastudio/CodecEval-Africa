#!/bin/bash
#SBATCH --job-name=nisqa_pilot
#SBATCH --output=batch_scripts/nisqa_pilot_%j.out
#SBATCH --error=batch_scripts/nisqa_pilot_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=02:00:00
#SBATCH --account=ufdatastudios
#SBATCH --partition=hpg-default

set -euo pipefail

source /orange/ufdatastudios/c.okocha/CodecEval-Africa/.venv/bin/activate
cd /orange/ufdatastudios/c.okocha/CodecEval-Africa

echo "NISQA pilot started: $(date)"
python scripts/evaluate_quality.py \
  --metric nisqa \
  --audio-dir outputs/afrispeech_dialog/Encodec_outputs/out_24kbps \
  --num-samples 10 \
  --output-dir results/NISQA/quality_metrics \
  --output-name afrispeech_dialog_Encodec_out_24kbps_nisqa_pilot

echo "NISQA pilot completed: $(date)"

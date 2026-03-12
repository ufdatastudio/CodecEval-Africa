#!/bin/bash
#SBATCH --job-name=wer_all_codecs
#SBATCH --output=batch_scripts/wer_all_codecs_%j.out
#SBATCH --error=batch_scripts/wer_all_codecs_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=48:00:00
#SBATCH --account=ufdatastudios
#SBATCH --partition=hpg-default

set -euo pipefail

source /orange/ufdatastudios/c.okocha/CodecEval-Africa/.venv/bin/activate
cd /orange/ufdatastudios/c.okocha/CodecEval-Africa

OUT_DIR="results/ASR/quality_metrics"
mkdir -p "$OUT_DIR"

WER_LIMIT="${WER_LIMIT:--1}"
ASR_MODEL="${ASR_MODEL:-openai/whisper-base}"
ASR_LANGUAGE="${ASR_LANGUAGE:-en}"

SUMMARY_FILE="$OUT_DIR/wer_all_codecs_summary_$(date +%Y%m%d_%H%M%S).txt"

echo "==================================================" | tee "$SUMMARY_FILE"
echo "WER Evaluation - All Codecs / All Datasets" | tee -a "$SUMMARY_FILE"
echo "Started: $(date)" | tee -a "$SUMMARY_FILE"
echo "Output dir: $OUT_DIR" | tee -a "$SUMMARY_FILE"
echo "Per-variant file limit: $WER_LIMIT" | tee -a "$SUMMARY_FILE"
echo "ASR model: $ASR_MODEL" | tee -a "$SUMMARY_FILE"
echo "ASR language: $ASR_LANGUAGE" | tee -a "$SUMMARY_FILE"
echo "==================================================" | tee -a "$SUMMARY_FILE"

TOTAL=0
SUCCESS=0
FAILED=0
SKIPPED=0

for dataset in afrispeech_dialog afrinames afrispeech_multilingual; do
  dataset_outputs="outputs/$dataset"
  transcript_csv="data/$dataset/metadata.csv"

  echo "" | tee -a "$SUMMARY_FILE"
  echo "Dataset: $dataset" | tee -a "$SUMMARY_FILE"

  if [ ! -d "$dataset_outputs" ]; then
    echo "  SKIP: missing outputs dir: $dataset_outputs" | tee -a "$SUMMARY_FILE"
    continue
  fi

  if [ ! -f "$transcript_csv" ]; then
    echo "  SKIP: no transcript CSV found at $transcript_csv" | tee -a "$SUMMARY_FILE"
    continue
  fi

  for codec_dir in "$dataset_outputs"/*_outputs; do
    [ -d "$codec_dir" ] || continue

    codec_name=$(basename "$codec_dir" | sed 's/_outputs$//')
    echo "  Codec: $codec_name" | tee -a "$SUMMARY_FILE"

    for variant_dir in "$codec_dir"/*; do
      [ -d "$variant_dir" ] || continue

      variant=$(basename "$variant_dir")
      wav_count=$(find "$variant_dir" -maxdepth 1 -type f -name "*.wav" | wc -l)
      TOTAL=$((TOTAL + 1))

      if [ "$wav_count" -eq 0 ]; then
        echo "    [$variant] SKIP: no wav files" | tee -a "$SUMMARY_FILE"
        SKIPPED=$((SKIPPED + 1))
        continue
      fi

      out_name="${dataset}_${codec_name}_${variant}_wer"
      json_path="$OUT_DIR/${out_name}.json"

      echo "    [$variant] Running WER on $wav_count files..." | tee -a "$SUMMARY_FILE"

      if python scripts/evaluate_quality.py \
        --metric wer \
        --audio-dir "$variant_dir" \
        --transcript-csv "$transcript_csv" \
        --asr-model "$ASR_MODEL" \
        --asr-language "$ASR_LANGUAGE" \
        --num-samples "$WER_LIMIT" \
        --output-dir "$OUT_DIR" \
        --output-name "$out_name"; then

        if [ -s "$json_path" ]; then
          wer=$(python - <<PY
import json
with open('$json_path', 'r') as f:
    d = json.load(f)
print(d.get('wer', 'nan'))
PY
)
          nfiles=$(python - <<PY
import json
with open('$json_path', 'r') as f:
    d = json.load(f)
print(d.get('num_scored_files', 0))
PY
)
          echo "    [$variant] OK: WER=$wer files=$nfiles" | tee -a "$SUMMARY_FILE"
          SUCCESS=$((SUCCESS + 1))
        else
          echo "    [$variant] FAIL: JSON missing: $json_path" | tee -a "$SUMMARY_FILE"
          FAILED=$((FAILED + 1))
        fi
      else
        echo "    [$variant] FAIL: evaluate_quality.py wer failed" | tee -a "$SUMMARY_FILE"
        FAILED=$((FAILED + 1))
      fi
    done
  done
done

echo "" | tee -a "$SUMMARY_FILE"
echo "==================================================" | tee -a "$SUMMARY_FILE"
echo "Completed: $(date)" | tee -a "$SUMMARY_FILE"
echo "Total: $TOTAL | Success: $SUCCESS | Failed: $FAILED | Skipped: $SKIPPED" | tee -a "$SUMMARY_FILE"
echo "Summary: $SUMMARY_FILE" | tee -a "$SUMMARY_FILE"
echo "==================================================" | tee -a "$SUMMARY_FILE"
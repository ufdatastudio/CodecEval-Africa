# CodecEval-Africa

A comprehensive benchmark for neural speech codecs on African-accented English and conversational audio.

## Overview

CodecEval-Africa evaluates 6 state-of-the-art neural speech codecs on balanced African speech datasets, providing comprehensive quality metrics and performance analysis.

## Supported Codecs

**Working Codecs (6 total):**
- **EnCodec** (`encodec_24khz`) - Meta's production neural codec
- **LanguageCodec**
- **UniCodec** (`unicodec`) - Simplified neural autoencoder with quantization
- **DAC** (`dac`) - High-quality residual neural codec
- **SemantiCodec** (`sematicodec`) - Semantic-aware codec with attention mechanisms
- **APCodec** (`apcodec`) - Adaptive perceptual codec

## Metrics

- **NISQA** - Speech quality assessment
- **ViSQOL** - Perceptual quality (spectral distance)
- **DNSMOS** - Noise suppression quality
- **Speaker Similarity** - MFCC cosine similarity
- **Prosody** - F0 RMSE for fundamental frequency preservation
- **ASR WER** - Word error rate (when transcripts available)

## Quickstart

### Environment Setup
```bash
# Create virtual environment and install dependencies
make env

# Activate environment
source .venv/bin/activate
```

### Running the Benchmark
```bash
# Encode/decode with all codecs
make run

# Compute quality metrics
make scores

# Generate visualizations
make plots
```

### GPU Acceleration
```bash
# Submit to SLURM for GPU processing
sbatch run_sample_metrics_b200.sh
```

## Dataset

- **Afri-Names**: 8 African accents × 35 clips each = 280 clips
- **AfriSpeech-Dialog**: 6 African accents × 35 clips each = 210 clips
- **Total**: ~490 audio clips for comprehensive evaluation

## Configuration

Edit `configs/benchmark.yml` to:
- Adjust bitrates (3, 6, 12, 18 kbps)
- Select codecs to test
- Configure evaluation conditions
- Set output directories

## Results

Results are saved to:
- `results/decoded/` - Encoded audio files
- `results/csv/benchmark.csv` - Quality metrics
- `results/figures/` - Visualizations and plots


## License

This project uses Hugging Face datasets. Please respect dataset licenses and do not redistribute audio files.

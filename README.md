# 🎙️ CodecEval-Africa

<div align="center">

**A comprehensive benchmark for neural speech codecs on African-accented English and conversational audio**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Supported Codecs](#-supported-codecs)
- [Evaluation Metrics](#-evaluation-metrics)
- [Datasets](#-datasets)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

**CodecEval-Africa** is a comprehensive benchmarking framework for evaluating state-of-the-art neural speech codecs on African-accented English and conversational audio. This project provides:

- **6 Neural Codecs** - Evaluation of leading speech compression models
- **6 Quality Metrics** - Comprehensive perceptual and objective quality assessment
- **African Speech Datasets** - Focused evaluation on African-accented English
- **Multi-Bitrate Analysis** - Performance evaluation across various compression rates
- **GPU Acceleration** - Optimized for high-performance computing environments

---

## ✨ Features

- 🎚️ **Multi-Codec Support** - EnCodec, DAC, LanguageCodec, UniCodec, SemantiCodec, WavTokenizer
- 📊 **Comprehensive Metrics** - NISQA, ViSQOL, DNSMOS, Speaker Similarity, Prosody, ASR WER
- 🌍 **African Speech Focus** - Specialized evaluation on African-accented English datasets
- ⚡ **GPU Accelerated** - Optimized batch processing for SLURM clusters
- 📈 **Bitrate Analysis** - Performance evaluation from ultra-low to high bitrates
- 🔧 **Modular Design** - Easy to extend with new codecs and metrics
- 📝 **Detailed Logging** - Comprehensive output for analysis and visualization

---

## 🔧 Supported Codecs

| Codec | Bitrates | Features | Hardware |
|-------|----------|----------|----------|
| **EnCodec** | 3.0, 6.0, 12.0, 24.0 kbps | Causal streaming, Meta's production codec | GPU |
| **DAC** | 8, 16, 24 kbps | High-quality residual codec, multiple sampling rates | GPU |
| **LanguageCodec** | ~6.6 kbps | Language model-based, 4 bandwidth variants | CPU |
| **UniCodec** | Variable (~0.35 kbps default) | Unified framework, configurable bandwidth | GPU |
| **SemantiCodec** | 0.31-1.40 kbps | Semantic-aware, ultra-low bitrate compression | CPU |
| **WavTokenizer** | Token-based | Multiple token-rate models (40-75 tokens/sec) | GPU |

### Codec Details

<details>
<summary><b>EnCodec</b> - Meta's Production Neural Codec</summary>

- **Bitrates**: 3.0, 6.0, 12.0, 24.0 kbps
- **Features**: Causal streaming support, real-time processing
- **Use Case**: Production-ready speech compression

</details>

<details>
<summary><b>DAC</b> - High-Quality Residual Neural Codec</summary>

- **Bitrates**: 8 kbps (16kHz), 16 kbps (24kHz), 24 kbps (44kHz)
- **Features**: Multiple sampling rate models, excellent speech quality
- **Use Case**: High-quality speech compression

</details>

<details>
<summary><b>LanguageCodec</b> - Language Model-Based Codec</summary>

- **Bitrates**: ~6.6 kbps (4 bandwidth variants: IDs 0-3)
- **Features**: Bandwidth embedding variations, language-aware compression
- **Use Case**: Language-optimized compression

</details>

<details>
<summary><b>UniCodec</b> - Unified Neural Codec Framework</summary>

- **Bitrates**: Variable (bandwidth IDs 0-3, ~0.35 kbps default)
- **Features**: Configurable bandwidth settings, unified framework
- **Use Case**: Flexible compression with configurable quality

</details>

<details>
<summary><b>SemantiCodec</b> - Semantic-Aware Codec</summary>

- **Bitrates**: 0.31, 0.63, 1.25, 0.33, 0.68, 1.40 kbps (6 configurations)
- **Features**: Semantic-aware compression with attention mechanisms
- **Use Case**: Ultra-low bitrate semantic compression

</details>

<details>
<summary><b>WavTokenizer</b> - Token-Based Audio Codec</summary>

- **Models**: Multiple configurations (40-75 tokens/sec)
- **Features**: Token-rate based compression, multiple model variants
- **Use Case**: Token-based audio representation

</details>

---

## 📊 Evaluation Metrics

| Metric | Description | Type |
|--------|-------------|------|
| **NISQA** | Speech quality assessment | Perceptual |
| **ViSQOL** | Perceptual quality (spectral distance) | Perceptual |
| **DNSMOS** | Noise suppression quality | Perceptual |
| **Speaker Similarity** | MFCC cosine similarity | Objective |
| **Prosody** | F0 RMSE for fundamental frequency preservation | Objective |
| **ASR WER** | Word error rate (when transcripts available) | Objective |

---

## 🌍 Datasets

### Afri-Names
- **Type**: African-accented English names dataset
- **Accents**: 8 African accents
- **Content**: Multiple audio clips per accent
- **Use Case**: Accent-specific evaluation

### AfriSpeech-Dialog
- **Type**: Conversational African speech dataset
- **Accents**: 6 African accents
- **Content**: Medical consultations and dialogues
- **Use Case**: Real-world conversational audio evaluation

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- SLURM cluster access (for batch processing)

### Installation

```bash
# Clone the repository
git clone https://github.com/ufdatastudio/CodecEval-Africa.git
cd CodecEval-Africa

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Set up cache directories (optional, for faster processing)
export HF_HOME="/path/to/.cache/huggingface"
export TRANSFORMERS_CACHE="/path/to/.cache/transformers"
export TORCH_HOME="/path/to/.cache/torch"
```

---

## 💻 Usage

### Individual Codec Evaluation

Submit batch jobs for each codec using SLURM:

```bash
# EnCodec evaluation
sbatch batch_scripts/run_encodec_afrinames.sh

# DAC evaluation
sbatch batch_scripts/run_dac_afrinames.sh

# LanguageCodec evaluation
sbatch batch_scripts/run_languagecodec_afrinames.sh

# UniCodec evaluation
sbatch batch_scripts/run_unicodec_afrinames.sh

# SemantiCodec evaluation
sbatch batch_scripts/run_semanticodec_afrinames.sh

# WavTokenizer evaluation
sbatch batch_scripts/run_wavtokenizer_afrinames.sh
```

### Metrics Evaluation

```bash
# Run quality metrics on decoded audio
python scripts/run_all_metrics_batch.py

# Generate analysis and visualizations
python scripts/analyze_all_metrics.py
python scripts/plot_reports.py
```

### GPU Acceleration

- **GPU-Accelerated Codecs**: EnCodec, DAC, UniCodec, WavTokenizer
- **CPU Codecs**: LanguageCodec, SemantiCodec
- Batch scripts are pre-configured for SLURM GPU partitions

---

## ⚙️ Configuration

### Codec-Specific Bitrates

| Codec | Supported Bitrates |
|-------|-------------------|
| EnCodec | 3, 6, 12, 24 kbps |
| DAC | 8, 16, 24 kbps (different sampling rates) |
| LanguageCodec | 4 bandwidth variants (~6.6 kbps each) |
| UniCodec | 4 bandwidth IDs (variable bitrates) |
| SemantiCodec | 6 ultra-low bitrate configurations (0.31-1.40 kbps) |
| WavTokenizer | Multiple token-rate models |

### Configuration Files

Edit `configs/benchmark.yml` to:
- Select codecs to test
- Configure evaluation conditions
- Set output directories
- Adjust bitrate settings

---

## 📁 Results

Results are automatically saved to the following directories:

```
results/
├── decoded/          # Encoded audio files
├── csv/             # Quality metrics (CSV format)
│   └── benchmark.csv
└── figures/         # Visualizations and plots
```

### Output Format

- **Audio Files**: Decoded audio in WAV format
- **Metrics CSV**: Comprehensive quality metrics for all codecs
- **Visualizations**: Performance plots and comparisons

---

## 📂 Project Structure

```
CodecEval-Africa/
├── batch_scripts/          # SLURM batch job scripts
├── code/                   # Core codec evaluation code
│   └── codecs/            # Codec runners
├── configs/               # Configuration files
├── data/                  # Dataset manifests
├── scripts/               # Evaluation and analysis scripts
├── results/               # Output results
│   ├── decoded/          # Decoded audio files
│   ├── csv/              # Metrics CSV files
│   └── figures/          # Visualization plots
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project uses Hugging Face datasets. Please respect dataset licenses and do not redistribute audio files.

See the [LICENSE](LICENSE) file for more details.

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

<div align="center">

**Made with ❤️ for African Speech Research**

</div>

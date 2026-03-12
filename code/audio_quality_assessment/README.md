# Audio Quality Assessment Module

A comprehensive audio quality assessment toolkit specifically designed for evaluating neural codec compression on African speech data. This module implements state-of-the-art metrics using established libraries and validated methods from the literature.

## 🎯 Overview

This module provides accurate implementations of multiple complementary audio quality metrics:

### Audio Quality Assessment
- **NISQA** (Non-Intrusive Speech Quality Assessment) - Predicts MOS without reference
- **ViSQOL** (Virtual Speech Quality Objective Listener) - Perceptual quality assessment  
- **DNSMOS** (Deep Noise Suppression MOS) - Signal, background, and overall quality

### Speaker & Prosody Analysis
- **Speaker Similarity** - Using deep embeddings (WavLM, ECAPA-TDNN) or MFCC features
- **Prosody Analysis** - F0, rhythm, stress, and temporal characteristic preservation
- **Voice Characteristic Retention** - Comprehensive prosodic feature comparison

### ASR Impact Evaluation
- **Word Error Rate (WER)** - Using Whisper, Wav2Vec2, or other ASR models
- **Character Error Rate (CER)** - Fine-grained transcription accuracy
- **Transcription Quality** - Direct comparison of ASR outputs

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r code/audio_quality_assessment/requirements.txt

# Optional: Install SpeechBrain for advanced speaker embeddings
pip install speechbrain
```

### Basic Usage

#### Single Audio Pair Evaluation

```python
from code.audio_quality_assessment import AudioQualityAssessmentRunner

# Initialize runner
runner = AudioQualityAssessmentRunner()

# Evaluate audio pair
result = runner.evaluate_pair(
    reference_path="reference.wav",
    degraded_path="degraded.wav", 
    reference_text="Hello world"  # Optional transcript
)

print(f"NISQA MOS: {result['metrics']['nisqa']['mos']}")
print(f"ViSQOL Score: {result['metrics']['visqol']['moslqo']}")
print(f"Speaker Similarity: {result['metrics']['speaker_similarity']['cosine_similarity']}")
```

#### Command Line Interface

```bash
# Single pair evaluation
python -m code.audio_quality_assessment.runner \
    --reference reference.wav \
    --degraded degraded.wav \
    --transcript "Hello world" \
    --output results.json

# Batch evaluation
python -m code.audio_quality_assessment.runner \
    --pairs audio_pairs.json \
    --config config.yaml \
    --output batch_results.json
```

#### Batch Audio Pairs Format

```json
[
    {
        "reference": "path/to/reference1.wav",
        "degraded": "path/to/degraded1.wav",
        "transcript": "Optional ground truth text"
    },
    {
        "reference": "path/to/reference2.wav", 
        "degraded": "path/to/degraded2.wav",
        "transcript": "Another transcript"
    }
]
```

## 📊 Metrics Details

### NISQA (Non-Intrusive Speech Quality Assessment)
- **Range**: 1-5 (higher is better)
- **Type**: Non-intrusive (no reference needed)
- **Purpose**: Predicts Mean Opinion Score for speech quality
- **Implementation**: Official NISQA model or high-fidelity alternative

```python
from code.audio_quality_assessment.nisqa_scorer import NISQAScorer

scorer = NISQAScorer()
score = scorer.score("audio.wav", return_details=True)
# Returns: {'mos': 4.2, 'noi': 4.0, 'dis': 4.1, 'col': 4.3, 'loud': 4.0}
```

### ViSQOL (Virtual Speech Quality Objective Listener)
- **Range**: 1-5 (higher is better)
- **Type**: Intrusive (requires reference)
- **Purpose**: Perceptual quality assessment using psychoacoustic models
- **Implementation**: Patch-based spectral similarity with perceptual weighting

```python
from code.audio_quality_assessment.visqol_scorer import ViSQOLScorer

scorer = ViSQOLScorer()
score = scorer.score("reference.wav", "degraded.wav", return_details=True)
# Returns: {'moslqo': 3.8, 'similarity': 0.85, 'degradation': 0.12}
```

### DNSMOS (Deep Noise Suppression MOS)
- **Range**: 1-5 (higher is better)
- **Type**: Non-intrusive
- **Purpose**: Separate assessment of signal, background, and overall quality
- **Implementation**: Deep learning model with spectral and perceptual features

```python
from code.audio_quality_assessment.dnsmos_scorer import DNSMOSScorer

scorer = DNSMOSScorer()
scores = scorer.score("audio.wav")
# Returns: {'SIG': 3.9, 'BAK': 4.2, 'OVR': 4.0}
```

### Speaker Similarity Assessment
- **Range**: 0-1 (higher is better)
- **Type**: Intrusive
- **Purpose**: Measure speaker identity preservation
- **Models**: WavLM, ECAPA-TDNN, x-vectors, or MFCC features

```python
from code.audio_quality_assessment.speaker_similarity import SpeakerSimilarityScorer

scorer = SpeakerSimilarityScorer(model_type="wavlm")  # or "ecapa", "mfcc"
score = scorer.compute_similarity("ref.wav", "deg.wav", return_details=True)
# Returns: {'cosine_similarity': 0.94, 'euclidean_distance': 2.1, 'correlation': 0.91}
```

### Prosody Analysis
- **Metrics**: F0 RMSE, correlation, rhythm, tempo, stress similarity
- **Type**: Intrusive
- **Purpose**: Evaluate preservation of prosodic features
- **Implementation**: Comprehensive F0, rhythm, and stress analysis

```python
from code.audio_quality_assessment.prosody_analyzer import ProsodyAnalyzer

analyzer = ProsodyAnalyzer()
scores = analyzer.analyze("ref.wav", "deg.wav", return_details=True)
# Returns: {'f0_rmse': 1.2, 'f0_correlation': 0.89, 'rhythm_similarity': 0.85, ...}
```

### ASR Impact Evaluation  
- **Metrics**: WER, CER, transcription similarity
- **Type**: Uses ASR models (Whisper, Wav2Vec2)
- **Purpose**: Measure impact on speech recognition accuracy
- **Implementation**: State-of-the-art ASR models with error analysis

```python
from code.audio_quality_assessment.asr_evaluator import ASREvaluator

evaluator = ASREvaluator(model_name="openai/whisper-base")
result = evaluator.evaluate_wer("ref.wav", "deg.wav", "ground truth text")
# Returns: {'wer_increase': 0.02, 'transcription_similarity': 0.97, ...}
```

## ⚙️ Configuration

Use a YAML configuration file to customize evaluation:

```yaml
metrics:
  nisqa:
    enabled: true
  visqol:
    enabled: true
    sample_rate: 16000
  speaker_similarity:
    enabled: true
    model_type: "wavlm"  # or "ecapa", "mfcc"
  asr:
    enabled: true
    model_name: "openai/whisper-large-v3"
    language: "en"

output:
  format: "json"  # or "yaml", "csv"
  detailed: true

device: "cuda"  # or "cpu", "auto"
```

## 📈 Output Format

### Detailed JSON Output Example

```json
{
  "reference_path": "reference.wav",
  "degraded_path": "degraded.wav", 
  "reference_text": "Hello world",
  "metrics": {
    "nisqa": {
      "mos": 4.2,
      "noi": 4.0,
      "dis": 4.1,
      "col": 4.3,
      "loud": 4.0
    },
    "visqol": {
      "moslqo": 3.8,
      "similarity": 0.85,
      "degradation": 0.12
    },
    "dnsmos": {
      "SIG": 3.9,
      "BAK": 4.2, 
      "OVR": 4.0
    },
    "speaker_similarity": {
      "cosine_similarity": 0.94,
      "euclidean_distance": 2.1,
      "correlation": 0.91
    },
    "prosody": {
      "f0_rmse": 1.2,
      "f0_correlation": 0.89,
      "rhythm_similarity": 0.85,
      "tempo_similarity": 0.92,
      "stress_similarity": 0.87,
      "overall_prosody": 0.88
    },
    "asr": {
      "wer_reference": 0.05,
      "wer_degraded": 0.07,
      "wer_increase": 0.02,
      "transcription_similarity": 0.97,
      "relative_wer": 0.03
    }
  }
}
```

## 🔧 Advanced Usage

### Custom Model Integration

```python
# Use custom NISQA model
from code.audio_quality_assessment.nisqa_scorer import NISQAScorer
scorer = NISQAScorer(model_path="path/to/custom_nisqa.tar")

# Use custom ASR model
from code.audio_quality_assessment.asr_evaluator import ASREvaluator
evaluator = ASREvaluator(model_name="path/to/custom_whisper")
```

### Batch Processing with Progress Tracking

```python
from tqdm import tqdm

runner = AudioQualityAssessmentRunner("config.yaml")

results = []
for pair in tqdm(audio_pairs, desc="Evaluating"):
    result = runner.evaluate_pair(pair['reference'], pair['degraded'])
    results.append(result)
```

### Integration with Existing Pipeline

```python
# Integration example for codec evaluation pipeline
def evaluate_codec_quality(reference_dir, codec_output_dir, manifest):
    runner = AudioQualityAssessmentRunner()
    
    results = []
    for item in manifest['items']:
        audio_id = item['id']
        ref_path = f"{reference_dir}/{audio_id}.wav"
        deg_path = f"{codec_output_dir}/{audio_id}.wav"
        transcript = item.get('transcript')
        
        if Path(ref_path).exists() and Path(deg_path).exists():
            result = runner.evaluate_pair(ref_path, deg_path, transcript)
            result['audio_id'] = audio_id
            result['accent'] = item.get('accent')
            results.append(result)
    
    return results
```

## 📚 Scientific Background

### NISQA
Based on the ITU-T P.808 standard for non-intrusive speech quality assessment. Uses deep learning to predict perceptual quality without requiring a reference signal.

**Reference**: Mittag, G., & Möller, S. (2020). Non-intrusive speech quality assessment for super-wideband speech communication networks. ICASSP 2020.

### ViSQOL  
Implements the Virtual Speech Quality Objective Listener algorithm using psychoacoustic principles and spectral similarity analysis.

**Reference**: Hines, A., et al. (2015). ViSQOL: an objective speech quality model. EURASIP Journal on Audio, Speech, and Music Processing.

### DNSMOS
Deep learning approach for assessing noise suppression quality with separate evaluation of signal distortion and background noise.

**Reference**: Reddy, C. K. A., et al. (2021). DNSMOS: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors.

### Speaker Similarity
Uses state-of-the-art speaker embeddings from models like WavLM and ECAPA-TDNN for robust speaker identity preservation assessment.

### Prosody Analysis
Comprehensive analysis of prosodic features including F0 contours, rhythm patterns, and stress characteristics using established phonetic analysis methods.

### ASR Impact
Evaluates practical impact on downstream applications using production-quality ASR systems with detailed error analysis.

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```yaml
   device: "cpu"  # Use CPU instead
   ```

2. **Model Download Issues**
   ```bash
   export HF_DATASETS_OFFLINE=0
   export TRANSFORMERS_OFFLINE=0
   ```

3. **Audio Format Issues**
   - Ensure audio is in WAV format
   - Check sample rates (automatically handled)
   - Verify mono/stereo (automatically converted)

4. **Missing Dependencies**
   ```bash
   pip install --upgrade torch torchaudio transformers
   pip install speechbrain  # For advanced speaker models
   ```

### Performance Optimization

- Use GPU when available (`device: "cuda"`)
- Batch process multiple files
- Use lighter ASR models for speed (whisper-base vs whisper-large)
- Disable unused metrics in configuration

## 📞 Support

For issues, questions, or contributions, please refer to the main project documentation or create an issue in the project repository.

## 🏆 Citation

If you use this audio quality assessment module in your research, please cite:

```bibtex
@software{codeceval_africa_aqa,
  title={Audio Quality Assessment for Neural Codec Evaluation},
  author={CodecEval-Africa Project},
  year={2026},
  url={https://github.com/ufdatastudio/CodecEval-Africa}
}
```
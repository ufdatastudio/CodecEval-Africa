# Audio Codec Output Verification Guide

## Overview
This guide explains how to verify that audio codec outputs are correct in terms of bitrate, bandwidth, sample rate, and format.

## Key Concepts

### 1. Model Bitrate vs File Bitrate

**IMPORTANT:** There's a critical distinction between:
- **Model Bitrate** (compressed): The bitrate of the learned representation (e.g., 3kbps, 6kbps)
- **File Bitrate** (uncompressed): The bitrate of the saved WAV file (e.g., 384kbps, 706kbps)

#### Why the Difference?
Neural audio codecs work in two stages:
1. **Encoding**: Audio → Compressed tokens/codes (model bitrate)
2. **Decoding**: Compressed tokens → Reconstructed WAV (file bitrate)

The saved WAV files store the **reconstructed audio** in uncompressed format, which has a much higher bitrate than the compressed representation.

#### Example Calculation
For a 24kHz, 16-bit, mono WAV file:
```
File Bitrate = Sample Rate × Bit Depth × Channels
             = 24000 Hz × 16 bits × 1 channel
             = 384,000 bits/second
             = 384 kbps
```

But the model might use only **6 kbps** for the compressed representation!

### 2. Codec-Specific Standards

#### DAC (Descript Audio Codec)
- **Model Bitrates**: 8, 16, 24 kbps
- **Sample Rate**: 44,100 Hz
- **File Bitrate (uncompressed WAV)**: ~706 kbps (44.1kHz × 16-bit × 1ch)
- **Channels**: 1 (mono)
- **Format**: WAV
- **Verification**: Check sample rate is 44.1kHz, file bitrate should be ~706 kbps

#### Encodec (Meta)
- **Model Bitrates**: 3, 6, 12, 24 kbps
- **Sample Rate**: 24,000 Hz
- **File Bitrate (uncompressed WAV)**: ~384 kbps (24kHz × 16-bit × 1ch)
- **Channels**: 1 (mono)
- **Format**: WAV
- **Verification**: Check sample rate is 24kHz, file bitrate should be ~384 kbps

#### SemantiCodec
- **Model Bitrates**: 0.31, 0.33, 0.63, 0.68, 1.25, 1.40 kbps (very low!)
- **Sample Rate**: 16,000 Hz
- **File Bitrate (uncompressed WAV)**: ~256 kbps (16kHz × 16-bit × 1ch)
- **Channels**: 1 (mono)
- **Format**: WAV
- **Note**: Uses semantic tokens, extremely low bitrate
- **Verification**: Check sample rate is 16kHz, file bitrate should be ~256 kbps

#### UniCodec
- **Model Bitrate**: 6.6 kbps
- **Sample Rate**: 24,000 Hz
- **File Bitrate (uncompressed WAV)**: ~384 kbps
- **Channels**: 1 (mono)
- **Format**: WAV

#### LanguageCodec
- **Bandwidth Levels**: 0, 1, 2, 3 (not direct bitrate)
- **Sample Rate**: 16,000 Hz
- **File Bitrate (uncompressed WAV)**: ~256 kbps
- **Channels**: 1 (mono)
- **Format**: WAV
- **Note**: Uses bandwidth parameter instead of bitrate

#### WavTokenizer
- **Models**: 
  - small-320-24k-4096
  - small-600-24k-4096
  - medium-speech-75token
  - medium-music-audio-75token
  - large-speech-75token
  - large-unify-40token
- **Sample Rate**: 24,000 Hz
- **File Bitrate (uncompressed WAV)**: ~384 kbps
- **Channels**: 1 (mono)
- **Format**: WAV
- **Note**: Token-based compression with different models

## Verification Methods

### Method 1: Sample Rate Verification
✅ **Easy and Definitive**

Each codec has a specific sample rate. This is the most straightforward check:

```bash
# Using soundfile (Python)
import soundfile as sf
info = sf.info('output.wav')
print(f"Sample rate: {info.samplerate} Hz")
print(f"Channels: {info.channels}")
print(f"Duration: {info.duration} sec")
```

```bash
# Using ffprobe (command line)
ffprobe -v error -show_entries stream=sample_rate,channels -of default=noprint_wrappers=1 output.wav
```

### Method 2: File Bitrate Calculation
✅ **Good for verifying uncompressed WAV format**

Calculate the file's bitrate:
```python
import os
file_size_bytes = os.path.getsize('output.wav')
duration_sec = info.duration
file_bitrate_kbps = (file_size_bytes * 8) / (duration_sec * 1000)
```

**Expected file bitrates for different sample rates (16-bit mono):**
- 16 kHz → 256 kbps
- 24 kHz → 384 kbps
- 44.1 kHz → 706 kbps
- 48 kHz → 768 kbps

### Method 3: Model Bitrate Verification
⚠️ **Requires access to intermediate representations**

To verify the actual model bitrate, you need to check:
- Number of codebook tokens used
- Token emission rate (tokens per second)
- Bits per token

This requires access to the encoded representation before decoding to WAV.

### Method 4: Audio Quality Metrics
✅ **Best for verifying reconstruction quality**

Use perceptual metrics:
- **PESQ** (Perceptual Evaluation of Speech Quality)
- **VISQOL** (Virtual Speech Quality Objective Listener)
- **NISQA** (Non-Intrusive Speech Quality Assessment)
- **SI-SDR** (Scale-Invariant Signal-to-Distortion Ratio)

```python
# Example with NISQA
from nisqa.NISQA_lib import nisqa
results = nisqa.predict('path/to/audio.wav')
```

## Verification Checklist

### ✅ Basic Checks
- [ ] Files exist and are readable
- [ ] All files are in WAV format
- [ ] Sample rate matches codec specification
- [ ] Files are mono (1 channel)
- [ ] No corrupted or 0-byte files
- [ ] File bitrate matches expected uncompressed bitrate for sample rate

### ✅ Advanced Checks
- [ ] Audio duration matches original
- [ ] No clipping or extreme values
- [ ] Perceptual quality metrics within acceptable range
- [ ] Spectral analysis shows no unexpected artifacts
- [ ] Compare file sizes across different bitrate settings

### ✅ Dataset-Specific Checks
- [ ] All original files have corresponding outputs
- [ ] Outputs organized by bitrate/bandwidth
- [ ] Consistent naming convention
- [ ] Metadata preserved (if applicable)

## Common Issues and Solutions

### Issue 1: Mixed Sample Rates
**Problem**: Files in same codec output have different sample rates
**Solution**: Check codec processing pipeline, ensure consistent preprocessing

### Issue 2: File Bitrate Lower Than Expected
**Problem**: File bitrate is lower than calculated uncompressed rate
**Solution**: May indicate compression or lower bit depth; verify file format and encoding

### Issue 3: File Bitrate Much Higher Than Model Bitrate
**This is NORMAL!** WAV files store uncompressed audio. The model bitrate refers to the compressed representation.

### Issue 4: Missing Files
**Problem**: Some audio files didn't get processed
**Solution**: Check error logs, verify input file compatibility, check codec configuration

## Tools and Scripts

### Available Scripts
1. **analyze_audio_outputs.py**: Comprehensive analysis of all codec outputs
2. **verify_codec_outputs.py**: Verification against expected specifications
3. **compare_codecs.sh**: Side-by-side comparison of codec outputs

### Quick Commands

```bash
# Analyze all outputs
python scripts/analyze_audio_outputs.py --save-json

# Verify outputs
python scripts/verify_codec_outputs.py --save-json

# Count files per codec
find outputs -name "*.wav" | grep -E "(DAC|Encodec|SemantiCodec)" | sort | uniq -c

# Check sample rates
for f in outputs/DAC_outputs/*/*.wav; do 
  soxi -r "$f"; 
done | sort | uniq -c

# Get file size statistics
find outputs -name "*.wav" -exec ls -lh {} \; | awk '{sum+=$5; n++} END {print "Average:", sum/n/1024, "KB"}'
```

## Summary

To verify codec outputs are accurate:

1. **Check Sample Rate** - Must match codec spec (16kHz, 24kHz, or 44.1kHz)
2. **Check File Bitrate** - Should match uncompressed WAV bitrate for that sample rate
3. **Don't Confuse** - File bitrate (uncompressed WAV) ≠ Model bitrate (compressed)
4. **Use Quality Metrics** - NISQA, PESQ, VISQOL to verify reconstruction quality
5. **Verify Organization** - Files properly organized by bitrate/bandwidth settings

The **most important verification** is that the sample rate is correct and the audio quality metrics show good reconstruction. The file bitrate being much higher than the model bitrate is expected and normal!

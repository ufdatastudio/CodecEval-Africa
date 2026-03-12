#!/usr/bin/env python3
"""
Simple audio quality test - checks basic properties and tests lightweight metrics.
"""

import soundfile as sf
import numpy as np
from pathlib import Path
import sys

def analyze_audio_properties(audio_path):
    """Analyze basic audio properties."""
    print(f"\n{'='*70}")
    print(f"Audio File Analysis")
    print(f"{'='*70}")
    print(f"File: {Path(audio_path).name}")
    print(f"Path: {audio_path}")
    
    # Load audio
    audio, sr = sf.read(audio_path)
    
    # Basic properties
    duration = len(audio) / sr
    print(f"\nBasic Properties:")
    print(f"  Sample Rate: {sr} Hz")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Samples: {len(audio):,}")
    print(f"  Channels: {1 if audio.ndim == 1 else audio.ndim}")
    
    # Statistical properties
    print(f"\nStatistical Properties:")
    print(f"  RMS Level: {np.sqrt(np.mean(audio**2)):.4f}")
    print(f"  Peak Level: {np.max(np.abs(audio)):.4f}")
    print(f"  Mean: {np.mean(audio):.6f}")
    print(f"  Std Dev: {np.std(audio):.4f}")
    
    # Dynamic range
    print(f"\nDynamic Range:")
    print(f"  Min: {np.min(audio):.4f}")
    print(f"  Max: {np.max(audio):.4f}")
    
    # Check for clipping
    clipping_threshold = 0.99
    clipped_samples = np.sum(np.abs(audio) > clipping_threshold)
    if clipped_samples > 0:
        print(f"  ⚠️  Clipped samples: {clipped_samples} ({100*clipped_samples/len(audio):.2f}%)")
    else:
        print(f"  ✓ No clipping detected")
    
    # Check for silence
    silence_threshold = 0.001
    silent_samples = np.sum(np.abs(audio) < silence_threshold)
    print(f"  Silent samples: {silent_samples} ({100*silent_samples/len(audio):.2f}%)")
    
    # Energy distribution
    energy = audio ** 2
    print(f"\nEnergy Distribution:")
    print(f"  Total Energy: {np.sum(energy):.2f}")
    print(f"  Mean Energy: {np.mean(energy):.6f}")
    
    # SNR estimate (simple)
    signal_power = np.mean(energy)
    noise_estimate = np.std(audio[int(len(audio)*0.9):])  # Last 10% as noise estimate
    if noise_estimate > 0:
        snr_estimate = 10 * np.log10(signal_power / (noise_estimate**2))
        print(f"  Estimated SNR: {snr_estimate:.2f} dB")
    
    print(f"{'='*70}\n")
    
    return {
        'sample_rate': sr,
        'duration': duration,
        'rms': float(np.sqrt(np.mean(audio**2))),
        'peak': float(np.max(np.abs(audio))),
        'clipped_samples': int(clipped_samples),
        'silent_samples': int(silent_samples)
    }


def compare_codec_outputs(base_audio, codec_outputs):
    """Compare multiple codec outputs."""
    print(f"\n{'='*70}")
    print(f"Codec Output Comparison")
    print(f"{'='*70}\n")
    
    results = {}
    for codec_name, audio_path in codec_outputs.items():
        print(f"{codec_name}:")
        result = analyze_audio_properties(audio_path)
        results[codec_name] = result
    
    return results


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python quick_audio_test.py <audio_file>")
        print("   or: python quick_audio_test.py <audio1> <audio2> <audio3> ...")
        sys.exit(1)
    
    print(f"\n{'#'*70}")
    print(f"Quick Audio Quality Test")
    print(f"{'#'*70}\n")
    
    if len(sys.argv) == 2:
        # Single file analysis
        analyze_audio_properties(sys.argv[1])
    else:
        # Multiple files - compare them
        codec_outputs = {}
        for i, audio_path in enumerate(sys.argv[1:], 1):
            codec_name = Path(audio_path).parent.name
            codec_outputs[f"{codec_name}"] = audio_path
        
        compare_codec_outputs(None, codec_outputs)
    
    print("✅ Analysis complete!\n")

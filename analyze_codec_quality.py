#!/usr/bin/env python3
"""
Comprehensive Codec Quality Analysis
Analyze compression quality using multiple metrics beyond file size
"""

import os
import sys
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Add the code directory to path
sys.path.append('/orange/ufdatastudios/c.okocha/CodecEval-Africa/code')

def calculate_snr(original, compressed):
    """Calculate Signal-to-Noise Ratio"""
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - compressed) ** 2)
    if noise_power == 0:
        return float('inf')
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def calculate_spectral_centroid(audio, sr):
    """Calculate spectral centroid"""
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    return np.mean(spectral_centroids)

def calculate_zero_crossing_rate(audio):
    """Calculate zero crossing rate"""
    return np.mean(librosa.feature.zero_crossing_rate(audio)[0])

def calculate_spectral_rolloff(audio, sr, roll_percent=0.85):
    """Calculate spectral rolloff"""
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=roll_percent)[0]
    return np.mean(spectral_rolloff)

def calculate_mel_cepstral_distortion(original, compressed, sr):
    """Calculate Mel-Cepstral Distortion"""
    # Extract mel-frequency cepstral coefficients
    mfcc_orig = librosa.feature.mfcc(y=original, sr=sr, n_mfcc=13)
    mfcc_comp = librosa.feature.mfcc(y=compressed, sr=sr, n_mfcc=13)
    
    # Calculate MCD
    mcd = np.mean(np.sqrt(2 * np.sum((mfcc_orig - mfcc_comp) ** 2, axis=0)))
    return mcd

def calculate_dynamic_range(audio):
    """Calculate dynamic range in dB"""
    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))
    if rms == 0:
        return 0
    dynamic_range = 20 * np.log10(peak / rms)
    return dynamic_range

def analyze_audio_quality(original_file, compressed_file, bitrate):
    """Comprehensive audio quality analysis"""
    print(f"  Analyzing {bitrate} kbps...")
    
    # Load audio files
    orig_audio, orig_sr = librosa.load(original_file, sr=None)
    comp_audio, comp_sr = librosa.load(compressed_file, sr=None)
    
    # Ensure same length
    min_len = min(len(orig_audio), len(comp_audio))
    orig_audio = orig_audio[:min_len]
    comp_audio = comp_audio[:min_len]
    
    # Calculate metrics
    metrics = {}
    
    # SNR
    metrics['snr'] = calculate_snr(orig_audio, comp_audio)
    
    # Spectral Centroid
    metrics['spectral_centroid_orig'] = calculate_spectral_centroid(orig_audio, orig_sr)
    metrics['spectral_centroid_comp'] = calculate_spectral_centroid(comp_audio, comp_sr)
    
    # Zero Crossing Rate
    metrics['zcr_orig'] = calculate_zero_crossing_rate(orig_audio)
    metrics['zcr_comp'] = calculate_zero_crossing_rate(comp_audio)
    
    # Spectral Rolloff
    metrics['spectral_rolloff_orig'] = calculate_spectral_rolloff(orig_audio, orig_sr)
    metrics['spectral_rolloff_comp'] = calculate_spectral_rolloff(comp_audio, comp_sr)
    
    # Mel-Cepstral Distortion
    metrics['mcd'] = calculate_mel_cepstral_distortion(orig_audio, comp_audio, orig_sr)
    
    # Dynamic Range
    metrics['dynamic_range_orig'] = calculate_dynamic_range(orig_audio)
    metrics['dynamic_range_comp'] = calculate_dynamic_range(comp_audio)
    
    # Correlation
    metrics['correlation'] = pearsonr(orig_audio, comp_audio)[0]
    
    return metrics

def analyze_codec_quality(codec_name, original_file, output_dir):
    """Analyze quality for a specific codec across bitrates"""
    print(f"\n{'='*80}")
    print(f"QUALITY ANALYSIS: {codec_name.upper()}")
    print(f"{'='*80}")
    
    bitrates = [1.5, 6, 24]
    all_metrics = {}
    
    for bitrate in bitrates:
        compressed_file = os.path.join(output_dir, f"*_{codec_name}_{bitrate}kbps.wav")
        # Find the actual file
        import glob
        files = glob.glob(compressed_file)
        if not files:
            print(f"  ❌ No file found for {codec_name} at {bitrate} kbps")
            continue
            
        compressed_file = files[0]
        
        try:
            metrics = analyze_audio_quality(original_file, compressed_file, bitrate)
            all_metrics[bitrate] = metrics
            
            print(f"  📊 {bitrate} kbps Metrics:")
            print(f"    SNR: {metrics['snr']:.2f} dB")
            print(f"    MCD: {metrics['mcd']:.3f}")
            print(f"    Correlation: {metrics['correlation']:.3f}")
            print(f"    Dynamic Range: {metrics['dynamic_range_orig']:.1f} → {metrics['dynamic_range_comp']:.1f} dB")
            print(f"    Spectral Centroid: {metrics['spectral_centroid_orig']:.0f} → {metrics['spectral_centroid_comp']:.0f} Hz")
            
        except Exception as e:
            print(f"  ❌ Error analyzing {bitrate} kbps: {e}")
    
    return all_metrics

def main():
    """Main analysis function"""
    print("🎵 COMPREHENSIVE CODEC QUALITY ANALYSIS")
    print("=" * 80)
    print("Analyzing compression quality using multiple metrics")
    print("=" * 80)
    
    # Configuration
    original_file = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/data/afrispeech_dialog/data/0adaefab-c0fa-4d55-9564-100d2bd5bd93_86a60667f1b75930c7844e37494b97f7_UxiL1B07.wav"
    output_dir = "/orange/ufdatastudios/c.okocha/CodecEval-Africa/individual_codec_test"
    
    if not os.path.exists(original_file):
        print(f"❌ Original file not found: {original_file}")
        return
    
    # Codecs to analyze
    codecs = ["encodec_24khz", "languagecodec", "dac", "sematicodec", "unicodec", "apcodec"]
    
    all_results = {}
    
    for codec in codecs:
        try:
            metrics = analyze_codec_quality(codec, original_file, output_dir)
            all_results[codec] = metrics
        except Exception as e:
            print(f"❌ Error analyzing {codec}: {e}")
    
    # Generate summary report
    print(f"\n{'='*80}")
    print("📊 QUALITY ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    for codec, metrics in all_results.items():
        if not metrics:
            continue
            
        print(f"\n{codec.upper()}:")
        print("-" * 40)
        
        # Check if quality improves with bitrate
        bitrates = sorted(metrics.keys())
        if len(bitrates) >= 2:
            snr_values = [metrics[b]['snr'] for b in bitrates if 'snr' in metrics[b]]
            mcd_values = [metrics[b]['mcd'] for b in bitrates if 'mcd' in metrics[b]]
            
            if len(snr_values) >= 2:
                snr_trend = "📈 Improving" if snr_values[-1] > snr_values[0] else "📉 Degrading"
                print(f"  SNR Trend: {snr_trend} ({snr_values[0]:.1f} → {snr_values[-1]:.1f} dB)")
            
            if len(mcd_values) >= 2:
                mcd_trend = "📉 Improving" if mcd_values[-1] < mcd_values[0] else "📈 Degrading"
                print(f"  MCD Trend: {mcd_trend} ({mcd_values[0]:.3f} → {mcd_values[-1]:.3f})")
    
    print(f"\n🎯 Analysis completed! Check quality trends above.")
    print("📈 Improving trends indicate proper bitrate scaling")
    print("📉 Degrading trends may indicate compression issues")

if __name__ == "__main__":
    main()

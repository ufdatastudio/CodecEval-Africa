#!/usr/bin/env python3
"""
Verify actual codec bitrate from discrete codes
This is the most accurate method for neural codecs
"""
import os
import sys
import torch
import wave
import numpy as np

def calculate_from_codes(codes, frame_rate, codebook_size, num_codebooks):
    """
    Calculate bitrate from discrete codes.
    
    Args:
        codes: Discrete codes tensor [B, K, T] where K=codebooks, T=frames
        frame_rate: Frame rate in Hz
        codebook_size: Size of codebook (vocab size)
        num_codebooks: Number of codebooks used
    
    Returns:
        bitrate_kbps: Actual bitrate in kbps
    """
    if isinstance(codes, torch.Tensor):
        codes = codes.cpu().numpy()
    
    # Get dimensions
    if codes.ndim == 3:
        B, K, T = codes.shape
        num_codebooks = K
    else:
        raise ValueError(f"Expected 3D codes tensor [B, K, T], got shape {codes.shape}")
    
    # Calculate bits per frame per codebook
    bits_per_code = np.log2(codebook_size)
    bits_per_frame = bits_per_code * num_codebooks
    
    # Calculate bitrate
    bitrate_bps = frame_rate * bits_per_frame
    bitrate_kbps = bitrate_bps / 1000.0
    
    return bitrate_kbps, {
        'num_codebooks': num_codebooks,
        'codebook_size': codebook_size,
        'frame_rate': frame_rate,
        'bits_per_code': bits_per_code,
        'bits_per_frame': bits_per_frame,
        'bitrate_kbps': bitrate_kbps
    }

def verify_wav_bitrate(file_path):
    """Calculate bitrate from decoded WAV file (PCM bitrate, not codec bitrate)."""
    file_size = os.path.getsize(file_path)
    
    with wave.open(file_path, 'rb') as wav:
        sample_rate = wav.getframerate()
        n_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        n_frames = wav.getnframes()
        duration = n_frames / sample_rate
    
    # Effective bitrate from file size
    effective_bitrate_kbps = (file_size * 8) / duration / 1000.0
    
    # PCM bitrate
    pcm_bitrate_kbps = sample_rate * n_channels * sample_width * 8 / 1000.0
    
    return {
        'file_size_bytes': file_size,
        'duration': duration,
        'sample_rate': sample_rate,
        'effective_bitrate_kbps': effective_bitrate_kbps,
        'pcm_bitrate_kbps': pcm_bitrate_kbps
    }

def compare_original_vs_compressed(original_file, compressed_file):
    """Compare original and compressed file sizes to estimate compression ratio."""
    orig_size = os.path.getsize(original_file)
    comp_size = os.path.getsize(compressed_file)
    
    # Get durations
    with wave.open(original_file, 'rb') as wav:
        orig_duration = wav.getnframes() / wav.getframerate()
    with wave.open(compressed_file, 'rb') as wav:
        comp_duration = wav.getnframes() / wav.getframerate()
    
    # Note: compressed_file is decoded WAV, so this comparison is not accurate
    # for codec bitrate. It shows file size but not actual codec bitrate.
    
    compression_ratio = orig_size / comp_size if comp_size > 0 else 0
    
    orig_bitrate = (orig_size * 8) / orig_duration / 1000.0
    comp_bitrate = (comp_size * 8) / comp_duration / 1000.0
    
    return {
        'original_size_bytes': orig_size,
        'compressed_size_bytes': comp_size,
        'compression_ratio': compression_ratio,
        'original_bitrate_kbps': orig_bitrate,
        'compressed_bitrate_kbps': comp_bitrate,
        'note': 'Compressed file is decoded WAV (PCM), not actual compressed format'
    }

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python verify_codec_bitrate.py <wav_file>")
        print("\nThis script verifies the PCM bitrate of decoded WAV files.")
        print("For actual codec bitrate, you need access to discrete codes.")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    info = verify_wav_bitrate(file_path)
    
    print(f"File: {os.path.basename(file_path)}")
    print(f"  File size: {info['file_size_bytes']:,} bytes ({info['file_size_bytes']/1024:.2f} KB)")
    print(f"  Duration: {info['duration']:.2f} seconds")
    print(f"  Sample rate: {info['sample_rate']} Hz")
    print(f"  Effective bitrate: {info['effective_bitrate_kbps']:.2f} kbps")
    print(f"  PCM bitrate: {info['pcm_bitrate_kbps']:.2f} kbps")
    print(f"\n⚠️  NOTE: This is decoded WAV (uncompressed PCM format).")
    print(f"   The actual codec bitrate would be much lower.")
    print(f"   To get actual codec bitrate, calculate from discrete codes.")


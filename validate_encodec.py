#!/usr/bin/env python3
"""
Comprehensive validation script to verify EnCodec is working properly.
"""

import sys
import os
sys.path.append('.')

import numpy as np
import soundfile as sf
import hashlib
from code.codecs.encodec_runner import EncodecRunner
from code.codecs.soundstream_runner import SoundStreamRunner

def get_file_hash(filepath):
    """Get SHA1 hash of file."""
    if not os.path.exists(filepath):
        return "FILE_NOT_FOUND"
    
    with open(filepath, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()[:8]

def get_audio_stats(filepath):
    """Get comprehensive audio statistics."""
    if not os.path.exists(filepath):
        return {"exists": False}
    
    try:
        wav, sr = sf.read(filepath)
        rms = np.sqrt(np.mean(wav**2))
        duration = len(wav) / sr
        max_val = np.max(np.abs(wav))
        min_val = np.min(wav)
        mean_val = np.mean(wav)
        
        return {
            "exists": True,
            "duration": duration,
            "rms": rms,
            "max": max_val,
            "min": min_val,
            "mean": mean_val,
            "shape": wav.shape,
            "sample_rate": sr,
            "file_size": os.path.getsize(filepath)
        }
    except Exception as e:
        return {"exists": True, "error": str(e)}

def validate_encodec_compression():
    """Validate that EnCodec is actually compressing audio."""
    print("=== ENCODEC VALIDATION ===")
    
    # Test file
    test_file = "data/afrispeech_dialog/data/247554f8-f233-4861-bc1a-8fc327b5d5df_2b500b633e5d5ecce35433cbbb859ddc_8bW4oSXn.wav"
    output_dir = "encodec_validation_output"
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get original file stats
    print("\n=== ORIGINAL FILE ===")
    original_stats = get_audio_stats(test_file)
    original_hash = get_file_hash(test_file)
    
    print(f"File: {test_file}")
    print(f"Size: {original_stats['file_size']:,} bytes")
    print(f"Duration: {original_stats['duration']:.2f}s")
    print(f"Sample Rate: {original_stats['sample_rate']} Hz")
    print(f"Shape: {original_stats['shape']}")
    print(f"RMS: {original_stats['rms']:.6f}")
    print(f"Hash: {original_hash}")
    
    # Test different bitrates
    bitrates = [3, 6, 12, 24]
    
    print(f"\n=== ENCODEC COMPRESSION TEST ===")
    results = []
    
    for bitrate in bitrates:
        print(f"\n--- Testing {bitrate} kbps ---")
        
        try:
            # EnCodec
            runner = EncodecRunner(bandwidth_kbps=bitrate, causal=True, sr=16000)
            encodec_output = f"{output_dir}/encodec_{bitrate}kbps.wav"
            runner.run(test_file, encodec_output)
            
            # Get stats
            encodec_stats = get_audio_stats(encodec_output)
            encodec_hash = get_file_hash(encodec_output)
            
            # Calculate compression ratio
            compression_ratio = original_stats['file_size'] / encodec_stats['file_size']
            
            print(f"✅ EnCodec {bitrate}kbps:")
            print(f"   Output: {encodec_output}")
            print(f"   Size: {encodec_stats['file_size']:,} bytes")
            print(f"   Compression: {compression_ratio:.2f}x")
            print(f"   Duration: {encodec_stats['duration']:.2f}s")
            print(f"   RMS: {encodec_stats['rms']:.6f}")
            print(f"   Hash: {encodec_hash}")
            
            results.append({
                "bitrate": bitrate,
                "codec": "encodec",
                "success": True,
                "compression_ratio": compression_ratio,
                "hash": encodec_hash,
                "size": encodec_stats['file_size']
            })
            
        except Exception as e:
            print(f"❌ EnCodec {bitrate}kbps: FAILED - {e}")
            results.append({
                "bitrate": bitrate,
                "codec": "encodec",
                "success": False,
                "error": str(e)
            })
    
    # Test SoundStream for comparison
    print(f"\n=== SOUNDSTREAM COMPARISON ===")
    try:
        runner = SoundStreamRunner(bitrate_kbps=6, sr=16000)
        soundstream_output = f"{output_dir}/soundstream_6kbps.wav"
        runner.run(test_file, soundstream_output)
        
        soundstream_stats = get_audio_stats(soundstream_output)
        soundstream_hash = get_file_hash(soundstream_output)
        soundstream_compression = original_stats['file_size'] / soundstream_stats['file_size']
        
        print(f"✅ SoundStream 6kbps:")
        print(f"   Output: {soundstream_output}")
        print(f"   Size: {soundstream_stats['file_size']:,} bytes")
        print(f"   Compression: {soundstream_compression:.2f}x")
        print(f"   Hash: {soundstream_hash}")
        
    except Exception as e:
        print(f"❌ SoundStream: FAILED - {e}")
    
    # Summary
    print(f"\n=== VALIDATION SUMMARY ===")
    successful_results = [r for r in results if r["success"]]
    print(f"Successful compressions: {len(successful_results)}/{len(results)}")
    
    if len(successful_results) > 0:
        print(f"\nCompression Analysis:")
        for result in successful_results:
            print(f"  {result['bitrate']}kbps: {result['compression_ratio']:.2f}x compression")
        
        # Check if different bitrates produce different outputs
        hashes = [r["hash"] for r in successful_results]
        unique_hashes = set(hashes)
        
        if len(unique_hashes) > 1:
            print(f"✅ GOOD: Different bitrates produce different outputs ({len(unique_hashes)} unique)")
        else:
            print(f"❌ PROBLEM: All bitrates produce identical outputs")
        
        # Check if outputs are different from original
        if original_hash not in hashes:
            print(f"✅ GOOD: All outputs differ from original (compression occurred)")
        else:
            print(f"❌ PROBLEM: Some outputs identical to original (no compression)")
    
    return results

if __name__ == "__main__":
    validate_encodec_compression()

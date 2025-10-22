#!/usr/bin/env python3
"""
Test ASR on afrispeech samples with proper ground truth transcripts.
"""

import os
import sys
import csv
import soundfile as sf
import jiwer
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from code.asr.wer_eval import ASRModel

def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate (CER)."""
    if not reference.strip():
        return 1.0 if hypothesis.strip() else 0.0
    
    ref_chars = reference.replace(" ", "")
    hyp_chars = hypothesis.replace(" ", "")
    
    if not ref_chars:
        return 1.0 if hyp_chars else 0.0
    
    cer = jiwer.cer(ref_chars, hyp_chars)
    return cer

def load_afrispeech_manifest(manifest_path: Path) -> list:
    """Load afrispeech manifest with audio and transcript paths."""
    manifest_data = []
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            manifest_data.append({
                'filename': row['filename'],
                'accent': row['accent'],
                'audio_path': row['audio_path'],
                'transcript_path': row['transcript_path'],
                'transcript': row['transcript']
            })
    
    return manifest_data

def main():
    """Test ASR on afrispeech samples."""
    
    print("=" * 80)
    print("AFRISPEECH ASR TEST")
    print("=" * 80)
    
    # Manifest file
    manifest_file = "data/afrispeech_samples/samples_manifest.csv"
    
    if not Path(manifest_file).exists():
        print(f"✗ Manifest file not found: {manifest_file}")
        print("Run download script first: sbatch run_download_afrispeech.sh")
        return
    
    # Load manifest
    print(f"Loading manifest: {manifest_file}")
    manifest_data = load_afrispeech_manifest(Path(manifest_file))
    print(f"✓ Loaded {len(manifest_data)} samples")
    
    # Show accent distribution
    accent_counts = {}
    for item in manifest_data:
        accent = item['accent']
        accent_counts[accent] = accent_counts.get(accent, 0) + 1
    
    print(f"\nAccent distribution:")
    for accent, count in sorted(accent_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {accent}: {count} samples")
    
    # Initialize ASR model
    print(f"\nLoading Whisper Large-v3 ASR model...")
    try:
        asr_model = ASRModel("openai/whisper-large-v3")
        print("✓ ASR model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading ASR model: {e}")
        return
    
    # Test on all samples
    results = []
    
    for i, item in enumerate(manifest_data):
        print(f"\n--- Sample {i+1}/{len(manifest_data)}: {item['accent']} ---")
        print(f"File: {item['filename']}")
        print(f"Reference: '{item['transcript']}'")
        
        # Check if audio file exists
        if not Path(item['audio_path']).exists():
            print(f"✗ Audio file not found: {item['audio_path']}")
            continue
        
        # Get audio info
        try:
            wav, sr = sf.read(item['audio_path'])
            duration = len(wav) / sr
            print(f"Audio: {duration:.2f}s, {sr}Hz")
        except Exception as e:
            print(f"✗ Error reading audio: {e}")
            continue
        
        # Transcribe
        print("Transcribing...")
        try:
            hypothesis = asr_model.transcribe(item['audio_path'])
            print(f"ASR Output: '{hypothesis}'")
        except Exception as e:
            print(f"✗ Error transcribing: {e}")
            continue
        
        # Compute metrics
        if hypothesis.strip():
            wer = jiwer.wer(item['transcript'], hypothesis)
            cer = compute_cer(item['transcript'], hypothesis)
            
            print(f"WER: {wer:.3f} ({wer*100:.1f}%)")
            print(f"CER: {cer:.3f} ({cer*100:.1f}%)")
            
            results.append({
                'filename': item['filename'],
                'accent': item['accent'],
                'reference': item['transcript'],
                'hypothesis': hypothesis,
                'wer': wer,
                'cer': cer,
                'duration': duration
            })
        else:
            print("✗ No transcription output")
    
    # Summary
    if results:
        print("\n" + "=" * 80)
        print("ASR PERFORMANCE SUMMARY")
        print("=" * 80)
        
        # Overall statistics
        avg_wer = sum(r['wer'] for r in results) / len(results)
        avg_cer = sum(r['cer'] for r in results) / len(results)
        
        print(f"Overall Performance:")
        print(f"  Average WER: {avg_wer:.3f} ({avg_wer*100:.1f}%)")
        print(f"  Average CER: {avg_cer:.3f} ({avg_cer*100:.1f}%)")
        print(f"  Total samples: {len(results)}")
        
        # By accent
        print(f"\nBy Accent:")
        accent_stats = {}
        for r in results:
            accent = r['accent']
            if accent not in accent_stats:
                accent_stats[accent] = []
            accent_stats[accent].append(r['wer'])
        
        for accent, wers in accent_stats.items():
            avg_accent_wer = sum(wers) / len(wers)
            print(f"  {accent}: {avg_accent_wer:.3f} WER ({len(wers)} samples)")
        
        # Best and worst
        if results:
            best_result = min(results, key=lambda x: x['wer'])
            worst_result = max(results, key=lambda x: x['wer'])
            
            print(f"\nBest Performance:")
            print(f"  {best_result['accent']}: {best_result['wer']:.3f} WER")
            print(f"  Reference: '{best_result['reference'][:60]}...'")
            print(f"  ASR Output: '{best_result['hypothesis'][:60]}...'")
            
            print(f"\nWorst Performance:")
            print(f"  {worst_result['accent']}: {worst_result['wer']:.3f} WER")
            print(f"  Reference: '{worst_result['reference'][:60]}...'")
            print(f"  ASR Output: '{worst_result['hypothesis'][:60]}...'")
    
    print(f"\nAfriSpeech ASR test complete!")

if __name__ == "__main__":
    main()


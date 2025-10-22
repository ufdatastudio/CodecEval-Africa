#!/usr/bin/env python3
"""
Create balanced manifests for AfriSpeech datasets.
Focus on AfriSpeech-Dialog and AfriSpeech-200 with balanced accent distribution.
"""

import os
import yaml
import csv
from pathlib import Path
from typing import List, Dict, Any
import random

def create_afrispeech_dialog_manifest():
    """Create balanced manifest for AfriSpeech-Dialog dataset."""
    
    metadata_file = "data/afrispeech_dialog/metadata.csv"
    manifest_file = "data/manifests/afrispeech_dialog_balanced.yaml"
    
    print("Creating AfriSpeech-Dialog balanced manifest...")
    
    # Read metadata
    samples = []
    with open(metadata_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({
                'file_name': row['file_name'],
                'transcript': row['transcript'],
                'accent': row['accent'],
                'country': row['country'],
                'domain': row['domain'],
                'duration': float(row.get('duration', 0.0))
            })
    
    print(f"Found {len(samples)} samples in AfriSpeech-Dialog")
    
    # Balance by accent (max 5 samples per accent to keep it manageable)
    balanced_samples = []
    accent_counts = {}
    
    for sample in samples:
        accent = sample['accent']
        if accent_counts.get(accent, 0) < 5:  # Limit to 5 per accent
            balanced_samples.append(sample)
            accent_counts[accent] = accent_counts.get(accent, 0) + 1
    
    print(f"Balanced to {len(balanced_samples)} samples")
    print("Accent distribution:")
    for accent, count in accent_counts.items():
        print(f"  {accent}: {count} samples")
    
    # Create manifest items
    items = []
    for i, sample in enumerate(balanced_samples):
        # Convert file path to local path
        audio_path = f"data/afrispeech_dialog/{sample['file_name']}"
        
        items.append({
            'id': f"afrispeech_dialog_{i:03d}",
            'path': audio_path,
            'transcript': sample['transcript'],
            'accent': sample['accent'],
            'country': sample['country'],
            'domain': sample['domain'],
            'duration': sample['duration']
        })
    
    # Create manifest
    manifest = {
        'dataset': 'afrispeech_dialog',
        'sampling_rate': 16000,  # Standard for AfriSpeech
        'description': 'AfriSpeech-Dialog balanced by accent (max 5 per accent)',
        'total_samples': len(items),
        'accents': list(accent_counts.keys()),
        'items': items
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(manifest_file), exist_ok=True)
    
    # Write manifest
    with open(manifest_file, 'w') as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Created manifest: {manifest_file}")
    return manifest_file

def create_afrispeech_200_manifest():
    """Create balanced manifest for AfriSpeech-200 dataset (test samples only)."""
    
    # Check if AfriSpeech-200 data exists
    afrispeech_200_dir = Path("data/afrispeech_full")
    if not afrispeech_200_dir.exists():
        print("⚠ AfriSpeech-200 data not found. Run afrispeech.py first.")
        return None
    
    manifest_file = "data/manifests/afrispeech_200_balanced.yaml"
    
    print("Creating AfriSpeech-200 balanced manifest...")
    
    # Target accents from the original script
    target_accents = [
        "hausa", "igbo", "yoruba", "swahili", "english", 
        "chichewa", "fulani", "dholuo"
    ]
    
    items = []
    accent_counts = {}
    
    # Find test audio files for each accent
    for accent in target_accents:
        accent_dir = afrispeech_200_dir / accent
        if not accent_dir.exists():
            print(f"⚠ Accent directory not found: {accent}")
            continue
            
        # Look for test audio files
        test_audio_dir = accent_dir / "audio" / "test"
        if not test_audio_dir.exists():
            print(f"⚠ Test audio directory not found: {test_audio_dir}")
            continue
            
        audio_files = list(test_audio_dir.glob("*.wav")) + list(test_audio_dir.glob("*.mp3"))
        
        # Limit to 10 samples per accent for balanced evaluation
        selected_files = random.sample(audio_files, min(10, len(audio_files)))
        
        for i, audio_file in enumerate(selected_files):
            # Find corresponding transcript
            transcript_content = ""
            test_transcript_dir = accent_dir / "transcript" / "test"
            
            # Try different transcript file extensions
            for ext in ['.csv', '.txt']:
                transcript_file = test_transcript_dir / f"{audio_file.stem}{ext}"
                if transcript_file.exists():
                    try:
                        if ext == '.csv':
                            # Read CSV transcript
                            with open(transcript_file, 'r', encoding='utf-8') as f:
                                reader = csv.DictReader(f)
                                for row in reader:
                                    if row.get('filename', '').replace('.wav', '') == audio_file.stem:
                                        transcript_content = row.get('transcript', '').strip()
                                        break
                        else:
                            # Read text transcript
                            with open(transcript_file, 'r', encoding='utf-8') as f:
                                transcript_content = f.read().strip()
                        break
                    except Exception as e:
                        print(f"⚠ Error reading transcript {transcript_file}: {e}")
                        continue
            
            if transcript_content:
                items.append({
                    'id': f"afrispeech_200_{accent}_{i:03d}",
                    'path': str(audio_file),
                    'transcript': transcript_content,
                    'accent': accent,
                    'country': '',  # Not specified in AfriSpeech-200
                    'domain': 'test',
                    'duration': 0.0  # Will be measured during processing
                })
                accent_counts[accent] = accent_counts.get(accent, 0) + 1
    
    print(f"Found {len(items)} samples in AfriSpeech-200")
    print("Accent distribution:")
    for accent, count in accent_counts.items():
        print(f"  {accent}: {count} samples")
    
    # Create manifest
    manifest = {
        'dataset': 'afrispeech_200',
        'sampling_rate': 16000,
        'description': 'AfriSpeech-200 test samples balanced by accent (max 10 per accent)',
        'total_samples': len(items),
        'accents': list(accent_counts.keys()),
        'items': items
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(manifest_file), exist_ok=True)
    
    # Write manifest
    with open(manifest_file, 'w') as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Created manifest: {manifest_file}")
    return manifest_file

def create_tiny_manifests():
    """Create tiny manifests for quick testing."""
    
    print("Creating tiny manifests for quick testing...")
    
    # Tiny AfriSpeech-Dialog manifest (5 samples)
    dialog_manifest = "data/manifests/afrispeech_dialog_tiny.yaml"
    
    # Read a few samples from the full manifest
    full_manifest_file = "data/manifests/afrispeech_dialog_balanced.yaml"
    if os.path.exists(full_manifest_file):
        with open(full_manifest_file, 'r') as f:
            full_manifest = yaml.safe_load(f)
        
        # Take first 5 samples
        tiny_items = full_manifest['items'][:5]
        
        tiny_manifest = {
            'dataset': 'afrispeech_dialog_tiny',
            'sampling_rate': 16000,
            'description': 'Tiny AfriSpeech-Dialog for quick testing (5 samples)',
            'total_samples': len(tiny_items),
            'items': tiny_items
        }
        
        os.makedirs(os.path.dirname(dialog_manifest), exist_ok=True)
        with open(dialog_manifest, 'w') as f:
            yaml.dump(tiny_manifest, f, default_flow_style=False, sort_keys=False)
        
        print(f"✓ Created tiny manifest: {dialog_manifest}")
    
    # Tiny AfriSpeech-200 manifest (5 samples)
    afrispeech_200_manifest = "data/manifests/afrispeech_200_tiny.yaml"
    
    full_200_manifest_file = "data/manifests/afrispeech_200_balanced.yaml"
    if os.path.exists(full_200_manifest_file):
        with open(full_200_manifest_file, 'r') as f:
            full_manifest = yaml.safe_load(f)
        
        # Take first 5 samples
        tiny_items = full_manifest['items'][:5]
        
        tiny_manifest = {
            'dataset': 'afrispeech_200_tiny',
            'sampling_rate': 16000,
            'description': 'Tiny AfriSpeech-200 for quick testing (5 samples)',
            'total_samples': len(tiny_items),
            'items': tiny_items
        }
        
        os.makedirs(os.path.dirname(afrispeech_200_manifest), exist_ok=True)
        with open(afrispeech_200_manifest, 'w') as f:
            yaml.dump(tiny_manifest, f, default_flow_style=False, sort_keys=False)
        
        print(f"✓ Created tiny manifest: {afrispeech_200_manifest}")

def main():
    """Create all AfriSpeech manifests."""
    
    print("=" * 60)
    print("AFRISPEECH MANIFEST CREATION")
    print("=" * 60)
    
    # Set random seed for reproducibility
    random.seed(2025)
    
    # Create balanced manifests
    create_afrispeech_dialog_manifest()
    print()
    
    create_afrispeech_200_manifest()
    print()
    
    # Create tiny manifests for testing
    create_tiny_manifests()
    
    print("=" * 60)
    print("MANIFEST CREATION COMPLETED")
    print("=" * 60)
    
    print("\nCreated manifests:")
    print("- data/manifests/afrispeech_dialog_balanced.yaml")
    print("- data/manifests/afrispeech_200_balanced.yaml")
    print("- data/manifests/afrispeech_dialog_tiny.yaml")
    print("- data/manifests/afrispeech_200_tiny.yaml")
    
    print("\nReady for AfriSpeech codec evaluation!")

if __name__ == "__main__":
    main()

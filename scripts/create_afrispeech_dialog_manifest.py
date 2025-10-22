#!/usr/bin/env python3
"""
Create balanced manifest for AfriSpeech-Dialog dataset.
Focus on balanced accent distribution for comprehensive codec evaluation.
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
    
    # Balance by accent (max 5 samples per accent to keep it manageable for initial evaluation)
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

def create_tiny_manifest():
    """Create tiny manifest for quick testing (3 samples)."""
    
    print("Creating tiny manifest for quick testing...")
    
    # Read the balanced manifest
    balanced_manifest_file = "data/manifests/afrispeech_dialog_balanced.yaml"
    if not os.path.exists(balanced_manifest_file):
        print("⚠ Balanced manifest not found. Run create_afrispeech_dialog_manifest() first.")
        return None
    
    with open(balanced_manifest_file, 'r') as f:
        balanced_manifest = yaml.safe_load(f)
    
    # Take first 3 samples for quick testing
    tiny_items = balanced_manifest['items'][:3]
    
    tiny_manifest = {
        'dataset': 'afrispeech_dialog_tiny',
        'sampling_rate': 16000,
        'description': 'Tiny AfriSpeech-Dialog for quick testing (3 samples)',
        'total_samples': len(tiny_items),
        'items': tiny_items
    }
    
    tiny_manifest_file = "data/manifests/afrispeech_dialog_tiny.yaml"
    os.makedirs(os.path.dirname(tiny_manifest_file), exist_ok=True)
    
    with open(tiny_manifest_file, 'w') as f:
        yaml.dump(tiny_manifest, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Created tiny manifest: {tiny_manifest_file}")
    return tiny_manifest_file

def main():
    """Create AfriSpeech-Dialog manifests."""
    
    print("=" * 60)
    print("AFRISPEECH-DIALOG MANIFEST CREATION")
    print("=" * 60)
    
    # Set random seed for reproducibility
    random.seed(2025)
    
    # Create balanced manifest
    balanced_manifest = create_afrispeech_dialog_manifest()
    print()
    
    # Create tiny manifest for testing
    tiny_manifest = create_tiny_manifest()
    
    print("=" * 60)
    print("MANIFEST CREATION COMPLETED")
    print("=" * 60)
    
    print("\nCreated manifests:")
    print("- data/manifests/afrispeech_dialog_balanced.yaml")
    print("- data/manifests/afrispeech_dialog_tiny.yaml")
    
    print(f"\nReady for AfriSpeech-Dialog codec evaluation!")
    print(f"Total experiments: 6 codecs × 5 bitrates × {len(yaml.safe_load(open(balanced_manifest))['items'])} samples = {6 * 5 * len(yaml.safe_load(open(balanced_manifest))['items'])} evaluations")

if __name__ == "__main__":
    main()

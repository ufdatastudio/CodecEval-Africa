#!/usr/bin/env python3
"""
Create a single audio manifest for testing the full pipeline.
"""

import yaml
import os
from pathlib import Path

def create_single_audio_manifest():
    """Create a manifest with just one audio file for testing."""
    
    # Load the balanced manifest to get one sample
    with open('data/manifests/afrispeech_dialog_balanced.yaml', 'r') as f:
        balanced_manifest = yaml.safe_load(f)
    
    # Take the first audio file
    single_item = balanced_manifest['items'][0]
    
    # Create single audio manifest
    single_manifest = {
        'dataset': 'afrispeech_dialog_single',
        'sampling_rate': 16000,
        'description': 'Single audio test for full pipeline validation',
        'total_samples': 1,
        'accents': [single_item.get('accent', 'Unknown')],
        'items': [single_item]
    }
    
    # Save to isolated directory
    test_dir = Path('data/manifests/test_single')
    test_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_path = test_dir / 'single_audio_test.yaml'
    with open(manifest_path, 'w') as f:
        yaml.dump(single_manifest, f, default_flow_style=False)
    
    print(f"✓ Created single audio manifest: {manifest_path}")
    print(f"  Audio: {single_item['id']}")
    print(f"  Path: {single_item['path']}")
    print(f"  Transcript: '{single_item['transcript'][:100]}...'")
    
    return str(manifest_path)

if __name__ == "__main__":
    manifest_path = create_single_audio_manifest()
    print(f"\nManifest created at: {manifest_path}")


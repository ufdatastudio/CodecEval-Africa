#!/usr/bin/env python3
"""
Test the fixed ASR implementation with ground truth transcripts.
"""

import sys
import os
sys.path.append('/orange/ufdatastudios/c.okocha/CodecEval-Africa')

from code.pipeline_fixed import _get_ground_truth_transcript, _compute_all_metrics
from code.asr.wer_eval import ASRModel
import yaml

def test_ground_truth_extraction():
    """Test extracting ground truth transcripts from manifest."""
    print("=== TESTING GROUND TRUTH EXTRACTION ===")
    
    # Load config
    with open('configs/afrispeech_dialog_only.yml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Load manifest
    manifest_path = cfg['datasets'][0]
    with open(manifest_path, 'r') as f:
        manifest = yaml.safe_load(f)
    
    # Test with first item
    test_item = {
        'id': manifest['items'][0]['id'],
        'dataset': 'afrispeech_dialog'
    }
    
    print(f"Testing item: {test_item['id']}")
    
    # Get ground truth transcript
    ground_truth = _get_ground_truth_transcript(test_item, cfg)
    
    if ground_truth:
        print(f"✓ Ground truth found: '{ground_truth[:100]}...'")
        return True
    else:
        print("✗ No ground truth transcript found")
        return False

def test_asr_with_ground_truth():
    """Test ASR WER calculation with ground truth."""
    print("\n=== TESTING ASR WITH GROUND TRUTH ===")
    
    # Load config
    with open('configs/afrispeech_dialog_only.yml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Load manifest
    manifest_path = cfg['datasets'][0]
    with open(manifest_path, 'r') as f:
        manifest = yaml.safe_load(f)
    
    # Test with first item
    test_item = {
        'id': manifest['items'][0]['id'],
        'dataset': 'afrispeech_dialog'
    }
    
    original_file = manifest['items'][0]['path']
    decoded_file = original_file  # Use same file for testing
    
    print(f"Testing ASR for: {test_item['id']}")
    print(f"Original: {original_file}")
    print(f"Decoded: {decoded_file}")
    
    # Initialize ASR model
    asr_model = ASRModel()
    
    # Test metrics computation
    try:
        metrics = _compute_all_metrics(original_file, decoded_file, asr_model, test_item, cfg)
        
        print(f"✓ Metrics computed successfully")
        print(f"WER: {metrics.get('wer', 'N/A')}")
        print(f"Ref text: '{metrics.get('ref_text', 'N/A')[:50]}...'")
        print(f"Hyp text: '{metrics.get('hyp_text', 'N/A')[:50]}...'")
        
        return True
    except Exception as e:
        print(f"✗ Error computing metrics: {e}")
        return False

if __name__ == "__main__":
    print("Testing fixed ASR implementation...")
    
    # Test 1: Ground truth extraction
    success1 = test_ground_truth_extraction()
    
    # Test 2: ASR with ground truth
    success2 = test_asr_with_ground_truth()
    
    if success1 and success2:
        print("\n✅ All tests passed! ASR implementation is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the implementation.")


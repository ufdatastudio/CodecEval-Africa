#!/usr/bin/env python3
"""
NISQA v2.0 Runner for CodecEval-Africa
Replaces the old NISQA runner with the new official implementation.
"""

import os
import sys
import subprocess
import tempfile
import pandas as pd
from pathlib import Path

def run_nisqa_v2(audio_file, output_dir=None):
    """
    Run NISQA v2.0 on a single audio file.
    
    Args:
        audio_file (str): Path to audio file
        output_dir (str): Output directory for results (optional)
    
    Returns:
        dict: NISQA metrics (mos, noisiness, discontinuity, coloration, loudness)
    """
    print(f"    Running NISQA v2.0 on: {os.path.basename(audio_file)}")
    
    # Create temporary output directory if not provided
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Run NISQA v2.0 prediction
        cmd = [
            "python", "NISQA/run_predict.py",
            "--mode", "predict_file",
            "--pretrained_model", "NISQA/weights/nisqa.tar",
            "--deg", audio_file,
            "--output_dir", output_dir
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            print(f"    NISQA v2.0 failed: {result.stderr}")
            return {
                'mos': float('nan'),
                'noisiness': float('nan'),
                'discontinuity': float('nan'),
                'coloration': float('nan'),
                'loudness': float('nan')
            }
        
        # Parse results from CSV
        results_file = Path(output_dir) / "NISQA_results.csv"
        if not results_file.exists():
            print(f"    NISQA v2.0 results file not found: {results_file}")
            return {
                'mos': float('nan'),
                'noisiness': float('nan'),
                'discontinuity': float('nan'),
                'coloration': float('nan'),
                'loudness': float('nan')
            }
        
        # Read results
        df = pd.read_csv(results_file)
        if len(df) == 0:
            print("    NISQA v2.0 returned empty results")
            return {
                'mos': float('nan'),
                'noisiness': float('nan'),
                'discontinuity': float('nan'),
                'coloration': float('nan'),
                'loudness': float('nan')
            }
        
        # Extract metrics
        row = df.iloc[0]
        metrics = {
            'mos': float(row['mos_pred']),
            'noisiness': float(row['noi_pred']),
            'discontinuity': float(row['dis_pred']),
            'coloration': float(row['col_pred']),
            'loudness': float(row['loud_pred'])
        }
        
        print(f"    NISQA v2.0 results: MOS={metrics['mos']:.3f}, Noise={metrics['noisiness']:.3f}")
        return metrics
        
    except subprocess.TimeoutExpired:
        print("    NISQA v2.0 timed out")
        return {
            'mos': float('nan'),
            'noisiness': float('nan'),
            'discontinuity': float('nan'),
            'coloration': float('nan'),
            'loudness': float('nan')
        }
    except Exception as e:
        print(f"    NISQA v2.0 error: {e}")
        return {
            'mos': float('nan'),
            'noisiness': float('nan'),
            'discontinuity': float('nan'),
            'coloration': float('nan'),
            'loudness': float('nan')
        }

def run_nisqa_v2_batch(audio_files, output_dir=None):
    """
    Run NISQA v2.0 on multiple audio files.
    
    Args:
        audio_files (list): List of audio file paths
        output_dir (str): Output directory for results (optional)
    
    Returns:
        list: List of NISQA metrics for each file
    """
    print(f"Running NISQA v2.0 on {len(audio_files)} files...")
    
    results = []
    for i, audio_file in enumerate(audio_files):
        print(f"  Processing {i+1}/{len(audio_files)}: {os.path.basename(audio_file)}")
        metrics = run_nisqa_v2(audio_file, output_dir)
        results.append(metrics)
    
    return results

if __name__ == "__main__":
    # Test the new NISQA runner
    import sys
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        print(f"Testing NISQA v2.0 on: {test_file}")
        metrics = run_nisqa_v2(test_file)
        print(f"Results: {metrics}")
    else:
        print("Usage: python nisqa_runner_v2.py <audio_file>")

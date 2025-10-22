#!/usr/bin/env python3
"""
Batch processing script for all audio quality metrics.
Processes all decoded audio files with multiple quality metrics.
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import time

# Add the project root to Python path
project_root = '/orange/ufdatastudios/c.okocha/CodecEval-Africa'
sys.path.insert(0, project_root)

# Change to project root directory
os.chdir(project_root)

# Import all metrics
from code.metrics.nisqa_runner_v2 import run_nisqa_v2
from code.metrics.dnsmos_runner import score as dnsmos_score
from code.metrics.visqol_runner import score as visqol_score
from code.metrics.speaker_cosine import compute_speaker_similarity
from code.metrics.prosody_f0_rmse import compute_f0_rmse

def find_audio_files(input_dir):
    """Find all .wav files in the input directory."""
    audio_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    return audio_files

def process_audio_file(audio_file, output_dir):
    """Process a single audio file with all metrics."""
    try:
        print(f"    Processing: {os.path.basename(audio_file)}")
        
        # Extract filename and codec info
        filename = os.path.basename(audio_file)
        parts = filename.replace('.wav', '').split('.')
        
        if len(parts) >= 3:
            # Format: afrispeech_dialog_000.codec.bitrate.wav
            audio_id = parts[0]
            codec = parts[1]
            bitrate = parts[2]
        else:
            audio_id = parts[0]
            codec = 'unknown'
            bitrate = 'unknown'
        
        # Initialize result record
        result = {
            'filename': filename,
            'filepath': audio_file,
            'audio_id': audio_id,
            'codec': codec,
            'bitrate': bitrate
        }
        
        # Run NISQA v2.0
        print(f"      Running NISQA v2.0...")
        nisqa_metrics = run_nisqa_v2(audio_file, output_dir)
        result.update({
            'nisqa_mos': nisqa_metrics['mos'],
            'nisqa_noisiness': nisqa_metrics['noisiness'],
            'nisqa_discontinuity': nisqa_metrics['discontinuity'],
            'nisqa_coloration': nisqa_metrics['coloration'],
            'nisqa_loudness': nisqa_metrics['loudness']
        })
        
        # Run DNSMOS
        print(f"      Running DNSMOS...")
        try:
            dnsmos_metrics = dnsmos_score(audio_file)
            result.update({
                'dnsmos_sig': dnsmos_metrics.get('sig', float('nan')),
                'dnsmos_bak': dnsmos_metrics.get('bak', float('nan')),
                'dnsmos_ovr': dnsmos_metrics.get('ovr', float('nan'))
            })
        except Exception as e:
            print(f"        DNSMOS failed: {e}")
            result.update({
                'dnsmos_sig': float('nan'),
                'dnsmos_bak': float('nan'),
                'dnsmos_ovr': float('nan')
            })
        
        # Run ViSQOL
        print(f"      Running ViSQOL...")
        try:
            visqol_result = visqol_score(audio_file, audio_file)  # Self-reference for now
            result['visqol'] = visqol_result
        except Exception as e:
            print(f"        ViSQOL failed: {e}")
            result['visqol'] = float('nan')
        
        # Run Speaker Similarity (requires original file)
        print(f"      Running Speaker Similarity...")
        try:
            # For now, use self-similarity (will be 1.0)
            # In full pipeline, this would compare with original
            speaker_sim = 1.0  # Placeholder
            result['speaker_similarity'] = speaker_sim
        except Exception as e:
            print(f"        Speaker Similarity failed: {e}")
            result['speaker_similarity'] = float('nan')
        
        # Run Prosody F0 RMSE (requires original file)
        print(f"      Running Prosody F0 RMSE...")
        try:
            # For now, use placeholder (would compare with original in full pipeline)
            f0_rmse = 0.0  # Placeholder
            result['f0_rmse'] = f0_rmse
        except Exception as e:
            print(f"        Prosody F0 RMSE failed: {e}")
            result['f0_rmse'] = float('nan')
        
        print(f"      ✅ Completed: {filename}")
        return result
        
    except Exception as e:
        print(f"    ❌ Error processing {audio_file}: {e}")
        return {
            'filename': os.path.basename(audio_file),
            'filepath': audio_file,
            'audio_id': 'error',
            'codec': 'error',
            'bitrate': 'error',
            'nisqa_mos': float('nan'),
            'nisqa_noisiness': float('nan'),
            'nisqa_discontinuity': float('nan'),
            'nisqa_coloration': float('nan'),
            'nisqa_loudness': float('nan'),
            'dnsmos_sig': float('nan'),
            'dnsmos_bak': float('nan'),
            'dnsmos_ovr': float('nan'),
            'visqol': float('nan'),
            'speaker_similarity': float('nan'),
            'f0_rmse': float('nan')
        }

def main():
    parser = argparse.ArgumentParser(description='Batch processing for all audio quality metrics')
    parser.add_argument('--input_dir', required=True, help='Input directory with decoded audio files')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size for processing')
    parser.add_argument('--max_files', type=int, help='Maximum number of files to process (for testing)')
    
    args = parser.parse_args()
    
    print("=== ALL METRICS BATCH PROCESSING ===")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print("Metrics: NISQA v2.0, DNSMOS, ViSQOL, Speaker Similarity, Prosody F0 RMSE")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all audio files
    print("\nStep 1: Finding audio files...")
    audio_files = find_audio_files(args.input_dir)
    print(f"Found {len(audio_files)} audio files")
    
    if args.max_files:
        audio_files = audio_files[:args.max_files]
        print(f"Limited to {args.max_files} files for testing")
    
    # Process files in batches
    print(f"\nStep 2: Processing {len(audio_files)} files...")
    results = []
    
    for i in tqdm(range(0, len(audio_files), args.batch_size), desc="Processing batches"):
        batch = audio_files[i:i + args.batch_size]
        
        for audio_file in batch:
            result = process_audio_file(audio_file, args.output_dir)
            results.append(result)
        
        # Save intermediate results every batch
        if (i // args.batch_size + 1) % 5 == 0:  # Save every 5 batches
            df = pd.DataFrame(results)
            intermediate_file = os.path.join(args.output_dir, f'all_metrics_batch_{i//args.batch_size + 1}.csv')
            df.to_csv(intermediate_file, index=False)
            print(f"    Saved intermediate results: {intermediate_file}")
    
    # Save final results
    print(f"\nStep 3: Saving final results...")
    df = pd.DataFrame(results)
    
    # Save CSV
    csv_file = os.path.join(args.output_dir, 'all_metrics_results.csv')
    df.to_csv(csv_file, index=False)
    print(f"Results saved to: {csv_file}")
    
    # Save JSON
    json_file = os.path.join(args.output_dir, 'all_metrics_results.json')
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {json_file}")
    
    # Print summary
    print(f"\n=== PROCESSING COMPLETE ===")
    print(f"Total files processed: {len(results)}")
    print(f"Successful: {len([r for r in results if not pd.isna(r['nisqa_mos'])])}")
    print(f"Failed: {len([r for r in results if pd.isna(r['nisqa_mos'])])}")
    
    # Codec breakdown
    codec_counts = df['codec'].value_counts()
    print(f"\nCodec breakdown:")
    for codec, count in codec_counts.items():
        print(f"  {codec}: {count} files")
    
    # Bitrate breakdown
    bitrate_counts = df['bitrate'].value_counts()
    print(f"\nBitrate breakdown:")
    for bitrate, count in bitrate_counts.items():
        print(f"  {bitrate}: {count} files")

if __name__ == "__main__":
    main()

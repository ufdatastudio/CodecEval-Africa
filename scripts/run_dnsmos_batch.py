#!/usr/bin/env python3
"""
Batch DNSMOS processing script.
Processes all decoded audio files with DNSMOS metrics.
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

from code.metrics.dnsmos_runner_real import score as dnsmos_score

def find_audio_files(input_dir):
    """Find all .wav files in the input directory."""
    audio_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    return audio_files

def process_audio_file(audio_file, output_dir):
    """Process a single audio file with DNSMOS."""
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
        
        # Run DNSMOS
        print(f"      Running DNSMOS...")
        dnsmos_metrics = dnsmos_score(audio_file)
        result.update({
            'dnsmos_sig': dnsmos_metrics.get('sig', float('nan')),
            'dnsmos_bak': dnsmos_metrics.get('bak', float('nan')),
            'dnsmos_ovr': dnsmos_metrics.get('ovr', float('nan'))
        })
        
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
            'dnsmos_sig': float('nan'),
            'dnsmos_bak': float('nan'),
            'dnsmos_ovr': float('nan')
        }

def main():
    parser = argparse.ArgumentParser(description='Batch DNSMOS processing')
    parser.add_argument('--input_dir', required=True, help='Input directory with decoded audio files')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for processing')
    parser.add_argument('--max_files', type=int, help='Maximum number of files to process (for testing)')
    
    args = parser.parse_args()
    
    print("=== DNSMOS BATCH PROCESSING ===")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print("Metrics: DNSMOS (Signal, Background, Overall)")
    
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
        if (i // args.batch_size + 1) % 10 == 0:  # Save every 10 batches
            df = pd.DataFrame(results)
            intermediate_file = os.path.join(args.output_dir, f'dnsmos_results_batch_{i//args.batch_size + 1}.csv')
            df.to_csv(intermediate_file, index=False)
            print(f"    Saved intermediate results: {intermediate_file}")
    
    # Save final results
    print(f"\nStep 3: Saving final results...")
    df = pd.DataFrame(results)
    
    # Save CSV
    csv_file = os.path.join(args.output_dir, 'dnsmos_results.csv')
    df.to_csv(csv_file, index=False)
    print(f"Results saved to: {csv_file}")
    
    # Save JSON
    json_file = os.path.join(args.output_dir, 'dnsmos_results.json')
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {json_file}")
    
    # Print summary
    print(f"\n=== PROCESSING COMPLETE ===")
    print(f"Total files processed: {len(results)}")
    print(f"Successful: {len([r for r in results if not pd.isna(r['dnsmos_ovr'])])}")
    print(f"Failed: {len([r for r in results if pd.isna(r['dnsmos_ovr'])])}")
    
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


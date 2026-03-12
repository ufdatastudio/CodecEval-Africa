#!/usr/bin/env python3
"""
Organize STOI results from JSON files and map them to datasets/codecs/bitrates.

This script properly parses all STOI result files and organizes them by:
1. Checking the degraded directory path in the logs
2. Extracting dataset, codec, and bitrate from file metadata
3. Creating comprehensive summaries
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import sys


def find_stoi_results(results_dir="results/quality_metrics"):
    """Find all STOI result JSON files."""
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Error: {results_dir} not found")
        return []
    
    stoi_files = list(results_path.glob("*stoi*.json"))
    return sorted(stoi_files, key=lambda x: x.stat().st_mtime)


def map_results_to_codecs(log_dir="batch_scripts"):
    """
    Parse batch job logs to map result files to codec/bitrate/dataset.
    Returns a dict mapping result filename to (dataset, codec, bitrate).
    """
    mapping = {}
    
    # Find all STOI log files
    log_path = Path(log_dir)
    err_files = list(log_path.glob("stoi_comp_*.err")) + list(log_path.glob("stoi_all_*.err"))
    
    for err_file in err_files:
        try:
            with open(err_file, 'r') as f:
                content = f.read()
            
            # Find patterns like:
            # "Degraded directory: outputs/afrispeech_dialog/Encodec_outputs/out_24kbps"
            # followed by "Results saved to: results/quality_metrics/stoi_out_24kbps_TIMESTAMP.json"
            
            pattern = r'Degraded directory: outputs/(\w+)/(\w+)_outputs/out_([\w.]+).*?Results saved to: (results/quality_metrics/stoi_out_[\w.]+_\d+\.json)'
            
            matches = re.findall(pattern, content, re.DOTALL)
            
            for dataset, codec, bitrate, result_file in matches:
                result_filename = Path(result_file).name
                mapping[result_filename] = (dataset, codec, bitrate)
                
        except Exception as e:
            print(f"Warning: Could not parse {err_file}: {e}")
            continue
    
    return mapping


def organize_results():
    """Main result organization function."""
    
    print("Scanning STOI result files...")
    stoi_files = find_stoi_results()
    
    if not stoi_files:
        print("No STOI results found")
        return
    
    print(f"Found {len(stoi_files)} result files")
    print("\nMapping results to codecs...")
    
    mapping = map_results_to_codecs()
    print(f"Mapped {len(mapping)} results from logs")
    
    # Organize by dataset -> codec -> bitrate
    organized = defaultdict(lambda: defaultdict(dict))
    unmapped = []
    
    for filepath in stoi_files:
        filename = filepath.name
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if filename in mapping:
                dataset, codec, bitrate = mapping[filename]
                organized[dataset][codec][bitrate] = {
                    'stoi': data['stoi'],
                    'stoi_std': data['stoi_std'],
                    'num_files': data['num_files'],
                    'file': str(filepath)
                }
            else:
                # Try to infer from old results
                unmapped.append((filename, data))
                
        except Exception as e:
            print(f"Warning: Could not read {filepath}: {e}")
    
    if unmapped:
        print(f"\nWarning: {len(unmapped)} files could not be mapped to codec/dataset")
    
    return organized


def print_organized_results(organized):
    """Print results in clean table format."""
    
    print("\n" + "="*100)
    print("STOI EVALUATION RESULTS - COMPREHENSIVE SUMMARY")
    print("="*100)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100)
    
    for dataset in sorted(organized.keys()):
        codecs = organized[dataset]
        
        print(f"\n📊 Dataset: {dataset.upper()}")
        print("-" * 100)
        print(f"{'Codec':<20} {'Bitrate':<12} {'STOI':<10} {'Std Dev':<10} {'Files':<8} {'Quality'}")
        print("-" * 100)
        
        # Collect all results for sorting
        all_results = []
        for codec, bitrates in codecs.items():
            for bitrate, data in bitrates.items():
                all_results.append((codec, bitrate, data))
        
        # Sort by codec, then numerically by bitrate
        def sort_key(item):
            codec, bitrate, _ = item
            try:
                bps = float(bitrate.replace('kbps', '').replace('k', ''))
            except:
                bps = 0
            return (codec, bps)
        
        all_results.sort(key=sort_key)
        
        # Print rows
        for codec, bitrate, data in all_results:
            stoi_val = data['stoi']
            std_val = data['stoi_std']
            num_files = data['num_files']
            
            # Quality rating
            if stoi_val >= 0.95:
                quality = "🟢 Excellent"
            elif stoi_val >= 0.90:
                quality = "🟡 Very Good"
            elif stoi_val >= 0.85:
                quality = "🟠 Good"
            elif stoi_val >= 0.75:
                quality = "🔴 Fair"
            else:
                quality = "⚫ Poor"
            
            print(f"{codec:<20} {bitrate:<12} {stoi_val:<10.4f} {std_val:<10.4f} {num_files:<8} {quality}")
        
        print("-" * 100)
    
    print("\n" + "="*100)


def create_summary_file(organized, output_file="results/stoi_comprehensive_summary.txt"):
    """Create a clean summary file."""
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("STOI EVALUATION RESULTS - COMPREHENSIVE SUMMARY\n")
        f.write("="*100 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("STOI (Short-Time Objective Intelligibility) Score Range: 0.0 - 1.0\n")
        f.write("Higher scores indicate better intelligibility\n")
        f.write("="*100 + "\n\n")
        
        for dataset in sorted(organized.keys()):
            codecs = organized[dataset]
            
            f.write(f"\nDataset: {dataset.upper()}\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Codec':<20} {'Bitrate':<12} {'STOI':<10} {'Std Dev':<10} {'Files':<8} {'Quality'}\n")
            f.write("-" * 100 + "\n")
            
            # Collect and sort
            all_results = []
            for codec, bitrates in codecs.items():
                for bitrate, data in bitrates.items():
                    all_results.append((codec, bitrate, data))
            
            def sort_key(item):
                codec, bitrate, _ = item
                try:
                    bps = float(bitrate.replace('kbps', '').replace('k', ''))
                except:
                    bps = 0
                return (codec, bps)
            
            all_results.sort(key=sort_key)
            
            for codec, bitrate, data in all_results:
                stoi_val = data['stoi']
                std_val = data['stoi_std']
                num_files = data['num_files']
                
                if stoi_val >= 0.95:
                    quality = "Excellent"
                elif stoi_val >= 0.90:
                    quality = "Very Good"
                elif stoi_val >= 0.85:
                    quality = "Good"
                elif stoi_val >= 0.75:
                    quality = "Fair"
                else:
                    quality = "Poor"
                
                f.write(f"{codec:<20} {bitrate:<12} {stoi_val:<10.4f} {std_val:<10.4f} {num_files:<8} {quality}\n")
            
            f.write("\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("Quality Scale:\n")
        f.write("  Excellent:  STOI ≥ 0.95\n")
        f.write("  Very Good:  0.90 ≤ STOI < 0.95\n")
        f.write("  Good:       0.85 ≤ STOI < 0.90\n")
        f.write("  Fair:       0.75 ≤ STOI < 0.85\n")
        f.write("  Poor:       STOI < 0.75\n")
        f.write("="*100 + "\n")
    
    print(f"\n✓ Summary saved to: {output_path}")
    return output_path


def main():
    """Main function."""
    
    organized = organize_results()
    
    if not organized:
        print("\nNo results to display")
        sys.exit(1)
    
    # Print to console
    print_organized_results(organized)
    
    # Save to file
    summary_file = create_summary_file(organized)
    
    print(f"\n✓ STOI results organized successfully!")
    print(f"✓ Summary: {summary_file}")


if __name__ == '__main__':
    main()

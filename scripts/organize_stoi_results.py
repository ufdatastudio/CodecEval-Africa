#!/usr/bin/env python3
"""
Organize and display STOI evaluation results in a clean, structured format.

This script:
1. Scans all STOI result files
2. Organizes them by dataset, codec, and bitrate
3. Creates summary tables and visualizations
4. Generates a clean report
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import sys

def parse_stoi_results(results_dir="results/quality_metrics"):
    """Parse all STOI result JSON files."""
    
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return {}
    
    # Find all STOI result files
    stoi_files = list(results_path.glob("*stoi*.json"))
    
    if not stoi_files:
        print(f"No STOI result files found in {results_dir}")
        return {}
    
    print(f"Found {len(stoi_files)} STOI result files")
    
    # Organize results by dataset and codec
    organized = defaultdict(lambda: defaultdict(dict))
    
    for filepath in stoi_files:
        try:
            with open(filepath, 'r') as f:
                result = json.load(f)
            
            # Extract metadata from filename or path
            # Format: stoi_out_<bitrate>_<timestamp>.json or <codec>_out_<bitrate>_stoi_<timestamp>.json
            filename = filepath.stem
            
            # Try to infer dataset/codec/bitrate from filename
            parts = filename.split('_')
            
            # Look for bitrate pattern
            bitrate = None
            codec = None
            dataset = None
            
            for i, part in enumerate(parts):
                if 'kbps' in part:
                    bitrate = part
                    # Check previous part for codec name
                    if i > 0 and parts[i-1] not in ['out', 'stoi']:
                        codec = parts[i-1]
            
            # If we can't parse, use generic names
            if not codec:
                codec = "Unknown"
            if not bitrate:
                bitrate = "unknown"
            if not dataset:
                dataset = "afrispeech_dialog"  # Default assumption
            
            # Store result
            organized[dataset][codec][bitrate] = {
                'stoi': result.get('stoi', 0.0),
                'stoi_std': result.get('stoi_std', 0.0),
                'num_files': result.get('num_files', 0),
                'file': str(filepath),
                'timestamp': filepath.stem.split('_')[-1] if '_' in filepath.stem else 'unknown'
            }
            
        except Exception as e:
            print(f"Warning: Could not parse {filepath}: {e}")
            continue
    
    return organized


def print_results_table(organized_results):
    """Print results in a clean table format."""
    
    if not organized_results:
        print("No results to display")
        return
    
    print("\n" + "="*100)
    print("STOI EVALUATION RESULTS SUMMARY")
    print("="*100)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100)
    
    for dataset, codecs in sorted(organized_results.items()):
        print(f"\n📊 Dataset: {dataset.upper()}")
        print("-" * 100)
        print(f"{'Codec':<20} {'Bitrate':<12} {'STOI':<12} {'Std Dev':<12} {'Files':<10} {'Result File'}")
        print("-" * 100)
        
        # Collect all results for this dataset
        all_results = []
        for codec, bitrates in sorted(codecs.items()):
            for bitrate, data in sorted(bitrates.items(), key=lambda x: x[0]):
                all_results.append((codec, bitrate, data))
        
        # Sort by codec, then by bitrate (numerically)
        def sort_key(item):
            codec, bitrate, _ = item
            # Extract numeric value from bitrate
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
            result_file = Path(data['file']).name
            
            # Color code STOI scores
            if stoi_val >= 0.9:
                quality = "🟢 EXCELLENT"
            elif stoi_val >= 0.8:
                quality = "🟡 GOOD"
            elif stoi_val >= 0.7:
                quality = "🟠 FAIR"
            else:
                quality = "🔴 POOR"
            
            print(f"{codec:<20} {bitrate:<12} {stoi_val:<12.4f} {std_val:<12.4f} {num_files:<10} {result_file}")
        
        print("-" * 100)
    
    print("\n" + "="*100)


def create_markdown_report(organized_results, output_file="results/STOI_RESULTS.md"):
    """Create a markdown report of STOI results."""
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("# STOI Evaluation Results\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("STOI (Short-Time Objective Intelligibility) measures speech intelligibility.\n")
        f.write("Scores range from 0 to 1, where higher values indicate better intelligibility.\n\n")
        
        for dataset, codecs in sorted(organized_results.items()):
            f.write(f"## Dataset: {dataset}\n\n")
            
            # Create table
            f.write("| Codec | Bitrate | STOI | Std Dev | # Files | Quality |\n")
            f.write("|-------|---------|------|---------|---------|----------|\n")
            
            # Collect and sort results
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
                
                # Quality rating
                if stoi_val >= 0.9:
                    quality = "🟢 Excellent"
                elif stoi_val >= 0.8:
                    quality = "🟡 Good"
                elif stoi_val >= 0.7:
                    quality = "🟠 Fair"
                else:
                    quality = "🔴 Poor"
                
                f.write(f"| {codec} | {bitrate} | {stoi_val:.4f} | {std_val:.4f} | {num_files} | {quality} |\n")
            
            f.write("\n")
        
        f.write("---\n\n")
        f.write("**Quality Scale:**\n")
        f.write("- 🟢 Excellent: STOI ≥ 0.9\n")
        f.write("- 🟡 Good: 0.8 ≤ STOI < 0.9\n")
        f.write("- 🟠 Fair: 0.7 ≤ STOI < 0.8\n")
        f.write("- 🔴 Poor: STOI < 0.7\n")
    
    print(f"\n✓ Markdown report saved to: {output_path}")


def main():
    """Main function."""
    
    # Parse results
    results = parse_stoi_results()
    
    if not results:
        print("No STOI results found. Run evaluations first.")
        sys.exit(1)
    
    # Display table
    print_results_table(results)
    
    # Create markdown report
    create_markdown_report(results)
    
    print("\n✓ STOI results organized successfully!")


if __name__ == '__main__':
    main()

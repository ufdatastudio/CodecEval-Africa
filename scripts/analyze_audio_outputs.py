#!/usr/bin/env python3
"""
Comprehensive Audio Output Analysis Script
Analyzes all codec outputs in the outputs directory and generates detailed statistics.
"""

import os
import sys
import json
import wave
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    print("Warning: soundfile not available, will use wave module only")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: librosa not available, limited analysis")


# Expected specifications for each codec
CODEC_SPECS = {
    'DAC': {
        '8kbps': {'bitrate': 8, 'sample_rate': 44100, 'tolerance': 0.5},
        '16kbps': {'bitrate': 16, 'sample_rate': 44100, 'tolerance': 0.5},
        '24kbps': {'bitrate': 24, 'sample_rate': 44100, 'tolerance': 0.5},
    },
    'Encodec': {
        '3kbps': {'bitrate': 3, 'sample_rate': 24000, 'tolerance': 0.3},
        '6kbps': {'bitrate': 6, 'sample_rate': 24000, 'tolerance': 0.3},
        '12kbps': {'bitrate': 12, 'sample_rate': 24000, 'tolerance': 0.3},
        '24kbps': {'bitrate': 24, 'sample_rate': 24000, 'tolerance': 0.3},
    },
    'SemantiCodec': {
        '0.31kbps': {'bitrate': 0.31, 'sample_rate': 16000, 'tolerance': 0.1},
        '0.33kbps': {'bitrate': 0.33, 'sample_rate': 16000, 'tolerance': 0.1},
        '0.63kbps': {'bitrate': 0.63, 'sample_rate': 16000, 'tolerance': 0.1},
        '0.68kbps': {'bitrate': 0.68, 'sample_rate': 16000, 'tolerance': 0.1},
        '1.25kbps': {'bitrate': 1.25, 'sample_rate': 16000, 'tolerance': 0.1},
        '1.40kbps': {'bitrate': 1.40, 'sample_rate': 16000, 'tolerance': 0.1},
    },
    'UniCodec': {
        '6.6kbps': {'bitrate': 6.6, 'sample_rate': 24000, 'tolerance': 0.5},
    },
    'LanguageCodec': {
        'bandwidth_0': {'bitrate': 0.25, 'sample_rate': 16000, 'tolerance': 0.1},
        'bandwidth_1': {'bitrate': 0.5, 'sample_rate': 16000, 'tolerance': 0.1},
        'bandwidth_2': {'bitrate': 1.0, 'sample_rate': 16000, 'tolerance': 0.1},
        'bandwidth_3': {'bitrate': 2.0, 'sample_rate': 16000, 'tolerance': 0.1},
    },
    'WavTokenizer': {
        'small-320-24k-4096': {'sample_rate': 24000, 'tokens_per_sec': 75},
        'small-600-24k-4096': {'sample_rate': 24000, 'tokens_per_sec': 75},
        'medium-speech-75token': {'sample_rate': 24000, 'tokens_per_sec': 75},
        'medium-music-audio-75token': {'sample_rate': 24000, 'tokens_per_sec': 75},
        'large-speech-75token': {'sample_rate': 24000, 'tokens_per_sec': 75},
        'large-unify-40token': {'sample_rate': 24000, 'tokens_per_sec': 40},
    }
}


def get_audio_info(filepath: str) -> Dict:
    """Extract detailed information from an audio file."""
    info = {
        'filepath': filepath,
        'exists': False,
        'file_size_bytes': 0,
        'file_size_kb': 0,
        'sample_rate': None,
        'channels': None,
        'duration_sec': None,
        'bit_depth': None,
        'calculated_bitrate_kbps': None,
        'format': None,
        'error': None
    }
    
    if not os.path.exists(filepath):
        info['error'] = 'File not found'
        return info
    
    info['exists'] = True
    info['file_size_bytes'] = os.path.getsize(filepath)
    info['file_size_kb'] = info['file_size_bytes'] / 1024
    
    # Try soundfile first (supports more formats)
    if SOUNDFILE_AVAILABLE:
        try:
            with sf.SoundFile(filepath) as audio_file:
                info['sample_rate'] = audio_file.samplerate
                info['channels'] = audio_file.channels
                info['duration_sec'] = len(audio_file) / audio_file.samplerate
                info['format'] = audio_file.format
                info['bit_depth'] = audio_file.subtype
                
                # Calculate actual bitrate
                if info['duration_sec'] > 0:
                    info['calculated_bitrate_kbps'] = (info['file_size_bytes'] * 8) / (info['duration_sec'] * 1000)
                return info
        except Exception as e:
            pass  # Fall back to wave
    
    # Try wave module (WAV files only)
    try:
        with wave.open(filepath, 'rb') as wav_file:
            info['sample_rate'] = wav_file.getframerate()
            info['channels'] = wav_file.getnchannels()
            info['duration_sec'] = wav_file.getnframes() / wav_file.getframerate()
            info['bit_depth'] = wav_file.getsampwidth() * 8
            info['format'] = 'WAV'
            
            # Calculate actual bitrate
            if info['duration_sec'] > 0:
                info['calculated_bitrate_kbps'] = (info['file_size_bytes'] * 8) / (info['duration_sec'] * 1000)
    except Exception as e:
        info['error'] = str(e)
    
    return info


def scan_directory(directory: str) -> List[Dict]:
    """Recursively scan directory for audio files."""
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.opus'}
    audio_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in audio_extensions:
                filepath = os.path.join(root, file)
                audio_files.append(filepath)
    
    return audio_files


def analyze_codec_outputs(outputs_dir: str) -> Dict:
    """Analyze all codec outputs and generate statistics."""
    results = defaultdict(lambda: defaultdict(list))
    
    # Scan each codec output directory
    codec_dirs = {
        'DAC': ['DAC_outputs'],
        'Encodec': ['Encodec_outputs'],
        'SemantiCodec': ['SemantiCodec_outputs'],
        'UniCodec': ['UniCodec_outputs'],
        'LanguageCodec': ['LanguageCodec_outputs'],
        'WavTokenizer': ['WavTokenizer_outputs']
    }
    
    # Also scan dataset-specific outputs
    dataset_dirs = ['afrinames', 'afrispeech_multilingual']
    
    for codec_name, dirs in codec_dirs.items():
        for codec_dir in dirs:
            # Check main outputs
            main_path = os.path.join(outputs_dir, codec_dir)
            if os.path.exists(main_path):
                print(f"\nAnalyzing {codec_name} in {main_path}...")
                audio_files = scan_directory(main_path)
                for audio_file in audio_files:
                    info = get_audio_info(audio_file)
                    rel_path = os.path.relpath(audio_file, outputs_dir)
                    info['relative_path'] = rel_path
                    results[codec_name]['main'].append(info)
            
            # Check dataset-specific outputs
            for dataset in dataset_dirs:
                dataset_path = os.path.join(outputs_dir, dataset, codec_dir)
                if os.path.exists(dataset_path):
                    print(f"Analyzing {codec_name} in {dataset_path}...")
                    audio_files = scan_directory(dataset_path)
                    for audio_file in audio_files:
                        info = get_audio_info(audio_file)
                        rel_path = os.path.relpath(audio_file, outputs_dir)
                        info['relative_path'] = rel_path
                        results[codec_name][dataset].append(info)
    
    return dict(results)


def generate_summary_stats(results: Dict) -> Dict:
    """Generate summary statistics from analysis results."""
    summary = {}
    
    for codec_name, datasets in results.items():
        codec_summary = {}
        
        for dataset_name, files in datasets.items():
            if not files:
                continue
            
            # Filter out files with errors
            valid_files = [f for f in files if f['exists'] and not f['error']]
            
            if not valid_files:
                codec_summary[dataset_name] = {'total_files': len(files), 'valid_files': 0, 'errors': len(files)}
                continue
            
            # Calculate statistics
            sample_rates = [f['sample_rate'] for f in valid_files if f['sample_rate']]
            durations = [f['duration_sec'] for f in valid_files if f['duration_sec']]
            bitrates = [f['calculated_bitrate_kbps'] for f in valid_files if f['calculated_bitrate_kbps']]
            file_sizes = [f['file_size_kb'] for f in valid_files]
            
            dataset_stats = {
                'total_files': len(files),
                'valid_files': len(valid_files),
                'errors': len(files) - len(valid_files),
                'sample_rates': {
                    'unique': list(set(sample_rates)) if sample_rates else [],
                    'mean': np.mean(sample_rates) if sample_rates else None,
                },
                'durations': {
                    'total_sec': sum(durations) if durations else 0,
                    'total_min': sum(durations) / 60 if durations else 0,
                    'mean_sec': np.mean(durations) if durations else None,
                    'min_sec': min(durations) if durations else None,
                    'max_sec': max(durations) if durations else None,
                },
                'bitrates': {
                    'mean_kbps': np.mean(bitrates) if bitrates else None,
                    'median_kbps': np.median(bitrates) if bitrates else None,
                    'min_kbps': min(bitrates) if bitrates else None,
                    'max_kbps': max(bitrates) if bitrates else None,
                    'std_kbps': np.std(bitrates) if bitrates else None,
                },
                'file_sizes': {
                    'total_kb': sum(file_sizes),
                    'total_mb': sum(file_sizes) / 1024,
                    'mean_kb': np.mean(file_sizes) if file_sizes else None,
                }
            }
            
            codec_summary[dataset_name] = dataset_stats
        
        summary[codec_name] = codec_summary
    
    return summary


def print_report(summary: Dict, results: Dict):
    """Print a formatted report of the analysis."""
    print("\n" + "="*80)
    print("AUDIO OUTPUT ANALYSIS REPORT")
    print("="*80)
    
    total_files = 0
    total_size_mb = 0
    total_duration_min = 0
    
    for codec_name, datasets in summary.items():
        print(f"\n{'='*80}")
        print(f"CODEC: {codec_name}")
        print(f"{'='*80}")
        
        for dataset_name, stats in datasets.items():
            print(f"\n  Dataset: {dataset_name}")
            print(f"  {'-'*76}")
            print(f"    Total Files: {stats['total_files']}")
            print(f"    Valid Files: {stats['valid_files']}")
            
            if stats['errors'] > 0:
                print(f"    ⚠️  Errors: {stats['errors']}")
            
            if stats['valid_files'] > 0:
                # Sample rates
                if stats['sample_rates']['unique']:
                    print(f"    Sample Rates: {stats['sample_rates']['unique']} Hz")
                
                # Durations
                print(f"    Total Duration: {stats['durations']['total_min']:.2f} minutes")
                print(f"    Mean Duration: {stats['durations']['mean_sec']:.2f} sec")
                print(f"    Duration Range: {stats['durations']['min_sec']:.2f} - {stats['durations']['max_sec']:.2f} sec")
                
                # Bitrates
                if stats['bitrates']['mean_kbps']:
                    print(f"    Mean Bitrate: {stats['bitrates']['mean_kbps']:.2f} kbps")
                    print(f"    Median Bitrate: {stats['bitrates']['median_kbps']:.2f} kbps")
                    print(f"    Bitrate Range: {stats['bitrates']['min_kbps']:.2f} - {stats['bitrates']['max_kbps']:.2f} kbps")
                    print(f"    Bitrate Std Dev: {stats['bitrates']['std_kbps']:.2f} kbps")
                
                # File sizes
                print(f"    Total Size: {stats['file_sizes']['total_mb']:.2f} MB")
                print(f"    Mean File Size: {stats['file_sizes']['mean_kb']:.2f} KB")
                
                # Check against expected specs
                verify_codec_specs(codec_name, dataset_name, stats, results[codec_name][dataset_name])
                
                # Update totals
                total_files += stats['valid_files']
                total_size_mb += stats['file_sizes']['total_mb']
                total_duration_min += stats['durations']['total_min']
    
    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    print(f"Total Audio Files: {total_files}")
    print(f"Total Storage: {total_size_mb:.2f} MB ({total_size_mb/1024:.2f} GB)")
    print(f"Total Audio Duration: {total_duration_min:.2f} minutes ({total_duration_min/60:.2f} hours)")
    print(f"{'='*80}\n")


def verify_codec_specs(codec_name: str, dataset_name: str, stats: Dict, files: List[Dict]):
    """Verify codec outputs against expected specifications."""
    if codec_name not in CODEC_SPECS:
        return
    
    # Try to determine which spec to use based on directory structure
    issues = []
    
    # Check sample rate consistency
    unique_sample_rates = stats['sample_rates']['unique']
    if len(unique_sample_rates) > 1:
        issues.append(f"⚠️  Multiple sample rates found: {unique_sample_rates}")
    
    # For each file, try to match against expected specs
    for spec_name, spec in CODEC_SPECS[codec_name].items():
        if 'bitrate' in spec:
            mean_bitrate = stats['bitrates']['mean_kbps']
            expected_bitrate = spec['bitrate']
            tolerance = spec.get('tolerance', 0.5)
            
            if mean_bitrate and abs(mean_bitrate - expected_bitrate) / expected_bitrate > tolerance:
                # Only warn if the spec name appears in file paths
                sample_paths = [f['relative_path'] for f in files[:5]]
                if any(spec_name.replace('kbps', '') in path or spec_name in path for path in sample_paths):
                    issues.append(f"⚠️  Bitrate mismatch for {spec_name}: expected {expected_bitrate} kbps, got {mean_bitrate:.2f} kbps")
        
        if 'sample_rate' in spec:
            expected_sr = spec['sample_rate']
            if unique_sample_rates and expected_sr not in unique_sample_rates:
                # Only warn if the spec name appears in file paths
                sample_paths = [f['relative_path'] for f in files[:5]]
                if any(spec_name in path for path in sample_paths):
                    issues.append(f"⚠️  Sample rate mismatch for {spec_name}: expected {expected_sr} Hz, got {unique_sample_rates}")
    
    if issues:
        print(f"    \n    SPEC VERIFICATION:")
        for issue in issues:
            print(f"    {issue}")


def save_detailed_report(results: Dict, summary: Dict, output_file: str):
    """Save detailed analysis to JSON file."""
    report = {
        'timestamp': Path(output_file).stem,
        'summary': summary,
        'detailed_results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n✓ Detailed report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze codec audio outputs')
    parser.add_argument('--outputs-dir', default='/orange/ufdatastudios/c.okocha/CodecEval-Africa/outputs',
                        help='Path to outputs directory')
    parser.add_argument('--save-json', action='store_true',
                        help='Save detailed report as JSON')
    parser.add_argument('--output-file', default='audio_analysis_report.json',
                        help='Output JSON filename')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.outputs_dir):
        print(f"Error: Outputs directory not found: {args.outputs_dir}")
        sys.exit(1)
    
    print(f"Analyzing audio outputs in: {args.outputs_dir}")
    print("This may take a few minutes depending on the number of files...\n")
    
    # Analyze all outputs
    results = analyze_codec_outputs(args.outputs_dir)
    
    # Generate summary statistics
    summary = generate_summary_stats(results)
    
    # Print report
    print_report(summary, results)
    
    # Save detailed report if requested
    if args.save_json:
        save_detailed_report(results, summary, args.output_file)
    
    print("\n✓ Analysis complete!")


if __name__ == '__main__':
    main()

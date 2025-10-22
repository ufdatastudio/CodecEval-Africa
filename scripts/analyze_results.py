#!/usr/bin/env python3
"""
Analyze CodecEval-Africa benchmark results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def load_results(csv_path):
    """Load benchmark results from CSV."""
    if not os.path.exists(csv_path):
        print(f"Results file not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} results from {csv_path}")
    return df

def analyze_codec_performance(df):
    """Analyze codec performance across metrics."""
    print("\nCODEC PERFORMANCE ANALYSIS")
    
    # Group by codec
    codec_stats = df.groupby('codec').agg({
        'visqol': ['mean', 'std'],
        'speaker_similarity': ['mean', 'std'],
        'f0_rmse': ['mean', 'std'],
        'nisqa_drop': ['mean', 'std']
    }).round(3)
    
    print("Codec Performance Summary:")
    print(codec_stats)
    
    return codec_stats

def analyze_bitrate_impact(df):
    """Analyze impact of bitrate on quality."""
    print("\nBITRATE IMPACT ANALYSIS")
    
    # Group by codec and bitrate
    bitrate_stats = df.groupby(['codec', 'kbps']).agg({
        'visqol': 'mean',
        'speaker_similarity': 'mean',
        'f0_rmse': 'mean'
    }).round(3)
    
    print("Quality vs Bitrate:")
    print(bitrate_stats)
    
    return bitrate_stats

def analyze_accent_performance(df):
    """Analyze performance across different accents."""
    print("\nACCENT PERFORMANCE ANALYSIS")
    
    if 'accent' in df.columns:
        accent_stats = df.groupby('accent').agg({
            'visqol': 'mean',
            'speaker_similarity': 'mean',
            'f0_rmse': 'mean'
        }).round(3)
        
        print("Performance by Accent:")
        print(accent_stats)
        return accent_stats
    else:
        print("No accent information available")
        return None

def create_visualizations(df, output_dir="results/figures"):
    """Create comprehensive visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Codec Performance Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # ViSQOL by codec
    df.boxplot(column='visqol', by='codec', ax=axes[0,0])
    axes[0,0].set_title('ViSQOL by Codec')
    axes[0,0].set_xlabel('Codec')
    axes[0,0].set_ylabel('ViSQOL Score')
    
    # Speaker Similarity by codec
    df.boxplot(column='speaker_similarity', by='codec', ax=axes[0,1])
    axes[0,1].set_title('Speaker Similarity by Codec')
    axes[0,1].set_xlabel('Codec')
    axes[0,1].set_ylabel('Speaker Similarity')
    
    # F0 RMSE by codec
    df.boxplot(column='f0_rmse', by='codec', ax=axes[1,0])
    axes[1,0].set_title('F0 RMSE by Codec')
    axes[1,0].set_xlabel('Codec')
    axes[1,0].set_ylabel('F0 RMSE')
    
    # NISQA Drop by codec
    df.boxplot(column='nisqa_drop', by='codec', ax=axes[1,1])
    axes[1,1].set_title('NISQA Quality Drop by Codec')
    axes[1,1].set_xlabel('Codec')
    axes[1,1].set_ylabel('NISQA Drop')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/codec_performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Bitrate Impact
    if 'kbps' in df.columns:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # ViSQOL vs Bitrate
        for codec in df['codec'].unique():
            codec_data = df[df['codec'] == codec]
            axes[0].plot(codec_data['kbps'], codec_data['visqol'], 'o-', label=codec)
        axes[0].set_xlabel('Bitrate (kbps)')
        axes[0].set_ylabel('ViSQOL Score')
        axes[0].set_title('ViSQOL vs Bitrate')
        axes[0].legend()
        axes[0].grid(True)
        
        # Speaker Similarity vs Bitrate
        for codec in df['codec'].unique():
            codec_data = df[df['codec'] == codec]
            axes[1].plot(codec_data['kbps'], codec_data['speaker_similarity'], 'o-', label=codec)
        axes[1].set_xlabel('Bitrate (kbps)')
        axes[1].set_ylabel('Speaker Similarity')
        axes[1].set_title('Speaker Similarity vs Bitrate')
        axes[1].legend()
        axes[1].grid(True)
        
        # F0 RMSE vs Bitrate
        for codec in df['codec'].unique():
            codec_data = df[df['codec'] == codec]
            axes[2].plot(codec_data['kbps'], codec_data['f0_rmse'], 'o-', label=codec)
        axes[2].set_xlabel('Bitrate (kbps)')
        axes[2].set_ylabel('F0 RMSE')
        axes[2].set_title('F0 RMSE vs Bitrate')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/bitrate_impact_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to {output_dir}/")

def generate_summary_report(df, output_file="results/analysis_summary.txt"):
    """Generate a comprehensive summary report."""
    with open(output_file, 'w') as f:
        f.write("CodecEval-Africa Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total evaluations: {len(df)}\n")
        f.write(f"Codecs tested: {df['codec'].nunique()}\n")
        f.write(f"Bitrates tested: {sorted(df['kbps'].unique())}\n")
        f.write(f"Datasets: {df['dataset'].nunique()}\n\n")
        
        f.write("Codec Performance Summary:\n")
        f.write("-" * 30 + "\n")
        
        for codec in df['codec'].unique():
            codec_data = df[df['codec'] == codec]
            f.write(f"\n{codec}:\n")
            f.write(f"  ViSQOL: {codec_data['visqol'].mean():.3f} ± {codec_data['visqol'].std():.3f}\n")
            f.write(f"  Speaker Similarity: {codec_data['speaker_similarity'].mean():.3f} ± {codec_data['speaker_similarity'].std():.3f}\n")
            f.write(f"  F0 RMSE: {codec_data['f0_rmse'].mean():.3f} ± {codec_data['f0_rmse'].std():.3f}\n")
            f.write(f"  NISQA Drop: {codec_data['nisqa_drop'].mean():.3f} ± {codec_data['nisqa_drop'].std():.3f}\n")
    
    print(f"Summary report saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze CodecEval-Africa results')
    parser.add_argument('--csv', default='results/csv/benchmark.csv', help='Path to results CSV')
    parser.add_argument('--output', default='results/figures', help='Output directory for figures')
    args = parser.parse_args()
    
    print("CodecEval-Africa Results Analysis")
    print("=" * 40)
    
    # Load results
    df = load_results(args.csv)
    if df is None:
        return
    
    # Perform analysis
    analyze_codec_performance(df)
    analyze_bitrate_impact(df)
    analyze_accent_performance(df)
    
    # Create visualizations
    create_visualizations(df, args.output)
    
    # Generate summary
    generate_summary_report(df)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()

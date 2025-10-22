#!/usr/bin/env python3
"""
Analyze NISQA v2.0 results and generate comprehensive reports.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add the project root to Python path
project_root = '/orange/ufdatastudios/c.okocha/CodecEval-Africa'
sys.path.insert(0, project_root)

# Change to project root directory
os.chdir(project_root)

def load_nisqa_results(results_dir):
    """Load NISQA results from CSV file."""
    csv_file = os.path.join(results_dir, 'nisqa_results.csv')
    if not os.path.exists(csv_file):
        print(f"ERROR: Results file not found: {csv_file}")
        return None
    
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} NISQA results")
    return df

def analyze_by_codec(df, output_dir):
    """Analyze NISQA metrics by codec."""
    print("\n=== ANALYSIS BY CODEC ===")
    
    # Group by codec
    codec_stats = df.groupby('codec').agg({
        'mos': ['mean', 'std', 'min', 'max'],
        'noisiness': ['mean', 'std'],
        'discontinuity': ['mean', 'std'],
        'coloration': ['mean', 'std'],
        'loudness': ['mean', 'std']
    }).round(3)
    
    print("Codec Statistics:")
    print(codec_stats)
    
    # Save codec analysis
    codec_file = os.path.join(output_dir, 'codec_analysis.csv')
    codec_stats.to_csv(codec_file)
    print(f"Codec analysis saved to: {codec_file}")
    
    return codec_stats

def analyze_by_bitrate(df, output_dir):
    """Analyze NISQA metrics by bitrate."""
    print("\n=== ANALYSIS BY BITRATE ===")
    
    # Group by bitrate
    bitrate_stats = df.groupby('bitrate').agg({
        'mos': ['mean', 'std', 'min', 'max'],
        'noisiness': ['mean', 'std'],
        'discontinuity': ['mean', 'std'],
        'coloration': ['mean', 'std'],
        'loudness': ['mean', 'std']
    }).round(3)
    
    print("Bitrate Statistics:")
    print(bitrate_stats)
    
    # Save bitrate analysis
    bitrate_file = os.path.join(output_dir, 'bitrate_analysis.csv')
    bitrate_stats.to_csv(bitrate_file)
    print(f"Bitrate analysis saved to: {bitrate_file}")
    
    return bitrate_stats

def create_visualizations(df, output_dir):
    """Create visualization plots for NISQA results."""
    print("\n=== CREATING VISUALIZATIONS ===")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('NISQA v2.0 Quality Metrics Analysis', fontsize=16)
    
    metrics = ['mos', 'noisiness', 'discontinuity', 'coloration', 'loudness']
    
    # Plot 1: MOS by Codec
    ax1 = axes[0, 0]
    df.boxplot(column='mos', by='codec', ax=ax1)
    ax1.set_title('Overall Quality (MOS) by Codec')
    ax1.set_xlabel('Codec')
    ax1.set_ylabel('MOS Score')
    
    # Plot 2: MOS by Bitrate
    ax2 = axes[0, 1]
    df.boxplot(column='mos', by='bitrate', ax=ax2)
    ax2.set_title('Overall Quality (MOS) by Bitrate')
    ax2.set_xlabel('Bitrate (kbps)')
    ax2.set_ylabel('MOS Score')
    
    # Plot 3: Quality Dimensions Heatmap
    ax3 = axes[0, 2]
    codec_metrics = df.groupby('codec')[metrics].mean()
    sns.heatmap(codec_metrics.T, annot=True, cmap='RdYlBu_r', ax=ax3)
    ax3.set_title('Quality Dimensions by Codec')
    
    # Plot 4: Noisiness by Codec
    ax4 = axes[1, 0]
    df.boxplot(column='noisiness', by='codec', ax=ax4)
    ax4.set_title('Noisiness by Codec')
    ax4.set_xlabel('Codec')
    ax4.set_ylabel('Noisiness Score')
    
    # Plot 5: Discontinuity by Codec
    ax5 = axes[1, 1]
    df.boxplot(column='discontinuity', by='codec', ax=ax5)
    ax5.set_title('Discontinuity by Codec')
    ax5.set_xlabel('Codec')
    ax5.set_ylabel('Discontinuity Score')
    
    # Plot 6: MOS vs Bitrate Scatter
    ax6 = axes[1, 2]
    for codec in df['codec'].unique():
        codec_data = df[df['codec'] == codec]
        ax6.scatter(codec_data['bitrate'], codec_data['mos'], label=codec, alpha=0.7)
    ax6.set_title('MOS vs Bitrate by Codec')
    ax6.set_xlabel('Bitrate (kbps)')
    ax6.set_ylabel('MOS Score')
    ax6.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, 'nisqa_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {plot_file}")
    
    plt.close()

def generate_summary_report(df, output_dir):
    """Generate a comprehensive summary report."""
    print("\n=== GENERATING SUMMARY REPORT ===")
    
    report_file = os.path.join(output_dir, 'summary_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("NISQA v2.0 ANALYSIS SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Basic statistics
        f.write(f"Total files analyzed: {len(df)}\n")
        f.write(f"Unique codecs: {df['codec'].nunique()}\n")
        f.write(f"Unique bitrates: {df['bitrate'].nunique()}\n")
        f.write(f"Codecs: {', '.join(df['codec'].unique())}\n")
        f.write(f"Bitrates: {', '.join(df['bitrate'].unique())}\n\n")
        
        # Overall quality statistics
        f.write("OVERALL QUALITY STATISTICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Mean MOS: {df['mos'].mean():.3f}\n")
        f.write(f"Std MOS: {df['mos'].std():.3f}\n")
        f.write(f"Min MOS: {df['mos'].min():.3f}\n")
        f.write(f"Max MOS: {df['mos'].max():.3f}\n\n")
        
        # Best and worst codecs
        codec_means = df.groupby('codec')['mos'].mean().sort_values(ascending=False)
        f.write("CODE RANKING BY OVERALL QUALITY (MOS)\n")
        f.write("-" * 40 + "\n")
        for i, (codec, mos) in enumerate(codec_means.items(), 1):
            f.write(f"{i}. {codec}: {mos:.3f}\n")
        
        f.write(f"\nBest codec: {codec_means.index[0]} ({codec_means.iloc[0]:.3f})\n")
        f.write(f"Worst codec: {codec_means.index[-1]} ({codec_means.iloc[-1]:.3f})\n\n")
        
        # Quality dimensions analysis
        f.write("QUALITY DIMENSIONS ANALYSIS\n")
        f.write("-" * 30 + "\n")
        for metric in ['noisiness', 'discontinuity', 'coloration', 'loudness']:
            f.write(f"{metric.capitalize()}:\n")
            f.write(f"  Mean: {df[metric].mean():.3f}\n")
            f.write(f"  Std: {df[metric].std():.3f}\n")
            f.write(f"  Range: {df[metric].min():.3f} - {df[metric].max():.3f}\n\n")
    
    print(f"Summary report saved to: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze NISQA v2.0 results')
    parser.add_argument('--results_dir', required=True, help='Directory with NISQA results')
    parser.add_argument('--output_dir', required=True, help='Output directory for analysis')
    
    args = parser.parse_args()
    
    print("=== NISQA v2.0 RESULTS ANALYSIS ===")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    df = load_nisqa_results(args.results_dir)
    if df is None:
        return
    
    # Run analyses
    analyze_by_codec(df, args.output_dir)
    analyze_by_bitrate(df, args.output_dir)
    create_visualizations(df, args.output_dir)
    generate_summary_report(df, args.output_dir)
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()

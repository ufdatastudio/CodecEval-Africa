#!/usr/bin/env python3
"""
Analyze DNSMOS results and generate comprehensive reports.
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

def load_dnsmos_results(results_dir):
    """Load DNSMOS results from CSV file."""
    csv_file = os.path.join(results_dir, 'dnsmos_results.csv')
    if not os.path.exists(csv_file):
        print(f"ERROR: Results file not found: {csv_file}")
        return None
    
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} DNSMOS results")
    return df

def analyze_by_codec(df, output_dir):
    """Analyze DNSMOS metrics by codec."""
    print("\n=== ANALYSIS BY CODEC ===")
    
    # Group by codec
    codec_stats = df.groupby('codec').agg({
        'dnsmos_sig': ['mean', 'std', 'min', 'max'],
        'dnsmos_bak': ['mean', 'std', 'min', 'max'],
        'dnsmos_ovr': ['mean', 'std', 'min', 'max']
    }).round(3)
    
    print("Codec Statistics:")
    print(codec_stats)
    
    # Save codec analysis
    codec_file = os.path.join(output_dir, 'codec_analysis.csv')
    codec_stats.to_csv(codec_file)
    print(f"Codec analysis saved to: {codec_file}")
    
    return codec_stats

def analyze_by_bitrate(df, output_dir):
    """Analyze DNSMOS metrics by bitrate."""
    print("\n=== ANALYSIS BY BITRATE ===")
    
    # Group by bitrate
    bitrate_stats = df.groupby('bitrate').agg({
        'dnsmos_sig': ['mean', 'std', 'min', 'max'],
        'dnsmos_bak': ['mean', 'std', 'min', 'max'],
        'dnsmos_ovr': ['mean', 'std', 'min', 'max']
    }).round(3)
    
    print("Bitrate Statistics:")
    print(bitrate_stats)
    
    # Save bitrate analysis
    bitrate_file = os.path.join(output_dir, 'bitrate_analysis.csv')
    bitrate_stats.to_csv(bitrate_file)
    print(f"Bitrate analysis saved to: {bitrate_file}")
    
    return bitrate_stats

def create_visualizations(df, output_dir):
    """Create visualization plots for DNSMOS results."""
    print("\n=== CREATING VISUALIZATIONS ===")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('DNSMOS Audio Quality Metrics Analysis', fontsize=16)
    
    # Plot 1: Overall by Codec
    ax1 = axes[0, 0]
    df.boxplot(column='dnsmos_ovr', by='codec', ax=ax1)
    ax1.set_title('DNSMOS Overall by Codec')
    ax1.set_xlabel('Codec')
    ax1.set_ylabel('Overall Score')
    
    # Plot 2: Signal by Codec
    ax2 = axes[0, 1]
    df.boxplot(column='dnsmos_sig', by='codec', ax=ax2)
    ax2.set_title('DNSMOS Signal by Codec')
    ax2.set_xlabel('Codec')
    ax2.set_ylabel('Signal Score')
    
    # Plot 3: Background by Codec
    ax3 = axes[0, 2]
    df.boxplot(column='dnsmos_bak', by='codec', ax=ax3)
    ax3.set_title('DNSMOS Background by Codec')
    ax3.set_xlabel('Codec')
    ax3.set_ylabel('Background Score')
    
    # Plot 4: Overall by Bitrate
    ax4 = axes[1, 0]
    df.boxplot(column='dnsmos_ovr', by='bitrate', ax=ax4)
    ax4.set_title('DNSMOS Overall by Bitrate')
    ax4.set_xlabel('Bitrate (kbps)')
    ax4.set_ylabel('Overall Score')
    
    # Plot 5: Signal vs Background Scatter
    ax5 = axes[1, 1]
    ax5.scatter(df['dnsmos_sig'], df['dnsmos_bak'], alpha=0.6)
    ax5.set_title('Signal vs Background')
    ax5.set_xlabel('Signal Score')
    ax5.set_ylabel('Background Score')
    
    # Plot 6: Overall vs Bitrate Scatter by Codec
    ax6 = axes[1, 2]
    for codec in df['codec'].unique():
        codec_data = df[df['codec'] == codec]
        ax6.scatter(codec_data['bitrate'], codec_data['dnsmos_ovr'], label=codec, alpha=0.7)
    ax6.set_title('Overall vs Bitrate by Codec')
    ax6.set_xlabel('Bitrate (kbps)')
    ax6.set_ylabel('Overall Score')
    ax6.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, 'dnsmos_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {plot_file}")
    plt.close()

def generate_summary_report(df, output_dir):
    """Generate a comprehensive summary report."""
    print("\n=== GENERATING SUMMARY REPORT ===")
    
    report_file = os.path.join(output_dir, 'summary_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("DNSMOS AUDIO QUALITY ANALYSIS SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Basic statistics
        f.write(f"Total files analyzed: {len(df)}\n")
        f.write(f"Unique codecs: {df['codec'].nunique()}\n")
        f.write(f"Unique bitrates: {df['bitrate'].nunique()}\n")
        f.write(f"Codecs: {', '.join(df['codec'].unique())}\n")
        f.write(f"Bitrates: {', '.join(df['bitrate'].unique())}\n\n")
        
        # DNSMOS statistics
        f.write("DNSMOS STATISTICS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Mean Overall: {df['dnsmos_ovr'].mean():.3f}\n")
        f.write(f"Std Overall: {df['dnsmos_ovr'].std():.3f}\n")
        f.write(f"Min Overall: {df['dnsmos_ovr'].min():.3f}\n")
        f.write(f"Max Overall: {df['dnsmos_ovr'].max():.3f}\n\n")
        
        f.write(f"Mean Signal: {df['dnsmos_sig'].mean():.3f}\n")
        f.write(f"Std Signal: {df['dnsmos_sig'].std():.3f}\n")
        f.write(f"Min Signal: {df['dnsmos_sig'].min():.3f}\n")
        f.write(f"Max Signal: {df['dnsmos_sig'].max():.3f}\n\n")
        
        f.write(f"Mean Background: {df['dnsmos_bak'].mean():.3f}\n")
        f.write(f"Std Background: {df['dnsmos_bak'].std():.3f}\n")
        f.write(f"Min Background: {df['dnsmos_bak'].min():.3f}\n")
        f.write(f"Max Background: {df['dnsmos_bak'].max():.3f}\n\n")
        
        # Best and worst codecs
        codec_means = df.groupby('codec')['dnsmos_ovr'].mean().sort_values(ascending=False)
        f.write("CODEC RANKING BY DNSMOS OVERALL\n")
        f.write("-" * 35 + "\n")
        for i, (codec, ovr) in enumerate(codec_means.items(), 1):
            f.write(f"{i}. {codec}: {ovr:.3f}\n")
        
        f.write(f"\nBest codec: {codec_means.index[0]} ({codec_means.iloc[0]:.3f})\n")
        f.write(f"Worst codec: {codec_means.index[-1]} ({codec_means.iloc[-1]:.3f})\n\n")
        
        # Bitrate analysis
        bitrate_means = df.groupby('bitrate')['dnsmos_ovr'].mean().sort_values(ascending=False)
        f.write("BITRATE RANKING BY DNSMOS OVERALL\n")
        f.write("-" * 35 + "\n")
        for i, (bitrate, ovr) in enumerate(bitrate_means.items(), 1):
            f.write(f"{i}. {bitrate}: {ovr:.3f}\n")
    
    print(f"Summary report saved to: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze DNSMOS results')
    parser.add_argument('--results_dir', required=True, help='Directory with DNSMOS results')
    parser.add_argument('--output_dir', required=True, help='Output directory for analysis')
    
    args = parser.parse_args()
    
    print("=== DNSMOS RESULTS ANALYSIS ===")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    df = load_dnsmos_results(args.results_dir)
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



#!/usr/bin/env python3
"""
Comprehensive analysis script for all audio quality metrics.
Analyzes NISQA v2.0, DNSMOS, ViSQOL, Speaker Similarity, and Prosody results.
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

def load_all_metrics_results(results_dir):
    """Load all metrics results from CSV file."""
    csv_file = os.path.join(results_dir, 'all_metrics_results.csv')
    if not os.path.exists(csv_file):
        print(f"ERROR: Results file not found: {csv_file}")
        return None
    
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} comprehensive metric results")
    return df

def analyze_metric_correlations(df, output_dir):
    """Analyze correlations between different metrics."""
    print("\n=== METRIC CORRELATIONS ===")
    
    # Select numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    metric_cols = [col for col in numeric_cols if any(metric in col for metric in 
                   ['nisqa', 'dnsmos', 'visqol', 'speaker', 'f0'])]
    
    if len(metric_cols) < 2:
        print("Not enough metrics for correlation analysis")
        return
    
    # Calculate correlations
    corr_matrix = df[metric_cols].corr()
    
    # Save correlation matrix
    corr_file = os.path.join(output_dir, 'metric_correlations.csv')
    corr_matrix.to_csv(corr_file)
    print(f"Correlation matrix saved to: {corr_file}")
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f')
    plt.title('Audio Quality Metrics Correlation Matrix')
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, 'metric_correlations.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Correlation heatmap saved to: {plot_file}")
    plt.close()

def analyze_by_codec_comprehensive(df, output_dir):
    """Comprehensive analysis by codec."""
    print("\n=== COMPREHENSIVE CODEC ANALYSIS ===")
    
    # Group by codec and calculate statistics for all metrics
    codec_stats = df.groupby('codec').agg({
        'nisqa_mos': ['mean', 'std', 'min', 'max'],
        'nisqa_noisiness': ['mean', 'std'],
        'nisqa_discontinuity': ['mean', 'std'],
        'nisqa_coloration': ['mean', 'std'],
        'nisqa_loudness': ['mean', 'std'],
        'dnsmos_sig': ['mean', 'std'],
        'dnsmos_bak': ['mean', 'std'],
        'dnsmos_ovr': ['mean', 'std'],
        'visqol': ['mean', 'std'],
        'speaker_similarity': ['mean', 'std'],
        'f0_rmse': ['mean', 'std']
    }).round(3)
    
    print("Comprehensive Codec Statistics:")
    print(codec_stats)
    
    # Save comprehensive codec analysis
    codec_file = os.path.join(output_dir, 'comprehensive_codec_analysis.csv')
    codec_stats.to_csv(codec_file)
    print(f"Comprehensive codec analysis saved to: {codec_file}")
    
    return codec_stats

def create_comprehensive_visualizations(df, output_dir):
    """Create comprehensive visualization plots."""
    print("\n=== CREATING COMPREHENSIVE VISUALIZATIONS ===")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle('Comprehensive Audio Quality Metrics Analysis', fontsize=16)
    
    # Plot 1: NISQA MOS by Codec
    ax1 = axes[0, 0]
    df.boxplot(column='nisqa_mos', by='codec', ax=ax1)
    ax1.set_title('NISQA MOS by Codec')
    ax1.set_xlabel('Codec')
    ax1.set_ylabel('MOS Score')
    
    # Plot 2: DNSMOS Overall by Codec
    ax2 = axes[0, 1]
    df.boxplot(column='dnsmos_ovr', by='codec', ax=ax2)
    ax2.set_title('DNSMOS Overall by Codec')
    ax2.set_xlabel('Codec')
    ax2.set_ylabel('DNSMOS Overall Score')
    
    # Plot 3: ViSQOL by Codec
    ax3 = axes[0, 2]
    df.boxplot(column='visqol', by='codec', ax=ax3)
    ax3.set_title('ViSQOL by Codec')
    ax3.set_xlabel('Codec')
    ax3.set_ylabel('ViSQOL Score')
    
    # Plot 4: NISQA Noisiness by Codec
    ax4 = axes[1, 0]
    df.boxplot(column='nisqa_noisiness', by='codec', ax=ax4)
    ax4.set_title('NISQA Noisiness by Codec')
    ax4.set_xlabel('Codec')
    ax4.set_ylabel('Noisiness Score')
    
    # Plot 5: NISQA Coloration by Codec
    ax5 = axes[1, 1]
    df.boxplot(column='nisqa_coloration', by='codec', ax=ax5)
    ax5.set_title('NISQA Coloration by Codec')
    ax5.set_xlabel('Codec')
    ax5.set_ylabel('Coloration Score')
    
    # Plot 6: Speaker Similarity by Codec
    ax6 = axes[1, 2]
    df.boxplot(column='speaker_similarity', by='codec', ax=ax6)
    ax6.set_title('Speaker Similarity by Codec')
    ax6.set_xlabel('Codec')
    ax6.set_ylabel('Speaker Similarity')
    
    # Plot 7: MOS vs Bitrate Scatter
    ax7 = axes[2, 0]
    for codec in df['codec'].unique():
        codec_data = df[df['codec'] == codec]
        ax7.scatter(codec_data['bitrate'], codec_data['nisqa_mos'], label=codec, alpha=0.7)
    ax7.set_title('NISQA MOS vs Bitrate by Codec')
    ax7.set_xlabel('Bitrate (kbps)')
    ax7.set_ylabel('MOS Score')
    ax7.legend()
    
    # Plot 8: DNSMOS vs NISQA MOS Scatter
    ax8 = axes[2, 1]
    ax8.scatter(df['dnsmos_ovr'], df['nisqa_mos'], alpha=0.6)
    ax8.set_title('DNSMOS vs NISQA MOS')
    ax8.set_xlabel('DNSMOS Overall')
    ax8.set_ylabel('NISQA MOS')
    
    # Plot 9: Quality Dimensions Heatmap
    ax9 = axes[2, 2]
    quality_dims = ['nisqa_mos', 'nisqa_noisiness', 'nisqa_discontinuity', 
                   'nisqa_coloration', 'nisqa_loudness']
    codec_quality = df.groupby('codec')[quality_dims].mean()
    sns.heatmap(codec_quality.T, annot=True, cmap='RdYlBu_r', ax=ax9)
    ax9.set_title('Quality Dimensions by Codec')
    
    plt.tight_layout()
    
    # Save comprehensive plot
    plot_file = os.path.join(output_dir, 'comprehensive_metrics_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Comprehensive visualization saved to: {plot_file}")
    plt.close()

def generate_comprehensive_report(df, output_dir):
    """Generate a comprehensive summary report."""
    print("\n=== GENERATING COMPREHENSIVE REPORT ===")
    
    report_file = os.path.join(output_dir, 'comprehensive_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("COMPREHENSIVE AUDIO QUALITY METRICS ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        # Basic statistics
        f.write(f"Total files analyzed: {len(df)}\n")
        f.write(f"Unique codecs: {df['codec'].nunique()}\n")
        f.write(f"Unique bitrates: {df['bitrate'].nunique()}\n")
        f.write(f"Codecs: {', '.join(df['codec'].unique())}\n")
        f.write(f"Bitrates: {', '.join(df['bitrate'].unique())}\n\n")
        
        # NISQA v2.0 statistics
        f.write("NISQA v2.0 QUALITY STATISTICS\n")
        f.write("-" * 35 + "\n")
        f.write(f"Mean MOS: {df['nisqa_mos'].mean():.3f}\n")
        f.write(f"Std MOS: {df['nisqa_mos'].std():.3f}\n")
        f.write(f"Min MOS: {df['nisqa_mos'].min():.3f}\n")
        f.write(f"Max MOS: {df['nisqa_mos'].max():.3f}\n\n")
        
        # DNSMOS statistics
        f.write("DNSMOS STATISTICS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Mean Overall: {df['dnsmos_ovr'].mean():.3f}\n")
        f.write(f"Mean Signal: {df['dnsmos_sig'].mean():.3f}\n")
        f.write(f"Mean Background: {df['dnsmos_bak'].mean():.3f}\n\n")
        
        # ViSQOL statistics
        f.write("ViSQOL STATISTICS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Mean ViSQOL: {df['visqol'].mean():.3f}\n")
        f.write(f"Std ViSQOL: {df['visqol'].std():.3f}\n\n")
        
        # Codec ranking by NISQA MOS
        codec_means = df.groupby('codec')['nisqa_mos'].mean().sort_values(ascending=False)
        f.write("CODEC RANKING BY NISQA MOS\n")
        f.write("-" * 30 + "\n")
        for i, (codec, mos) in enumerate(codec_means.items(), 1):
            f.write(f"{i}. {codec}: {mos:.3f}\n")
        
        f.write(f"\nBest codec: {codec_means.index[0]} ({codec_means.iloc[0]:.3f})\n")
        f.write(f"Worst codec: {codec_means.index[-1]} ({codec_means.iloc[-1]:.3f})\n\n")
        
        # Quality dimensions analysis
        f.write("NISQA v2.0 QUALITY DIMENSIONS\n")
        f.write("-" * 30 + "\n")
        for metric in ['nisqa_noisiness', 'nisqa_discontinuity', 'nisqa_coloration', 'nisqa_loudness']:
            f.write(f"{metric.replace('nisqa_', '').capitalize()}:\n")
            f.write(f"  Mean: {df[metric].mean():.3f}\n")
            f.write(f"  Std: {df[metric].std():.3f}\n")
            f.write(f"  Range: {df[metric].min():.3f} - {df[metric].max():.3f}\n\n")
    
    print(f"Comprehensive report saved to: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze comprehensive audio quality metrics')
    parser.add_argument('--results_dir', required=True, help='Directory with all metrics results')
    parser.add_argument('--output_dir', required=True, help='Output directory for analysis')
    
    args = parser.parse_args()
    
    print("=== COMPREHENSIVE AUDIO QUALITY METRICS ANALYSIS ===")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    df = load_all_metrics_results(args.results_dir)
    if df is None:
        return
    
    # Run comprehensive analyses
    analyze_metric_correlations(df, args.output_dir)
    analyze_by_codec_comprehensive(df, args.output_dir)
    create_comprehensive_visualizations(df, args.output_dir)
    generate_comprehensive_report(df, args.output_dir)
    
    print(f"\n=== COMPREHENSIVE ANALYSIS COMPLETE ===")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()



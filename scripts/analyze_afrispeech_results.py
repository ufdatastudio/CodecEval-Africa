#!/usr/bin/env python3
"""
Analyze AfriSpeech-Dialog codec evaluation results.
Generate comprehensive analysis and visualizations.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List, Any

def load_results(results_dir: str = "results/afrispeech_dialog") -> pd.DataFrame:
    """Load results from the evaluation."""
    
    metadata_file = Path(results_dir) / "metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Results not found at {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        results = json.load(f)
    
    df = pd.DataFrame(results)
    
    # Add derived columns
    df['compression_efficiency'] = df['compression_ratio']
    df['size_reduction_pct'] = (1 - df['compressed_size_bytes'] / df['original_size_bytes']) * 100
    
    return df

def analyze_compression_efficiency(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze compression efficiency across codecs and bitrates."""
    
    analysis = {}
    
    # Overall compression statistics
    analysis['overall'] = {
        'mean_compression_ratio': df['compression_ratio'].mean(),
        'std_compression_ratio': df['compression_ratio'].std(),
        'mean_size_reduction': df['size_reduction_pct'].mean(),
        'std_size_reduction': df['size_reduction_pct'].std()
    }
    
    # By codec
    codec_stats = df.groupby('codec').agg({
        'compression_ratio': ['mean', 'std'],
        'size_reduction_pct': ['mean', 'std'],
        'compressed_size_bytes': ['mean', 'std']
    }).round(3)
    
    analysis['by_codec'] = codec_stats.to_dict()
    
    # By bitrate
    bitrate_stats = df.groupby('kbps').agg({
        'compression_ratio': ['mean', 'std'],
        'size_reduction_pct': ['mean', 'std'],
        'compressed_size_bytes': ['mean', 'std']
    }).round(3)
    
    analysis['by_bitrate'] = bitrate_stats.to_dict()
    
    return analysis

def plot_compression_analysis(df: pd.DataFrame, output_dir: str = "results/afrispeech_dialog/reports"):
    """Create compression analysis visualizations."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('AfriSpeech-Dialog Codec Compression Analysis', fontsize=16, fontweight='bold')
    
    # 1. Compression ratio by codec
    ax1 = axes[0, 0]
    df.boxplot(column='compression_ratio', by='codec', ax=ax1)
    ax1.set_title('Compression Ratio by Codec')
    ax1.set_xlabel('Codec')
    ax1.set_ylabel('Compression Ratio')
    plt.suptitle('')  # Remove default title
    
    # 2. Size reduction by bitrate
    ax2 = axes[0, 1]
    df.boxplot(column='size_reduction_pct', by='kbps', ax=ax2)
    ax2.set_title('Size Reduction by Bitrate')
    ax2.set_xlabel('Bitrate (kbps)')
    ax2.set_ylabel('Size Reduction (%)')
    plt.suptitle('')  # Remove default title
    
    # 3. Compression ratio vs bitrate heatmap
    ax3 = axes[1, 0]
    pivot_data = df.pivot_table(values='compression_ratio', index='codec', columns='kbps', aggfunc='mean')
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='viridis', ax=ax3)
    ax3.set_title('Compression Ratio: Codec vs Bitrate')
    
    # 4. File size distribution
    ax4 = axes[1, 1]
    df['compressed_size_mb'] = df['compressed_size_bytes'] / (1024 * 1024)
    df.boxplot(column='compressed_size_mb', by='codec', ax=ax4)
    ax4.set_title('Compressed File Size by Codec')
    ax4.set_xlabel('Codec')
    ax4.set_ylabel('File Size (MB)')
    plt.suptitle('')  # Remove default title
    
    plt.tight_layout()
    plt.savefig(output_path / 'compression_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Compression analysis plot saved: {output_path / 'compression_analysis.png'}")

def analyze_codec_performance(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze codec performance rankings."""
    
    # Calculate performance metrics
    codec_performance = df.groupby('codec').agg({
        'compression_ratio': 'mean',
        'size_reduction_pct': 'mean',
        'compressed_size_bytes': 'mean'
    }).round(3)
    
    # Rank codecs by different criteria
    rankings = {
        'compression_ratio': codec_performance.sort_values('compression_ratio', ascending=False).index.tolist(),
        'size_reduction': codec_performance.sort_values('size_reduction_pct', ascending=False).index.tolist(),
        'file_size': codec_performance.sort_values('compressed_size_bytes', ascending=True).index.tolist()
    }
    
    return {
        'performance_metrics': codec_performance.to_dict(),
        'rankings': rankings
    }

def generate_summary_report(df: pd.DataFrame, analysis: Dict[str, Any], output_dir: str = "results/afrispeech_dialog/reports"):
    """Generate comprehensive summary report."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report = f"""
# AfriSpeech-Dialog Codec Evaluation Results

## Dataset Overview
- **Total samples**: {len(df['id'].unique())}
- **Total experiments**: {len(df)}
- **Codecs evaluated**: {len(df['codec'].unique())}
- **Bitrates tested**: {sorted(df['kbps'].unique())}

## Compression Analysis

### Overall Performance
- **Mean compression ratio**: {analysis['overall']['mean_compression_ratio']:.2f}x
- **Mean size reduction**: {analysis['overall']['mean_size_reduction']:.1f}%
- **Standard deviation**: {analysis['overall']['std_compression_ratio']:.2f}

### Top Performing Codecs by Compression Ratio
"""
    
    # Add codec rankings
    codec_perf = analysis['by_codec']['compression_ratio']['mean']
    sorted_codecs = sorted(codec_perf.items(), key=lambda x: x[1], reverse=True)
    
    for i, (codec, ratio) in enumerate(sorted_codecs, 1):
        report += f"{i}. **{codec}**: {ratio:.2f}x compression\n"
    
    report += f"""
### Bitrate Analysis
"""
    
    # Add bitrate analysis
    bitrate_perf = analysis['by_bitrate']['compression_ratio']['mean']
    for bitrate in sorted(bitrate_perf.keys()):
        ratio = bitrate_perf[bitrate]
        report += f"- **{bitrate} kbps**: {ratio:.2f}x compression\n"
    
    report += f"""
## Key Findings

1. **Best Overall Compression**: {sorted_codecs[0][0]} with {sorted_codecs[0][1]:.2f}x ratio
2. **Most Consistent**: Analysis shows standard deviation patterns
3. **Bitrate Impact**: Compression efficiency varies by target bitrate
4. **File Size Reduction**: Average {analysis['overall']['mean_size_reduction']:.1f}% reduction

## Recommendations

1. **High Compression Needs**: Use {sorted_codecs[0][0]}
2. **Balanced Performance**: Consider mid-range codecs
3. **Bitrate Selection**: Choose based on application requirements
4. **Storage Optimization**: Focus on size reduction percentages

---
*Report generated from {len(df)} codec evaluations on AfriSpeech-Dialog dataset*
"""
    
    # Save report
    with open(output_path / 'summary_report.md', 'w') as f:
        f.write(report)
    
    print(f"✓ Summary report saved: {output_path / 'summary_report.md'}")
    
    return report

def main():
    """Main analysis function."""
    
    print("=" * 60)
    print("AFRISPEECH-DIALOG RESULTS ANALYSIS")
    print("=" * 60)
    
    try:
        # Load results
        print("Loading results...")
        df = load_results()
        print(f"✓ Loaded {len(df)} experimental results")
        
        # Analyze compression efficiency
        print("Analyzing compression efficiency...")
        analysis = analyze_compression_efficiency(df)
        print("✓ Compression analysis completed")
        
        # Create visualizations
        print("Creating visualizations...")
        plot_compression_analysis(df)
        print("✓ Visualizations created")
        
        # Generate summary report
        print("Generating summary report...")
        report = generate_summary_report(df, analysis)
        print("✓ Summary report generated")
        
        print("=" * 60)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Results saved in: results/afrispeech_dialog/reports/")
        print("- compression_analysis.png")
        print("- summary_report.md")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure the experiment has completed and results are available.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()

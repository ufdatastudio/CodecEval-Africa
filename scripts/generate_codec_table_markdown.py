#!/usr/bin/env python3
"""
Generate Markdown table format from codec statistics
"""

import json
from pathlib import Path

BASE_DIR = Path('/orange/ufdatastudios/c.okocha/CodecEval-Africa')
OUTPUTS_DIR = BASE_DIR / 'outputs'
STATS_PATH = OUTPUTS_DIR / 'codec_stats.json'
MD_PATH = OUTPUTS_DIR / 'codec_stats_table.md'

def format_value(val):
    """Format value for table display"""
    if val is None or val == '':
        return '-'
    if isinstance(val, float):
        return f"{val:.2f}" if val >= 0.01 else f"{val:.4f}"
    return str(val)

def main():
    # Load stats
    with open(STATS_PATH, 'r') as f:
        stats_data = json.load(f)
    
    stats = stats_data['stats']
    rows = []
    
    for codec_name, variants in stats.items():
        for variant_label, data in variants.items():
            rows.append({
                'codec': codec_name,
                'variant': variant_label,
                'num_files': data['num_files'],
                'avg_duration_sec': data['avg_duration_sec'],
                'total_duration_sec': data['total_duration_sec'],
                'avg_file_size_mb': data['avg_file_size_mb'],
                'sample_rate_hz': data['typical_sample_rate'],
                'sample_rates': ', '.join(map(str, data['sample_rates'])),
                'channels': ', '.join(map(str, data['channels'])),
                'bits_per_sample': ', '.join(map(str, data['bits_per_sample'])),
                'token_rate': data.get('token_rate', ''),
                'semantic_vocab_size': data.get('semantic_vocab_size', ''),
                'codec_bitrate_kbps': data.get('codec_bitrate_kbps', ''),
                'codec_bitrate_label': data.get('codec_bitrate_label', ''),
                'bandwidth_id': data.get('bandwidth_id', ''),
                'tokens_per_sec': data.get('tokens_per_sec', ''),
                'target_sample_rate': data.get('target_sample_rate', ''),
            })
    
    # Generate Markdown table
    md_lines = [
        "# Codec Statistics Summary",
        "",
        f"**Total Codecs:** {stats_data['metadata']['total_codecs']}  ",
        f"**Total Variants:** {stats_data['metadata']['total_variants']}  ",
        f"**Total Files:** {stats_data['metadata']['total_files']}  ",
        "",
        "## Summary Table",
        "",
        "| Codec | Variant | Files | Avg Duration (s) | Total Duration (s) | Avg File Size (MB) | Sample Rate (Hz) | Bitrate | Token Rate | Vocab Size | Bandwidth ID | Tokens/sec |",
        "|-------|---------|-------|-------------------|-------------------|-------------------|------------------|---------|------------|------------|--------------|------------|"
    ]
    
    for row in rows:
        bitrate = row['codec_bitrate_label'] or row['codec_bitrate_kbps'] or '-'
        if bitrate and bitrate != '-' and not str(bitrate).endswith('kbps'):
            bitrate = f"{bitrate} kbps"
        
        md_lines.append(
            f"| {row['codec']} | {row['variant']} | {row['num_files']} | "
            f"{format_value(row['avg_duration_sec'])} | {format_value(row['total_duration_sec'])} | "
            f"{format_value(row['avg_file_size_mb'])} | {row['sample_rate_hz']} | "
            f"{bitrate} | {format_value(row['token_rate'])} | {format_value(row['semantic_vocab_size'])} | "
            f"{format_value(row['bandwidth_id'])} | {format_value(row['tokens_per_sec'])} |"
        )
    
    # Add detailed section
    md_lines.extend([
        "",
        "## Detailed Information",
        ""
    ])
    
    for codec_name, variants in stats.items():
        md_lines.append(f"### {codec_name}")
        md_lines.append("")
        for variant_label, data in variants.items():
            md_lines.append(f"#### {variant_label}")
            md_lines.append("")
            md_lines.append(f"- **Files:** {data['num_files']}")
            md_lines.append(f"- **Average Duration:** {data['avg_duration_sec']:.2f} seconds")
            md_lines.append(f"- **Total Duration:** {data['total_duration_sec']:.2f} seconds ({data['total_duration_sec']/3600:.2f} hours)")
            md_lines.append(f"- **Average File Size:** {data['avg_file_size_mb']:.4f} MB")
            md_lines.append(f"- **Sample Rate:** {data['typical_sample_rate']} Hz")
            md_lines.append(f"- **Channels:** {', '.join(map(str, data['channels']))}")
            md_lines.append(f"- **Bits per Sample:** {', '.join(map(str, data['bits_per_sample']))}")
            
            # Codec-specific
            if 'token_rate' in data:
                md_lines.append(f"- **Token Rate:** {data['token_rate']}")
            if 'semantic_vocab_size' in data:
                md_lines.append(f"- **Semantic Vocab Size:** {data['semantic_vocab_size']}")
            if 'codec_bitrate_label' in data:
                md_lines.append(f"- **Codec Bitrate:** {data['codec_bitrate_label']}")
            if 'bandwidth_id' in data:
                md_lines.append(f"- **Bandwidth ID:** {data['bandwidth_id']}")
            if 'tokens_per_sec' in data:
                md_lines.append(f"- **Tokens per Second:** {data['tokens_per_sec']}")
            if 'target_sample_rate' in data:
                md_lines.append(f"- **Target Sample Rate:** {data['target_sample_rate']} Hz")
            
            md_lines.append("")
    
    # Write to file
    with open(MD_PATH, 'w') as f:
        f.write('\n'.join(md_lines))
    
    print(f"Generated Markdown table: {MD_PATH}")
    print(f"  Total rows: {len(rows)}")

if __name__ == '__main__':
    main()


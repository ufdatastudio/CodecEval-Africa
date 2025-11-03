#!/usr/bin/env python3
import os
import json
from pathlib import Path
import soundfile as sf

BASE_DIR = Path('/orange/ufdatastudios/c.okocha/CodecEval-Africa')
OUTPUTS_DIR = BASE_DIR / 'outputs'
MANIFEST_PATH = OUTPUTS_DIR / 'codec_manifest.json'

# Known folder roots
SEMANTI_ROOT = OUTPUTS_DIR / 'SemantiCodec_outputs'
UNICODEC_ROOT = OUTPUTS_DIR / 'UniCodec_outputs'
WT_ROOT = OUTPUTS_DIR / 'WavTokenizer_outputs'
APCODEC_ROOT = OUTPUTS_DIR / 'APCodec_outputs'

# Helpers

def wav_info(path: Path):
    try:
        with sf.SoundFile(str(path), 'r') as f:
            sr = f.samplerate
            ch = f.channels
            subtype = f.subtype
            duration = len(f) / sr if sr > 0 else 0.0
        size_bytes = path.stat().st_size
        # bits per sample from subtype
        bits = 16
        if '24' in subtype:
            bits = 24
        elif '32' in subtype:
            bits = 32
        return {
            'sample_rate': sr,
            'channels': ch,
            'subtype': subtype,
            'bits_per_sample': bits,
            'duration_sec': duration,
            'file_size_bytes': size_bytes,
        }
    except Exception as e:
        return {'error': str(e)}

# Static maps for SemantiCodec from script config
SEMANTIC_LABEL_MAP = {
    '0.31kbps': {'token_rate': 25, 'semantic_vocab_size': 4096},
    '0.63kbps': {'token_rate': 50, 'semantic_vocab_size': 4096},
    '1.25kbps': {'token_rate': 100, 'semantic_vocab_size': 4096},
    '0.33kbps': {'token_rate': 25, 'semantic_vocab_size': 8192},
    '0.68kbps': {'token_rate': 50, 'semantic_vocab_size': 16384},
    '1.40kbps': {'token_rate': 100, 'semantic_vocab_size': 32768},
}

# Static hints for WavTokenizer models (based on names we used)
WT_MODEL_HINTS = {
    'WavTokenizer_small-600-24k-4096': {'tokens_per_sec': 40},
    'WavTokenizer_small-320-24k-4096': {'tokens_per_sec': 75},
    'WavTokenizer_large-unify-40token': {'tokens_per_sec': 40},
    'WavTokenizer_large-speech-75token': {'tokens_per_sec': 75},
    'WavTokenizer_medium-speech-75token': {'tokens_per_sec': 75},
    'WavTokenizer_medium-music-audio-75token': {'tokens_per_sec': 75},
}

# Parse APCodec label like out_6kbps_24khz or out_12kbps_48khz

def parse_apcodec_label(label: str):
    # expect e.g., out_6kbps_24khz
    parts = label.replace('out_', '').split('_')
    info = {}
    for p in parts:
        if p.endswith('kbps'):
            try:
                info['codec_bitrate_kbps'] = float(p.replace('kbps', ''))
            except ValueError:
                pass
        if p.endswith('khz'):
            try:
                info['target_sample_rate'] = int(p.replace('khz', '')) * 1000
            except ValueError:
                pass
    return info

# UniCodec bandwidth id from folder name patterns

def parse_unicodec_label(label: str):
    # handle out_bwX or out_*. If numeric at end, store as bandwidth_id
    out = {}
    if label.startswith('out_bw'):
        try:
            out['bandwidth_id'] = int(label.replace('out_bw', ''))
        except ValueError:
            pass
    return out


def scan_folder(root: Path, codec_name: str):
    entries = []
    if not root.exists():
        return entries
    # Iterate subfolders as variants; if files directly inside, treat root as variant
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if not subdirs:
        subdirs = [root]
    for sub in subdirs:
        variant_label = sub.name
        for wav in sub.rglob('*.wav'):
            meta = wav_info(wav)
            rec = {
                'codec': codec_name,
                'variant': variant_label,
                'path': str(wav),
            }
            rec.update(meta)
            # enrich per codec
            if codec_name == 'SemantiCodec':
                if variant_label in SEMANTIC_LABEL_MAP:
                    rec.update(SEMANTIC_LABEL_MAP[variant_label])
                    rec['codec_bitrate_label'] = variant_label
            elif codec_name == 'APCodec':
                rec.update(parse_apcodec_label(variant_label))
            elif codec_name == 'UniCodec':
                rec.update(parse_unicodec_label(variant_label))
            elif codec_name == 'WavTokenizer':
                hint = WT_MODEL_HINTS.get(variant_label)
                if hint:
                    rec.update(hint)
            entries.append(rec)
    return entries


def main():
    all_entries = []
    # Scan each known codec
    all_entries += scan_folder(SEMANTI_ROOT, 'SemantiCodec')
    all_entries += scan_folder(UNICODEC_ROOT, 'UniCodec')
    all_entries += scan_folder(WT_ROOT, 'WavTokenizer')
    all_entries += scan_folder(APCODEC_ROOT, 'APCodec')

    # Build summary per codec/variant
    summary = {}
    for e in all_entries:
        key = (e['codec'], e['variant'])
        s = summary.setdefault(e['codec'], {})
        v = s.setdefault(e['variant'], {
            'num_files': 0,
            'total_duration_sec': 0.0,
            'sample_rates': set(),
            'channels': set(),
            'bits_per_sample': set(),
        })
        v['num_files'] += 1
        v['total_duration_sec'] += e.get('duration_sec', 0.0) or 0.0
        if 'sample_rate' in e:
            v['sample_rates'].add(e['sample_rate'])
        if 'channels' in e:
            v['channels'].add(e['channels'])
        if 'bits_per_sample' in e:
            v['bits_per_sample'].add(e['bits_per_sample'])
        # carry some known attributes if present (first one wins)
        for attr in ['token_rate', 'semantic_vocab_size', 'bandwidth_id', 'tokens_per_sec', 'codec_bitrate_kbps', 'target_sample_rate']:
            if attr in e and attr not in v:
                v[attr] = e[attr]

    # convert sets to sorted lists
    for codec, variants in summary.items():
        for vlabel, v in variants.items():
            v['sample_rates'] = sorted(list(v['sample_rates']))
            v['channels'] = sorted(list(v['channels']))
            v['bits_per_sample'] = sorted(list(v['bits_per_sample']))

    manifest = {
        'base_dir': str(BASE_DIR),
        'outputs_dir': str(OUTPUTS_DIR),
        'num_records': len(all_entries),
        'files': all_entries,
        'summary': summary,
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest: {MANIFEST_PATH} ({len(all_entries)} records)")


if __name__ == '__main__':
    main()

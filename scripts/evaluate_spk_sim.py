#!/usr/bin/env python3
"""Speaker similarity evaluation for paired WAV directories.

Uses MFCC-based embeddings + cosine similarity and writes JSON summaries.
"""

import argparse
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np
import soundfile as sf
from scipy.signal import stft, resample_poly


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _safe_float(value):
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _load_mono_16k(audio_path: Path):
    audio, sr = sf.read(str(audio_path), always_2d=False)
    if audio is None:
        return None
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if sr != 16000:
        gcd = math.gcd(int(sr), 16000)
        up = 16000 // gcd
        down = int(sr) // gcd
        audio = resample_poly(audio, up, down).astype(np.float32)
    if audio is None or len(audio) == 0:
        return None
    return audio


def _extract_mfcc_embedding(audio: np.ndarray):
    if audio is None or len(audio) == 0:
        return None

    n_fft = 400
    hop = 160
    _, _, zxx = stft(audio, fs=16000, nperseg=n_fft, noverlap=n_fft - hop, nfft=n_fft, boundary=None)
    power = np.abs(zxx) ** 2
    if power is None or power.size == 0:
        return None

    # Lightweight log-spectral embedding (13 bins grouped from low to high frequencies)
    n_bins = power.shape[0]
    groups = np.array_split(np.arange(n_bins), 13)
    feats = []
    for g in groups:
        band = np.log(np.mean(power[g, :], axis=0) + 1e-8)
        feats.append(np.mean(band))
        feats.append(np.std(band))

    # Delta-like temporal dynamics from frame-wise energy
    energy = np.log(np.mean(power, axis=0) + 1e-8)
    if energy.size > 1:
        delta = np.diff(energy)
        feats.append(float(np.mean(delta)))
        feats.append(float(np.std(delta)))
    else:
        feats.extend([0.0, 0.0])

    emb = np.asarray(feats, dtype=np.float32)
    return emb


def _cosine_similarity(emb1: np.ndarray, emb2: np.ndarray):
    denom = (np.linalg.norm(emb1) * np.linalg.norm(emb2)) + 1e-8
    sim = float(np.dot(emb1, emb2) / denom)
    # Map from [-1, 1] -> [0, 1]
    return float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0))


def _euclidean_distance(emb1: np.ndarray, emb2: np.ndarray):
    return float(np.linalg.norm(emb1 - emb2))


def _correlation_similarity(emb1: np.ndarray, emb2: np.ndarray):
    corr = np.corrcoef(emb1, emb2)[0, 1]
    if np.isnan(corr):
        return None
    return float(np.clip((float(corr) + 1.0) / 2.0, 0.0, 1.0))


def evaluate_spk_sim(ref_dir: Path, deg_dir: Path, output_file: Path, model_type: str = 'mfcc', limit: int = -1):
    ref_files = {p.name: p for p in sorted(ref_dir.glob('*.wav'))}
    deg_files = {p.name: p for p in sorted(deg_dir.glob('*.wav'))}

    matched_names = sorted(set(ref_files.keys()).intersection(deg_files.keys()))
    if limit > 0:
        matched_names = matched_names[:limit]

    logger.info('Reference files: %d', len(ref_files))
    logger.info('Degraded files:  %d', len(deg_files))
    logger.info('Matched files:   %d', len(matched_names))

    if model_type != 'mfcc':
        logger.warning("Only 'mfcc' is implemented in this runner; falling back to mfcc")
        model_type = 'mfcc'

    rows = []
    cosine_scores = []

    for name in matched_names:
        ref_path = ref_files[name]
        deg_path = deg_files[name]
        try:
            ref_audio = _load_mono_16k(ref_path)
            deg_audio = _load_mono_16k(deg_path)
            ref_emb = _extract_mfcc_embedding(ref_audio)
            deg_emb = _extract_mfcc_embedding(deg_audio)

            if ref_emb is None or deg_emb is None:
                raise ValueError('Could not extract MFCC embedding')

            result = {
                'cosine_similarity': _cosine_similarity(ref_emb, deg_emb),
                'euclidean_distance': _euclidean_distance(ref_emb, deg_emb),
                'correlation': _correlation_similarity(ref_emb, deg_emb),
            }
        except Exception as exc:
            result = {
                'cosine_similarity': None,
                'euclidean_distance': None,
                'correlation': None,
                'error': str(exc),
            }

        cosine = _safe_float(result.get('cosine_similarity'))
        euclidean = _safe_float(result.get('euclidean_distance'))
        correlation = _safe_float(result.get('correlation'))

        row = {
            'filename': name,
            'cosine_similarity': cosine,
            'euclidean_distance': euclidean,
            'correlation': correlation,
        }
        if 'error' in result:
            row['error'] = str(result['error'])

        rows.append(row)
        if cosine is not None:
            cosine_scores.append(cosine)

    summary = {
        'metric': 'spk_sim',
        'model_type': model_type,
        'reference_dir': str(ref_dir),
        'degraded_dir': str(deg_dir),
        'timestamp': datetime.now().isoformat(),
        'num_reference_files': len(ref_files),
        'num_degraded_files': len(deg_files),
        'num_matched_files': len(matched_names),
        'num_scored_files': len(cosine_scores),
        'spk_sim': mean(cosine_scores) if cosine_scores else None,
        'spk_sim_std': stdev(cosine_scores) if len(cosine_scores) > 1 else 0.0 if len(cosine_scores) == 1 else None,
        'spk_sim_median': median(cosine_scores) if cosine_scores else None,
        'spk_sim_min': min(cosine_scores) if cosine_scores else None,
        'spk_sim_max': max(cosine_scores) if cosine_scores else None,
        'per_file': rows,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('w') as f:
        json.dump(summary, f, indent=2)

    logger.info('Saved SPK-SIM results to: %s', output_file)
    if summary['spk_sim'] is not None:
        logger.info('SPK-SIM mean: %.4f over %d files', summary['spk_sim'], summary['num_scored_files'])

    return summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate speaker similarity for paired WAV directories.')
    parser.add_argument('--ref-dir', required=True, help='Reference WAV directory')
    parser.add_argument('--deg-dir', required=True, help='Degraded WAV directory')
    parser.add_argument('--output-dir', default='results/SPK-SIM', help='Output directory')
    parser.add_argument('--output-name', required=True, help='Output JSON name without extension')
    parser.add_argument('--model-type', default='mfcc', choices=['mfcc'], help='Speaker model type')
    parser.add_argument('--limit', type=int, default=-1, help='Optional limit on number of matched files')
    args = parser.parse_args()

    ref_dir = Path(args.ref_dir)
    deg_dir = Path(args.deg_dir)
    output_file = Path(args.output_dir) / f"{args.output_name}.json"

    if not ref_dir.exists():
        raise FileNotFoundError(f'Reference directory not found: {ref_dir}')
    if not deg_dir.exists():
        raise FileNotFoundError(f'Degraded directory not found: {deg_dir}')

    evaluate_spk_sim(ref_dir, deg_dir, output_file, model_type=args.model_type, limit=args.limit)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""ACC/mAP evaluation via retrieval over reference embeddings.

For each degraded file, all reference files are ranked by cosine similarity.
The matching filename is treated as the relevant item.
"""

import argparse
import json
import logging
import math
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
from scipy.signal import stft, resample_poly

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'audio_quality_assessment'))

from speaker_similarity import SpeakerSimilarityScorer


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


def _extract_embedding(audio: np.ndarray):
    if audio is None or len(audio) == 0:
        return None

    n_fft = 400
    hop = 160
    _, _, zxx = stft(audio, fs=16000, nperseg=n_fft, noverlap=n_fft - hop, nfft=n_fft, boundary=None)
    power = np.abs(zxx) ** 2
    if power is None or power.size == 0:
        return None

    n_bins = power.shape[0]
    groups = np.array_split(np.arange(n_bins), 13)
    feats = []
    for group in groups:
        band = np.log(np.mean(power[group, :], axis=0) + 1e-8)
        feats.append(np.mean(band))
        feats.append(np.std(band))

    energy = np.log(np.mean(power, axis=0) + 1e-8)
    if energy.size > 1:
        delta = np.diff(energy)
        feats.append(float(np.mean(delta)))
        feats.append(float(np.std(delta)))
    else:
        feats.extend([0.0, 0.0])

    return np.asarray(feats, dtype=np.float32)


def _cosine_similarity(emb1: np.ndarray, emb2: np.ndarray):
    denom = (np.linalg.norm(emb1) * np.linalg.norm(emb2)) + 1e-8
    sim = float(np.dot(emb1, emb2) / denom)
    return float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0))


def _bootstrap_ci(values: np.ndarray, n_bootstrap: int = 200, seed: int = 42):
    if values is None or len(values) == 0 or n_bootstrap <= 1:
        return [float('nan'), float('nan')]
    rng = np.random.default_rng(seed)
    n = len(values)
    samples = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        samples.append(float(np.mean(values[idx])))
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def evaluate_acc_map(
    ref_dir: Path,
    deg_dir: Path,
    output_file: Path,
    model_type: str = 'mfcc',
    limit: int = -1,
    seed: int = 42,
    n_bootstrap: int = 200,
):
    ref_files = {p.name: p for p in sorted(ref_dir.glob('*.wav'))}
    deg_files = {p.name: p for p in sorted(deg_dir.glob('*.wav'))}

    matched_names = sorted(set(ref_files.keys()).intersection(deg_files.keys()))
    if limit > 0:
        matched_names = matched_names[:limit]

    logger.info('Reference files: %d', len(ref_files))
    logger.info('Degraded files:  %d', len(deg_files))
    logger.info('Matched files:   %d', len(matched_names))

    if len(matched_names) < 2:
        raise RuntimeError('Need at least 2 matched files for ACC/mAP retrieval evaluation')

    scorer = SpeakerSimilarityScorer(model_type=model_type)
    logger.info('ACC/mAP backend requested: %s (effective: %s)', model_type, scorer.model_type)
    if model_type != scorer.model_type:
        logger.warning('Requested backend %s fell back to %s', model_type, scorer.model_type)

    ref_emb = {}
    deg_emb = {}

    for name in matched_names:
        try:
            if scorer.model_type == 'mfcc':
                r = _extract_embedding(_load_mono_16k(ref_files[name]))
                d = _extract_embedding(_load_mono_16k(deg_files[name]))
            else:
                r = scorer._extract_speaker_embedding(str(ref_files[name]))
                d = scorer._extract_speaker_embedding(str(deg_files[name]))
        except Exception:
            r = None
            d = None

        if r is not None:
            ref_emb[name] = r
        if d is not None:
            deg_emb[name] = d

    usable = sorted(set(ref_emb.keys()).intersection(deg_emb.keys()))
    if len(usable) < 2:
        raise RuntimeError('Too few usable files after embedding extraction')

    hit_top1 = []
    reciprocal_ranks = []
    per_file = []

    for query_name in usable:
        query_emb = deg_emb[query_name]
        scored = []
        for cand_name in usable:
            sim = _cosine_similarity(ref_emb[cand_name], query_emb)
            scored.append((cand_name, sim))
        scored.sort(key=lambda x: x[1], reverse=True)

        rank = None
        for idx, (cand_name, _) in enumerate(scored, start=1):
            if cand_name == query_name:
                rank = idx
                break

        if rank is None:
            continue

        rr = 1.0 / float(rank)
        top1 = 1.0 if rank == 1 else 0.0

        hit_top1.append(top1)
        reciprocal_ranks.append(rr)

        top5 = [name for name, _ in scored[:5]]
        per_file.append({
            'file': query_name,
            'rank_of_correct': int(rank),
            'reciprocal_rank': float(rr),
            'top1_correct': bool(top1 == 1.0),
            'top5_candidates': top5,
        })

    if len(hit_top1) == 0:
        raise RuntimeError('No valid queries scored for ACC/mAP')

    top1_arr = np.asarray(hit_top1, dtype=np.float64)
    rr_arr = np.asarray(reciprocal_ranks, dtype=np.float64)
    acc = float(np.mean(top1_arr))
    map_score = float(np.mean(rr_arr))
    acc_ci = _bootstrap_ci(top1_arr, n_bootstrap=n_bootstrap, seed=seed)
    map_ci = _bootstrap_ci(rr_arr, n_bootstrap=n_bootstrap, seed=seed)

    summary = {
        'metric': 'acc_map',
        'metrics': {
            'acc': acc,
            'acc_percent': 100.0 * acc,
            'acc_ci95': acc_ci,
            'acc_percent_ci95': [None if np.isnan(x) else 100.0 * x for x in acc_ci],
            'map': map_score,
            'map_percent': 100.0 * map_score,
            'map_ci95': map_ci,
            'map_percent_ci95': [None if np.isnan(x) else 100.0 * x for x in map_ci],
            'mrr': map_score,
        },
        'model_type': scorer.model_type,
        'model_type_requested': model_type,
        'reference_dir': str(ref_dir),
        'degraded_dir': str(deg_dir),
        'timestamp': datetime.now().isoformat(),
        'num_reference_files': len(ref_files),
        'num_degraded_files': len(deg_files),
        'num_matched_files': len(matched_names),
        'num_usable_files': len(usable),
        'bootstrap_samples': int(n_bootstrap),
        'num_queries': int(len(hit_top1)),
        'per_file': per_file,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('w') as f:
        json.dump(summary, f, indent=2)

    logger.info('Saved ACC/mAP results to: %s', output_file)
    logger.info('ACC: %.4f (%.2f%%), mAP: %.4f (%.2f%%)', acc, 100.0 * acc, map_score, 100.0 * map_score)
    return summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate ACC/mAP retrieval for paired WAV directories.')
    parser.add_argument('--ref-dir', required=True, help='Reference WAV directory')
    parser.add_argument('--deg-dir', required=True, help='Degraded WAV directory')
    parser.add_argument('--output-dir', default='results/ACC_MAP/metrics', help='Output directory')
    parser.add_argument('--output-name', required=True, help='Output JSON name without extension')
    parser.add_argument('--model-type', default='mfcc', choices=['mfcc', 'wavlm', 'ecapa', 'xvector'], help='Speaker model type')
    parser.add_argument('--limit', type=int, default=-1, help='Optional limit on number of matched files')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for bootstrap resampling')
    parser.add_argument('--bootstrap-samples', type=int, default=200, help='Number of bootstrap resamples for CI estimation')
    args = parser.parse_args()

    ref_dir = Path(args.ref_dir)
    deg_dir = Path(args.deg_dir)
    output_file = Path(args.output_dir) / f"{args.output_name}.json"

    if not ref_dir.exists():
        raise FileNotFoundError(f'Reference directory not found: {ref_dir}')
    if not deg_dir.exists():
        raise FileNotFoundError(f'Degraded directory not found: {deg_dir}')

    evaluate_acc_map(
        ref_dir=ref_dir,
        deg_dir=deg_dir,
        output_file=output_file,
        model_type=args.model_type,
        limit=args.limit,
        seed=args.seed,
        n_bootstrap=args.bootstrap_samples,
    )


if __name__ == '__main__':
    main()

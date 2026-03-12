#!/usr/bin/env python3
"""ASV-style application-level evaluation: EER and minDCF.

This script evaluates speaker-verification discriminability between
reference and degraded audio using embedding cosine scores.
"""

import argparse
import json
import logging
import math
import random
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


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _compute_eer(scores: np.ndarray, labels: np.ndarray):
    if len(scores) == 0:
        return float('nan'), float('nan')

    thresholds = np.unique(scores)
    thresholds = np.concatenate(([thresholds.min() - 1e-8], thresholds, [thresholds.max() + 1e-8]))

    pos = np.sum(labels == 1)
    neg = np.sum(labels == 0)
    if pos == 0 or neg == 0:
        return float('nan'), float('nan')

    fars = []
    frrs = []
    for thr in thresholds:
        pred = scores >= thr
        fp = np.sum((pred == 1) & (labels == 0))
        fn = np.sum((pred == 0) & (labels == 1))
        far = fp / neg
        frr = fn / pos
        fars.append(far)
        frrs.append(frr)

    fars = np.asarray(fars)
    frrs = np.asarray(frrs)
    idx = int(np.argmin(np.abs(fars - frrs)))
    eer = float((fars[idx] + frrs[idx]) / 2.0)
    eer_thr = float(thresholds[idx])
    return eer, eer_thr


def _compute_min_dcf(scores: np.ndarray, labels: np.ndarray, p_target=0.01, c_miss=1.0, c_fa=1.0):
    if len(scores) == 0:
        return float('nan'), float('nan')

    thresholds = np.unique(scores)
    thresholds = np.concatenate(([thresholds.min() - 1e-8], thresholds, [thresholds.max() + 1e-8]))

    pos = np.sum(labels == 1)
    neg = np.sum(labels == 0)
    if pos == 0 or neg == 0:
        return float('nan'), float('nan')

    best = None
    best_thr = None
    denom = min(c_miss * p_target, c_fa * (1.0 - p_target))
    denom = max(denom, 1e-12)

    for thr in thresholds:
        pred = scores >= thr
        fp = np.sum((pred == 1) & (labels == 0))
        fn = np.sum((pred == 0) & (labels == 1))
        fpr = fp / neg
        fnr = fn / pos
        dcf = c_miss * fnr * p_target + c_fa * fpr * (1.0 - p_target)
        norm_dcf = dcf / denom
        if best is None or norm_dcf < best:
            best = float(norm_dcf)
            best_thr = float(thr)

    return best, best_thr


def _bootstrap_ci(scores: np.ndarray, labels: np.ndarray, n_bootstrap: int = 200, seed: int = 42):
    """Bootstrap 95% CIs for EER and minDCF."""
    if len(scores) == 0 or n_bootstrap <= 1:
        return {
            'eer_ci95': [float('nan'), float('nan')],
            'min_dcf_ci95': [float('nan'), float('nan')],
        }

    rng = np.random.default_rng(seed)
    n = len(scores)
    eers = []
    dcfs = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        s = scores[idx]
        y = labels[idx]
        if np.sum(y == 1) == 0 or np.sum(y == 0) == 0:
            continue
        eer, _ = _compute_eer(s, y)
        dcf, _ = _compute_min_dcf(s, y)
        if not np.isnan(eer):
            eers.append(eer)
        if not np.isnan(dcf):
            dcfs.append(dcf)

    if len(eers) < 5 or len(dcfs) < 5:
        return {
            'eer_ci95': [float('nan'), float('nan')],
            'min_dcf_ci95': [float('nan'), float('nan')],
        }

    eer_ci = [float(np.percentile(eers, 2.5)), float(np.percentile(eers, 97.5))]
    dcf_ci = [float(np.percentile(dcfs, 2.5)), float(np.percentile(dcfs, 97.5))]

    return {
        'eer_ci95': eer_ci,
        'min_dcf_ci95': dcf_ci,
    }


def evaluate_asv(
    ref_dir: Path,
    deg_dir: Path,
    output_file: Path,
    model_type: str = 'mfcc',
    impostors_per_file: int = 10,
    impostor_mode: str = 'all',
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
        raise RuntimeError('Need at least 2 matched files to build target/impostor ASV trials')

    random.seed(seed)
    rng = np.random.default_rng(seed)

    scorer = SpeakerSimilarityScorer(model_type=model_type)
    logger.info('ASV backend requested: %s (effective: %s)', model_type, scorer.model_type)
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
        if r is not None and d is not None:
            ref_emb[name] = r
            deg_emb[name] = d

    usable = sorted(set(ref_emb.keys()).intersection(deg_emb.keys()))
    if len(usable) < 2:
        raise RuntimeError('Too few usable files after embedding extraction')

    labels = []
    scores = []
    trials = []

    for name in usable:
        target_score = _cosine_similarity(ref_emb[name], deg_emb[name])
        labels.append(1)
        scores.append(target_score)
        trials.append({'file': name, 'label': 'target', 'score': float(target_score)})

        pool = [x for x in usable if x != name]
        if impostor_mode == 'all':
            chosen = pool
        else:
            n_imp = min(impostors_per_file, len(pool))
            if n_imp <= 0:
                continue
            chosen = rng.choice(pool, size=n_imp, replace=False)
        for imp_name in chosen:
            imp_score = _cosine_similarity(ref_emb[imp_name], deg_emb[name])
            labels.append(0)
            scores.append(imp_score)
            trials.append({'file': name, 'impostor_ref': imp_name, 'label': 'impostor', 'score': float(imp_score)})

    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)

    target_scores = scores[labels == 1]
    impostor_scores = scores[labels == 0]

    eer, eer_thr = _compute_eer(scores, labels)
    min_dcf, min_dcf_thr = _compute_min_dcf(scores, labels)
    ci = _bootstrap_ci(scores, labels, n_bootstrap=n_bootstrap, seed=seed)

    separation_margin = float(np.min(target_scores) - np.max(impostor_scores)) if len(target_scores) and len(impostor_scores) else float('nan')
    overlap = bool(separation_margin <= 0) if not np.isnan(separation_margin) else None

    summary = {
        'metric': 'asv',
        'metrics': {
            'eer': eer,
            'eer_percent': None if np.isnan(eer) else 100.0 * eer,
            'eer_threshold': eer_thr,
            'eer_ci95': ci['eer_ci95'],
            'eer_percent_ci95': [None if np.isnan(x) else 100.0 * x for x in ci['eer_ci95']],
            'min_dcf': min_dcf,
            'min_dcf_threshold': min_dcf_thr,
            'min_dcf_ci95': ci['min_dcf_ci95'],
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
        'impostors_per_file': int(impostors_per_file),
        'impostor_mode': impostor_mode,
        'bootstrap_samples': int(n_bootstrap),
        'num_trials': int(len(scores)),
        'num_target_trials': int(np.sum(labels == 1)),
        'num_impostor_trials': int(np.sum(labels == 0)),
        'diagnostics': {
            'target_min': float(np.min(target_scores)) if len(target_scores) else None,
            'target_max': float(np.max(target_scores)) if len(target_scores) else None,
            'impostor_min': float(np.min(impostor_scores)) if len(impostor_scores) else None,
            'impostor_max': float(np.max(impostor_scores)) if len(impostor_scores) else None,
            'separation_margin_targetMin_minus_impostorMax': separation_margin,
            'has_overlap': overlap,
        },
        'score_stats': {
            'mean': float(np.mean(scores)) if len(scores) else None,
            'std': float(np.std(scores)) if len(scores) else None,
            'min': float(np.min(scores)) if len(scores) else None,
            'max': float(np.max(scores)) if len(scores) else None,
            'target_mean': float(np.mean(scores[labels == 1])) if np.any(labels == 1) else None,
            'impostor_mean': float(np.mean(scores[labels == 0])) if np.any(labels == 0) else None,
        },
        'trials': trials,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('w') as f:
        json.dump(summary, f, indent=2)

    logger.info('Saved ASV results to: %s', output_file)
    logger.info(
        'EER: %.4f (%.2f%%), minDCF: %.4f, overlap=%s, margin=%.6f',
        summary['metrics']['eer'],
        summary['metrics']['eer_percent'],
        summary['metrics']['min_dcf'],
        summary['diagnostics']['has_overlap'],
        summary['diagnostics']['separation_margin_targetMin_minus_impostorMax'] if summary['diagnostics']['separation_margin_targetMin_minus_impostorMax'] is not None else float('nan'),
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate ASV-style EER/minDCF for paired WAV directories.')
    parser.add_argument('--ref-dir', required=True, help='Reference WAV directory')
    parser.add_argument('--deg-dir', required=True, help='Degraded WAV directory')
    parser.add_argument('--output-dir', default='results/ASV/metrics', help='Output directory')
    parser.add_argument('--output-name', required=True, help='Output JSON name without extension')
    parser.add_argument('--model-type', default='mfcc', choices=['mfcc', 'wavlm', 'ecapa', 'xvector'], help='Speaker model type')
    parser.add_argument('--impostors-per-file', type=int, default=10, help='Number of impostor trials per file')
    parser.add_argument('--impostor-mode', choices=['sampled', 'all'], default='all', help='Use sampled impostors or all possible impostors')
    parser.add_argument('--limit', type=int, default=-1, help='Optional limit on number of matched files')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for impostor sampling')
    parser.add_argument('--bootstrap-samples', type=int, default=200, help='Number of bootstrap resamples for CI estimation')
    args = parser.parse_args()

    ref_dir = Path(args.ref_dir)
    deg_dir = Path(args.deg_dir)
    output_file = Path(args.output_dir) / f"{args.output_name}.json"

    if not ref_dir.exists():
        raise FileNotFoundError(f'Reference directory not found: {ref_dir}')
    if not deg_dir.exists():
        raise FileNotFoundError(f'Degraded directory not found: {deg_dir}')

    evaluate_asv(
        ref_dir=ref_dir,
        deg_dir=deg_dir,
        output_file=output_file,
        model_type=args.model_type,
        impostors_per_file=args.impostors_per_file,
        impostor_mode=args.impostor_mode,
        limit=args.limit,
        seed=args.seed,
        n_bootstrap=args.bootstrap_samples,
    )


if __name__ == '__main__':
    main()

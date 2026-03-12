#!/usr/bin/env python3
"""
Unified Quality Metrics Evaluation Script

Supports:
- NISQA (non-intrusive, no reference needed)
- UTMOS (non-intrusive, no reference needed)
- PESQ (intrusive, requires reference) 
- STOI (intrusive, requires reference)
- F0 RMSE (intrusive, requires reference)
- ViSQOL (intrusive, requires reference)
- Prosody (intrusive, requires reference)

Usage:
  # Non-intrusive (NISQA only)
  python evaluate_quality.py --metric nisqa --audio-dir path/to/audio
    python evaluate_quality.py --metric utmos --audio-dir path/to/audio

  # Intrusive (PESQ/STOI - requires reference)
  python evaluate_quality.py --metric pesq --ref-dir path/to/original --deg-dir path/to/compressed
  python evaluate_quality.py --metric stoi --ref-dir path/to/original --deg-dir path/to/compressed
    python evaluate_quality.py --metric f0 --ref-dir path/to/original --deg-dir path/to/compressed
    python evaluate_quality.py --metric visqol --ref-dir path/to/original --deg-dir path/to/compressed
    python evaluate_quality.py --metric prosody --ref-dir path/to/original --deg-dir path/to/compressed

  # All intrusive metrics
  python evaluate_quality.py --metric all --ref-dir path/to/original --deg-dir path/to/compressed
"""

import sys
import os
import json
import argparse
import logging
import csv
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'code' / 'audio_quality_assessment'))


def evaluate_nisqa(audio_dir, output_file, num_samples=-1):
    """Run NISQA evaluation (non-intrusive)."""
    from glob import glob
    from nisqa_scorer import NISQAScorer
    import soundfile as sf
    
    logger.info("="*70)
    logger.info("NISQA Quality Assessment (Non-Intrusive)")
    logger.info("="*70)
    
    # Get audio files
    audio_files = sorted(glob(os.path.join(audio_dir, "*.wav")))
    if num_samples > 0:
        audio_files = audio_files[:num_samples]
    audio_files = [str(Path(path).resolve()) for path in audio_files]
    
    logger.info(f"Audio directory: {audio_dir}")
    logger.info(f"Number of files: {len(audio_files)}")
    logger.info(f"Output file: {output_file}")
    logger.info("")
    
    # Initialize scorer
    logger.info("Loading NISQA scorer...")
    scorer = NISQAScorer()
    logger.info("✓ NISQA scorer loaded")
    logger.info("")
    
    # Process files in one official NISQA prediction pass
    results = []
    batch_failed = False
    try:
        score_map = scorer.batch_score(audio_files, return_details=True)
    except Exception as e:
        logger.error(f"NISQA batch prediction failed: {e}")
        score_map = {}
        batch_failed = True

    for audio_path in audio_files:
        filename = Path(audio_path).name
        try:
            info = sf.info(audio_path)
            if batch_failed:
                try:
                    result = scorer.score(audio_path, return_details=True)
                except Exception as e:
                    logger.error(f"Failed individual NISQA score for {filename}: {e}")
                    result = {
                        'mos': float('nan'),
                        'noi': float('nan'),
                        'dis': float('nan'),
                        'col': float('nan'),
                        'loud': float('nan'),
                    }
            else:
                result = score_map.get(audio_path, {
                    'mos': float('nan'),
                    'noi': float('nan'),
                    'dis': float('nan'),
                    'col': float('nan'),
                    'loud': float('nan'),
                })
            result['file'] = audio_path
            result['filename'] = filename
            result['duration'] = info.duration
            result['sample_rate'] = info.samplerate
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
    
    # Save results
    os.makedirs(Path(output_file).parent, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    if results:
        import numpy as np
        mos_scores = [r['mos'] for r in results if not (r['mos'] != r['mos'])]
        logger.info("")
        logger.info("="*70)
        logger.info("Summary")
        logger.info("="*70)
        logger.info(f"Files processed: {len(results)}")
        if mos_scores:
            logger.info(f"Average MOS: {np.mean(mos_scores):.3f} ± {np.std(mos_scores):.3f}")
            logger.info(f"MOS range: {np.min(mos_scores):.3f} - {np.max(mos_scores):.3f}")
        else:
            logger.info("Average MOS: nan ± nan")
            logger.info("MOS range: nan - nan")
        logger.info(f"Results saved to: {output_file}")
        logger.info("="*70)
    
    return results


def evaluate_utmos(audio_dir, output_file, num_samples=-1, max_audio_seconds=10.0):
    """Run UTMOS evaluation (non-intrusive)."""
    from glob import glob
    from utmos_scorer import UTMOSScorer

    logger.info("="*70)
    logger.info("UTMOS Quality Assessment (Non-Intrusive)")
    logger.info("="*70)

    audio_files = sorted(glob(os.path.join(audio_dir, "*.wav")))
    if num_samples > 0:
        audio_files = audio_files[:num_samples]

    logger.info(f"Audio directory: {audio_dir}")
    logger.info(f"Number of files: {len(audio_files)}")
    logger.info(f"Output file: {output_file}")
    logger.info("")

    scorer = UTMOSScorer(max_duration_seconds=max_audio_seconds)

    # If sampling limit requested, score a temp list by copying selection logic through temp dir semantics
    if num_samples > 0:
        import tempfile
        import shutil
        with tempfile.TemporaryDirectory() as tmp_dir:
            for src in audio_files:
                shutil.copy2(src, os.path.join(tmp_dir, Path(src).name))
            result = scorer.score_batch(tmp_dir)
    else:
        result = scorer.score_batch(audio_dir)

    os.makedirs(Path(output_file).parent, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info("")
    logger.info(f"✓ Results saved to: {output_file}")

    return result


def evaluate_pesq(ref_dir, deg_dir, output_file):
    """Run PESQ evaluation (intrusive)."""
    from pesq_scorer import PESQScorer
    
    logger.info("="*70)
    logger.info("PESQ Quality Assessment (Intrusive)")
    logger.info("="*70)
    logger.info(f"Reference directory: {ref_dir}")
    logger.info(f"Degraded directory: {deg_dir}")
    logger.info("")
    
    scorer = PESQScorer(mode='wb')
    result = scorer.score_batch(ref_dir, deg_dir)
    
    # Save results
    os.makedirs(Path(output_file).parent, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info("")
    logger.info(f"✓ Results saved to: {output_file}")
    
    return result


def evaluate_stoi(ref_dir, deg_dir, output_file):
    """Run STOI evaluation (intrusive)."""
    from stoi_scorer import STOIScorer
    
    logger.info("="*70)
    logger.info("STOI Quality Assessment (Intrusive)")
    logger.info("="*70)
    logger.info(f"Reference directory: {ref_dir}")
    logger.info(f"Degraded directory: {deg_dir}")
    logger.info("")
    
    scorer = STOIScorer(extended=False)
    result = scorer.score_batch(ref_dir, deg_dir)
    
    # Save results
    os.makedirs(Path(output_file).parent, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info("")
    logger.info(f"✓ Results saved to: {output_file}")
    
    return result


def evaluate_f0(ref_dir, deg_dir, output_file, num_samples=-1):
    """Run F0 RMSE evaluation (intrusive)."""
    from f0_scorer import F0Scorer

    logger.info("="*70)
    logger.info("F0 RMSE Prosody Assessment (Intrusive)")
    logger.info("="*70)
    logger.info(f"Reference directory: {ref_dir}")
    logger.info(f"Degraded directory: {deg_dir}")
    logger.info("")

    scorer = F0Scorer()
    result = scorer.score_batch(ref_dir, deg_dir, num_samples=num_samples)

    os.makedirs(Path(output_file).parent, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info("")
    logger.info(f"✓ Results saved to: {output_file}")

    return result


def evaluate_visqol(ref_dir, deg_dir, output_file, num_samples=-1):
    """Run ViSQOL evaluation (intrusive)."""
    from glob import glob
    from visqol_scorer import ViSQOLScorer

    logger.info("="*70)
    logger.info("ViSQOL Quality Assessment (Intrusive)")
    logger.info("="*70)
    logger.info(f"Reference directory: {ref_dir}")
    logger.info(f"Degraded directory: {deg_dir}")
    logger.info("")

    ref_map = {Path(p).name: p for p in sorted(glob(os.path.join(ref_dir, "*.wav")))}
    deg_files = sorted(glob(os.path.join(deg_dir, "*.wav")))
    if num_samples > 0:
        deg_files = deg_files[:num_samples]

    scorer = ViSQOLScorer()
    per_file = []
    mos_values = []

    for deg_path in deg_files:
        name = Path(deg_path).name
        ref_path = ref_map.get(name)
        if not ref_path:
            per_file.append({'filename': name, 'moslqo': float('nan'), 'error': 'reference file not found'})
            continue
        try:
            score = scorer.score(ref_path, deg_path, return_details=True)
            row = {'filename': name, **score}
            per_file.append(row)
            if not isinstance(score.get('moslqo'), float) or not (score.get('moslqo') != score.get('moslqo')):
                mos_values.append(float(score.get('moslqo')))
        except Exception as e:
            per_file.append({'filename': name, 'moslqo': float('nan'), 'error': str(e)})

    import numpy as np
    result = {
        'metric': 'visqol',
        'reference_dir': os.path.abspath(ref_dir),
        'degraded_dir': os.path.abspath(deg_dir),
        'num_ref_files': len(ref_map),
        'num_deg_files': len(deg_files),
        'num_scored_files': len(mos_values),
        'moslqo': float(np.mean(mos_values)) if mos_values else float('nan'),
        'moslqo_std': float(np.std(mos_values)) if mos_values else float('nan'),
        'moslqo_min': float(np.min(mos_values)) if mos_values else float('nan'),
        'moslqo_max': float(np.max(mos_values)) if mos_values else float('nan'),
        'per_file': per_file,
    }

    os.makedirs(Path(output_file).parent, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info("")
    logger.info(f"✓ Results saved to: {output_file}")
    return result


def evaluate_prosody(ref_dir, deg_dir, output_file, num_samples=-1):
    """Run full Prosody evaluation (intrusive)."""
    from glob import glob
    from prosody_analyzer import ProsodyAnalyzer

    logger.info("="*70)
    logger.info("Prosody Assessment (Intrusive)")
    logger.info("="*70)
    logger.info(f"Reference directory: {ref_dir}")
    logger.info(f"Degraded directory: {deg_dir}")
    logger.info("")

    ref_map = {Path(p).name: p for p in sorted(glob(os.path.join(ref_dir, "*.wav")))}
    deg_files = sorted(glob(os.path.join(deg_dir, "*.wav")))
    if num_samples > 0:
        deg_files = deg_files[:num_samples]

    analyzer = ProsodyAnalyzer()
    per_file = []
    overall_values = []
    f0_values = []

    for deg_path in deg_files:
        name = Path(deg_path).name
        ref_path = ref_map.get(name)
        if not ref_path:
            per_file.append({'filename': name, 'overall_prosody': float('nan'), 'error': 'reference file not found'})
            continue
        try:
            score = analyzer.analyze(ref_path, deg_path, return_details=True)
            row = {'filename': name, **score}
            per_file.append(row)
            op = score.get('overall_prosody')
            f0 = score.get('f0_rmse')
            if isinstance(op, float) and not (op != op):
                overall_values.append(float(op))
            if isinstance(f0, float) and not (f0 != f0):
                f0_values.append(float(f0))
        except Exception as e:
            per_file.append({'filename': name, 'overall_prosody': float('nan'), 'error': str(e)})

    import numpy as np
    result = {
        'metric': 'prosody',
        'reference_dir': os.path.abspath(ref_dir),
        'degraded_dir': os.path.abspath(deg_dir),
        'num_ref_files': len(ref_map),
        'num_deg_files': len(deg_files),
        'num_scored_files': len(overall_values),
        'overall_prosody': float(np.mean(overall_values)) if overall_values else float('nan'),
        'overall_prosody_std': float(np.std(overall_values)) if overall_values else float('nan'),
        'f0_rmse': float(np.mean(f0_values)) if f0_values else float('nan'),
        'f0_rmse_std': float(np.std(f0_values)) if f0_values else float('nan'),
        'per_file': per_file,
    }

    os.makedirs(Path(output_file).parent, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info("")
    logger.info(f"✓ Results saved to: {output_file}")
    return result


def _load_transcript_map(transcript_csv):
    """Load filename -> transcript map from CSV metadata."""
    transcript_map = {}

    with open(transcript_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or 'file_name' not in reader.fieldnames or 'transcript' not in reader.fieldnames:
            raise ValueError("Transcript CSV must contain 'file_name' and 'transcript' columns")

        for row in reader:
            raw_name = (row.get('file_name') or '').strip()
            transcript = (row.get('transcript') or '').strip()
            if not raw_name or not transcript:
                continue

            filename = Path(raw_name).name
            transcript_map[filename] = transcript

    return transcript_map


def evaluate_wer(audio_dir, transcript_csv, output_file, num_samples=-1,
                 asr_model='openai/whisper-base', language='en'):
    """Run WER evaluation (application-level ASR metric)."""
    from glob import glob
    from asr_evaluator import ASREvaluator

    logger.info("="*70)
    logger.info("WER Assessment (Application-Level, ASR)")
    logger.info("="*70)
    logger.info(f"Audio directory: {audio_dir}")
    logger.info(f"Transcript CSV: {transcript_csv}")
    logger.info(f"ASR model: {asr_model}")
    logger.info(f"Language: {language}")
    logger.info("")

    transcript_map = _load_transcript_map(transcript_csv)

    audio_files = sorted(glob(os.path.join(audio_dir, "*.wav")))
    if num_samples > 0:
        audio_files = audio_files[:num_samples]

    logger.info(f"Found {len(audio_files)} audio files")
    logger.info(f"Loaded transcripts for {len(transcript_map)} files")

    evaluator = ASREvaluator(model_name=asr_model, language=language)

    per_file = []
    wer_values = []
    cer_values = []
    skipped_no_transcript = 0

    for audio_path in audio_files:
        filename = Path(audio_path).name
        transcript = transcript_map.get(filename)
        if not transcript:
            skipped_no_transcript += 1
            continue

        result = evaluator.evaluate_degraded_only(
            degraded_audio=audio_path,
            reference_text=transcript,
            return_details=True
        )

        row = {
            'file': audio_path,
            'filename': filename,
            **result
        }
        per_file.append(row)

        wer_score = result.get('wer_degraded')
        cer_score = result.get('cer_degraded')
        if isinstance(wer_score, (int, float)) and not (wer_score != wer_score):
            wer_values.append(float(wer_score))
        if isinstance(cer_score, (int, float)) and not (cer_score != cer_score):
            cer_values.append(float(cer_score))

    import numpy as np
    summary = {
        'metric': 'wer',
        'audio_dir': os.path.abspath(audio_dir),
        'transcript_csv': os.path.abspath(transcript_csv),
        'asr_model': asr_model,
        'language': language,
        'num_audio_files': len(audio_files),
        'num_scored_files': len(per_file),
        'num_skipped_no_transcript': skipped_no_transcript,
        'wer': float(np.mean(wer_values)) if wer_values else float('nan'),
        'wer_std': float(np.std(wer_values)) if wer_values else float('nan'),
        'wer_min': float(np.min(wer_values)) if wer_values else float('nan'),
        'wer_max': float(np.max(wer_values)) if wer_values else float('nan'),
        'cer': float(np.mean(cer_values)) if cer_values else float('nan'),
        'cer_std': float(np.std(cer_values)) if cer_values else float('nan'),
        'per_file': per_file,
    }

    os.makedirs(Path(output_file).parent, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("")
    logger.info("="*70)
    logger.info("Summary")
    logger.info("="*70)
    logger.info(f"Files scored: {summary['num_scored_files']}/{summary['num_audio_files']}")
    logger.info(f"Skipped (no transcript): {summary['num_skipped_no_transcript']}")
    logger.info(f"Mean WER: {summary['wer']:.4f}")
    logger.info(f"Mean CER: {summary['cer']:.4f}")
    logger.info(f"Results saved to: {output_file}")
    logger.info("="*70)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Unified quality metrics evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--metric', required=True, 
                        choices=['nisqa', 'utmos', 'pesq', 'stoi', 'f0', 'visqol', 'prosody', 'wer', 'all'],
                        help='Which metric to evaluate')
    
    # For NISQA (non-intrusive)
    parser.add_argument('--audio-dir', help='Directory with audio files (for NISQA)')
    parser.add_argument('--num-samples', type=int, default=-1,
                        help='Number of samples to process (-1 for all)')
    parser.add_argument('--utmos-max-seconds', type=float, default=10.0,
                        help='Maximum duration (seconds) per clip for UTMOS scoring')
    parser.add_argument('--transcript-csv',
                        help='CSV containing file_name and transcript columns (required for WER)')
    parser.add_argument('--asr-model', default='openai/whisper-base',
                        help='ASR model to use for WER evaluation')
    parser.add_argument('--asr-language', default='en',
                        help='Language code for ASR model')
    
    # For PESQ/STOI (intrusive)
    parser.add_argument('--ref-dir', help='Reference (original) audio directory')
    parser.add_argument('--deg-dir', help='Degraded (compressed) audio directory')
    
    # Output
    parser.add_argument('--output-dir', default='results/quality_metrics',
                        help='Output directory for results')
    parser.add_argument('--output-name', help='Custom output filename (without extension)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.metric in ['nisqa', 'utmos', 'wer'] and not args.audio_dir:
        parser.error("--audio-dir is required for NISQA/UTMOS/WER")

    if args.metric == 'wer' and not args.transcript_csv:
        parser.error("--transcript-csv is required for WER")
    
    if args.metric in ['pesq', 'stoi', 'f0', 'visqol', 'prosody', 'all'] and (not args.ref_dir or not args.deg_dir):
        parser.error("--ref-dir and --deg-dir are required for PESQ/STOI/F0/ViSQOL/Prosody")
    
    # Generate output filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        if args.metric == 'nisqa':
            output_name = args.output_name or f"nisqa_{Path(args.audio_dir).name}_{timestamp}"
            output_file = os.path.join(args.output_dir, f"{output_name}.json")
            evaluate_nisqa(args.audio_dir, output_file, args.num_samples)

        elif args.metric == 'utmos':
            output_name = args.output_name or f"utmos_{Path(args.audio_dir).name}_{timestamp}"
            output_file = os.path.join(args.output_dir, f"{output_name}.json")
            evaluate_utmos(args.audio_dir, output_file, args.num_samples, args.utmos_max_seconds)
            
        elif args.metric == 'pesq':
            output_name = args.output_name or f"pesq_{Path(args.deg_dir).name}_{timestamp}"
            output_file = os.path.join(args.output_dir, f"{output_name}.json")
            evaluate_pesq(args.ref_dir, args.deg_dir, output_file)
            
        elif args.metric == 'stoi':
            output_name = args.output_name or f"stoi_{Path(args.deg_dir).name}_{timestamp}"
            output_file = os.path.join(args.output_dir, f"{output_name}.json")
            evaluate_stoi(args.ref_dir, args.deg_dir, output_file)

        elif args.metric == 'f0':
            output_name = args.output_name or f"f0_{Path(args.deg_dir).name}_{timestamp}"
            output_file = os.path.join(args.output_dir, f"{output_name}.json")
            evaluate_f0(args.ref_dir, args.deg_dir, output_file, args.num_samples)

        elif args.metric == 'visqol':
            output_name = args.output_name or f"visqol_{Path(args.deg_dir).name}_{timestamp}"
            output_file = os.path.join(args.output_dir, f"{output_name}.json")
            evaluate_visqol(args.ref_dir, args.deg_dir, output_file, args.num_samples)

        elif args.metric == 'prosody':
            output_name = args.output_name or f"prosody_{Path(args.deg_dir).name}_{timestamp}"
            output_file = os.path.join(args.output_dir, f"{output_name}.json")
            evaluate_prosody(args.ref_dir, args.deg_dir, output_file, args.num_samples)

        elif args.metric == 'wer':
            output_name = args.output_name or f"wer_{Path(args.audio_dir).name}_{timestamp}"
            output_file = os.path.join(args.output_dir, f"{output_name}.json")
            evaluate_wer(
                args.audio_dir,
                args.transcript_csv,
                output_file,
                args.num_samples,
                asr_model=args.asr_model,
                language=args.asr_language,
            )
            
        elif args.metric == 'all':
            # Run all intrusive metrics
            base_name = args.output_name or f"metrics_{Path(args.deg_dir).name}_{timestamp}"
            
            logger.info("Running all intrusive quality metrics...")
            logger.info("")
            
            pesq_file = os.path.join(args.output_dir, f"pesq_{base_name}.json")
            stoi_file = os.path.join(args.output_dir, f"stoi_{base_name}.json")
            
            pesq_result = evaluate_pesq(args.ref_dir, args.deg_dir, pesq_file)
            logger.info("")
            stoi_result = evaluate_stoi(args.ref_dir, args.deg_dir, stoi_file)
            
            # Combined summary
            logger.info("")
            logger.info("="*70)
            logger.info("Combined Summary")
            logger.info("="*70)
            if 'wb_pesq' in pesq_result:
                logger.info(f"PESQ WB: {pesq_result['wb_pesq']:.3f}")
            if 'nb_pesq' in pesq_result:
                logger.info(f"PESQ NB: {pesq_result['nb_pesq']:.3f}")
            logger.info(f"PESQ:    {pesq_result['pesq']:.3f}")
            logger.info(f"STOI:    {stoi_result['stoi']:.4f}")
            logger.info(f"Files:   {pesq_result['num_files']}")
            logger.info("="*70)
            
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

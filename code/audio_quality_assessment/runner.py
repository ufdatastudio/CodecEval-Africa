"""
Audio Quality Assessment Runner

This script provides a comprehensive evaluation of audio quality
using all implemented metrics for neural codec evaluation.

Usage:
    python -m code.audio_quality_assessment.runner --config config.yaml
    python -m code.audio_quality_assessment.runner --reference ref.wav --degraded deg.wav
"""

import argparse
import json
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings("ignore")

from .nisqa_scorer import NISQAScorer
from .visqol_scorer import ViSQOLScorer
from .dnsmos_scorer import DNSMOSScorer
from .speaker_similarity import SpeakerSimilarityScorer
from .prosody_analyzer import ProsodyAnalyzer
from .asr_evaluator import ASREvaluator

class AudioQualityAssessmentRunner:
    """
    Comprehensive audio quality assessment runner.
    
    Evaluates audio quality using multiple complementary metrics
    designed specifically for neural codec evaluation.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize assessment runner.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize scorers based on config
        self.scorers = self._initialize_scorers()
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file or use defaults."""
        default_config = {
            'metrics': {
                'nisqa': {'enabled': True},
                'visqol': {'enabled': True},
                'dnsmos': {'enabled': True},
                'speaker_similarity': {
                    'enabled': True,
                    'model_type': 'mfcc'  # or 'wavlm', 'ecapa', 'xvector'
                },
                'prosody': {'enabled': True},
                'asr': {
                    'enabled': True,
                    'model_name': 'openai/whisper-base',
                    'language': 'en'
                }
            },
            'output': {
                'format': 'json',  # 'json', 'csv', 'yaml'
                'detailed': True,
                'save_path': None
            },
            'device': 'auto'  # 'auto', 'cuda', 'cpu'
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                
                # Merge configs (user config overrides defaults)
                config = self._merge_configs(default_config, user_config)
                self.logger.info(f"Loaded configuration from {config_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to load config {config_path}: {e}")
                self.logger.info("Using default configuration")
                config = default_config
        else:
            config = default_config
            
        return config
    
    def _merge_configs(self, default: Dict, user: Dict) -> Dict:
        """Recursively merge user config into default config."""
        merged = default.copy()
        
        for key, value in user.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _initialize_scorers(self) -> Dict:
        """Initialize all enabled scorers."""
        scorers = {}
        device = self.config.get('device', 'auto')
        
        if device == 'auto':
            device = None  # Let each scorer auto-detect
        
        metrics_config = self.config.get('metrics', {})
        
        # NISQA
        if metrics_config.get('nisqa', {}).get('enabled', True):
            try:
                scorers['nisqa'] = NISQAScorer(device=device)
                self.logger.info("NISQA scorer initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize NISQA: {e}")
        
        # ViSQOL  
        if metrics_config.get('visqol', {}).get('enabled', True):
            try:
                scorers['visqol'] = ViSQOLScorer()
                self.logger.info("ViSQOL scorer initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize ViSQOL: {e}")
        
        # DNSMOS
        if metrics_config.get('dnsmos', {}).get('enabled', True):
            try:
                scorers['dnsmos'] = DNSMOSScorer(device=device)
                self.logger.info("DNSMOS scorer initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize DNSMOS: {e}")
        
        # Speaker Similarity
        speaker_config = metrics_config.get('speaker_similarity', {})
        if speaker_config.get('enabled', True):
            try:
                model_type = speaker_config.get('model_type', 'mfcc')
                scorers['speaker_similarity'] = SpeakerSimilarityScorer(
                    model_type=model_type, 
                    device=device
                )
                self.logger.info(f"Speaker similarity scorer initialized ({model_type})")
            except Exception as e:
                self.logger.error(f"Failed to initialize speaker similarity: {e}")
        
        # Prosody Analysis
        if metrics_config.get('prosody', {}).get('enabled', True):
            try:
                scorers['prosody'] = ProsodyAnalyzer()
                self.logger.info("Prosody analyzer initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize prosody analyzer: {e}")
        
        # ASR Evaluation
        asr_config = metrics_config.get('asr', {})
        if asr_config.get('enabled', True):
            try:
                model_name = asr_config.get('model_name', 'openai/whisper-base')
                language = asr_config.get('language', 'en')
                scorers['asr'] = ASREvaluator(
                    model_name=model_name,
                    device=device,
                    language=language
                )
                self.logger.info(f"ASR evaluator initialized ({model_name})")
            except Exception as e:
                self.logger.error(f"Failed to initialize ASR evaluator: {e}")
        
        return scorers
    
    def evaluate_pair(self, reference_path: str, degraded_path: str,
                     reference_text: Optional[str] = None) -> Dict:
        """
        Evaluate a single reference-degraded audio pair.
        
        Args:
            reference_path: Path to reference audio
            degraded_path: Path to degraded audio
            reference_text: Ground truth transcript (optional)
            
        Returns:
            Dictionary containing all evaluation results
        """
        results = {
            'reference_path': reference_path,
            'degraded_path': degraded_path,
            'reference_text': reference_text,
            'metrics': {}
        }
        
        detailed = self.config.get('output', {}).get('detailed', True)
        
        # NISQA (non-intrusive, only needs degraded audio)
        if 'nisqa' in self.scorers:
            try:
                nisqa_score = self.scorers['nisqa'].score(degraded_path, return_details=detailed)
                results['metrics']['nisqa'] = nisqa_score
                self.logger.info(f"NISQA: {nisqa_score.get('mos', 'N/A')}")
            except Exception as e:
                self.logger.error(f"NISQA evaluation failed: {e}")
                results['metrics']['nisqa'] = {'mos': float('nan')}
        
        # ViSQOL (intrusive, needs both reference and degraded)
        if 'visqol' in self.scorers:
            try:
                visqol_score = self.scorers['visqol'].score(
                    reference_path, degraded_path, return_details=detailed
                )
                results['metrics']['visqol'] = visqol_score
                self.logger.info(f"ViSQOL: {visqol_score.get('moslqo', 'N/A')}")
            except Exception as e:
                self.logger.error(f"ViSQOL evaluation failed: {e}")
                results['metrics']['visqol'] = {'moslqo': float('nan')}
        
        # DNSMOS (non-intrusive, only needs degraded audio)
        if 'dnsmos' in self.scorers:
            try:
                dnsmos_score = self.scorers['dnsmos'].score(degraded_path)
                results['metrics']['dnsmos'] = dnsmos_score
                self.logger.info(f"DNSMOS: SIG={dnsmos_score.get('SIG', 'N/A')}, "
                               f"BAK={dnsmos_score.get('BAK', 'N/A')}, "
                               f"OVR={dnsmos_score.get('OVR', 'N/A')}")
            except Exception as e:
                self.logger.error(f"DNSMOS evaluation failed: {e}")
                results['metrics']['dnsmos'] = {'SIG': float('nan'), 'BAK': float('nan'), 'OVR': float('nan')}
        
        # Speaker Similarity
        if 'speaker_similarity' in self.scorers:
            try:
                speaker_score = self.scorers['speaker_similarity'].compute_similarity(
                    reference_path, degraded_path, return_details=detailed
                )
                results['metrics']['speaker_similarity'] = speaker_score
                self.logger.info(f"Speaker Similarity: {speaker_score.get('cosine_similarity', 'N/A')}")
            except Exception as e:
                self.logger.error(f"Speaker similarity evaluation failed: {e}")
                results['metrics']['speaker_similarity'] = {'cosine_similarity': float('nan')}
        
        # Prosody Analysis
        if 'prosody' in self.scorers:
            try:
                prosody_score = self.scorers['prosody'].analyze(
                    reference_path, degraded_path, return_details=detailed
                )
                results['metrics']['prosody'] = prosody_score
                self.logger.info(f"Prosody: F0_RMSE={prosody_score.get('f0_rmse', 'N/A')}, "
                               f"Overall={prosody_score.get('overall_prosody', 'N/A')}")
            except Exception as e:
                self.logger.error(f"Prosody evaluation failed: {e}")
                results['metrics']['prosody'] = {'f0_rmse': float('nan'), 'overall_prosody': float('nan')}
        
        # ASR Evaluation
        if 'asr' in self.scorers:
            try:
                asr_score = self.scorers['asr'].evaluate_wer(
                    reference_path, degraded_path, reference_text, return_details=detailed
                )
                results['metrics']['asr'] = asr_score
                self.logger.info(f"ASR: Relative_WER={asr_score.get('relative_wer', 'N/A')}, "
                               f"Transcript_Sim={asr_score.get('transcription_similarity', 'N/A')}")
            except Exception as e:
                self.logger.error(f"ASR evaluation failed: {e}")
                results['metrics']['asr'] = {'relative_wer': float('nan'), 'transcription_similarity': float('nan')}
        
        return results
    
    def evaluate_batch(self, audio_pairs: List[Dict]) -> List[Dict]:
        """
        Evaluate multiple audio pairs.
        
        Args:
            audio_pairs: List of dictionaries with 'reference', 'degraded', and optional 'transcript'
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for i, pair in enumerate(audio_pairs):
            self.logger.info(f"Evaluating pair {i+1}/{len(audio_pairs)}")
            
            result = self.evaluate_pair(
                pair['reference'],
                pair['degraded'],
                pair.get('transcript')
            )
            
            results.append(result)
        
        return results
    
    def save_results(self, results: List[Dict], output_path: str):
        """Save evaluation results to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        format_type = self.config.get('output', {}).get('format', 'json')
        
        if format_type == 'json':
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        elif format_type == 'yaml':
            with open(output_path, 'w') as f:
                yaml.dump(results, f, default_flow_style=False)
        elif format_type == 'csv':
            import pandas as pd
            # Flatten results for CSV
            flattened = self._flatten_results(results)
            df = pd.DataFrame(flattened)
            df.to_csv(output_path, index=False)
        
        self.logger.info(f"Results saved to {output_path}")
    
    def _flatten_results(self, results: List[Dict]) -> List[Dict]:
        """Flatten nested results for CSV export."""
        flattened = []
        
        for result in results:
            flat_result = {
                'reference_path': result['reference_path'],
                'degraded_path': result['degraded_path'],
                'reference_text': result.get('reference_text', '')
            }
            
            # Flatten metrics
            for metric_name, metric_data in result.get('metrics', {}).items():
                if isinstance(metric_data, dict):
                    for key, value in metric_data.items():
                        flat_result[f"{metric_name}_{key}"] = value
                else:
                    flat_result[metric_name] = metric_data
            
            flattened.append(flat_result)
        
        return flattened


def main():
    """Command line interface for audio quality assessment."""
    parser = argparse.ArgumentParser(description="Audio Quality Assessment for Neural Codec Evaluation")
    
    # Input options
    parser.add_argument('--reference', type=str, help="Path to reference audio file")
    parser.add_argument('--degraded', type=str, help="Path to degraded audio file") 
    parser.add_argument('--transcript', type=str, help="Ground truth transcript (optional)")
    parser.add_argument('--pairs', type=str, help="Path to JSON file with audio pairs")
    
    # Configuration
    parser.add_argument('--config', type=str, help="Path to configuration YAML file")
    
    # Output
    parser.add_argument('--output', type=str, help="Output file path")
    parser.add_argument('--format', choices=['json', 'yaml', 'csv'], default='json',
                       help="Output format")
    
    # Logging
    parser.add_argument('--verbose', action='store_true', help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize runner
    runner = AudioQualityAssessmentRunner(args.config)
    
    # Override output format if specified
    if args.format:
        runner.config['output']['format'] = args.format
    
    # Evaluate audio
    if args.reference and args.degraded:
        # Single pair evaluation
        result = runner.evaluate_pair(args.reference, args.degraded, args.transcript)
        results = [result]
        
    elif args.pairs:
        # Batch evaluation
        with open(args.pairs, 'r') as f:
            pairs = json.load(f)
        results = runner.evaluate_batch(pairs)
        
    else:
        parser.error("Must specify either --reference/--degraded or --pairs")
    
    # Output results
    if args.output:
        runner.save_results(results, args.output)
    else:
        # Print to stdout
        if runner.config['output']['format'] == 'json':
            print(json.dumps(results, indent=2, default=str))
        elif runner.config['output']['format'] == 'yaml':
            print(yaml.dump(results, default_flow_style=False))


if __name__ == "__main__":
    main()
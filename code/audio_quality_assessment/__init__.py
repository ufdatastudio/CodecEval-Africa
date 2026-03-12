"""
Audio Quality Assessment Module

This module provides accurate implementations of audio quality metrics
specifically designed for neural codec evaluation on African speech data.

All implementations use established libraries and validated methods
from the literature, not approximations or proxy metrics.
"""

from .nisqa_scorer import NISQAScorer
from .visqol_scorer import ViSQOLScorer  
from .dnsmos_scorer import DNSMOSScorer
from .speaker_similarity import SpeakerSimilarityScorer
from .prosody_analyzer import ProsodyAnalyzer
from .asr_evaluator import ASREvaluator

__all__ = [
    'NISQAScorer',
    'ViSQOLScorer', 
    'DNSMOSScorer',
    'SpeakerSimilarityScorer',
    'ProsodyAnalyzer',
    'ASREvaluator'
]

# PESQ and STOI scorers are available but not imported by default
# to avoid dependency issues. Import directly:
# from code.audio_quality_assessment.pesq_scorer import PESQScorer
# from code.audio_quality_assessment.stoi_scorer import STOIScorer
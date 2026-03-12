"""
ASR Impact Evaluation Implementation

This module provides accurate ASR (Automatic Speech Recognition) impact
evaluation for measuring how codec compression affects speech recognition
performance.

Uses state-of-the-art ASR models like Whisper, Wav2Vec2, and others
to compute Word Error Rate (WER) and other ASR quality metrics.
"""

import torch
import torchaudio
import numpy as np
import soundfile as sf
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import warnings
import jiwer
from dataclasses import dataclass

warnings.filterwarnings("ignore")

@dataclass
class ASRResult:
    """Container for ASR evaluation results."""
    reference_text: str
    hypothesis_text: str
    wer: float
    cer: float  # Character Error Rate
    insertions: int
    deletions: int
    substitutions: int
    hits: int

class ASREvaluator:
    """
    ASR Impact evaluator for measuring codec effects on speech recognition.
    
    Evaluates how well speech recognition systems perform on
    codec-compressed audio compared to reference audio.
    """
    
    def __init__(self, model_name: str = "openai/whisper-large-v3", 
                 device: Optional[str] = None, language: str = "en"):
        """
        Initialize ASR evaluator.
        
        Args:
            model_name: ASR model to use ('whisper-large-v3', 'wav2vec2', etc.)
            device: Device to run model on ('cuda' or 'cpu'). Auto-detected if None.
            language: Language code for ASR (e.g., 'en', 'fr', 'es')
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.language = language
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.logger.info(f"ASR evaluator using {model_name} on {self.device}")

        self.torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.pipeline_device_id = 0 if self.device.type == "cuda" else -1
        self.whisper_generate_kwargs = {
            'max_new_tokens': 256,
            'num_beams': 1,
            'do_sample': False,
            'repetition_penalty': 1.0,
        }
        
        # Load ASR model
        self.asr_pipeline = self._load_asr_model()
        
        # Set target sample rate based on model
        if "whisper" in model_name.lower():
            self.target_sr = 16000
        elif "wav2vec2" in model_name.lower():
            self.target_sr = 16000
        else:
            self.target_sr = 16000
    
    def _load_asr_model(self):
        """Load the specified ASR model."""
        try:
            if "whisper" in self.model_name.lower():
                return self._load_whisper_model()
            elif "wav2vec2" in self.model_name.lower():
                return self._load_wav2vec2_model()
            else:
                return self._load_transformers_asr_model()
                
        except Exception as e:
            self.logger.error(f"Failed to load ASR model {self.model_name}: {e}")
            raise e
    
    def _load_whisper_model(self):
        """Load Whisper ASR model."""
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
            
            # Load model and processor
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            model.to(self.device)
            
            processor = AutoProcessor.from_pretrained(self.model_name)
            
            # Create pipeline
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                chunk_length_s=30,
                batch_size=8,
                return_timestamps=False,
                torch_dtype=self.torch_dtype,
                device=self.pipeline_device_id,
                generate_kwargs=self.whisper_generate_kwargs,
            )
            
            self.logger.info(f"Whisper model {self.model_name} loaded successfully")
            return pipe
            
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {e}")
            raise e
    
    def _load_wav2vec2_model(self):
        """Load Wav2Vec2 ASR model."""
        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline
            
            # Create pipeline
            pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                device=0 if self.device.type == "cuda" else -1
            )
            
            self.logger.info(f"Wav2Vec2 model {self.model_name} loaded successfully")
            return pipe
            
        except Exception as e:
            self.logger.error(f"Failed to load Wav2Vec2 model: {e}")
            raise e
    
    def _load_transformers_asr_model(self):
        """Load generic transformers ASR model."""
        try:
            from transformers import pipeline
            
            pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                device=0 if self.device.type == "cuda" else -1
            )
            
            self.logger.info(f"ASR model {self.model_name} loaded successfully")
            return pipe
            
        except Exception as e:
            self.logger.error(f"Failed to load ASR model: {e}")
            raise e
    
    def evaluate_wer(self, reference_audio: str, degraded_audio: str,
                     reference_text: Optional[str] = None,
                     return_details: bool = True) -> Dict[str, float]:
        """
        Evaluate WER (Word Error Rate) impact of codec compression.
        
        Args:
            reference_audio: Path to reference audio file
            degraded_audio: Path to degraded audio file  
            reference_text: Ground truth transcript (if available)
            return_details: If True, returns detailed metrics
            
        Returns:
            Dictionary containing:
            - wer_reference: WER of reference audio vs ground truth
            - wer_degraded: WER of degraded audio vs ground truth
            - wer_increase: WER increase due to compression
            - cer_reference: CER of reference audio (if details requested)
            - cer_degraded: CER of degraded audio (if details requested)
            - transcription_similarity: Direct similarity between transcriptions
        """
        try:
            # Transcribe both audio files
            ref_result = self._transcribe_audio(reference_audio)
            deg_result = self._transcribe_audio(degraded_audio)
            
            if ref_result is None or deg_result is None:
                return self._get_nan_results()
            
            # Extract transcription text
            ref_transcript = ref_result.get('text', '')
            deg_transcript = deg_result.get('text', '')
            
            results = {}
            
            # If ground truth is available, compute absolute WER
            if reference_text:
                ref_asr_result = self._compute_error_rates(reference_text, ref_transcript)
                deg_asr_result = self._compute_error_rates(reference_text, deg_transcript)
                
                results.update({
                    'wer_reference': ref_asr_result.wer,
                    'wer_degraded': deg_asr_result.wer,
                    'wer_increase': deg_asr_result.wer - ref_asr_result.wer,
                })
                
                if return_details:
                    results.update({
                        'cer_reference': ref_asr_result.cer,
                        'cer_degraded': deg_asr_result.cer,
                        'insertions_ref': ref_asr_result.insertions,
                        'deletions_ref': ref_asr_result.deletions,
                        'substitutions_ref': ref_asr_result.substitutions,
                        'insertions_deg': deg_asr_result.insertions,
                        'deletions_deg': deg_asr_result.deletions,
                        'substitutions_deg': deg_asr_result.substitutions,
                    })
            
            # Compute transcription similarity (how similar are the two ASR outputs)
            transcript_similarity = self._compute_transcription_similarity(ref_transcript, deg_transcript)
            results['transcription_similarity'] = transcript_similarity
            
            # Relative WER (using reference transcription as "ground truth")
            relative_asr_result = self._compute_error_rates(ref_transcript, deg_transcript)
            results.update({
                'relative_wer': relative_asr_result.wer,
                'relative_cer': relative_asr_result.cer
            })
            
            if return_details:
                results.update({
                    'reference_transcript': ref_transcript,
                    'degraded_transcript': deg_transcript,
                    'ground_truth': reference_text or ''
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error evaluating WER: {e}")
            return self._get_nan_results()
    
    def _transcribe_audio(self, audio_path: str) -> Optional[Dict]:
        """Transcribe audio file using loaded ASR model."""
        try:
            # Load and preprocess audio
            audio, sr = sf.read(audio_path, always_2d=False)
            
            # Convert to mono if stereo
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            # Normalize dtype for stable Whisper/transformers inference
            audio = np.asarray(audio, dtype=np.float32)
            
            # Resample if needed
            if sr != self.target_sr:
                audio_tensor = torch.from_numpy(audio).to(dtype=torch.float32)
                audio = torchaudio.transforms.Resample(sr, self.target_sr)(audio_tensor).numpy()

            audio = np.ascontiguousarray(audio, dtype=np.float32)
            
            # Transcribe using pipeline
            if "whisper" in self.model_name.lower():
                language_token = 'english' if self.language.lower() in ['en', 'english'] else self.language
                generate_kwargs = dict(self.whisper_generate_kwargs)
                generate_kwargs['language'] = language_token
                try:
                    result = self.asr_pipeline(audio, generate_kwargs=generate_kwargs)
                except RuntimeError as e:
                    err = str(e)
                    if "expected scalar type Double but found Float" in err:
                        result = self.asr_pipeline(
                            np.ascontiguousarray(audio, dtype=np.float64),
                            generate_kwargs=generate_kwargs,
                        )
                    elif "expected scalar type Float but found Double" in err:
                        result = self.asr_pipeline(
                            np.ascontiguousarray(audio, dtype=np.float32),
                            generate_kwargs=generate_kwargs,
                        )
                    else:
                        raise
            else:
                result = self.asr_pipeline(audio)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error transcribing {audio_path}: {e}")
            return None
    
    def _compute_error_rates(self, reference: str, hypothesis: str) -> ASRResult:
        """Compute detailed error rates between reference and hypothesis."""
        # Clean text (normalize whitespace, case)
        ref_clean = self._clean_text(reference)
        hyp_clean = self._clean_text(hypothesis)
        
        # Compute WER
        wer = jiwer.wer(ref_clean, hyp_clean)
        
        # Compute CER
        cer = jiwer.cer(ref_clean, hyp_clean)
        
        # Get detailed alignment for error analysis
        alignment = jiwer.process_words(ref_clean, hyp_clean)
        
        return ASRResult(
            reference_text=ref_clean,
            hypothesis_text=hyp_clean,
            wer=wer,
            cer=cer,
            insertions=alignment.insertions,
            deletions=alignment.deletions,
            substitutions=alignment.substitutions,
            hits=alignment.hits
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean text for consistent comparison."""
        import re
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _compute_transcription_similarity(self, ref_transcript: str, deg_transcript: str) -> float:
        """Compute similarity between two transcriptions."""
        try:
            # Clean texts
            ref_clean = self._clean_text(ref_transcript)
            deg_clean = self._clean_text(deg_transcript)
            
            # Use inverse of normalized edit distance as similarity
            if not ref_clean and not deg_clean:
                return 1.0  # Both empty
            
            if not ref_clean or not deg_clean:
                return 0.0  # One empty
            
            # Compute edit distance
            edit_distance = jiwer.wer(ref_clean, deg_clean)
            
            # Convert to similarity (1 - normalized_edit_distance)
            similarity = max(0.0, 1.0 - edit_distance)
            
            return similarity
            
        except Exception as e:
            self.logger.error(f"Error computing transcription similarity: {e}")
            return 0.0
    
    def batch_evaluate(self, audio_pairs: List[Tuple[str, str]], 
                      reference_texts: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Evaluate WER for multiple audio pairs.
        
        Args:
            audio_pairs: List of (reference_audio, degraded_audio) tuples
            reference_texts: Optional list of ground truth transcripts
            
        Returns:
            Dictionary mapping pair identifiers to evaluation results
        """
        results = {}
        
        for i, (ref_audio, deg_audio) in enumerate(audio_pairs):
            pair_id = f"pair_{i+1}_{Path(ref_audio).stem}_{Path(deg_audio).stem}"
            
            ref_text = reference_texts[i] if reference_texts else None
            
            results[pair_id] = self.evaluate_wer(
                ref_audio, deg_audio, ref_text, return_details=False
            )
        
        return results
    
    def _get_nan_results(self) -> Dict[str, float]:
        """Return NaN results for error cases."""
        return {
            'wer_reference': float('nan'),
            'wer_degraded': float('nan'),
            'wer_increase': float('nan'),
            'cer_reference': float('nan'),
            'cer_degraded': float('nan'),
            'transcription_similarity': float('nan'),
            'relative_wer': float('nan'),
            'relative_cer': float('nan')
        }

    def evaluate_degraded_only(self, degraded_audio: str, reference_text: str,
                               return_details: bool = True) -> Dict[str, float]:
        """
        Evaluate ASR on a degraded audio clip against ground-truth text only.

        Args:
            degraded_audio: Path to degraded audio file
            reference_text: Ground truth transcript
            return_details: If True, include transcript details in output

        Returns:
            Dictionary containing degraded WER/CER and optional transcript details.
        """
        try:
            deg_result = self._transcribe_audio(degraded_audio)
            if deg_result is None:
                return {
                    'wer_degraded': float('nan'),
                    'cer_degraded': float('nan'),
                }

            deg_transcript = deg_result.get('text', '')
            asr_result = self._compute_error_rates(reference_text, deg_transcript)

            result = {
                'wer_degraded': asr_result.wer,
                'cer_degraded': asr_result.cer,
            }

            if return_details:
                result.update({
                    'degraded_transcript': deg_transcript,
                    'ground_truth': reference_text,
                    'insertions_deg': asr_result.insertions,
                    'deletions_deg': asr_result.deletions,
                    'substitutions_deg': asr_result.substitutions,
                })

            return result

        except Exception as e:
            self.logger.error(f"Error evaluating degraded-only WER: {e}")
            return {
                'wer_degraded': float('nan'),
                'cer_degraded': float('nan'),
            }
    
    def evaluate_from_manifest(self, manifest_path: str, codec_output_dir: str) -> Dict[str, Dict[str, float]]:
        """
        Evaluate ASR impact using a manifest file.
        
        Args:
            manifest_path: Path to dataset manifest YAML file
            codec_output_dir: Directory containing codec outputs
            
        Returns:
            Dictionary mapping audio IDs to evaluation results
        """
        try:
            import yaml
            
            # Load manifest
            with open(manifest_path, 'r') as f:
                manifest = yaml.safe_load(f)
            
            results = {}
            
            for item in manifest.get('items', []):
                audio_id = item['id']
                transcript = item.get('transcript', '')
                
                # Find reference and degraded audio files
                ref_audio = None  # Would need to be provided or inferred
                deg_audio = Path(codec_output_dir) / f"{audio_id}.wav"
                
                if deg_audio.exists():
                    if ref_audio and Path(ref_audio).exists():
                        result = self.evaluate_wer(str(ref_audio), str(deg_audio), transcript)
                    else:
                        # Evaluate only degraded audio against transcript
                        deg_result = self._transcribe_audio(str(deg_audio))
                        if deg_result:
                            deg_transcript = deg_result.get('text', '')
                            asr_result = self._compute_error_rates(transcript, deg_transcript)
                            result = {
                                'wer_degraded': asr_result.wer,
                                'cer_degraded': asr_result.cer,
                                'transcription_similarity': 0.0  # No reference to compare
                            }
                        else:
                            result = self._get_nan_results()
                    
                    results[audio_id] = result
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error evaluating from manifest: {e}")
            return {}


def compute_wer(reference_audio: str, degraded_audio: str, ground_truth: str) -> float:
    """
    Legacy interface for WER computation.
    
    Args:
        reference_audio: Path to reference audio
        degraded_audio: Path to degraded audio
        ground_truth: Ground truth transcript
        
    Returns:
        WER increase due to compression
    """
    evaluator = ASREvaluator()
    result = evaluator.evaluate_wer(reference_audio, degraded_audio, ground_truth, return_details=False)
    return result.get('wer_increase', float('nan'))


if __name__ == "__main__":
    # Example usage
    evaluator = ASREvaluator(model_name="openai/whisper-base")
    
    # Test on sample files
    ref_file = "reference.wav"
    deg_file = "degraded.wav"
    ground_truth = "Hello, this is a test sentence."
    
    if Path(ref_file).exists() and Path(deg_file).exists():
        results = evaluator.evaluate_wer(ref_file, deg_file, ground_truth, return_details=True)
        print(f"ASR evaluation results: {results}")
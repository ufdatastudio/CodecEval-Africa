import os
import torch
from jiwer import wer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from typing import List, Union

class ASRModel:
    """ASR model for computing WER on audio files using Whisper."""
    
    def __init__(self, model_name: str = "openai/whisper-large-v3"):
        """Initialize ASR model."""
        print(f"Loading ASR model: {model_name}")
        
        # Setup device and dtype
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        print(f"Using device: {self.device}, dtype: {self.torch_dtype}")
        
        # Load model and processor
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self.model.to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Create pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=False,  # We don't need timestamps for WER
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        
        print(f"ASR model loaded on: {self.device}")
    
    def transcribe(self, audio_path: str, language: str = "en") -> str:
        """Transcribe audio file to text."""
        try:
            # Use Whisper pipeline for transcription
            result = self.pipe(
                audio_path,
                generate_kwargs={
                    "language": language,
                    "task": "transcribe",  # or "translate" if needed
                },
                return_timestamps=False,
            )
            
            # Extract text from result
            if isinstance(result, dict) and "text" in result:
                return result["text"].strip()
            elif isinstance(result, str):
                return result.strip()
            else:
                print(f"Unexpected result format: {type(result)}")
                return ""
            
        except Exception as e:
            print(f"Error transcribing {audio_path}: {e}")
            return ""

def compute_wer(ref_texts: Union[str, List[str]], hyp_texts: Union[str, List[str]]) -> float:
    """Compute Word Error Rate between reference and hypothesis texts."""
    return wer(ref_texts, hyp_texts)

def compute_wer_from_audio(ref_audio: str, hyp_audio: str, asr_model: ASRModel = None) -> dict:
    """Compute WER by transcribing audio files."""
    if asr_model is None:
        asr_model = ASRModel()
    
    # Transcribe both audio files
    ref_text = asr_model.transcribe(ref_audio)
    hyp_text = asr_model.transcribe(hyp_audio)
    
    # Compute WER
    wer_score = compute_wer(ref_text, hyp_text)
    
    return {
        "wer": wer_score,
        "ref_text": ref_text,
        "hyp_text": hyp_text,
        "ref_audio": ref_audio,
        "hyp_audio": hyp_audio
    }

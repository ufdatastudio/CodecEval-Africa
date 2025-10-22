import numpy as np
import torch
import torchaudio
from typing import Optional, Tuple

def compute_speaker_similarity(ref_audio: str, hyp_audio: str) -> float:
    """
    Compute speaker similarity using cosine similarity of MFCC features.
    
    This is a simplified implementation that uses MFCC features as a proxy
    for speaker embeddings until a full speaker verification model is integrated.
    
    Args:
        ref_audio: Path to reference audio file
        hyp_audio: Path to hypothesis audio file
        
    Returns:
        Similarity score (0-1, higher = more similar)
    """
    try:
        # Extract MFCC features from both audio files
        ref_features = _extract_mfcc_features(ref_audio)
        hyp_features = _extract_mfcc_features(hyp_audio)
        
        if ref_features is None or hyp_features is None:
            return 0.0
        
        # Compute cosine similarity
        similarity = _cosine_similarity(ref_features, hyp_features)
        
        return float(similarity)
        
    except Exception as e:
        print(f"Error computing speaker similarity: {e}")
        return 0.0

def _extract_mfcc_features(audio_path: str) -> Optional[np.ndarray]:
    """Extract MFCC features from audio file."""
    try:
        # Load audio using soundfile (more reliable)
        import soundfile as sf
        audio, sample_rate = sf.read(audio_path, always_2d=False)
        
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            import librosa
            audio = librosa.resample(audio.astype(float), orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        # Convert to tensor for MFCC computation
        waveform = torch.from_numpy(audio).float().unsqueeze(0)
        
        # Compute MFCC features
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=13,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23}
        )
        
        mfcc = mfcc_transform(waveform)
        
        # Average over time to get speaker embedding
        speaker_embedding = mfcc.mean(dim=-1).squeeze().numpy()
        
        return speaker_embedding
        
    except Exception as e:
        print(f"Error extracting MFCC features from {audio_path}: {e}")
        return None

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Normalize vectors
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    
    # Compute cosine similarity
    similarity = np.dot(a_norm, b_norm)
    
    return float(similarity)

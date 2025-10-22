import soundfile as sf
import numpy as np

def load_wav(path, sr):
    audio, sr0 = sf.read(path, always_2d=False)
    if sr0 != sr:
        import librosa
        audio = librosa.resample(audio.astype(float), orig_sr=sr0, target_sr=sr)
    if audio.ndim == 1:
        audio = audio[None, :]  # [1, T]
    return audio

def save_wav(audio, path, sr):
    x = np.asarray(audio)
    if x.ndim == 2 and x.shape[0] == 1:
        x = x[0]
    sf.write(path, x, sr)

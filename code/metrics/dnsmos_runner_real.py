import os
import sys
import numpy as np
import librosa
import soundfile as sf
import onnxruntime as ort
from typing import Optional, Dict

# Add the DNSMOS directory to the path
DNSMOS_ROOT = "/orange/ufdatastudios/c.okocha/DNSMOS/DNS-Challenge/DNSMOS"
sys.path.insert(0, DNSMOS_ROOT)

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01

class RealDNSMOS:
    def __init__(self, personalized_MOS=False):
        """
        Initialize the real DNSMOS model.
        
        Args:
            personalized_MOS: If True, use personalized DNSMOS (pDNSMOS)
        """
        self.personalized_MOS = personalized_MOS
        
        # Set up ONNX runtime options
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        
        # Model paths
        p808_model_path = os.path.join(DNSMOS_ROOT, 'DNSMOS', 'model_v8.onnx')
        
        if personalized_MOS:
            primary_model_path = os.path.join(DNSMOS_ROOT, 'pDNSMOS', 'sig_bak_ovr.onnx')
        else:
            primary_model_path = os.path.join(DNSMOS_ROOT, 'DNSMOS', 'sig_bak_ovr.onnx')
        
        # Initialize ONNX sessions
        self.onnx_sess = ort.InferenceSession(
            primary_model_path, 
            sess_options=so, 
            providers=['CPUExecutionProvider']
        )
        self.p808_onnx_sess = ort.InferenceSession(
            p808_model_path, 
            sess_options=so, 
            providers=['CPUExecutionProvider']
        )
    
    def audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        """Compute mel-spectrogram features."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=frame_size+1, 
            hop_length=hop_length, n_mels=n_mels
        )
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max)+40)/40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        """Apply polynomial fitting to raw scores."""
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021,  0.005101  ,  1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296,  0.02751166,  1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499,  0.44276479, -0.1644611 ,  0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
            p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439 ])
            p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def compute_score(self, audio_path: str) -> Dict[str, float]:
        """
        Compute DNSMOS scores for a single audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary with SIG, BAK, OVR, and P808_MOS scores
        """
        try:
            # Load audio
            aud, input_fs = sf.read(audio_path, always_2d=False)
            if np.ndim(aud) > 1:
                aud = np.mean(aud, axis=1)
            
            # Resample if needed
            if input_fs != SAMPLING_RATE:
                audio = librosa.resample(aud, orig_sr=input_fs, target_sr=SAMPLING_RATE, axis=0)
            else:
                audio = aud
            
            actual_audio_len = len(audio)
            len_samples = int(INPUT_LENGTH * SAMPLING_RATE)
            
            # Pad audio if too short
            while len(audio) < len_samples:
                audio = np.append(audio, audio)
            
            # Process in segments
            num_hops = int(np.floor(len(audio)/SAMPLING_RATE) - INPUT_LENGTH) + 1
            hop_len_samples = SAMPLING_RATE
            
            predicted_mos_sig_seg = []
            predicted_mos_bak_seg = []
            predicted_mos_ovr_seg = []
            predicted_p808_mos = []
            
            for idx in range(num_hops):
                audio_seg = audio[int(idx*hop_len_samples) : int((idx+INPUT_LENGTH)*hop_len_samples)]
                if len(audio_seg) < len_samples:
                    continue
                
                # Prepare input features
                input_features = np.array(audio_seg).astype('float32')[np.newaxis, :]
                p808_input_features = np.array(
                    self.audio_melspec(audio=audio_seg[:-160])
                ).astype('float32')[np.newaxis, :, :]
                
                # Run inference
                oi = {'input_1': input_features}
                p808_oi = {'input_1': p808_input_features}
                
                p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
                mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
                
                # Apply polynomial fitting
                mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(
                    mos_sig_raw, mos_bak_raw, mos_ovr_raw, self.personalized_MOS
                )
                
                predicted_mos_sig_seg.append(mos_sig)
                predicted_mos_bak_seg.append(mos_bak)
                predicted_mos_ovr_seg.append(mos_ovr)
                predicted_p808_mos.append(p808_mos)
            
            # Return average scores
            return {
                'sig': float(np.mean(predicted_mos_sig_seg)),
                'bak': float(np.mean(predicted_mos_bak_seg)),
                'ovr': float(np.mean(predicted_mos_ovr_seg)),
                'p808_mos': float(np.mean(predicted_p808_mos))
            }
            
        except Exception as e:
            print(f"Error computing DNSMOS scores for {audio_path}: {e}")
            return {
                'sig': float('nan'),
                'bak': float('nan'),
                'ovr': float('nan'),
                'p808_mos': float('nan')
            }

# Global instance for caching
_dnsmos_instance = None

def score(wav_path: str, sr: Optional[int] = None, personalized_MOS: bool = False) -> Dict[str, float]:
    """
    Compute DNSMOS scores for speech quality assessment.
    
    Args:
        wav_path: Path to audio file
        sr: Sample rate (ignored, DNSMOS uses 16kHz)
        personalized_MOS: If True, use personalized DNSMOS
        
    Returns:
        Dictionary with SIG, BAK, OVR, and P808_MOS scores
    """
    global _dnsmos_instance
    
    # Initialize instance if needed
    if _dnsmos_instance is None or _dnsmos_instance.personalized_MOS != personalized_MOS:
        _dnsmos_instance = RealDNSMOS(personalized_MOS=personalized_MOS)
    
    return _dnsmos_instance.compute_score(wav_path)

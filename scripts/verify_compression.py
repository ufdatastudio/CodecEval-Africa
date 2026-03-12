#!/usr/bin/env python3
"""
Codec Compression Verification Script
Verifies actual compression by analyzing encoded token representations.

This script modifies codec runners to save and analyze intermediate representations.
"""

import os
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Tuple
import json


class CompressionVerifier:
    """Verify actual compression ratios and bitrates from codec token representations."""
    
    def __init__(self, codec_name: str, output_dir: str):
        self.codec_name = codec_name
        self.output_dir = output_dir
        self.results = []
    
    def calculate_bitrate_from_codes(self, codes: torch.Tensor, duration: float, 
                                     codebook_size: int = None) -> Dict:
        """
        Calculate actual bitrate from encoded codes.
        
        Args:
            codes: Tensor of encoded tokens/codes [num_codebooks, sequence_length]
            duration: Audio duration in seconds
            codebook_size: Size of codebook (for calculating bits per token)
        
        Returns:
            Dictionary with bitrate information
        """
        num_codebooks, seq_length = codes.shape
        
        # Calculate bits per token
        if codebook_size:
            bits_per_token = np.log2(codebook_size)
        else:
            bits_per_token = 10  # Default assumption (1024 codebook size)
        
        # Calculate frame rate (tokens per second)
        frame_rate = seq_length / duration
        
        # Calculate bitrate per codebook
        bitrate_per_codebook = frame_rate * bits_per_token
        
        # Total bitrate
        total_bitrate = bitrate_per_codebook * num_codebooks
        
        return {
            'num_codebooks': num_codebooks,
            'sequence_length': seq_length,
            'duration_sec': duration,
            'frame_rate_hz': frame_rate,
            'bits_per_token': bits_per_token,
            'bitrate_per_codebook_kbps': bitrate_per_codebook / 1000,
            'total_bitrate_kbps': total_bitrate / 1000,
            'compression_info': {
                'num_codes': num_codebooks * seq_length,
                'total_bits': num_codebooks * seq_length * bits_per_token
            }
        }
    
    def verify_encodec_compression(self, audio_path: str, model, target_bandwidth: float) -> Dict:
        """Verify Encodec compression."""
        from encodec.utils import convert_audio
        
        # Load audio
        wav, sr = sf.read(audio_path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = torch.tensor(wav, dtype=torch.float32)
        
        # Prepare for model
        x = convert_audio(
            wav.unsqueeze(0).unsqueeze(0),
            sr,
            model.sample_rate,
            model.channels
        )
        
        # Encode
        model.set_target_bandwidth(target_bandwidth)
        with torch.no_grad():
            encoded_frames = model.encode(x.to(model.device))
        
        # Get codes
        codes = encoded_frames[0][0]  # [num_codebooks, seq_length]
        
        # Calculate duration
        duration = len(wav) / sr
        
        # Calculate actual bitrate
        bitrate_info = self.calculate_bitrate_from_codes(
            codes.cpu(), 
            duration,
            codebook_size=1024  # Encodec uses 1024 codebook size
        )
        
        bitrate_info['target_bandwidth_kbps'] = target_bandwidth
        bitrate_info['audio_path'] = audio_path
        bitrate_info['codec'] = 'Encodec'
        
        # Calculate error
        actual_bitrate = bitrate_info['total_bitrate_kbps']
        bitrate_info['bitrate_error_percent'] = abs(actual_bitrate - target_bandwidth) / target_bandwidth * 100
        
        return bitrate_info
    
    def verify_dac_compression(self, audio_path: str, model) -> Dict:
        """Verify DAC compression."""
        import dac
        
        # Load audio
        signal, sr = dac.utils.load(audio_path, sample_rate=model.sample_rate)
        
        # Encode
        with torch.no_grad():
            z, codes, latents, _, _ = model.encode(signal.to(model.device))
        
        # codes shape: [batch, num_codebooks, seq_length]
        codes = codes[0]  # Remove batch dimension
        
        duration = signal.shape[-1] / model.sample_rate
        
        # DAC uses different codebook sizes depending on model
        bitrate_info = self.calculate_bitrate_from_codes(
            codes.cpu(),
            duration,
            codebook_size=1024  # DAC typically uses 1024
        )
        
        bitrate_info['audio_path'] = audio_path
        bitrate_info['codec'] = 'DAC'
        
        return bitrate_info
    
    def verify_languagecodec_compression(self, audio_path: str, model, bandwidth: int) -> Dict:
        """Verify LanguageCodec compression."""
        # Load audio
        wav, sr = sf.read(audio_path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        
        # Resample if needed
        if sr != 16000:
            import torchaudio
            wav = torchaudio.functional.resample(
                torch.tensor(wav).unsqueeze(0), sr, 16000
            ).squeeze(0).numpy()
            sr = 16000
        
        # Encode
        wav_tensor = torch.tensor(wav).unsqueeze(0).unsqueeze(0).to(model.device)
        with torch.no_grad():
            codes = model.encode(wav_tensor, bandwidth=bandwidth)
        
        # codes shape varies by bandwidth
        codes = codes[0] if isinstance(codes, list) else codes
        
        duration = len(wav) / sr
        
        bitrate_info = self.calculate_bitrate_from_codes(
            codes.cpu() if torch.is_tensor(codes) else torch.tensor(codes),
            duration,
            codebook_size=1024
        )
        
        bitrate_info['bandwidth'] = bandwidth
        bitrate_info['audio_path'] = audio_path
        bitrate_info['codec'] = 'LanguageCodec'
        
        return bitrate_info
    
    def verify_wavtokenizer_compression(self, audio_path: str, model) -> Dict:
        """Verify WavTokenizer compression."""
        from decoder.pretrained import WavTokenizer
        
        # Load audio
        wav, sr = sf.read(audio_path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        
        # Resample to 24kHz
        if sr != 24000:
            import torchaudio
            wav = torchaudio.functional.resample(
                torch.tensor(wav).unsqueeze(0), sr, 24000
            ).squeeze(0).numpy()
            sr = 24000
        
        # Encode
        wav_tensor = torch.tensor(wav).unsqueeze(0).to(model.device)
        with torch.no_grad():
            features, discrete_code = model.encode_infer(wav_tensor)
        
        # discrete_code shape: [batch, layers, seq_length]
        codes = discrete_code[0]  # Remove batch
        
        duration = len(wav) / sr
        
        bitrate_info = self.calculate_bitrate_from_codes(
            codes.cpu(),
            duration,
            codebook_size=4096  # WavTokenizer uses 4096 codebook
        )
        
        bitrate_info['audio_path'] = audio_path
        bitrate_info['codec'] = 'WavTokenizer'
        
        return bitrate_info


def verify_codec_compression_batch(codec_name: str, audio_dir: str, 
                                   config: Dict, output_file: str = None):
    """
    Verify compression for a batch of audio files.
    
    Args:
        codec_name: Name of codec (Encodec, DAC, LanguageCodec, WavTokenizer)
        audio_dir: Directory containing audio files
        config: Configuration for codec (bitrates, bandwidths, etc.)
        output_file: Path to save results JSON
    """
    verifier = CompressionVerifier(codec_name, audio_dir)
    
    # Get audio files
    audio_files = list(Path(audio_dir).rglob("*.wav"))[:10]  # Sample 10 files
    
    print(f"Verifying {codec_name} compression on {len(audio_files)} files...")
    
    results = []
    
    if codec_name == 'Encodec':
        from encodec import EncodecModel
        model = EncodecModel.encodec_model_24khz()
        model.eval()
        
        for audio_path in audio_files:
            for bandwidth in config.get('bitrates', [3.0, 6.0, 12.0, 24.0]):
                try:
                    result = verifier.verify_encodec_compression(
                        str(audio_path), model, bandwidth
                    )
                    results.append(result)
                    print(f"  {audio_path.name} @ {bandwidth}kbps: "
                          f"Actual: {result['total_bitrate_kbps']:.2f}kbps, "
                          f"Error: {result['bitrate_error_percent']:.1f}%")
                except Exception as e:
                    print(f"  Error: {e}")
    
    elif codec_name == 'DAC':
        import dac
        model_path = dac.utils.download(model_type="44khz")
        model = dac.DAC.load(model_path)
        model.eval()
        
        for audio_path in audio_files:
            try:
                result = verifier.verify_dac_compression(str(audio_path), model)
                results.append(result)
                print(f"  {audio_path.name}: "
                      f"Actual: {result['total_bitrate_kbps']:.2f}kbps")
            except Exception as e:
                print(f"  Error: {e}")
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n✓ Compression verification saved to {output_file}")
    
    # Print summary
    if results:
        avg_bitrate = np.mean([r['total_bitrate_kbps'] for r in results])
        print(f"\nSummary:")
        print(f"  Average actual bitrate: {avg_bitrate:.2f} kbps")
        if 'bitrate_error_percent' in results[0]:
            avg_error = np.mean([r['bitrate_error_percent'] for r in results])
            print(f"  Average error: {avg_error:.1f}%")
    
    return results


if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify codec compression')
    parser.add_argument('--codec', required=True, 
                        choices=['Encodec', 'DAC', 'LanguageCodec', 'WavTokenizer'],
                        help='Codec to verify')
    parser.add_argument('--audio-dir', required=True,
                        help='Directory with audio files')
    parser.add_argument('--output', default='compression_verification.json',
                        help='Output JSON file')
    
    args = parser.parse_args()
    
    config = {
        'Encodec': {'bitrates': [3.0, 6.0, 12.0, 24.0]},
        'DAC': {},
        'LanguageCodec': {'bandwidths': [0, 1, 2, 3]},
        'WavTokenizer': {}
    }
    
    verify_codec_compression_batch(
        args.codec,
        args.audio_dir,
        config.get(args.codec, {}),
        args.output
    )

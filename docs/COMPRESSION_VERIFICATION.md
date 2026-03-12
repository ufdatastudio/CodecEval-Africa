# Compression Verification Strategy

## The Problem

Your current verification script (**verify_codec_outputs.py**) only checks:
- ✅ Sample rate
- ✅ Channels  
- ✅ File bitrate (uncompressed WAV)

But it **DOESN'T verify actual compression** because it only looks at decoded WAV files, not the intermediate compressed representations.

## Understanding Codec Compression

### How Neural Audio Codecs Work

```
Input WAV → [Encoder] → Tokens/Codes → [Decoder] → Output WAV
              (3-24 kbps)                             (384 kbps)
              ↑ COMPRESSION HERE ↑
```

The compression happens in the **Token/Code** stage, but your current outputs only save the final decoded WAV.

### Example: Encodec at 6 kbps

1. **Input**: 10 seconds of 24kHz audio
2. **Encoding**: 
   - Uses Residual Vector Quantization (RVQ)
   - Produces codes: `[num_codebooks, sequence_length]`
   - Example: `[4, 750]` = 4 codebooks, 750 frames
3. **Bitrate Calculation**:
   ```
   Frame rate = 750 frames / 10 sec = 75 Hz
   Bits per token = log2(1024) = 10 bits (codebook size = 1024)
   Bitrate per codebook = 75 Hz × 10 bits = 750 bps
   Total bitrate = 4 codebooks × 750 bps = 3000 bps = 3 kbps
   ```
4. **Decoding**: Codes → 24kHz 16-bit WAV = 384 kbps file

## Verification Methods

### ❌ Method 1: Check WAV File Bitrate (What you're doing now)
**Limitation**: Only verifies the decoded output, not the compression

```python
file_bitrate = (file_size * 8) / duration
# Result: ~384 kbps for 24kHz audio
# Doesn't tell you about compression!
```

### ✅ Method 2: Calculate from Token Representations (Accurate)

**Required**: Access to encoded tokens during encoding

```python
def calculate_actual_bitrate(codes, duration):
    """
    codes: tensor of shape [num_codebooks, sequence_length]
    """
    num_codebooks, seq_length = codes.shape
    frame_rate = seq_length / duration
    bits_per_token = log2(codebook_size)  # Usually 10 bits for 1024 codebook
    
    bitrate = num_codebooks * frame_rate * bits_per_token
    return bitrate / 1000  # kbps
```

### ✅ Method 3: File Size of Compressed Format (If saved)

If you save the compressed tokens to disk:

```python
compressed_size_bits = os.path.getsize('codes.bin') * 8
bitrate = compressed_size_bits / duration / 1000  # kbps
```

## Current Implementation Gap

Your codec runners **don't save the intermediate representations**:

```python
# In code/codecs/encodec_runner.py
encoded = model.encode(x)        # Compression happens here
decoded = model.decode(encoded)  # Immediately decoded
save_wav(decoded, path, sr)      # Only save decoded WAV
# ❌ The 'encoded' codes are never saved or analyzed!
```

## Solutions

### Solution 1: Modify Codec Runners to Save Codes

Update each codec runner to save the encoded representations:

```python
def encodec_reconstruct_with_codes(model, wav, chunk_sec=10.0, device='cuda'):
    """Modified to save and analyze codes."""
    sr = model.sample_rate
    chunk_len = int(chunk_sec * sr)
    decoded_chunks = []
    all_codes = []
    
    for start in range(0, len(wav), chunk_len):
        end = min(start + chunk_len, len(wav))
        x = wav[start:end].unsqueeze(0).unsqueeze(0).to(device)
        
        # Encode
        encoded = model.encode(x)
        codes = encoded[0][0]  # Extract codes [num_codebooks, seq_len]
        all_codes.append(codes)
        
        # Calculate bitrate for this chunk
        duration = (end - start) / sr
        bitrate = calculate_bitrate_from_codes(codes, duration)
        
        # Decode
        decoded = model.decode(encoded)
        decoded_chunks.append(decoded.squeeze().cpu())
    
    # Save codes for verification
    all_codes = torch.cat(all_codes, dim=1)
    return torch.cat(decoded_chunks), all_codes
```

### Solution 2: Use the Compression Verification Script

I've created **scripts/verify_compression.py** that:
- ✅ Loads the codec model
- ✅ Re-encodes sample audio files
- ✅ Extracts the token representations
- ✅ Calculates actual bitrate from tokens
- ✅ Compares against target bitrate

**Usage**:
```bash
# Verify Encodec compression
python scripts/verify_compression.py \
  --codec Encodec \
  --audio-dir data/afri_names_150_flat \
  --output encodec_compression_check.json

# Verify DAC compression
python scripts/verify_compression.py \
  --codec DAC \
  --audio-dir data/afri_names_150_flat \
  --output dac_compression_check.json
```

### Solution 3: Indirect Verification via File Sizes

Compare output file sizes across different bitrate settings:

```python
# Expected: Higher bitrate → Larger file (if model works correctly)
# But wait... WAV files are uncompressed!
# So this won't work unless you save compressed format
```

**This doesn't work** for WAV outputs because they're all uncompressed at the same sample rate.

## Codec-Specific Verification

### Encodec
```python
from encodec import EncodecModel
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # 6 kbps

encoded_frames = model.encode(audio)
codes = encoded_frames[0][0]  # [num_codebooks, seq_length]

# Verify num_codebooks matches bandwidth:
# 3 kbps → 2 codebooks
# 6 kbps → 4 codebooks  
# 12 kbps → 8 codebooks
# 24 kbps → 16 codebooks
```

### DAC
```python
import dac
model = dac.DAC.load("path/to/model")

z, codes, latents, commitment_loss, codebook_loss = model.encode(signal)
# codes: [batch, num_codebooks, seq_length]

# DAC at different bitrates uses different numbers of codebooks
```

### LanguageCodec
```python
# Uses bandwidth parameter (0-3)
codes = model.encode(audio, bandwidth=2)
# Different bandwidths use different numbers of codebooks
```

### WavTokenizer
```python
features, discrete_code = model.encode_infer(audio)
# discrete_code: [batch, num_layers, seq_length]
# Codebook size: 4096 (12 bits per token)
```

## Recommended Verification Workflow

1. **Quick Check** (Current): Verify sample rates and file formats
   ```bash
   python scripts/verify_codec_outputs.py --save-json
   ```

2. **Deep Verification** (New): Verify actual compression on sample files
   ```bash
   python scripts/verify_compression.py --codec Encodec --audio-dir data/sample
   ```

3. **Quality Metrics**: Verify reconstruction quality
   ```bash
   python code/audio_quality_assessment/run_nisqa.py
   ```

## Key Formulas

### Bitrate from Codes
```
Bitrate (kbps) = (num_codebooks × frame_rate × bits_per_token) / 1000

Where:
- num_codebooks: Number of quantization levels (varies by target bitrate)
- frame_rate: Frames per second = sequence_length / duration
- bits_per_token: log2(codebook_size)
  - 1024 codebook → 10 bits
  - 2048 codebook → 11 bits  
  - 4096 codebook → 12 bits
```

### Expected Frame Rates (for 75Hz models)
```
24kHz audio, 75Hz frame rate:
- 1 second audio → 75 frames
- 10 second audio → 750 frames
- Hop size: 24000 / 75 = 320 samples
```

## Summary

Your current verification script **cannot verify compression** because:
1. ❌ It only checks decoded WAV files
2. ❌ WAV files are uncompressed (always same bitrate for same sample rate)
3. ❌ The compressed tokens are never saved or analyzed

To **truly verify compression**, you need to:
1. ✅ Extract encoded token representations during encoding
2. ✅ Count codebooks and sequence length
3. ✅ Calculate actual bitrate from token dimensions
4. ✅ Compare against target bitrate

Use the new **verify_compression.py** script I created to do this!

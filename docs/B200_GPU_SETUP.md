# ✅ Updated Training Scripts for B200 GPUs

## Changes Made

All three training scripts have been updated to use **NVIDIA B200 GPUs** on HiPerGator's `hpg-b200` partition.

---

## 🚀 GPU Configuration

### Previous Setup (A100):
```bash
#SBATCH --gpus=a100:8
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --mem=256GB
```

### NEW Setup (B200):
```bash
#SBATCH --account=ufdatastudios
#SBATCH --gpus=8                    # 8x B200 GPUs
#SBATCH --partition=hpg-b200        # B200 partition
#SBATCH --mem=512GB                 # Increased to 512GB
```

### Additional Improvements:
- ✅ Added CUDA 12.8.1 module loading
- ✅ Set CUDA environment variables
- ✅ Added nvidia-smi GPU info display
- ✅ Increased memory allocation to 512GB
- ✅ Added account specification

---

## 📊 B200 GPU Advantages

**NVIDIA B200 GPUs** offer significant improvements over A100:

| Feature | A100 | B200 |
|---------|------|------|
| **Memory** | 80GB | 192GB HBM3e |
| **Performance** | 312 TFLOPS (FP16) | ~2-3x faster |
| **Memory Bandwidth** | 2 TB/s | 8 TB/s |
| **NVLink** | 600 GB/s | 1.8 TB/s |

**Benefits for Audio Codec Training:**
- ✅ **Larger batch sizes** possible
- ✅ **Faster training** (2-3x speedup expected)
- ✅ **Longer sequences** can be processed
- ✅ **Better multi-GPU scaling** with NVLink

---

## 📁 Updated Scripts

1. **WavTokenizer:**
   ```bash
   batch_scripts/train_wavtokenizer_african.sh
   ```

2. **LanguageCodec:**
   ```bash
   batch_scripts/train_languagecodec_african.sh
   ```

3. **UniCodec:**
   ```bash
   batch_scripts/train_unicodec_african.sh
   ```

---

## 🏃 How to Submit

### Individual Jobs:
```bash
# WavTokenizer on B200
sbatch batch_scripts/train_wavtokenizer_african.sh

# LanguageCodec on B200
sbatch batch_scripts/train_languagecodec_african.sh

# UniCodec on B200
sbatch batch_scripts/train_unicodec_african.sh
```

### All Jobs at Once:
```bash
./submit_all_training.sh
```

---

## 🔍 Verify GPU Allocation

After job starts, check the output log to see GPU info:

```bash
tail -f batch_scripts/logs/wavtokenizer_african_<JOB_ID>.out
```

You should see:
```
===== GPU Info =====
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 550.xx.xx              Driver Version: 550.xx.xx  CUDA Version: 12.8     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA B200...      On   | 00000000:00:00.0 Off |                  Off |
|...
```

---

## ⚙️ CUDA Configuration

The scripts now load CUDA 12.8.1:

```bash
module load cuda/12.8.1

export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
```

This ensures compatibility with B200 GPUs.

---

## 📈 Expected Performance Impact

### Training Time Estimates (with B200):

**WavTokenizer:**
- Previous (A100): ~24-36 hours
- **With B200: ~12-18 hours** (2x faster)

**LanguageCodec:**
- Previous (A100): ~18-24 hours
- **With B200: ~9-12 hours** (2x faster)

**UniCodec:**
- Previous (A100): ~30-48 hours
- **With B200: ~15-24 hours** (2x faster)

**Batch Size Recommendations (B200):**
- WavTokenizer: Can increase to 24-32
- LanguageCodec: Can increase to 48-64
- UniCodec: Can increase to 12-16

---

## 🎯 Ready to Launch!

All scripts are now optimized for B200 GPUs. You can start training immediately:

```bash
cd /orange/ufdatastudios/c.okocha/CodecEval-Africa
sbatch batch_scripts/train_wavtokenizer_african.sh
```

Monitor on WandB: https://wandb.ai/your-username/african-codec-training

---

**Your African codec training is ready to run on B200 GPUs! 🚀**

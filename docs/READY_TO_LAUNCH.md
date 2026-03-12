# 🎉 African Codec Training - Complete Setup Summary

**Status: ✅ READY TO LAUNCH!**

---

## ✅ What's Complete

### 1. **Dataset Preparation** ✅
- **8,082 audio files** (~50+ hours of African speech)
- **Train/Val split:** 7,274 / 808 files (90/10)
- **Sources:** AfriSpeech Dialog, AfriNames, HF datasets (afri-names, med-convo-nig, afrispeech-parliament)
- **Languages:** Nigerian, Kenyan, South African, Ghanaian, Zimbabwean accents
- **Format:** Ready for all three codecs (WavTokenizer, LanguageCodec, UniCodec)

### 2. **PyTorch Environment** ✅
- **PyTorch 2.5.1+cu124** (B200 compatible)
- **CUDA 12.4** (works with system CUDA 12.8.1)
- **TorchAudio 2.5.1** (latest)
- **WandB installed** for experiment tracking

### 3. **Training Configurations** ✅
- **WavTokenizer** config with African dataset
- **LanguageCodec** config with African dataset
- **UniCodec** config with African dataset (with domain labels)
- All configs include WandB logging

### 4. **SLURM Batch Scripts** ✅
- **B200 GPU optimized** (8x B200 per job)
- **hpg-b200 partition** configured
- **CUDA 12.8.1** module loading
- **512GB memory** allocation
- **72-hour time limit**
- **WandB integration** with your API key

---

## 📊 System Configuration

### GPUs:
- **Type:** NVIDIA B200 (Blackwell architecture)
- **Count:** 8 per job
- **Memory:** 192GB per GPU
- **Partition:** hpg-b200

### Software:
```
PyTorch:    2.5.1+cu124
CUDA:       12.4 (runtime) / 12.8.1 (system)
Python:     3.9+
WandB:      0.25.0
```

### Account:
- **SLURM Account:** ufdatastudios
- **User:** c.okocha@ufl.edu

---

## 🚀 How to Submit Training

### Option 1: Submit Individual Jobs

**WavTokenizer:**
```bash
cd /orange/ufdatastudios/c.okocha/CodecEval-Africa
sbatch batch_scripts/train_wavtokenizer_african.sh
```

**LanguageCodec:**
```bash
cd /orange/ufdatastudios/c.okocha/CodecEval-Africa
sbatch batch_scripts/train_languagecodec_african.sh
```

**UniCodec:**
```bash
cd /orange/ufdatastudios/c.okocha/CodecEval-Africa
sbatch batch_scripts/train_unicodec_african.sh
```

### Option 2: Submit All at Once
```bash
cd /orange/ufdatastudios/c.okocha/CodecEval-Africa
./submit_all_training.sh
```

---

## 📈 Expected Training Times (B200)

| Model | Batch Size | Segment | Estimated Time |
|-------|------------|---------|----------------|
| **WavTokenizer** | 16 (train) | 3s | ~12-18 hours |
| **LanguageCodec** | 32 (train) | 1s | ~9-12 hours |
| **UniCodec** | 8 (train) | 10s | ~15-24 hours |

---

## 📊 Monitoring Training

### WandB Dashboard
**Project:** `african-codec-training`  
**URL:** https://wandb.ai/your-username/african-codec-training

**Logged Metrics:**
- Training/validation loss
- PESQ scores
- UTMOS scores
- Periodicity metrics
- Learning rate
- GPU utilization

### SLURM Commands
```bash
# Check job status
squeue -u c.okocha

# View live logs
tail -f batch_scripts/logs/wavtokenizer_african_<JOB_ID>.out
tail -f batch_scripts/logs/languagecodec_african_<JOB_ID>.out
tail -f batch_scripts/logs/unicodec_african_<JOB_ID>.out

# Cancel a job
scancel <JOB_ID>

# Job details
scontrol show job <JOB_ID>
```

---

## 📁 File Locations

### Dataset Files:
```
/orange/ufdatastudios/c.okocha/CodecEval-Africa/data/codec_training_data/
├── african_speech_train.txt (7,274 files)
├── african_speech_val.txt (808 files)
├── african_speech_train_unicodec.txt (with domain labels)
└── african_speech_val_unicodec.txt (with domain labels)
```

### Training Scripts:
```
/orange/ufdatastudios/c.okocha/CodecEval-Africa/batch_scripts/
├── train_wavtokenizer_african.sh
├── train_languagecodec_african.sh
└── train_unicodec_african.sh
```

### Training Configs:
```
WavTokenizer/configs/african_finetuning.yaml
Languagecodec/configs/african_finetuning.yaml
UniCodec/configs/african_finetuning.yaml
```

### Model Checkpoints (after training):
```
WavTokenizer/logs/african_finetuning/wavtokenizer_african/version_X/checkpoints/
Languagecodec/logs/african_finetuning/languagecodec_african/version_X/checkpoints/
UniCodec/logs/african_finetuning/unicodec_african/version_X/checkpoints/
```

### SLURM Logs:
```
batch_scripts/logs/
├── wavtokenizer_african_<JOB_ID>.out
├── languagecodec_african_<JOB_ID>.out
└── unicodec_african_<JOB_ID>.out
```

---

## 🎯 Training Configuration Details

### WavTokenizer
- **Pretrained:** small-600-24k-4096
- **Learning Rate:** 5e-5 (finetuning)
- **Batch Size:** 16 (train), 4 (val)
- **Segment Length:** 3 seconds (72,000 samples)
- **Max Steps:** 50,000

### LanguageCodec
- **Pretrained:** languagecodec_paper_8nq
- **Learning Rate:** 5e-5 (finetuning)
- **Batch Size:** 32 (train), 8 (val)
- **Segment Length:** 1 second (24,000 samples)
- **Max Steps:** 50,000

### UniCodec
- **Pretrained:** unicodec
- **Learning Rate:** 5e-5 (finetuning)
- **Batch Size:** 8 (train), 2 (val)
- **Segment Length:** 10 seconds (240,000 samples)
- **Max Steps:** 50,000

---

## 🔧 Troubleshooting

### If training fails to start:
```bash
# Check SLURM logs
cat batch_scripts/logs/<model>_african_<JOB_ID>.err

# Verify PyTorch
cd /orange/ufdatastudios/c.okocha/CodecEval-Africa
source .venv/bin/activate
python3 -c "import torch; print(torch.__version__, torch.version.cuda)"

# Verify dataset
ls -lh data/codec_training_data/african_speech_train.txt
head -5 data/codec_training_data/african_speech_train.txt
```

### If WandB fails:
```bash
# Re-authenticate
cd /orange/ufdatastudios/c.okocha/CodecEval-Africa
source .venv/bin/activate
wandb login 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
```

### If PyTorch needs reinstall:
```bash
./upgrade_pytorch_b200.sh
```

---

## 📚 Documentation

All documentation is available in the project root:

- **TRAINING_GUIDE.md** - Complete training guide
- **B200_GPU_SETUP.md** - B200 GPU configuration details
- **PYTORCH_B200_UPGRADE.md** - PyTorch upgrade documentation
- **DATASET_READY.md** - Dataset preparation summary
- **data/codec_training_data/README.md** - Dataset documentation

---

## ✉️ Email Notifications

You'll receive emails at **c.okocha@ufl.edu** for:
- ✅ Job start (BEGIN)
- ✅ Job completion (END)
- ✅ Job failure (FAIL)

---

## 🎓 After Training Completes

1. **Find best checkpoint:**
```bash
ls -lht WavTokenizer/logs/african_finetuning/*/version_*/checkpoints/ | head -5
```

2. **Evaluate on test data:**
```bash
# Use the inference scripts in scripts/ directory
```

3. **Compare results in WandB:**
- View training curves
- Compare metrics across models
- Download model artifacts

4. **Run inference on AfriSpeech test sets:**
```bash
# Use the reconstructed audio for downstream tasks
```

---

## 🚀 Quick Start Checklist

- ✅ Dataset prepared (8,082 files)
- ✅ PyTorch upgraded (2.5.1+cu124)
- ✅ Training scripts configured (B200)
- ✅ WandB integrated
- ✅ SLURM scripts ready
- ✅ Documentation complete

**Everything is ready! Just run:**
```bash
cd /orange/ufdatastudios/c.okocha/CodecEval-Africa
sbatch batch_scripts/train_wavtokenizer_african.sh
```

---

**🎉 Your African Speech Codec Training System is Fully Configured and Ready to Launch! 🚀**

Monitor your training at: https://wandb.ai/your-username/african-codec-training

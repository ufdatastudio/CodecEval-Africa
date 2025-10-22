# Batch Scripts Directory

This directory contains all SLURM batch scripts for running CodecEval-Africa experiments on the HPC cluster.

## Script Categories

### 🔧 **Pipeline & Diagnostics**
- `run_encoding_diagnosis.sh` - Diagnose encoding pipeline issues
- `run_test_pipeline_fix.sh` - Test pipeline fixes
- `run_cuda_fix_test.sh` - Test CUDA tensor fixes

### 🎵 **Main Experiments**
- `run_afrispeech_dialog_experiments.sh` - Full AfriSpeech-Dialog evaluation
- `run_afrispeech_experiments.sh` - General AfriSpeech experiments
- `run_codec_eval.sh` - Codec evaluation pipeline

### 📊 **Metrics Analysis**
- `run_all_metrics.sh` - Run all metrics (NISQA, DNSMOS, ViSQOL, Speaker, Prosody)
- `run_all_metrics_test.sh` - Test all metrics on subset
- `run_nisqa_only.sh` - NISQA v2.0 analysis only
- `run_nisqa_test.sh` - Test NISQA on subset
- `run_dnsmos_only.sh` - DNSMOS analysis only
- `run_dnsmos_test.sh` - Test DNSMOS on subset

### 🔄 **Continuation Scripts**
- `run_single_audio_metrics_continue.sh` - Continue metrics from existing decoded files

## Usage

All scripts are designed to run on the `hpg-turin` partition with GPU support:

```bash
# Submit a job
sbatch batch_scripts/run_script_name.sh

# Check job status
squeue -u $USER

# View job output
cat script_name_JOBID.out
cat script_name_JOBID.err
```

## Script Dependencies

- **Python Environment**: `.venv/bin/activate`
- **GPU Support**: `#SBATCH --gpus=1`
- **Partition**: `hpg-turin`
- **Memory**: 8-16GB depending on script

## Recent Fixes

- ✅ **Pipeline Fixed**: Now uses codec registry system
- ✅ **CUDA Issues**: EnCodec/SoundStream use CPU fallback
- ✅ **Metrics Working**: NISQA v2.0, DNSMOS, ViSQOL all functional

## Job Monitoring

```bash
# Check all your jobs
squeue -u $USER

# Cancel a job
scancel JOBID

# View job details
scontrol show job JOBID
```

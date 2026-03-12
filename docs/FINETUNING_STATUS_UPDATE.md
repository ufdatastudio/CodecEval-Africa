# Finetuning Status Update

**Date**: 2026-02-25  
**Status**: Jobs Submitted and Pending in Queue

## Current Job IDs

| Model | Job ID | Status | Partition |
|---|---|---|---|
| WavTokenizer | 25660177 | PENDING (Priority) | hpg-turin |
| LanguageCodec | 25660178 | PENDING (Priority) | hpg-turin |
| UniCodec | 25660179 | PENDING (Priority) | hpg-turin |

## Issues Resolved

### Round 1-6 Failures (Jobs 25598989-25658400):
All jobs failed due to multiple cascading issues that have now been fixed:

1. **B200 GPU Incompatibility** ✓ FIXED
   - PyTorch 2.5.1+cu124 doesn't support B200 GPUs (sm_100)
   - **Solution**: Switched from `hpg-b200` to `hpg-turin` partition

2. **Missing Dependencies** ✓ FIXED
   - `transformers` missing in LanguageCodec
   - `matplotlib` missing in LanguageCodec
   - `tensorboard` missing in all three venvs
   - **Solution**: Installed all missing packages

3. **PyTorch Lightning Strategy Error** ✓ FIXED
   - `strategy: auto` not supported in older PyTorch Lightning
   - **Solution**: Removed strategy parameter, let Lightning auto-detect

4. **Import Errors in UniCodec** ✓ FIXED
   - `from vocos.discriminator_dac import DACDiscriminator` → should be `from decoder.discriminator_dac`
   - `from vocos.loss import DACGANLoss` → should be `from decoder.loss`
   - **Solution**: Fixed imports to use local decoder modules

5. **Sox Library Path** ✓ FIXED
   - `OSError: libsox.so: cannot open shared object file`
   - **Solution**: Added explicit `LD_LIBRARY_PATH=/apps/sox/14.4.2/lib`

6. **GPU Allocation** ✓ FIXED
   - Originally requested 2 GPUs with ddp strategy
   - **Solution**: Changed to 1 GPU with auto strategy

## Current Configuration

### SLURM Settings (All Three Models):
```bash
Partition: hpg-turin
GPUs: 1
CPUs: 8
Memory: 64GB
Time: 6 hours
Account: ufdatastudios
```

### Training Settings:
```yaml
max_steps: 2000
log_every_n_steps: 50
val_check_interval: 500
devices: 1
accelerator: gpu
```

### Virtual Environments:
- WavTokenizer: `.venv_wavtokenizer39` (Python 3.9.17, PyTorch 2.5.1+cu124)
- LanguageCodec: `.venv_languagecodec` (Python 3.10.18, PyTorch 2.5.1+cu124)
- UniCodec: `.venv_unicodec39` (Python 3.9.17, PyTorch 2.5.1+cu124)

### Logging:
- **TensorBoard**: Local logs in each model's `logs/african_finetuning/` directory
- **WandB**: Project `african-codec-training` (account: `c-okocha-university-of-florida`)

## Dataset

- **Training**: `/orange/ufdatastudios/c.okocha/CodecEval-Africa/data/codec_training_data/african_speech_train.txt`
- **Validation**: `/orange/ufdatastudios/c.okocha/CodecEval-Africa/data/codec_training_data/african_speech_val.txt`
- **UniCodec**: Uses labeled versions (`*_unicodec.txt`)

**Total Samples**: ~45,000+ African speech audio files from:
- AfriSpeech Parliament
- Afri-Names  
- Med-Convo-Nig
- AfriSpeech Dialog
- Additional local datasets

## Next Steps

1. Wait for jobs to start (currently pending in queue due to priority)
2. Monitor logs for successful training initialization
3. Track metrics via WandB: https://wandb.ai/c-okocha-university-of-florida/african-codec-training
4. After 2000 steps (~few hours), evaluate finetuned models

## Monitoring Commands

```bash
# Check job status
squeue -u c.okocha

# Check job history
sacct -u c.okocha --format=JobID,JobName,State,ExitCode,Elapsed,NodeList -S 2026-02-25

# View latest logs
tail -f batch_scripts/logs/wavtokenizer_african_25660177.out
tail -f batch_scripts/logs/languagecodec_african_25660178.out
tail -f batch_scripts/logs/unicodec_african_25660179.out

# View WandB runs
# https://wandb.ai/c-okocha-university-of-florida/african-codec-training
```

## Previous Attempts Summary

| Attempt | Jobs | Issue | Fix |
|---|---|---|---|
| 1 | 25598989-91 | B200 GPU sm_100 not supported | Switch to hpg-turin |
| 2 | 25649477-79 | Missing PyTorch (uninstall error) | Reinstall PyTorch 2.5.1 |
| 3 | 25650067-69 | `strategy: auto` not supported | Remove strategy param |
| 4 | 25658398-400 | Missing deps, import errors, sox | Fix all remaining issues |
| 5 | 25660177-79 | **Current** - In queue | Waiting to start |

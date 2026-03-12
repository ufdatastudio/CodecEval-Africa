# CodecEval-Africa: Next Steps Research Plan

## Current Status
**6 Working Codecs Implemented:**
- EnCodec (encodec_24khz) - Meta's production neural codec
- SoundStream (soundstream_impl) - EnCodec-based non-causal
- UniCodec (unicodec) - Simplified neural autoencoder
- DAC (dac) - High-quality residual neural codec
- SemantiCodec (sematicodec) - Semantic-aware with attention
- APCodec (apcodec) - Adaptive perceptual codec

**Metrics Implemented:**
- NISQA (speech quality)
- ViSQOL (perceptual quality)
- DNSMOS (noise suppression)
- Speaker Similarity (MFCC cosine)
- Prosody (F0 RMSE)
- ASR WER (when transcripts available)

## Phase 1: Comprehensive Benchmarking

### 1.1 Full Dataset Evaluation
**Goal:** Run complete evaluation on balanced African datasets

**Tasks:**
- [ ] Update configs/benchmark.yml to use full manifests (not tiny)
- [ ] Run encode/decode on all 6 codecs across multiple bitrates
- [ ] Test on Afri-Names (8 accents × 35 clips each = 280 clips)
- [ ] Test on AfriSpeech-Dialog (6 accents × 35 clips each = 210 clips)
- [ ] Total: ~490 audio clips × 6 codecs × 4 bitrates = ~11,760 evaluations

**Commands:**
```bash
# Update config to full datasets
python scripts/create_balanced_manifests.py  # Generate full manifests
python -m code.pipeline --config configs/benchmark.yml --stage encode_decode
python -m code.pipeline --config configs/benchmark.yml --stage metrics
```

### 1.2 Multi-Condition Testing
**Goal:** Test robustness across different conditions

**Tasks:**
- [ ] Clean audio (baseline)
- [ ] Noisy audio (20dB, 10dB SNR)
- [ ] Reverberant audio (small room)
- [ ] Packet loss simulation (3%, 5%, 10%)

**Implementation:**
- [ ] Update pipeline to apply conditions during encoding
- [ ] Test each codec under each condition
- [ ] Measure quality degradation

### 1.3 Bitrate Analysis
**Goal:** Understand quality vs. bitrate trade-offs

**Tasks:**
- [ ] Test all codecs at: 3, 6, 12, 18 kbps
- [ ] Measure quality metrics at each bitrate
- [ ] Identify optimal bitrate for each codec
- [ ] Compare efficiency across codecs

## Phase 2: Analysis & Insights

### 2.1 Accent-Specific Analysis
**Goal:** Understand codec performance across African accents

**Tasks:**
- [ ] Group results by accent (Nigerian, Ghanaian, Kenyan, etc.)
- [ ] Identify which codecs work best for which accents
- [ ] Analyze accent-specific quality patterns
- [ ] Study cultural/linguistic factors

### 2.2 Codec Comparison
**Goal:** Comprehensive codec evaluation

**Tasks:**
- [ ] Create performance matrices (codec × metric × condition)
- [ ] Identify best codec for each use case
- [ ] Analyze computational efficiency (RTF, memory)
- [ ] Study quality vs. speed trade-offs

### 2.3 Network Condition Analysis
**Goal:** Understand robustness to real-world conditions

**Tasks:**
- [ ] Analyze performance under packet loss
- [ ] Study noise robustness
- [ ] Evaluate reverb handling
- [ ] Identify most robust codecs

## Phase 3: Publication & Dissemination

### 3.1 Visualization & Reporting
**Goal:** Create compelling visualizations and reports

**Tasks:**
- [ ] Generate comprehensive plots and charts
- [ ] Create interactive dashboards
- [ ] Write detailed analysis reports
- [ ] Prepare presentation materials

### 3.2 Academic Output
**Goal:** Contribute to academic literature

**Tasks:**
- [ ] Write research paper
- [ ] Prepare conference presentation
- [ ] Create open-source benchmark
- [ ] Share datasets and code

## Immediate Next Steps (This Week)

### Priority 1: Full Benchmark Run
```bash
# 1. Update config to full datasets
cp configs/benchmark.yml configs/benchmark_full.yml
# Edit to use full manifests instead of tiny

# 2. Run full evaluation
sbatch run_metrics_b200.sh  # This will take several hours

# 3. Analyze results
python scripts/analyze_results.py results/csv/benchmark.csv
```

### Priority 2: Create Analysis Scripts
- [ ] Write comprehensive analysis script
- [ ] Create visualization functions
- [ ] Generate comparison tables
- [ ] Prepare summary statistics

### Priority 3: Documentation
- [ ] Document all codecs and their characteristics
- [ ] Create user guide for the benchmark
- [ ] Write methodology section
- [ ] Prepare README updates

## Success Metrics

**Technical Metrics:**
- [ ] All 6 codecs successfully evaluated
- [ ] Complete dataset coverage (490+ clips)
- [ ] Multiple condition testing
- [ ] Comprehensive quality analysis

**Research Impact:**
- [ ] Novel insights about African speech codecs
- [ ] Practical recommendations for developers
- [ ] Open-source benchmark for community
- [ ] Academic publication ready

## Timeline

**Week 1:** Full benchmark run + initial analysis
**Week 2:** Deep analysis + visualizations
**Week 3:** Report writing + paper preparation
**Week 4:** Final review + submission

---

**Ready to start? Let's begin with the full benchmark run!**

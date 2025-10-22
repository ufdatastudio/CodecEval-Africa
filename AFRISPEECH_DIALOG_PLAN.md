# AfriSpeech-Dialog Codec Evaluation Plan

## 🎯 Focus: AfriSpeech-Dialog Dataset Only

### Dataset Overview
- **Size**: 49 audio samples (~1.8GB)
- **Accents**: 12 African accents (Hausa, Swahili, Isoko, Yoruba, Sesotho, etc.)
- **Content**: Medical consultations and dialogues
- **Duration**: Variable (real conversations)
- **Quality**: Real-world speech with natural prosody

### Experimental Scope
```
Experiments = 6 codecs × 5 bitrates × 49 samples = 1,470 total evaluations
```

## 🔧 Codecs Under Evaluation

1. **EnCodec** (Meta) - Causal neural codec
2. **SoundStream** (Google) - Non-causal neural codec  
3. **UniCodec** (Tencent) - Unified framework
4. **DAC** (Descript) - High-quality residual codec
5. **SemantiCodec** (Meta) - Semantic-aware with attention
6. **APCodec** (Adaptive Perceptual) - Perceptual loss-based

## 📊 Bitrate Analysis

**Target Bitrates**: [3, 6, 12, 18, 24] kbps
- **3 kbps**: Ultra-low bandwidth (mobile/emergency)
- **6 kbps**: Low bandwidth (basic mobile)
- **12 kbps**: Standard quality (typical mobile)
- **18 kbps**: High quality (good mobile/wifi)
- **24 kbps**: Premium quality (high-speed connections)

## 📈 Key Metrics

### Compression Analysis
- **File size tracking**: Original vs. compressed sizes
- **Compression ratios**: Efficiency analysis
- **Bitrate accuracy**: Target vs. actual bitrates

### Quality Metrics
- **NISQA**: Speech quality prediction
- **ViSQOL**: Perceptual quality assessment
- **DNSMOS**: Noise suppression quality

### ASR Impact
- **WER**: Word Error Rate with Whisper Large-v3
- **CER**: Character Error Rate for detailed analysis
- **Medical terminology**: Specialized vocabulary preservation

### Speaker & Prosodic Analysis
- **Speaker similarity**: Voice preservation (cosine similarity)
- **F0 RMSE**: Pitch preservation accuracy
- **Prosodic patterns**: Natural speech rhythm

## 🎯 Expected Outcomes

### Primary Deliverables
1. **Codec Performance Rankings** for African speech
2. **Accent-Specific Analysis** across 12 African accents
3. **Bitrate Recommendations** for different use cases
4. **Compression Efficiency Study** with file size analysis

### Research Insights
1. **Best codec for African languages**
2. **Optimal bitrates for medical consultations**
3. **Accent-specific performance variations**
4. **Real-world deployment guidelines**

## 📅 Execution Plan

### Phase 1: Setup (This Week)
- [x] Update pipeline with compression size tracking
- [ ] Create balanced AfriSpeech-Dialog manifest
- [ ] Test all 6 codecs on sample files
- [ ] Validate metrics and pipeline

### Phase 2: Full Evaluation (Next Week)
- [ ] Run complete AfriSpeech-Dialog evaluation
- [ ] Process all 1,470 codec evaluations
- [ ] Compute comprehensive metrics
- [ ] Generate initial results

### Phase 3: Analysis (Following Week)
- [ ] Accent-specific performance analysis
- [ ] Statistical significance testing
- [ ] Generate publication-quality figures
- [ ] Write results summary

## 🔧 Technical Implementation

### Pipeline Updates
- ✅ Added compression size tracking to pipeline
- ✅ Enhanced metadata with file size information
- ✅ Updated codec registry with all 6 codecs

### Configuration
- Focused config for AfriSpeech-Dialog only
- 5 bitrates for comprehensive analysis
- All 6 neural codecs included
- Complete metrics suite

### Quality Assurance
- Balanced accent distribution (max 5 per accent)
- Reproducible results with fixed seeds
- Comprehensive error handling
- Progress tracking and logging

## 📊 Sample Distribution

### By Accent (AfriSpeech-Dialog)
- **Hausa**: 11 samples (22.4%)
- **Swahili**: 9 samples (18.4%)
- **Isoko**: 7 samples (14.3%)
- **Yoruba**: 6 samples (12.2%)
- **Sesotho**: 6 samples (12.2%)
- **Others**: 10 samples (20.5%)

### By Use Case
- **Medical consultations**: Primary focus
- **Conversational dialogues**: Secondary
- **Real-world scenarios**: Natural speech patterns

## 🚀 Next Steps

### Immediate Actions
1. **Create AfriSpeech-Dialog manifest** with balanced accent distribution
2. **Test pipeline** with compression size tracking
3. **Validate all 6 codecs** on sample files
4. **Run full evaluation** (1,470 experiments)

### Success Metrics
- **Pipeline runs successfully** on all samples
- **Compression ratios calculated** for all codecs
- **Quality metrics computed** across all bitrates
- **Accent-specific analysis** completed
- **Results documented** with clear insights

This focused plan ensures we get comprehensive results on AfriSpeech-Dialog before expanding to other datasets.

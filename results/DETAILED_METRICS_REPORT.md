# Comprehensive Metrics Report (Signal-Level + Application-Level)

Generated: 2026-02-21T16:54:39

This report aggregates all available metric JSON artifacts and reports dataset- and codec-level means plus full variant breakdowns.

## Datasets (from evaluated artifacts)

| Dataset | Variants represented | Typical evaluated files per variant (min–max) |
|---|---:|---:|
| afrispeech_dialog | 30 | 0-2401 |
| afrinames | 30 | 0-22500 |
| afrispeech_multilingual | 30 | 0-10000 |

## Metric Families

- **Signal-level:** NISQA, UTMOS, ViSQOL, STOI, Prosody, F0_RMSE, SPK_SIM
- **Application-level:** ASV_v2 (EER/minDCF), ACC_MAP_v2 (ACC/mAP), WER_B200 (WER/CER)

## Coverage Summary

| Metric | Level | Variant files parsed | Datasets covered | Codecs covered |
|---|---|---:|---|---|
| NISQA | signal-level | 90 | afrinames, afrispeech_dialog, afrispeech_multilingual | DAC, Encodec, FocalCodec, LanguageCodec, SemantiCodec, UniCodec, WavTokenizer |
| UTMOS | signal-level | 90 | afrinames, afrispeech_dialog, afrispeech_multilingual | DAC, Encodec, FocalCodec, LanguageCodec, SemantiCodec, UniCodec, WavTokenizer |
| ViSQOL | signal-level | 90 | afrinames, afrispeech_dialog, afrispeech_multilingual | DAC, Encodec, FocalCodec, LanguageCodec, SemantiCodec, UniCodec, WavTokenizer |
| STOI | signal-level | 90 | afrinames, afrispeech_dialog, afrispeech_multilingual | DAC, Encodec, FocalCodec, LanguageCodec, SemantiCodec, UniCodec, WavTokenizer |
| Prosody | signal-level | 90 | afrinames, afrispeech_dialog, afrispeech_multilingual | DAC, Encodec, FocalCodec, LanguageCodec, SemantiCodec, UniCodec, WavTokenizer |
| F0_RMSE | signal-level | 90 | afrinames, afrispeech_dialog, afrispeech_multilingual | DAC, Encodec, FocalCodec, LanguageCodec, SemantiCodec, UniCodec, WavTokenizer |
| SPK_SIM | signal-level | 90 | afrinames, afrispeech_dialog, afrispeech_multilingual | DAC, Encodec, FocalCodec, LanguageCodec, SemantiCodec, UniCodec, WavTokenizer |
| ASV_v2 | application-level | 90 | afrinames, afrispeech_dialog, afrispeech_multilingual | DAC, Encodec, FocalCodec, LanguageCodec, SemantiCodec, UniCodec, WavTokenizer |
| ACC_MAP_v2 | application-level | 90 | afrinames, afrispeech_dialog, afrispeech_multilingual | DAC, Encodec, FocalCodec, LanguageCodec, SemantiCodec, UniCodec, WavTokenizer |
| WER_B200 | application-level | 30 | afrispeech_dialog | DAC, Encodec, FocalCodec, LanguageCodec, SemantiCodec, UniCodec, WavTokenizer |

## NISQA (signal-level)

### Dataset × Codec Means

| Dataset | Codec | Variants | score | mean_num_scored_files |
|---|---|---|---|---|
| afrinames | DAC | 3 | 3.1549 | 115.00 |
| afrinames | Encodec | 4 | 3.1082 | 150.00 |
| afrinames | FocalCodec | 6 | 3.4535 | 150.00 |
| afrinames | LanguageCodec | 4 | 3.1798 | 67.00 |
| afrinames | SemantiCodec | 6 | 3.1258 | 150.00 |
| afrinames | UniCodec | 1 | 3.2303 | 150.00 |
| afrinames | WavTokenizer | 6 | 3.3774 | 150.00 |
| afrispeech_dialog | DAC | 3 | 3.2592 | 49.00 |
| afrispeech_dialog | Encodec | 4 | 3.1015 | 49.00 |
| afrispeech_dialog | FocalCodec | 6 | 3.0433 | 49.00 |
| afrispeech_dialog | LanguageCodec | 4 | 3.4042 | 49.00 |
| afrispeech_dialog | SemantiCodec | 6 | 3.0814 | 49.00 |
| afrispeech_dialog | UniCodec | 1 | 1.9300 | 49.00 |
| afrispeech_dialog | WavTokenizer | 6 | 3.2010 | 49.00 |
| afrispeech_multilingual | DAC | 3 | 2.8172 | 8.00 |
| afrispeech_multilingual | Encodec | 4 | 2.5578 | 100.00 |
| afrispeech_multilingual | FocalCodec | 6 | 3.0024 | 100.00 |
| afrispeech_multilingual | LanguageCodec | 4 | 2.7418 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 6 | 2.4492 | 100.00 |
| afrispeech_multilingual | UniCodec | 1 | 2.9262 | 100.00 |
| afrispeech_multilingual | WavTokenizer | 6 | 2.9329 | 100.00 |

### Variant-Level Results

| Dataset | Codec | Variant | score | num_scored_files |
|---|---|---|---|---|
| afrinames | DAC | out_16kbps | 3.0236 | 115.00 |
| afrinames | DAC | out_24kbps | 3.1841 | 115.00 |
| afrinames | DAC | out_8kbps | 3.2571 | 115.00 |
| afrinames | Encodec | out_12kbps | 3.2332 | 150.00 |
| afrinames | Encodec | out_24kbps | 3.2849 | 150.00 |
| afrinames | Encodec | out_3kbps | 2.8349 | 150.00 |
| afrinames | Encodec | out_6kbps | 3.0797 | 150.00 |
| afrinames | FocalCodec | focalcodec_12_5hz | 3.4691 | 150.00 |
| afrinames | FocalCodec | focalcodec_25hz | 3.5805 | 150.00 |
| afrinames | FocalCodec | focalcodec_50hz | 3.6573 | 150.00 |
| afrinames | FocalCodec | focalcodec_50hz_2k_causal | 3.3627 | 150.00 |
| afrinames | FocalCodec | focalcodec_50hz_4k_causal | 3.2948 | 150.00 |
| afrinames | FocalCodec | focalcodec_50hz_65k_causal | 3.3567 | 150.00 |
| afrinames | LanguageCodec | bandwidth_0 | 3.1989 | 67.00 |
| afrinames | LanguageCodec | bandwidth_1 | 3.1837 | 67.00 |
| afrinames | LanguageCodec | bandwidth_2 | 3.1689 | 67.00 |
| afrinames | LanguageCodec | bandwidth_3 | 3.1676 | 67.00 |
| afrinames | SemantiCodec | 0.31kbps | 3.2798 | 150.00 |
| afrinames | SemantiCodec | 0.33kbps | 3.2024 | 150.00 |
| afrinames | SemantiCodec | 0.63kbps | 3.0933 | 150.00 |
| afrinames | SemantiCodec | 0.68kbps | 3.0979 | 150.00 |
| afrinames | SemantiCodec | 1.25kbps | 3.0290 | 150.00 |
| afrinames | SemantiCodec | 1.40kbps | 3.0526 | 150.00 |
| afrinames | UniCodec | out_6.6kbps | 3.2303 | 150.00 |
| afrinames | WavTokenizer | WavTokenizer_large-speech-75token | 3.3887 | 150.00 |
| afrinames | WavTokenizer | WavTokenizer_large-unify-40token | 3.1702 | 150.00 |
| afrinames | WavTokenizer | WavTokenizer_medium-music-audio-75token | 2.9432 | 150.00 |
| afrinames | WavTokenizer | WavTokenizer_medium-speech-75token | 3.3111 | 150.00 |
| afrinames | WavTokenizer | WavTokenizer_small-320-24k-4096 | 3.5853 | 150.00 |
| afrinames | WavTokenizer | WavTokenizer_small-600-24k-4096 | 3.8656 | 150.00 |
| afrispeech_dialog | DAC | out_16kbps | 3.2351 | 49.00 |
| afrispeech_dialog | DAC | out_24kbps | 3.3294 | 49.00 |
| afrispeech_dialog | DAC | out_8kbps | 3.2131 | 49.00 |
| afrispeech_dialog | Encodec | out_12kbps | 3.2396 | 49.00 |
| afrispeech_dialog | Encodec | out_24kbps | 3.3582 | 49.00 |
| afrispeech_dialog | Encodec | out_3kbps | 2.7390 | 49.00 |
| afrispeech_dialog | Encodec | out_6kbps | 3.0693 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_12_5hz | 2.9637 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_25hz | 3.0400 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_50hz | 3.1703 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_50hz_2k_causal | 3.0265 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_50hz_4k_causal | 2.9994 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_50hz_65k_causal | 3.0599 | 49.00 |
| afrispeech_dialog | LanguageCodec | bandwidth_0 | 3.4059 | 49.00 |
| afrispeech_dialog | LanguageCodec | bandwidth_1 | 3.3993 | 49.00 |
| afrispeech_dialog | LanguageCodec | bandwidth_2 | 3.4014 | 49.00 |
| afrispeech_dialog | LanguageCodec | bandwidth_3 | 3.4102 | 49.00 |
| afrispeech_dialog | SemantiCodec | 0.31kbps | 3.2018 | 49.00 |
| afrispeech_dialog | SemantiCodec | 0.33kbps | 3.1104 | 49.00 |
| afrispeech_dialog | SemantiCodec | 0.63kbps | 3.0974 | 49.00 |
| afrispeech_dialog | SemantiCodec | 0.68kbps | 3.0813 | 49.00 |
| afrispeech_dialog | SemantiCodec | 1.25kbps | 3.0008 | 49.00 |
| afrispeech_dialog | SemantiCodec | 1.40kbps | 2.9967 | 49.00 |
| afrispeech_dialog | UniCodec | out_6.6kbps | 1.9300 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_large-speech-75token | 3.0682 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_large-unify-40token | 3.0705 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_medium-music-audio-75token | 2.9999 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_medium-speech-75token | 3.3137 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_small-320-24k-4096 | 3.3307 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_small-600-24k-4096 | 3.4230 | 49.00 |
| afrispeech_multilingual | DAC | out_16kbps | 2.5610 | 8.00 |
| afrispeech_multilingual | DAC | out_24kbps | 2.9582 | 8.00 |
| afrispeech_multilingual | DAC | out_8kbps | 2.9325 | 8.00 |
| afrispeech_multilingual | Encodec | out_12kbps | 2.6472 | 100.00 |
| afrispeech_multilingual | Encodec | out_24kbps | 2.6696 | 100.00 |
| afrispeech_multilingual | Encodec | out_3kbps | 2.3613 | 100.00 |
| afrispeech_multilingual | Encodec | out_6kbps | 2.5533 | 100.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_12_5hz | 2.9625 | 100.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_25hz | 3.0112 | 100.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_50hz | 3.0870 | 100.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_50hz_2k_causal | 3.0855 | 100.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_50hz_4k_causal | 2.8786 | 100.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_50hz_65k_causal | 2.9898 | 100.00 |
| afrispeech_multilingual | LanguageCodec | bandwidth_0 | 2.7523 | 100.00 |
| afrispeech_multilingual | LanguageCodec | bandwidth_1 | 2.7447 | 100.00 |
| afrispeech_multilingual | LanguageCodec | bandwidth_2 | 2.7395 | 100.00 |
| afrispeech_multilingual | LanguageCodec | bandwidth_3 | 2.7308 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 0.31kbps | 2.4612 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 0.33kbps | 2.4679 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 0.63kbps | 2.4207 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 0.68kbps | 2.4561 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 1.25kbps | 2.4378 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 1.40kbps | 2.4515 | 100.00 |
| afrispeech_multilingual | UniCodec | out_6.6kbps | 2.9262 | 100.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_large-speech-75token | 2.9628 | 100.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_large-unify-40token | 2.6975 | 100.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_medium-music-audio-75token | 2.6718 | 100.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_medium-speech-75token | 2.8840 | 100.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_small-320-24k-4096 | 3.0373 | 100.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_small-600-24k-4096 | 3.3442 | 100.00 |

## UTMOS (signal-level)

### Dataset × Codec Means

| Dataset | Codec | Variants | score | mean_num_scored_files |
|---|---|---|---|---|
| afrinames | DAC | 3 | 2.4154 | 115.00 |
| afrinames | Encodec | 4 | 1.9574 | 150.00 |
| afrinames | FocalCodec | 6 | 2.9186 | 150.00 |
| afrinames | LanguageCodec | 4 | 2.3601 | 67.00 |
| afrinames | SemantiCodec | 6 | 1.8799 | 150.00 |
| afrinames | UniCodec | 1 | 2.4401 | 150.00 |
| afrinames | WavTokenizer | 6 | 2.2987 | 150.00 |
| afrispeech_dialog | DAC | 3 | 2.2101 | 49.00 |
| afrispeech_dialog | Encodec | 4 | 1.8258 | 49.00 |
| afrispeech_dialog | FocalCodec | 6 | 2.4683 | 49.00 |
| afrispeech_dialog | LanguageCodec | 4 | 2.1743 | 49.00 |
| afrispeech_dialog | SemantiCodec | 6 | 1.7420 | 49.00 |
| afrispeech_dialog | UniCodec | 1 | 1.4159 | 49.00 |
| afrispeech_dialog | WavTokenizer | 6 | 2.0273 | 49.00 |
| afrispeech_multilingual | DAC | 3 | 1.8802 | 8.00 |
| afrispeech_multilingual | Encodec | 4 | 1.5534 | 100.00 |
| afrispeech_multilingual | FocalCodec | 6 | 2.1753 | 100.00 |
| afrispeech_multilingual | LanguageCodec | 4 | 1.7781 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 6 | 1.4834 | 100.00 |
| afrispeech_multilingual | UniCodec | 1 | 1.8831 | 100.00 |
| afrispeech_multilingual | WavTokenizer | 6 | 1.8006 | 100.00 |

### Variant-Level Results

| Dataset | Codec | Variant | score | num_scored_files |
|---|---|---|---|---|
| afrinames | DAC | out_16kbps | 2.4506 | 115.00 |
| afrinames | DAC | out_24kbps | 2.4020 | 115.00 |
| afrinames | DAC | out_8kbps | 2.3936 | 115.00 |
| afrinames | Encodec | out_12kbps | 2.1389 | 150.00 |
| afrinames | Encodec | out_24kbps | 2.2357 | 150.00 |
| afrinames | Encodec | out_3kbps | 1.5589 | 150.00 |
| afrinames | Encodec | out_6kbps | 1.8960 | 150.00 |
| afrinames | FocalCodec | focalcodec_12_5hz | 3.4348 | 150.00 |
| afrinames | FocalCodec | focalcodec_25hz | 3.2164 | 150.00 |
| afrinames | FocalCodec | focalcodec_50hz | 2.9669 | 150.00 |
| afrinames | FocalCodec | focalcodec_50hz_2k_causal | 2.6641 | 150.00 |
| afrinames | FocalCodec | focalcodec_50hz_4k_causal | 2.6231 | 150.00 |
| afrinames | FocalCodec | focalcodec_50hz_65k_causal | 2.6063 | 150.00 |
| afrinames | LanguageCodec | bandwidth_0 | 2.3573 | 67.00 |
| afrinames | LanguageCodec | bandwidth_1 | 2.3644 | 67.00 |
| afrinames | LanguageCodec | bandwidth_2 | 2.3626 | 67.00 |
| afrinames | LanguageCodec | bandwidth_3 | 2.3561 | 67.00 |
| afrinames | SemantiCodec | 0.31kbps | 1.8994 | 150.00 |
| afrinames | SemantiCodec | 0.33kbps | 1.9088 | 150.00 |
| afrinames | SemantiCodec | 0.63kbps | 1.8564 | 150.00 |
| afrinames | SemantiCodec | 0.68kbps | 1.8690 | 150.00 |
| afrinames | SemantiCodec | 1.25kbps | 1.8583 | 150.00 |
| afrinames | SemantiCodec | 1.40kbps | 1.8876 | 150.00 |
| afrinames | UniCodec | out_6.6kbps | 2.4401 | 150.00 |
| afrinames | WavTokenizer | WavTokenizer_large-speech-75token | 2.5142 | 150.00 |
| afrinames | WavTokenizer | WavTokenizer_large-unify-40token | 2.2801 | 150.00 |
| afrinames | WavTokenizer | WavTokenizer_medium-music-audio-75token | 1.7216 | 150.00 |
| afrinames | WavTokenizer | WavTokenizer_medium-speech-75token | 2.3257 | 150.00 |
| afrinames | WavTokenizer | WavTokenizer_small-320-24k-4096 | 2.3621 | 150.00 |
| afrinames | WavTokenizer | WavTokenizer_small-600-24k-4096 | 2.5887 | 150.00 |
| afrispeech_dialog | DAC | out_16kbps | 2.2610 | 49.00 |
| afrispeech_dialog | DAC | out_24kbps | 2.1830 | 49.00 |
| afrispeech_dialog | DAC | out_8kbps | 2.1865 | 49.00 |
| afrispeech_dialog | Encodec | out_12kbps | 1.9718 | 49.00 |
| afrispeech_dialog | Encodec | out_24kbps | 2.1087 | 49.00 |
| afrispeech_dialog | Encodec | out_3kbps | 1.4806 | 49.00 |
| afrispeech_dialog | Encodec | out_6kbps | 1.7423 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_12_5hz | 2.8830 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_25hz | 2.7609 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_50hz | 2.5546 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_50hz_2k_causal | 2.2238 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_50hz_4k_causal | 2.1919 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_50hz_65k_causal | 2.1955 | 49.00 |
| afrispeech_dialog | LanguageCodec | bandwidth_0 | 2.1686 | 49.00 |
| afrispeech_dialog | LanguageCodec | bandwidth_1 | 2.1804 | 49.00 |
| afrispeech_dialog | LanguageCodec | bandwidth_2 | 2.1754 | 49.00 |
| afrispeech_dialog | LanguageCodec | bandwidth_3 | 2.1729 | 49.00 |
| afrispeech_dialog | SemantiCodec | 0.31kbps | 1.7418 | 49.00 |
| afrispeech_dialog | SemantiCodec | 0.33kbps | 1.7498 | 49.00 |
| afrispeech_dialog | SemantiCodec | 0.63kbps | 1.7406 | 49.00 |
| afrispeech_dialog | SemantiCodec | 0.68kbps | 1.7599 | 49.00 |
| afrispeech_dialog | SemantiCodec | 1.25kbps | 1.7182 | 49.00 |
| afrispeech_dialog | SemantiCodec | 1.40kbps | 1.7419 | 49.00 |
| afrispeech_dialog | UniCodec | out_6.6kbps | 1.4159 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_large-speech-75token | 2.1634 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_large-unify-40token | 1.9184 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_medium-music-audio-75token | 1.6391 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_medium-speech-75token | 2.0914 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_small-320-24k-4096 | 2.0736 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_small-600-24k-4096 | 2.2777 | 49.00 |
| afrispeech_multilingual | DAC | out_16kbps | 1.8756 | 8.00 |
| afrispeech_multilingual | DAC | out_24kbps | 1.8934 | 8.00 |
| afrispeech_multilingual | DAC | out_8kbps | 1.8717 | 8.00 |
| afrispeech_multilingual | Encodec | out_12kbps | 1.6422 | 100.00 |
| afrispeech_multilingual | Encodec | out_24kbps | 1.6996 | 100.00 |
| afrispeech_multilingual | Encodec | out_3kbps | 1.3700 | 100.00 |
| afrispeech_multilingual | Encodec | out_6kbps | 1.5017 | 100.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_12_5hz | 2.5828 | 100.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_25hz | 2.4290 | 100.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_50hz | 2.2008 | 100.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_50hz_2k_causal | 1.9862 | 100.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_50hz_4k_causal | 1.9164 | 100.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_50hz_65k_causal | 1.9365 | 100.00 |
| afrispeech_multilingual | LanguageCodec | bandwidth_0 | 1.7767 | 100.00 |
| afrispeech_multilingual | LanguageCodec | bandwidth_1 | 1.7834 | 100.00 |
| afrispeech_multilingual | LanguageCodec | bandwidth_2 | 1.7731 | 100.00 |
| afrispeech_multilingual | LanguageCodec | bandwidth_3 | 1.7790 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 0.31kbps | 1.4825 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 0.33kbps | 1.4999 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 0.63kbps | 1.4743 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 0.68kbps | 1.4963 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 1.25kbps | 1.4612 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 1.40kbps | 1.4859 | 100.00 |
| afrispeech_multilingual | UniCodec | out_6.6kbps | 1.8831 | 100.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_large-speech-75token | 1.9500 | 100.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_large-unify-40token | 1.7744 | 100.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_medium-music-audio-75token | 1.4588 | 100.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_medium-speech-75token | 1.8724 | 100.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_small-320-24k-4096 | 1.7847 | 100.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_small-600-24k-4096 | 1.9632 | 100.00 |

## ViSQOL (signal-level)

### Dataset × Codec Means

| Dataset | Codec | Variants | score | mean_num_scored_files |
|---|---|---|---|---|
| afrinames | DAC | 3 | 4.9786 | 115.00 |
| afrinames | Encodec | 4 | 4.9389 | 150.00 |
| afrinames | FocalCodec | 6 | 4.7786 | 150.00 |
| afrinames | LanguageCodec | 4 | 2.7688 | 150.00 |
| afrinames | SemantiCodec | 6 | 4.8287 | 150.00 |
| afrinames | UniCodec | 1 | 4.9006 | 150.00 |
| afrinames | WavTokenizer | 6 | 4.8575 | 150.00 |
| afrispeech_dialog | DAC | 3 | 4.9825 | 49.00 |
| afrispeech_dialog | Encodec | 4 | 4.9518 | 49.00 |
| afrispeech_dialog | FocalCodec | 6 | 4.8135 | 49.00 |
| afrispeech_dialog | LanguageCodec | 4 | 4.9663 | 49.00 |
| afrispeech_dialog | SemantiCodec | 6 | 4.6854 | 49.00 |
| afrispeech_dialog | UniCodec | 1 | 4.8714 | 49.00 |
| afrispeech_dialog | WavTokenizer | 6 | 4.9042 | 49.00 |
| afrispeech_multilingual | DAC | 3 | 4.9748 | 8.00 |
| afrispeech_multilingual | Encodec | 4 | 4.9503 | 100.00 |
| afrispeech_multilingual | FocalCodec | 6 | 4.7922 | 100.00 |
| afrispeech_multilingual | LanguageCodec | 4 | 4.9585 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 6 | 4.8474 | 100.00 |
| afrispeech_multilingual | UniCodec | 1 | 4.9082 | 100.00 |
| afrispeech_multilingual | WavTokenizer | 6 | 4.8630 | 100.00 |

### Variant-Level Results

| Dataset | Codec | Variant | score | num_scored_files |
|---|---|---|---|---|
| afrinames | DAC | out_16kbps | 4.9945 | 115.00 |
| afrinames | DAC | out_24kbps | 4.9701 | 115.00 |
| afrinames | DAC | out_8kbps | 4.9713 | 115.00 |
| afrinames | Encodec | out_12kbps | 4.9525 | 150.00 |
| afrinames | Encodec | out_24kbps | 4.9617 | 150.00 |
| afrinames | Encodec | out_3kbps | 4.9069 | 150.00 |
| afrinames | Encodec | out_6kbps | 4.9345 | 150.00 |
| afrinames | FocalCodec | focalcodec_12_5hz | 4.7387 | 150.00 |
| afrinames | FocalCodec | focalcodec_25hz | 4.7937 | 150.00 |
| afrinames | FocalCodec | focalcodec_50hz | 4.7931 | 150.00 |
| afrinames | FocalCodec | focalcodec_50hz_2k_causal | 4.7727 | 150.00 |
| afrinames | FocalCodec | focalcodec_50hz_4k_causal | 4.7759 | 150.00 |
| afrinames | FocalCodec | focalcodec_50hz_65k_causal | 4.7974 | 150.00 |
| afrinames | LanguageCodec | bandwidth_0 | 2.7689 | 150.00 |
| afrinames | LanguageCodec | bandwidth_1 | 2.7688 | 150.00 |
| afrinames | LanguageCodec | bandwidth_2 | 2.7688 | 150.00 |
| afrinames | LanguageCodec | bandwidth_3 | 2.7689 | 150.00 |
| afrinames | SemantiCodec | 0.31kbps | 4.7504 | 150.00 |
| afrinames | SemantiCodec | 0.33kbps | 4.7572 | 150.00 |
| afrinames | SemantiCodec | 0.63kbps | 4.8427 | 150.00 |
| afrinames | SemantiCodec | 0.68kbps | 4.8484 | 150.00 |
| afrinames | SemantiCodec | 1.25kbps | 4.8840 | 150.00 |
| afrinames | SemantiCodec | 1.40kbps | 4.8896 | 150.00 |
| afrinames | UniCodec | out_6.6kbps | 4.9006 | 150.00 |
| afrinames | WavTokenizer | WavTokenizer_large-speech-75token | 4.9004 | 150.00 |
| afrinames | WavTokenizer | WavTokenizer_large-unify-40token | 4.8247 | 150.00 |
| afrinames | WavTokenizer | WavTokenizer_medium-music-audio-75token | 4.8542 | 150.00 |
| afrinames | WavTokenizer | WavTokenizer_medium-speech-75token | 4.8855 | 150.00 |
| afrinames | WavTokenizer | WavTokenizer_small-320-24k-4096 | 4.8739 | 150.00 |
| afrinames | WavTokenizer | WavTokenizer_small-600-24k-4096 | 4.8060 | 150.00 |
| afrispeech_dialog | DAC | out_16kbps | 4.9957 | 49.00 |
| afrispeech_dialog | DAC | out_24kbps | 4.9755 | 49.00 |
| afrispeech_dialog | DAC | out_8kbps | 4.9761 | 49.00 |
| afrispeech_dialog | Encodec | out_12kbps | 4.9611 | 49.00 |
| afrispeech_dialog | Encodec | out_24kbps | 4.9701 | 49.00 |
| afrispeech_dialog | Encodec | out_3kbps | 4.9288 | 49.00 |
| afrispeech_dialog | Encodec | out_6kbps | 4.9473 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_12_5hz | 4.8495 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_25hz | 4.8751 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_50hz | 4.5387 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_50hz_2k_causal | 4.8675 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_50hz_4k_causal | 4.8698 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_50hz_65k_causal | 4.8805 | 49.00 |
| afrispeech_dialog | LanguageCodec | bandwidth_0 | 4.9663 | 49.00 |
| afrispeech_dialog | LanguageCodec | bandwidth_1 | 4.9663 | 49.00 |
| afrispeech_dialog | LanguageCodec | bandwidth_2 | 4.9663 | 49.00 |
| afrispeech_dialog | LanguageCodec | bandwidth_3 | 4.9663 | 49.00 |
| afrispeech_dialog | SemantiCodec | 0.31kbps | 4.6695 | 49.00 |
| afrispeech_dialog | SemantiCodec | 0.33kbps | 4.6724 | 49.00 |
| afrispeech_dialog | SemantiCodec | 0.63kbps | 4.6869 | 49.00 |
| afrispeech_dialog | SemantiCodec | 0.68kbps | 4.6855 | 49.00 |
| afrispeech_dialog | SemantiCodec | 1.25kbps | 4.6983 | 49.00 |
| afrispeech_dialog | SemantiCodec | 1.40kbps | 4.6997 | 49.00 |
| afrispeech_dialog | UniCodec | out_6.6kbps | 4.8714 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_large-speech-75token | 4.9268 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_large-unify-40token | 4.8870 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_medium-music-audio-75token | 4.9036 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_medium-speech-75token | 4.9154 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_small-320-24k-4096 | 4.9123 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_small-600-24k-4096 | 4.8801 | 49.00 |
| afrispeech_multilingual | DAC | out_16kbps | 4.9935 | 8.00 |
| afrispeech_multilingual | DAC | out_24kbps | 4.9644 | 8.00 |
| afrispeech_multilingual | DAC | out_8kbps | 4.9667 | 8.00 |
| afrispeech_multilingual | Encodec | out_12kbps | 4.9627 | 100.00 |
| afrispeech_multilingual | Encodec | out_24kbps | 4.9711 | 100.00 |
| afrispeech_multilingual | Encodec | out_3kbps | 4.9207 | 100.00 |
| afrispeech_multilingual | Encodec | out_6kbps | 4.9467 | 100.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_12_5hz | 4.7598 | 100.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_25hz | 4.8049 | 100.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_50hz | 4.8354 | 100.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_50hz_2k_causal | 4.7762 | 100.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_50hz_4k_causal | 4.7769 | 100.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_50hz_65k_causal | 4.8001 | 100.00 |
| afrispeech_multilingual | LanguageCodec | bandwidth_0 | 4.9586 | 100.00 |
| afrispeech_multilingual | LanguageCodec | bandwidth_1 | 4.9584 | 100.00 |
| afrispeech_multilingual | LanguageCodec | bandwidth_2 | 4.9584 | 100.00 |
| afrispeech_multilingual | LanguageCodec | bandwidth_3 | 4.9585 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 0.31kbps | 4.7833 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 0.33kbps | 4.7894 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 0.63kbps | 4.8575 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 0.68kbps | 4.8623 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 1.25kbps | 4.8935 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 1.40kbps | 4.8984 | 100.00 |
| afrispeech_multilingual | UniCodec | out_6.6kbps | 4.9082 | 100.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_large-speech-75token | 4.9011 | 100.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_large-unify-40token | 4.8347 | 100.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_medium-music-audio-75token | 4.8654 | 100.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_medium-speech-75token | 4.8881 | 100.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_small-320-24k-4096 | 4.8726 | 100.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_small-600-24k-4096 | 4.8159 | 100.00 |

## STOI (signal-level)

### Dataset × Codec Means

| Dataset | Codec | Variants | score | mean_num_scored_files |
|---|---|---|---|---|
| afrinames | DAC | 3 | 0.9713 | 46.00 |
| afrinames | Encodec | 4 | 0.9321 | 67.00 |
| afrinames | FocalCodec | 6 | 0.7187 | 67.00 |
| afrinames | LanguageCodec | 4 | 0.9491 | 67.00 |
| afrinames | SemantiCodec | 6 | 0.7815 | 67.00 |
| afrinames | UniCodec | 1 | 0.8629 | 67.00 |
| afrinames | WavTokenizer | 6 | 0.8195 | 67.00 |
| afrispeech_dialog | DAC | 3 | 0.9597 | 49.00 |
| afrispeech_dialog | Encodec | 4 | 0.8979 | 49.00 |
| afrispeech_dialog | FocalCodec | 6 | 0.5905 | 49.00 |
| afrispeech_dialog | LanguageCodec | 4 | 0.9241 | 49.00 |
| afrispeech_dialog | SemantiCodec | 6 | 0.3508 | 49.00 |
| afrispeech_dialog | UniCodec | 1 | 0.7101 | 49.00 |
| afrispeech_dialog | WavTokenizer | 6 | 0.7632 | 49.00 |
| afrispeech_multilingual | DAC | 3 | 0.9680 | 8.00 |
| afrispeech_multilingual | Encodec | 4 | 0.9098 | 100.00 |
| afrispeech_multilingual | FocalCodec | 6 | 0.6677 | 100.00 |
| afrispeech_multilingual | LanguageCodec | 4 | 0.9314 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 6 | 0.7331 | 100.00 |
| afrispeech_multilingual | UniCodec | 1 | 0.8292 | 100.00 |
| afrispeech_multilingual | WavTokenizer | 6 | 0.7713 | 100.00 |

### Variant-Level Results

| Dataset | Codec | Variant | score | num_scored_files |
|---|---|---|---|---|
| afrinames | DAC | out_16kbps | 0.9934 | 46.00 |
| afrinames | DAC | out_24kbps | 0.9588 | 46.00 |
| afrinames | DAC | out_8kbps | 0.9616 | 46.00 |
| afrinames | Encodec | out_12kbps | 0.9543 | 67.00 |
| afrinames | Encodec | out_24kbps | 0.9658 | 67.00 |
| afrinames | Encodec | out_3kbps | 0.8816 | 67.00 |
| afrinames | Encodec | out_6kbps | 0.9265 | 67.00 |
| afrinames | FocalCodec | focalcodec_12_5hz | 0.6917 | 67.00 |
| afrinames | FocalCodec | focalcodec_25hz | 0.7461 | 67.00 |
| afrinames | FocalCodec | focalcodec_50hz | 0.7067 | 67.00 |
| afrinames | FocalCodec | focalcodec_50hz_2k_causal | 0.7143 | 67.00 |
| afrinames | FocalCodec | focalcodec_50hz_4k_causal | 0.7149 | 67.00 |
| afrinames | FocalCodec | focalcodec_50hz_65k_causal | 0.7383 | 67.00 |
| afrinames | LanguageCodec | bandwidth_0 | 0.9493 | 67.00 |
| afrinames | LanguageCodec | bandwidth_1 | 0.9488 | 67.00 |
| afrinames | LanguageCodec | bandwidth_2 | 0.9493 | 67.00 |
| afrinames | LanguageCodec | bandwidth_3 | 0.9491 | 67.00 |
| afrinames | SemantiCodec | 0.31kbps | 0.7092 | 67.00 |
| afrinames | SemantiCodec | 0.33kbps | 0.7127 | 67.00 |
| afrinames | SemantiCodec | 0.63kbps | 0.7897 | 67.00 |
| afrinames | SemantiCodec | 0.68kbps | 0.7983 | 67.00 |
| afrinames | SemantiCodec | 1.25kbps | 0.8366 | 67.00 |
| afrinames | SemantiCodec | 1.40kbps | 0.8427 | 67.00 |
| afrinames | UniCodec | out_6.6kbps | 0.8629 | 67.00 |
| afrinames | WavTokenizer | WavTokenizer_large-speech-75token | 0.8632 | 67.00 |
| afrinames | WavTokenizer | WavTokenizer_large-unify-40token | 0.8009 | 67.00 |
| afrinames | WavTokenizer | WavTokenizer_medium-music-audio-75token | 0.8085 | 67.00 |
| afrinames | WavTokenizer | WavTokenizer_medium-speech-75token | 0.8464 | 67.00 |
| afrinames | WavTokenizer | WavTokenizer_small-320-24k-4096 | 0.8280 | 67.00 |
| afrinames | WavTokenizer | WavTokenizer_small-600-24k-4096 | 0.7700 | 67.00 |
| afrispeech_dialog | DAC | out_16kbps | 0.9905 | 49.00 |
| afrispeech_dialog | DAC | out_24kbps | 0.9431 | 49.00 |
| afrispeech_dialog | DAC | out_8kbps | 0.9454 | 49.00 |
| afrispeech_dialog | Encodec | out_12kbps | 0.9250 | 49.00 |
| afrispeech_dialog | Encodec | out_24kbps | 0.9454 | 49.00 |
| afrispeech_dialog | Encodec | out_3kbps | 0.8343 | 49.00 |
| afrispeech_dialog | Encodec | out_6kbps | 0.8870 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_12_5hz | 0.6447 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_25hz | 0.7036 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_50hz | 0.1414 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_50hz_2k_causal | 0.6740 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_50hz_4k_causal | 0.6782 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_50hz_65k_causal | 0.7012 | 49.00 |
| afrispeech_dialog | LanguageCodec | bandwidth_0 | 0.9241 | 49.00 |
| afrispeech_dialog | LanguageCodec | bandwidth_1 | 0.9239 | 49.00 |
| afrispeech_dialog | LanguageCodec | bandwidth_2 | 0.9240 | 49.00 |
| afrispeech_dialog | LanguageCodec | bandwidth_3 | 0.9242 | 49.00 |
| afrispeech_dialog | SemantiCodec | 0.31kbps | 0.3362 | 49.00 |
| afrispeech_dialog | SemantiCodec | 0.33kbps | 0.3372 | 49.00 |
| afrispeech_dialog | SemantiCodec | 0.63kbps | 0.3519 | 49.00 |
| afrispeech_dialog | SemantiCodec | 0.68kbps | 0.3541 | 49.00 |
| afrispeech_dialog | SemantiCodec | 1.25kbps | 0.3617 | 49.00 |
| afrispeech_dialog | SemantiCodec | 1.40kbps | 0.3634 | 49.00 |
| afrispeech_dialog | UniCodec | out_6.6kbps | 0.7101 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_large-speech-75token | 0.8073 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_large-unify-40token | 0.7314 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_medium-music-audio-75token | 0.7556 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_medium-speech-75token | 0.7860 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_small-320-24k-4096 | 0.7772 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_small-600-24k-4096 | 0.7217 | 49.00 |
| afrispeech_multilingual | DAC | out_16kbps | 0.9922 | 8.00 |
| afrispeech_multilingual | DAC | out_24kbps | 0.9529 | 8.00 |
| afrispeech_multilingual | DAC | out_8kbps | 0.9591 | 8.00 |
| afrispeech_multilingual | Encodec | out_12kbps | 0.9371 | 100.00 |
| afrispeech_multilingual | Encodec | out_24kbps | 0.9536 | 100.00 |
| afrispeech_multilingual | Encodec | out_3kbps | 0.8476 | 100.00 |
| afrispeech_multilingual | Encodec | out_6kbps | 0.9010 | 100.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_12_5hz | 0.6254 | 100.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_25hz | 0.6913 | 100.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_50hz | 0.7330 | 100.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_50hz_2k_causal | 0.6465 | 100.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_50hz_4k_causal | 0.6419 | 100.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_50hz_65k_causal | 0.6679 | 100.00 |
| afrispeech_multilingual | LanguageCodec | bandwidth_0 | 0.9315 | 100.00 |
| afrispeech_multilingual | LanguageCodec | bandwidth_1 | 0.9312 | 100.00 |
| afrispeech_multilingual | LanguageCodec | bandwidth_2 | 0.9314 | 100.00 |
| afrispeech_multilingual | LanguageCodec | bandwidth_3 | 0.9314 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 0.31kbps | 0.6557 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 0.33kbps | 0.6596 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 0.63kbps | 0.7431 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 0.68kbps | 0.7483 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 1.25kbps | 0.7920 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 1.40kbps | 0.8002 | 100.00 |
| afrispeech_multilingual | UniCodec | out_6.6kbps | 0.8292 | 100.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_large-speech-75token | 0.8194 | 100.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_large-unify-40token | 0.7537 | 100.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_medium-music-audio-75token | 0.7661 | 100.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_medium-speech-75token | 0.8024 | 100.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_small-320-24k-4096 | 0.7733 | 100.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_small-600-24k-4096 | 0.7126 | 100.00 |

## Prosody (signal-level)

### Dataset × Codec Means

| Dataset | Codec | Variants | overall_prosody | f0_rmse | mean_num_scored_files |
|---|---|---|---|---|---|
| afrinames | DAC | 3 | 1.2071 | 2.2642 | 115.00 |
| afrinames | Encodec | 4 | 1.4589 | 3.7967 | 150.00 |
| afrinames | FocalCodec | 6 | 2.1174 | 7.6591 | 150.00 |
| afrinames | LanguageCodec | 4 | 0.5957 | 2.9907 | 150.00 |
| afrinames | SemantiCodec | 6 | 1.7142 | 5.3765 | 150.00 |
| afrinames | UniCodec | 1 | 1.6395 | 4.8833 | 150.00 |
| afrinames | WavTokenizer | 6 | 1.7056 | 5.2908 | 150.00 |
| afrispeech_dialog | DAC | 3 | 1.4132 | 3.2174 | 49.00 |
| afrispeech_dialog | Encodec | 4 | 1.7072 | 4.8955 | 49.00 |
| afrispeech_dialog | FocalCodec | 6 | 2.3937 | 9.0331 | 49.00 |
| afrispeech_dialog | LanguageCodec | 4 | 1.5229 | 3.8107 | 49.00 |
| afrispeech_dialog | SemantiCodec | 6 | 2.1989 | 7.9768 | 49.00 |
| afrispeech_dialog | UniCodec | 1 | 2.6451 | 10.1916 | 49.00 |
| afrispeech_dialog | WavTokenizer | 6 | 2.0532 | 6.8384 | 49.00 |
| afrispeech_multilingual | DAC | 3 | 1.1551 | 1.9836 | 8.00 |
| afrispeech_multilingual | Encodec | 4 | 1.5436 | 4.3340 | 100.00 |
| afrispeech_multilingual | FocalCodec | 6 | 2.2012 | 8.1570 | 100.00 |
| afrispeech_multilingual | LanguageCodec | 4 | 1.3184 | 2.9180 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 6 | 1.6918 | 5.2294 | 100.00 |
| afrispeech_multilingual | UniCodec | 1 | 1.6458 | 4.9083 | 100.00 |
| afrispeech_multilingual | WavTokenizer | 6 | 1.7381 | 5.4715 | 100.00 |

### Variant-Level Results

| Dataset | Codec | Variant | overall_prosody | f0_rmse | num_scored_files |
|---|---|---|---|---|---|
| afrinames | DAC | out_16kbps | 1.1064 | 1.6652 | 115.00 |
| afrinames | DAC | out_24kbps | 1.2387 | 2.4646 | 115.00 |
| afrinames | DAC | out_8kbps | 1.2761 | 2.6629 | 115.00 |
| afrinames | Encodec | out_12kbps | 1.4111 | 3.5020 | 150.00 |
| afrinames | Encodec | out_24kbps | 1.3638 | 3.2156 | 150.00 |
| afrinames | Encodec | out_3kbps | 1.5757 | 4.5142 | 150.00 |
| afrinames | Encodec | out_6kbps | 1.4850 | 3.9549 | 150.00 |
| afrinames | FocalCodec | focalcodec_12_5hz | 2.1593 | 7.9665 | 150.00 |
| afrinames | FocalCodec | focalcodec_25hz | 2.0589 | 7.3514 | 150.00 |
| afrinames | FocalCodec | focalcodec_50hz | 1.9716 | 6.8563 | 150.00 |
| afrinames | FocalCodec | focalcodec_50hz_2k_causal | 2.1739 | 7.9478 | 150.00 |
| afrinames | FocalCodec | focalcodec_50hz_4k_causal | 2.2023 | 8.1028 | 150.00 |
| afrinames | FocalCodec | focalcodec_50hz_65k_causal | 2.1382 | 7.7296 | 150.00 |
| afrinames | LanguageCodec | bandwidth_0 | 0.5959 | 2.9883 | 150.00 |
| afrinames | LanguageCodec | bandwidth_1 | 0.5917 | 2.9491 | 150.00 |
| afrinames | LanguageCodec | bandwidth_2 | 0.5946 | 2.9782 | 150.00 |
| afrinames | LanguageCodec | bandwidth_3 | 0.6008 | 3.0471 | 150.00 |
| afrinames | SemantiCodec | 0.31kbps | 1.7954 | 5.9496 | 150.00 |
| afrinames | SemantiCodec | 0.33kbps | 1.7983 | 5.9590 | 150.00 |
| afrinames | SemantiCodec | 0.63kbps | 1.7116 | 5.3493 | 150.00 |
| afrinames | SemantiCodec | 0.68kbps | 1.7105 | 5.3227 | 150.00 |
| afrinames | SemantiCodec | 1.25kbps | 1.6491 | 4.9259 | 150.00 |
| afrinames | SemantiCodec | 1.40kbps | 1.6205 | 4.7522 | 150.00 |
| afrinames | UniCodec | out_6.6kbps | 1.6395 | 4.8833 | 150.00 |
| afrinames | WavTokenizer | WavTokenizer_large-speech-75token | 1.5920 | 4.5736 | 150.00 |
| afrinames | WavTokenizer | WavTokenizer_large-unify-40token | 1.7139 | 5.3906 | 150.00 |
| afrinames | WavTokenizer | WavTokenizer_medium-music-audio-75token | 1.6747 | 5.1530 | 150.00 |
| afrinames | WavTokenizer | WavTokenizer_medium-speech-75token | 1.6067 | 4.6788 | 150.00 |
| afrinames | WavTokenizer | WavTokenizer_small-320-24k-4096 | 1.7545 | 5.5340 | 150.00 |
| afrinames | WavTokenizer | WavTokenizer_small-600-24k-4096 | 1.8917 | 6.4147 | 150.00 |
| afrispeech_dialog | DAC | out_16kbps | 1.2235 | 2.1941 | 49.00 |
| afrispeech_dialog | DAC | out_24kbps | 1.4869 | 3.6222 | 49.00 |
| afrispeech_dialog | DAC | out_8kbps | 1.5293 | 3.8360 | 49.00 |
| afrispeech_dialog | Encodec | out_12kbps | 1.6340 | 4.4682 | 49.00 |
| afrispeech_dialog | Encodec | out_24kbps | 1.5404 | 3.9590 | 49.00 |
| afrispeech_dialog | Encodec | out_3kbps | 1.8936 | 5.9611 | 49.00 |
| afrispeech_dialog | Encodec | out_6kbps | 1.7609 | 5.1939 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_12_5hz | 2.4371 | 9.1899 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_25hz | 2.3342 | 8.5498 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_50hz | 2.4779 | 10.2741 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_50hz_2k_causal | 2.3937 | 8.8591 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_50hz_4k_causal | 2.3872 | 8.8314 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_50hz_65k_causal | 2.3321 | 8.4942 | 49.00 |
| afrispeech_dialog | LanguageCodec | bandwidth_0 | 1.5229 | 3.8081 | 49.00 |
| afrispeech_dialog | LanguageCodec | bandwidth_1 | 1.5235 | 3.8095 | 49.00 |
| afrispeech_dialog | LanguageCodec | bandwidth_2 | 1.5209 | 3.7993 | 49.00 |
| afrispeech_dialog | LanguageCodec | bandwidth_3 | 1.5243 | 3.8257 | 49.00 |
| afrispeech_dialog | SemantiCodec | 0.31kbps | 2.2255 | 8.1862 | 49.00 |
| afrispeech_dialog | SemantiCodec | 0.33kbps | 2.2275 | 8.1980 | 49.00 |
| afrispeech_dialog | SemantiCodec | 0.63kbps | 2.2141 | 8.0454 | 49.00 |
| afrispeech_dialog | SemantiCodec | 0.68kbps | 2.2102 | 8.0037 | 49.00 |
| afrispeech_dialog | SemantiCodec | 1.25kbps | 2.1605 | 7.7349 | 49.00 |
| afrispeech_dialog | SemantiCodec | 1.40kbps | 2.1557 | 7.6928 | 49.00 |
| afrispeech_dialog | UniCodec | out_6.6kbps | 2.6451 | 10.1916 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_large-speech-75token | 1.9878 | 6.4319 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_large-unify-40token | 2.0889 | 7.0846 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_medium-music-audio-75token | 1.9393 | 6.2302 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_medium-speech-75token | 2.0816 | 6.9116 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_small-320-24k-4096 | 1.9964 | 6.5344 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_small-600-24k-4096 | 2.2254 | 7.8378 | 49.00 |
| afrispeech_multilingual | DAC | out_16kbps | 1.0834 | 1.5181 | 8.00 |
| afrispeech_multilingual | DAC | out_24kbps | 1.1772 | 2.1199 | 8.00 |
| afrispeech_multilingual | DAC | out_8kbps | 1.2047 | 2.3129 | 8.00 |
| afrispeech_multilingual | Encodec | out_12kbps | 1.5126 | 4.1280 | 100.00 |
| afrispeech_multilingual | Encodec | out_24kbps | 1.4533 | 3.7834 | 100.00 |
| afrispeech_multilingual | Encodec | out_3kbps | 1.6361 | 4.9106 | 100.00 |
| afrispeech_multilingual | Encodec | out_6kbps | 1.5725 | 4.5141 | 100.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_12_5hz | 2.2120 | 8.3481 | 100.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_25hz | 2.1630 | 7.9112 | 100.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_50hz | 1.9252 | 6.6412 | 100.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_50hz_2k_causal | 2.3073 | 8.7358 | 100.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_50hz_4k_causal | 2.3063 | 8.7187 | 100.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_50hz_65k_causal | 2.2933 | 8.5872 | 100.00 |
| afrispeech_multilingual | LanguageCodec | bandwidth_0 | 1.3173 | 2.9059 | 100.00 |
| afrispeech_multilingual | LanguageCodec | bandwidth_1 | 1.3210 | 2.9260 | 100.00 |
| afrispeech_multilingual | LanguageCodec | bandwidth_2 | 1.3183 | 2.9146 | 100.00 |
| afrispeech_multilingual | LanguageCodec | bandwidth_3 | 1.3170 | 2.9256 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 0.31kbps | 1.7677 | 5.7836 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 0.33kbps | 1.7390 | 5.6473 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 0.63kbps | 1.6734 | 5.1312 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 0.68kbps | 1.7076 | 5.2536 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 1.25kbps | 1.6475 | 4.8430 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 1.40kbps | 1.6156 | 4.7178 | 100.00 |
| afrispeech_multilingual | UniCodec | out_6.6kbps | 1.6458 | 4.9083 | 100.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_large-speech-75token | 1.6528 | 4.9261 | 100.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_large-unify-40token | 1.6461 | 5.0820 | 100.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_medium-music-audio-75token | 1.7163 | 5.3331 | 100.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_medium-speech-75token | 1.6646 | 5.0471 | 100.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_small-320-24k-4096 | 1.8153 | 5.8295 | 100.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_small-600-24k-4096 | 1.9332 | 6.6109 | 100.00 |

## F0_RMSE (signal-level)

### Dataset × Codec Means

| Dataset | Codec | Variants | score | mean_num_scored_files |
|---|---|---|---|---|
| afrinames | DAC | 3 | 47.8907 | 6.67 |
| afrinames | Encodec | 4 | 54.3656 | 10.00 |
| afrinames | FocalCodec | 6 | NaN | 0.00 |
| afrinames | LanguageCodec | 4 | NaN | 0.00 |
| afrinames | SemantiCodec | 6 | 66.2222 | 10.00 |
| afrinames | UniCodec | 1 | 62.1950 | 10.00 |
| afrinames | WavTokenizer | 6 | 65.4613 | 10.00 |
| afrispeech_dialog | DAC | 3 | 54.3074 | 6.67 |
| afrispeech_dialog | Encodec | 4 | 68.2444 | 10.00 |
| afrispeech_dialog | FocalCodec | 6 | NaN | 0.00 |
| afrispeech_dialog | LanguageCodec | 4 | 65.3883 | 10.00 |
| afrispeech_dialog | SemantiCodec | 6 | 103.6788 | 10.00 |
| afrispeech_dialog | UniCodec | 1 | 116.4562 | 10.00 |
| afrispeech_dialog | WavTokenizer | 6 | 92.9183 | 10.00 |
| afrispeech_multilingual | DAC | 3 | 43.2769 | 5.33 |
| afrispeech_multilingual | Encodec | 4 | NaN | 0.00 |
| afrispeech_multilingual | FocalCodec | 6 | NaN | 0.00 |
| afrispeech_multilingual | LanguageCodec | 4 | NaN | 0.00 |
| afrispeech_multilingual | SemantiCodec | 6 | 68.2695 | 10.00 |
| afrispeech_multilingual | UniCodec | 1 | NaN | 0.00 |
| afrispeech_multilingual | WavTokenizer | 6 | NaN | 0.00 |

### Variant-Level Results

| Dataset | Codec | Variant | score | num_scored_files |
|---|---|---|---|---|
| afrinames | DAC | out_16kbps | 45.5325 | 10.00 |
| afrinames | DAC | out_24kbps | NaN | 0.00 |
| afrinames | DAC | out_8kbps | 50.2489 | 10.00 |
| afrinames | Encodec | out_12kbps | 53.4415 | 10.00 |
| afrinames | Encodec | out_24kbps | 50.1892 | 10.00 |
| afrinames | Encodec | out_3kbps | 56.9690 | 10.00 |
| afrinames | Encodec | out_6kbps | 56.8625 | 10.00 |
| afrinames | FocalCodec | focalcodec_12_5hz | NaN | 0.00 |
| afrinames | FocalCodec | focalcodec_25hz | NaN | 0.00 |
| afrinames | FocalCodec | focalcodec_50hz | NaN | 0.00 |
| afrinames | FocalCodec | focalcodec_50hz_2k_causal | NaN | 0.00 |
| afrinames | FocalCodec | focalcodec_50hz_4k_causal | NaN | 0.00 |
| afrinames | FocalCodec | focalcodec_50hz_65k_causal | NaN | 0.00 |
| afrinames | LanguageCodec | bandwidth_0 | NaN | 0.00 |
| afrinames | LanguageCodec | bandwidth_1 | NaN | 0.00 |
| afrinames | LanguageCodec | bandwidth_2 | NaN | 0.00 |
| afrinames | LanguageCodec | bandwidth_3 | NaN | 0.00 |
| afrinames | SemantiCodec | 0.31kbps | 67.1587 | 10.00 |
| afrinames | SemantiCodec | 0.33kbps | 71.7854 | 10.00 |
| afrinames | SemantiCodec | 0.63kbps | 66.9648 | 10.00 |
| afrinames | SemantiCodec | 0.68kbps | 64.4656 | 10.00 |
| afrinames | SemantiCodec | 1.25kbps | 64.6636 | 10.00 |
| afrinames | SemantiCodec | 1.40kbps | 62.2949 | 10.00 |
| afrinames | UniCodec | out_6.6kbps | 62.1950 | 10.00 |
| afrinames | WavTokenizer | WavTokenizer_large-speech-75token | 63.3001 | 10.00 |
| afrinames | WavTokenizer | WavTokenizer_large-unify-40token | 65.6742 | 10.00 |
| afrinames | WavTokenizer | WavTokenizer_medium-music-audio-75token | 64.9215 | 10.00 |
| afrinames | WavTokenizer | WavTokenizer_medium-speech-75token | 65.2722 | 10.00 |
| afrinames | WavTokenizer | WavTokenizer_small-320-24k-4096 | 65.3352 | 10.00 |
| afrinames | WavTokenizer | WavTokenizer_small-600-24k-4096 | 68.2645 | 10.00 |
| afrispeech_dialog | DAC | out_16kbps | 45.2015 | 10.00 |
| afrispeech_dialog | DAC | out_24kbps | NaN | 0.00 |
| afrispeech_dialog | DAC | out_8kbps | 63.4133 | 10.00 |
| afrispeech_dialog | Encodec | out_12kbps | 63.0477 | 10.00 |
| afrispeech_dialog | Encodec | out_24kbps | 57.2361 | 10.00 |
| afrispeech_dialog | Encodec | out_3kbps | 80.8880 | 10.00 |
| afrispeech_dialog | Encodec | out_6kbps | 71.8059 | 10.00 |
| afrispeech_dialog | FocalCodec | focalcodec_12_5hz | NaN | 0.00 |
| afrispeech_dialog | FocalCodec | focalcodec_25hz | NaN | 0.00 |
| afrispeech_dialog | FocalCodec | focalcodec_50hz | NaN | 0.00 |
| afrispeech_dialog | FocalCodec | focalcodec_50hz_2k_causal | NaN | 0.00 |
| afrispeech_dialog | FocalCodec | focalcodec_50hz_4k_causal | NaN | 0.00 |
| afrispeech_dialog | FocalCodec | focalcodec_50hz_65k_causal | NaN | 0.00 |
| afrispeech_dialog | LanguageCodec | bandwidth_0 | 65.4828 | 10.00 |
| afrispeech_dialog | LanguageCodec | bandwidth_1 | 65.3217 | 10.00 |
| afrispeech_dialog | LanguageCodec | bandwidth_2 | 65.3281 | 10.00 |
| afrispeech_dialog | LanguageCodec | bandwidth_3 | 65.4204 | 10.00 |
| afrispeech_dialog | SemantiCodec | 0.31kbps | 103.7573 | 10.00 |
| afrispeech_dialog | SemantiCodec | 0.33kbps | 103.2490 | 10.00 |
| afrispeech_dialog | SemantiCodec | 0.63kbps | 106.5402 | 10.00 |
| afrispeech_dialog | SemantiCodec | 0.68kbps | 106.0515 | 10.00 |
| afrispeech_dialog | SemantiCodec | 1.25kbps | 101.1927 | 10.00 |
| afrispeech_dialog | SemantiCodec | 1.40kbps | 101.2822 | 10.00 |
| afrispeech_dialog | UniCodec | out_6.6kbps | 116.4562 | 10.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_large-speech-75token | 90.4474 | 10.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_large-unify-40token | 94.1441 | 10.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_medium-music-audio-75token | 88.4150 | 10.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_medium-speech-75token | 88.5629 | 10.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_small-320-24k-4096 | 91.1757 | 10.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_small-600-24k-4096 | 104.7645 | 10.00 |
| afrispeech_multilingual | DAC | out_16kbps | NaN | 0.00 |
| afrispeech_multilingual | DAC | out_24kbps | 40.3125 | 8.00 |
| afrispeech_multilingual | DAC | out_8kbps | 46.2413 | 8.00 |
| afrispeech_multilingual | Encodec | out_12kbps | NaN | 0.00 |
| afrispeech_multilingual | Encodec | out_24kbps | NaN | 0.00 |
| afrispeech_multilingual | Encodec | out_3kbps | NaN | 0.00 |
| afrispeech_multilingual | Encodec | out_6kbps | NaN | 0.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_12_5hz | NaN | 0.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_25hz | NaN | 0.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_50hz | NaN | 0.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_50hz_2k_causal | NaN | 0.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_50hz_4k_causal | NaN | 0.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_50hz_65k_causal | NaN | 0.00 |
| afrispeech_multilingual | LanguageCodec | bandwidth_0 | NaN | 0.00 |
| afrispeech_multilingual | LanguageCodec | bandwidth_1 | NaN | 0.00 |
| afrispeech_multilingual | LanguageCodec | bandwidth_2 | NaN | 0.00 |
| afrispeech_multilingual | LanguageCodec | bandwidth_3 | NaN | 0.00 |
| afrispeech_multilingual | SemantiCodec | 0.31kbps | 74.2830 | 10.00 |
| afrispeech_multilingual | SemantiCodec | 0.33kbps | 71.7502 | 10.00 |
| afrispeech_multilingual | SemantiCodec | 0.63kbps | 65.9099 | 10.00 |
| afrispeech_multilingual | SemantiCodec | 0.68kbps | 66.1677 | 10.00 |
| afrispeech_multilingual | SemantiCodec | 1.25kbps | 64.7303 | 10.00 |
| afrispeech_multilingual | SemantiCodec | 1.40kbps | 66.7755 | 10.00 |
| afrispeech_multilingual | UniCodec | out_6.6kbps | NaN | 0.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_large-speech-75token | NaN | 0.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_large-unify-40token | NaN | 0.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_medium-music-audio-75token | NaN | 0.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_medium-speech-75token | NaN | 0.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_small-320-24k-4096 | NaN | 0.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_small-600-24k-4096 | NaN | 0.00 |

## SPK_SIM (signal-level)

### Dataset × Codec Means

| Dataset | Codec | Variants | score | mean_num_scored_files |
|---|---|---|---|---|
| afrinames | DAC | 3 | NaN | NaN |
| afrinames | Encodec | 4 | NaN | NaN |
| afrinames | FocalCodec | 6 | NaN | NaN |
| afrinames | LanguageCodec | 4 | NaN | NaN |
| afrinames | SemantiCodec | 6 | NaN | NaN |
| afrinames | UniCodec | 1 | NaN | NaN |
| afrinames | WavTokenizer | 6 | NaN | NaN |
| afrispeech_dialog | DAC | 3 | NaN | NaN |
| afrispeech_dialog | Encodec | 4 | NaN | NaN |
| afrispeech_dialog | FocalCodec | 6 | NaN | NaN |
| afrispeech_dialog | LanguageCodec | 4 | NaN | NaN |
| afrispeech_dialog | SemantiCodec | 6 | NaN | NaN |
| afrispeech_dialog | UniCodec | 1 | NaN | NaN |
| afrispeech_dialog | WavTokenizer | 6 | NaN | NaN |
| afrispeech_multilingual | DAC | 3 | NaN | NaN |
| afrispeech_multilingual | Encodec | 4 | NaN | NaN |
| afrispeech_multilingual | FocalCodec | 6 | NaN | NaN |
| afrispeech_multilingual | LanguageCodec | 4 | NaN | NaN |
| afrispeech_multilingual | SemantiCodec | 6 | NaN | NaN |
| afrispeech_multilingual | UniCodec | 1 | NaN | NaN |
| afrispeech_multilingual | WavTokenizer | 6 | NaN | NaN |

### Variant-Level Results

| Dataset | Codec | Variant | score | num_scored_files |
|---|---|---|---|---|
| afrinames | DAC | out_16kbps | NaN | NaN |
| afrinames | DAC | out_24kbps | NaN | NaN |
| afrinames | DAC | out_8kbps | NaN | NaN |
| afrinames | Encodec | out_12kbps | NaN | NaN |
| afrinames | Encodec | out_24kbps | NaN | NaN |
| afrinames | Encodec | out_3kbps | NaN | NaN |
| afrinames | Encodec | out_6kbps | NaN | NaN |
| afrinames | FocalCodec | focalcodec_12_5hz | NaN | NaN |
| afrinames | FocalCodec | focalcodec_25hz | NaN | NaN |
| afrinames | FocalCodec | focalcodec_50hz | NaN | NaN |
| afrinames | FocalCodec | focalcodec_50hz_2k_causal | NaN | NaN |
| afrinames | FocalCodec | focalcodec_50hz_4k_causal | NaN | NaN |
| afrinames | FocalCodec | focalcodec_50hz_65k_causal | NaN | NaN |
| afrinames | LanguageCodec | bandwidth_0 | NaN | NaN |
| afrinames | LanguageCodec | bandwidth_1 | NaN | NaN |
| afrinames | LanguageCodec | bandwidth_2 | NaN | NaN |
| afrinames | LanguageCodec | bandwidth_3 | NaN | NaN |
| afrinames | SemantiCodec | 0.31kbps | NaN | NaN |
| afrinames | SemantiCodec | 0.33kbps | NaN | NaN |
| afrinames | SemantiCodec | 0.63kbps | NaN | NaN |
| afrinames | SemantiCodec | 0.68kbps | NaN | NaN |
| afrinames | SemantiCodec | 1.25kbps | NaN | NaN |
| afrinames | SemantiCodec | 1.40kbps | NaN | NaN |
| afrinames | UniCodec | out_6.6kbps | NaN | NaN |
| afrinames | WavTokenizer | WavTokenizer_large-speech-75token | NaN | NaN |
| afrinames | WavTokenizer | WavTokenizer_large-unify-40token | NaN | NaN |
| afrinames | WavTokenizer | WavTokenizer_medium-music-audio-75token | NaN | NaN |
| afrinames | WavTokenizer | WavTokenizer_medium-speech-75token | NaN | NaN |
| afrinames | WavTokenizer | WavTokenizer_small-320-24k-4096 | NaN | NaN |
| afrinames | WavTokenizer | WavTokenizer_small-600-24k-4096 | NaN | NaN |
| afrispeech_dialog | DAC | out_16kbps | NaN | NaN |
| afrispeech_dialog | DAC | out_24kbps | NaN | NaN |
| afrispeech_dialog | DAC | out_8kbps | NaN | NaN |
| afrispeech_dialog | Encodec | out_12kbps | NaN | NaN |
| afrispeech_dialog | Encodec | out_24kbps | NaN | NaN |
| afrispeech_dialog | Encodec | out_3kbps | NaN | NaN |
| afrispeech_dialog | Encodec | out_6kbps | NaN | NaN |
| afrispeech_dialog | FocalCodec | focalcodec_12_5hz | NaN | NaN |
| afrispeech_dialog | FocalCodec | focalcodec_25hz | NaN | NaN |
| afrispeech_dialog | FocalCodec | focalcodec_50hz | NaN | NaN |
| afrispeech_dialog | FocalCodec | focalcodec_50hz_2k_causal | NaN | NaN |
| afrispeech_dialog | FocalCodec | focalcodec_50hz_4k_causal | NaN | NaN |
| afrispeech_dialog | FocalCodec | focalcodec_50hz_65k_causal | NaN | NaN |
| afrispeech_dialog | LanguageCodec | bandwidth_0 | NaN | NaN |
| afrispeech_dialog | LanguageCodec | bandwidth_1 | NaN | NaN |
| afrispeech_dialog | LanguageCodec | bandwidth_2 | NaN | NaN |
| afrispeech_dialog | LanguageCodec | bandwidth_3 | NaN | NaN |
| afrispeech_dialog | SemantiCodec | 0.31kbps | NaN | NaN |
| afrispeech_dialog | SemantiCodec | 0.33kbps | NaN | NaN |
| afrispeech_dialog | SemantiCodec | 0.63kbps | NaN | NaN |
| afrispeech_dialog | SemantiCodec | 0.68kbps | NaN | NaN |
| afrispeech_dialog | SemantiCodec | 1.25kbps | NaN | NaN |
| afrispeech_dialog | SemantiCodec | 1.40kbps | NaN | NaN |
| afrispeech_dialog | UniCodec | out_6.6kbps | NaN | NaN |
| afrispeech_dialog | WavTokenizer | WavTokenizer_large-speech-75token | NaN | NaN |
| afrispeech_dialog | WavTokenizer | WavTokenizer_large-unify-40token | NaN | NaN |
| afrispeech_dialog | WavTokenizer | WavTokenizer_medium-music-audio-75token | NaN | NaN |
| afrispeech_dialog | WavTokenizer | WavTokenizer_medium-speech-75token | NaN | NaN |
| afrispeech_dialog | WavTokenizer | WavTokenizer_small-320-24k-4096 | NaN | NaN |
| afrispeech_dialog | WavTokenizer | WavTokenizer_small-600-24k-4096 | NaN | NaN |
| afrispeech_multilingual | DAC | out_16kbps | NaN | NaN |
| afrispeech_multilingual | DAC | out_24kbps | NaN | NaN |
| afrispeech_multilingual | DAC | out_8kbps | NaN | NaN |
| afrispeech_multilingual | Encodec | out_12kbps | NaN | NaN |
| afrispeech_multilingual | Encodec | out_24kbps | NaN | NaN |
| afrispeech_multilingual | Encodec | out_3kbps | NaN | NaN |
| afrispeech_multilingual | Encodec | out_6kbps | NaN | NaN |
| afrispeech_multilingual | FocalCodec | focalcodec_12_5hz | NaN | NaN |
| afrispeech_multilingual | FocalCodec | focalcodec_25hz | NaN | NaN |
| afrispeech_multilingual | FocalCodec | focalcodec_50hz | NaN | NaN |
| afrispeech_multilingual | FocalCodec | focalcodec_50hz_2k_causal | NaN | NaN |
| afrispeech_multilingual | FocalCodec | focalcodec_50hz_4k_causal | NaN | NaN |
| afrispeech_multilingual | FocalCodec | focalcodec_50hz_65k_causal | NaN | NaN |
| afrispeech_multilingual | LanguageCodec | bandwidth_0 | NaN | NaN |
| afrispeech_multilingual | LanguageCodec | bandwidth_1 | NaN | NaN |
| afrispeech_multilingual | LanguageCodec | bandwidth_2 | NaN | NaN |
| afrispeech_multilingual | LanguageCodec | bandwidth_3 | NaN | NaN |
| afrispeech_multilingual | SemantiCodec | 0.31kbps | NaN | NaN |
| afrispeech_multilingual | SemantiCodec | 0.33kbps | NaN | NaN |
| afrispeech_multilingual | SemantiCodec | 0.63kbps | NaN | NaN |
| afrispeech_multilingual | SemantiCodec | 0.68kbps | NaN | NaN |
| afrispeech_multilingual | SemantiCodec | 1.25kbps | NaN | NaN |
| afrispeech_multilingual | SemantiCodec | 1.40kbps | NaN | NaN |
| afrispeech_multilingual | UniCodec | out_6.6kbps | NaN | NaN |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_large-speech-75token | NaN | NaN |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_large-unify-40token | NaN | NaN |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_medium-music-audio-75token | NaN | NaN |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_medium-speech-75token | NaN | NaN |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_small-320-24k-4096 | NaN | NaN |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_small-600-24k-4096 | NaN | NaN |

## ASV_v2 (application-level)

### Dataset × Codec Means

| Dataset | Codec | Variants | eer_percent | min_dcf | mean_num_scored_files |
|---|---|---|---|---|---|
| afrinames | DAC | 3 | 0.0114 | 0.0137 | 13225.00 |
| afrinames | Encodec | 4 | 0.3697 | 0.0633 | 22500.00 |
| afrinames | FocalCodec | 6 | 22.1189 | 0.9662 | 22500.00 |
| afrinames | LanguageCodec | 4 | 0.5258 | 0.1604 | 4489.00 |
| afrinames | SemantiCodec | 6 | 2.7912 | 0.3837 | 22500.00 |
| afrinames | UniCodec | 1 | 0.8188 | 0.1310 | 22500.00 |
| afrinames | WavTokenizer | 6 | 1.2960 | 0.2789 | 22500.00 |
| afrispeech_dialog | DAC | 3 | 0.0142 | 0.0068 | 2401.00 |
| afrispeech_dialog | Encodec | 4 | 0.0000 | 0.0000 | 2401.00 |
| afrispeech_dialog | FocalCodec | 6 | 16.6206 | 0.9705 | 2401.00 |
| afrispeech_dialog | LanguageCodec | 4 | 0.0000 | 0.0000 | 2401.00 |
| afrispeech_dialog | SemantiCodec | 6 | 1.3428 | 0.5542 | 2401.00 |
| afrispeech_dialog | UniCodec | 1 | 20.4082 | 1.0000 | 2401.00 |
| afrispeech_dialog | WavTokenizer | 6 | 0.5315 | 0.1888 | 2401.00 |
| afrispeech_multilingual | DAC | 3 | 0.0000 | 0.0000 | 64.00 |
| afrispeech_multilingual | Encodec | 4 | 0.0707 | 0.0950 | 10000.00 |
| afrispeech_multilingual | FocalCodec | 6 | 24.1187 | 0.9983 | 10000.00 |
| afrispeech_multilingual | LanguageCodec | 4 | 0.0278 | 0.0525 | 10000.00 |
| afrispeech_multilingual | SemantiCodec | 6 | 0.7971 | 0.4300 | 10000.00 |
| afrispeech_multilingual | UniCodec | 1 | 1.0000 | 0.3900 | 10000.00 |
| afrispeech_multilingual | WavTokenizer | 6 | 1.4630 | 0.4050 | 10000.00 |

### Variant-Level Results

| Dataset | Codec | Variant | eer_percent | min_dcf | num_scored_files |
|---|---|---|---|---|---|
| afrinames | DAC | out_16kbps | 0.0000 | 0.0000 | 13225.00 |
| afrinames | DAC | out_24kbps | 0.0305 | 0.0336 | 13225.00 |
| afrinames | DAC | out_8kbps | 0.0038 | 0.0076 | 13225.00 |
| afrinames | Encodec | out_12kbps | 0.1655 | 0.0399 | 22500.00 |
| afrinames | Encodec | out_24kbps | 0.5235 | 0.0266 | 22500.00 |
| afrinames | Encodec | out_3kbps | 0.6667 | 0.1221 | 22500.00 |
| afrinames | Encodec | out_6kbps | 0.1230 | 0.0644 | 22500.00 |
| afrinames | FocalCodec | focalcodec_12_5hz | 29.3602 | 0.9844 | 22500.00 |
| afrinames | FocalCodec | focalcodec_25hz | 19.2908 | 0.9488 | 22500.00 |
| afrinames | FocalCodec | focalcodec_50hz | 14.0000 | 0.8909 | 22500.00 |
| afrinames | FocalCodec | focalcodec_50hz_2k_causal | 25.4407 | 0.9889 | 22500.00 |
| afrinames | FocalCodec | focalcodec_50hz_4k_causal | 23.8412 | 0.9933 | 22500.00 |
| afrinames | FocalCodec | focalcodec_50hz_65k_causal | 20.7808 | 0.9911 | 22500.00 |
| afrinames | LanguageCodec | bandwidth_0 | 0.3392 | 0.1642 | 4489.00 |
| afrinames | LanguageCodec | bandwidth_1 | 0.3618 | 0.1642 | 4489.00 |
| afrinames | LanguageCodec | bandwidth_2 | 0.2488 | 0.1343 | 4489.00 |
| afrinames | LanguageCodec | bandwidth_3 | 1.1533 | 0.1791 | 4489.00 |
| afrinames | SemantiCodec | 0.31kbps | 3.4139 | 0.5154 | 22500.00 |
| afrinames | SemantiCodec | 0.33kbps | 4.0000 | 0.4683 | 22500.00 |
| afrinames | SemantiCodec | 0.63kbps | 3.3333 | 0.3617 | 22500.00 |
| afrinames | SemantiCodec | 0.68kbps | 2.0000 | 0.3864 | 22500.00 |
| afrinames | SemantiCodec | 1.25kbps | 2.0000 | 0.2863 | 22500.00 |
| afrinames | SemantiCodec | 1.40kbps | 2.0000 | 0.2842 | 22500.00 |
| afrinames | UniCodec | out_6.6kbps | 0.8188 | 0.1310 | 22500.00 |
| afrinames | WavTokenizer | WavTokenizer_large-speech-75token | 0.6667 | 0.1399 | 22500.00 |
| afrinames | WavTokenizer | WavTokenizer_large-unify-40token | 1.3333 | 0.3327 | 22500.00 |
| afrinames | WavTokenizer | WavTokenizer_medium-music-audio-75token | 1.1790 | 0.2574 | 22500.00 |
| afrinames | WavTokenizer | WavTokenizer_medium-speech-75token | 0.7472 | 0.0999 | 22500.00 |
| afrinames | WavTokenizer | WavTokenizer_small-320-24k-4096 | 0.6711 | 0.2730 | 22500.00 |
| afrinames | WavTokenizer | WavTokenizer_small-600-24k-4096 | 3.1790 | 0.5707 | 22500.00 |
| afrispeech_dialog | DAC | out_16kbps | 0.0000 | 0.0000 | 2401.00 |
| afrispeech_dialog | DAC | out_24kbps | 0.0000 | 0.0000 | 2401.00 |
| afrispeech_dialog | DAC | out_8kbps | 0.0425 | 0.0204 | 2401.00 |
| afrispeech_dialog | Encodec | out_12kbps | 0.0000 | 0.0000 | 2401.00 |
| afrispeech_dialog | Encodec | out_24kbps | 0.0000 | 0.0000 | 2401.00 |
| afrispeech_dialog | Encodec | out_3kbps | 0.0000 | 0.0000 | 2401.00 |
| afrispeech_dialog | Encodec | out_6kbps | 0.0000 | 0.0000 | 2401.00 |
| afrispeech_dialog | FocalCodec | focalcodec_12_5hz | 20.1318 | 0.9809 | 2401.00 |
| afrispeech_dialog | FocalCodec | focalcodec_25hz | 14.2857 | 0.9209 | 2401.00 |
| afrispeech_dialog | FocalCodec | focalcodec_50hz | 14.3707 | 0.9617 | 2401.00 |
| afrispeech_dialog | FocalCodec | focalcodec_50hz_2k_causal | 18.3673 | 0.9796 | 2401.00 |
| afrispeech_dialog | FocalCodec | focalcodec_50hz_4k_causal | 16.6029 | 1.0000 | 2401.00 |
| afrispeech_dialog | FocalCodec | focalcodec_50hz_65k_causal | 15.9651 | 0.9796 | 2401.00 |
| afrispeech_dialog | LanguageCodec | bandwidth_0 | 0.0000 | 0.0000 | 2401.00 |
| afrispeech_dialog | LanguageCodec | bandwidth_1 | 0.0000 | 0.0000 | 2401.00 |
| afrispeech_dialog | LanguageCodec | bandwidth_2 | 0.0000 | 0.0000 | 2401.00 |
| afrispeech_dialog | LanguageCodec | bandwidth_3 | 0.0000 | 0.0000 | 2401.00 |
| afrispeech_dialog | SemantiCodec | 0.31kbps | 1.9345 | 0.8418 | 2401.00 |
| afrispeech_dialog | SemantiCodec | 0.33kbps | 1.9345 | 0.8291 | 2401.00 |
| afrispeech_dialog | SemantiCodec | 0.63kbps | 2.0408 | 0.7551 | 2401.00 |
| afrispeech_dialog | SemantiCodec | 0.68kbps | 2.0408 | 0.7972 | 2401.00 |
| afrispeech_dialog | SemantiCodec | 1.25kbps | 0.0638 | 0.0408 | 2401.00 |
| afrispeech_dialog | SemantiCodec | 1.40kbps | 0.0425 | 0.0612 | 2401.00 |
| afrispeech_dialog | UniCodec | out_6.6kbps | 20.4082 | 1.0000 | 2401.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_large-speech-75token | 0.1488 | 0.0612 | 2401.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_large-unify-40token | 0.3614 | 0.3291 | 2401.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_medium-music-audio-75token | 0.4677 | 0.2487 | 2401.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_medium-speech-75token | 0.0638 | 0.0204 | 2401.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_small-320-24k-4096 | 0.1063 | 0.1033 | 2401.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_small-600-24k-4096 | 2.0408 | 0.3699 | 2401.00 |
| afrispeech_multilingual | DAC | out_16kbps | 0.0000 | 0.0000 | 64.00 |
| afrispeech_multilingual | DAC | out_24kbps | 0.0000 | 0.0000 | 64.00 |
| afrispeech_multilingual | DAC | out_8kbps | 0.0000 | 0.0000 | 64.00 |
| afrispeech_multilingual | Encodec | out_12kbps | 0.0606 | 0.0500 | 10000.00 |
| afrispeech_multilingual | Encodec | out_24kbps | 0.0505 | 0.0500 | 10000.00 |
| afrispeech_multilingual | Encodec | out_3kbps | 0.1061 | 0.2000 | 10000.00 |
| afrispeech_multilingual | Encodec | out_6kbps | 0.0657 | 0.0800 | 10000.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_12_5hz | 32.0000 | 0.9900 | 10000.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_25hz | 19.8232 | 1.0000 | 10000.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_50hz | 14.1010 | 1.0000 | 10000.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_50hz_2k_causal | 29.9596 | 1.0000 | 10000.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_50hz_4k_causal | 25.8283 | 1.0000 | 10000.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_50hz_65k_causal | 23.0000 | 1.0000 | 10000.00 |
| afrispeech_multilingual | LanguageCodec | bandwidth_0 | 0.0253 | 0.0500 | 10000.00 |
| afrispeech_multilingual | LanguageCodec | bandwidth_1 | 0.0253 | 0.0500 | 10000.00 |
| afrispeech_multilingual | LanguageCodec | bandwidth_2 | 0.0152 | 0.0300 | 10000.00 |
| afrispeech_multilingual | LanguageCodec | bandwidth_3 | 0.0455 | 0.0800 | 10000.00 |
| afrispeech_multilingual | SemantiCodec | 0.31kbps | 1.0000 | 0.5400 | 10000.00 |
| afrispeech_multilingual | SemantiCodec | 0.33kbps | 1.0051 | 0.4600 | 10000.00 |
| afrispeech_multilingual | SemantiCodec | 0.63kbps | 0.8333 | 0.4500 | 10000.00 |
| afrispeech_multilingual | SemantiCodec | 0.68kbps | 0.9141 | 0.4000 | 10000.00 |
| afrispeech_multilingual | SemantiCodec | 1.25kbps | 0.7879 | 0.3800 | 10000.00 |
| afrispeech_multilingual | SemantiCodec | 1.40kbps | 0.2424 | 0.3500 | 10000.00 |
| afrispeech_multilingual | UniCodec | out_6.6kbps | 1.0000 | 0.3900 | 10000.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_large-speech-75token | 0.1414 | 0.2400 | 10000.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_large-unify-40token | 4.8586 | 0.2900 | 10000.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_medium-music-audio-75token | 0.7576 | 0.4500 | 10000.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_medium-speech-75token | 0.1313 | 0.1700 | 10000.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_small-320-24k-4096 | 0.8889 | 0.4500 | 10000.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_small-600-24k-4096 | 2.0000 | 0.8300 | 10000.00 |

## ACC_MAP_v2 (application-level)

### Dataset × Codec Means

| Dataset | Codec | Variants | acc_percent | map_percent | mean_num_scored_files |
|---|---|---|---|---|---|
| afrinames | DAC | 3 | 100.0000 | 100.0000 | 115.00 |
| afrinames | Encodec | 4 | 99.5000 | 99.7500 | 150.00 |
| afrinames | FocalCodec | 6 | 16.6667 | 28.0318 | 150.00 |
| afrinames | LanguageCodec | 4 | 100.0000 | 100.0000 | 67.00 |
| afrinames | SemantiCodec | 6 | 93.2222 | 95.8961 | 150.00 |
| afrinames | UniCodec | 1 | 100.0000 | 100.0000 | 150.00 |
| afrinames | WavTokenizer | 6 | 96.3333 | 98.1111 | 150.00 |
| afrispeech_dialog | DAC | 3 | 100.0000 | 100.0000 | 49.00 |
| afrispeech_dialog | Encodec | 4 | 100.0000 | 100.0000 | 49.00 |
| afrispeech_dialog | FocalCodec | 6 | 30.9524 | 46.3036 | 49.00 |
| afrispeech_dialog | LanguageCodec | 4 | 100.0000 | 100.0000 | 49.00 |
| afrispeech_dialog | SemantiCodec | 6 | 91.4966 | 95.6916 | 49.00 |
| afrispeech_dialog | UniCodec | 1 | 6.1224 | 21.8577 | 49.00 |
| afrispeech_dialog | WavTokenizer | 6 | 98.2993 | 99.0930 | 49.00 |
| afrispeech_multilingual | DAC | 3 | 100.0000 | 100.0000 | 8.00 |
| afrispeech_multilingual | Encodec | 4 | 100.0000 | 100.0000 | 100.00 |
| afrispeech_multilingual | FocalCodec | 6 | 9.8333 | 21.3858 | 100.00 |
| afrispeech_multilingual | LanguageCodec | 4 | 100.0000 | 100.0000 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 6 | 98.1667 | 99.0000 | 100.00 |
| afrispeech_multilingual | UniCodec | 1 | 100.0000 | 100.0000 | 100.00 |
| afrispeech_multilingual | WavTokenizer | 6 | 96.3333 | 97.6794 | 100.00 |

### Variant-Level Results

| Dataset | Codec | Variant | acc_percent | map_percent | num_scored_files |
|---|---|---|---|---|---|
| afrinames | DAC | out_16kbps | 100.0000 | 100.0000 | 115.00 |
| afrinames | DAC | out_24kbps | 100.0000 | 100.0000 | 115.00 |
| afrinames | DAC | out_8kbps | 100.0000 | 100.0000 | 115.00 |
| afrinames | Encodec | out_12kbps | 100.0000 | 100.0000 | 150.00 |
| afrinames | Encodec | out_24kbps | 100.0000 | 100.0000 | 150.00 |
| afrinames | Encodec | out_3kbps | 98.6667 | 99.3333 | 150.00 |
| afrinames | Encodec | out_6kbps | 99.3333 | 99.6667 | 150.00 |
| afrinames | FocalCodec | focalcodec_12_5hz | 8.0000 | 16.9676 | 150.00 |
| afrinames | FocalCodec | focalcodec_25hz | 22.0000 | 35.1276 | 150.00 |
| afrinames | FocalCodec | focalcodec_50hz | 40.0000 | 53.1923 | 150.00 |
| afrinames | FocalCodec | focalcodec_50hz_2k_causal | 8.0000 | 18.6349 | 150.00 |
| afrinames | FocalCodec | focalcodec_50hz_4k_causal | 12.0000 | 21.8079 | 150.00 |
| afrinames | FocalCodec | focalcodec_50hz_65k_causal | 10.0000 | 22.4604 | 150.00 |
| afrinames | LanguageCodec | bandwidth_0 | 100.0000 | 100.0000 | 67.00 |
| afrinames | LanguageCodec | bandwidth_1 | 100.0000 | 100.0000 | 67.00 |
| afrinames | LanguageCodec | bandwidth_2 | 100.0000 | 100.0000 | 67.00 |
| afrinames | LanguageCodec | bandwidth_3 | 100.0000 | 100.0000 | 67.00 |
| afrinames | SemantiCodec | 0.31kbps | 88.0000 | 92.6698 | 150.00 |
| afrinames | SemantiCodec | 0.33kbps | 87.3333 | 92.0778 | 150.00 |
| afrinames | SemantiCodec | 0.63kbps | 94.0000 | 96.2291 | 150.00 |
| afrinames | SemantiCodec | 0.68kbps | 96.0000 | 97.6222 | 150.00 |
| afrinames | SemantiCodec | 1.25kbps | 97.3333 | 98.4444 | 150.00 |
| afrinames | SemantiCodec | 1.40kbps | 96.6667 | 98.3333 | 150.00 |
| afrinames | UniCodec | out_6.6kbps | 100.0000 | 100.0000 | 150.00 |
| afrinames | WavTokenizer | WavTokenizer_large-speech-75token | 100.0000 | 100.0000 | 150.00 |
| afrinames | WavTokenizer | WavTokenizer_large-unify-40token | 94.6667 | 97.2222 | 150.00 |
| afrinames | WavTokenizer | WavTokenizer_medium-music-audio-75token | 98.0000 | 99.0000 | 150.00 |
| afrinames | WavTokenizer | WavTokenizer_medium-speech-75token | 100.0000 | 100.0000 | 150.00 |
| afrinames | WavTokenizer | WavTokenizer_small-320-24k-4096 | 97.3333 | 98.5556 | 150.00 |
| afrinames | WavTokenizer | WavTokenizer_small-600-24k-4096 | 88.0000 | 93.8889 | 150.00 |
| afrispeech_dialog | DAC | out_16kbps | 100.0000 | 100.0000 | 49.00 |
| afrispeech_dialog | DAC | out_24kbps | 100.0000 | 100.0000 | 49.00 |
| afrispeech_dialog | DAC | out_8kbps | 100.0000 | 100.0000 | 49.00 |
| afrispeech_dialog | Encodec | out_12kbps | 100.0000 | 100.0000 | 49.00 |
| afrispeech_dialog | Encodec | out_24kbps | 100.0000 | 100.0000 | 49.00 |
| afrispeech_dialog | Encodec | out_3kbps | 100.0000 | 100.0000 | 49.00 |
| afrispeech_dialog | Encodec | out_6kbps | 100.0000 | 100.0000 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_12_5hz | 30.6122 | 42.4613 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_25hz | 42.8571 | 58.0615 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_50hz | 46.9388 | 59.7017 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_50hz_2k_causal | 20.4082 | 35.6669 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_50hz_4k_causal | 22.4490 | 40.8029 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_50hz_65k_causal | 22.4490 | 41.1274 | 49.00 |
| afrispeech_dialog | LanguageCodec | bandwidth_0 | 100.0000 | 100.0000 | 49.00 |
| afrispeech_dialog | LanguageCodec | bandwidth_1 | 100.0000 | 100.0000 | 49.00 |
| afrispeech_dialog | LanguageCodec | bandwidth_2 | 100.0000 | 100.0000 | 49.00 |
| afrispeech_dialog | LanguageCodec | bandwidth_3 | 100.0000 | 100.0000 | 49.00 |
| afrispeech_dialog | SemantiCodec | 0.31kbps | 85.7143 | 92.8571 | 49.00 |
| afrispeech_dialog | SemantiCodec | 0.33kbps | 85.7143 | 92.5170 | 49.00 |
| afrispeech_dialog | SemantiCodec | 0.63kbps | 87.7551 | 93.8776 | 49.00 |
| afrispeech_dialog | SemantiCodec | 0.68kbps | 89.7959 | 94.8980 | 49.00 |
| afrispeech_dialog | SemantiCodec | 1.25kbps | 100.0000 | 100.0000 | 49.00 |
| afrispeech_dialog | SemantiCodec | 1.40kbps | 100.0000 | 100.0000 | 49.00 |
| afrispeech_dialog | UniCodec | out_6.6kbps | 6.1224 | 21.8577 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_large-speech-75token | 100.0000 | 100.0000 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_large-unify-40token | 97.9592 | 98.9796 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_medium-music-audio-75token | 97.9592 | 98.9796 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_medium-speech-75token | 100.0000 | 100.0000 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_small-320-24k-4096 | 100.0000 | 100.0000 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_small-600-24k-4096 | 93.8776 | 96.5986 | 49.00 |
| afrispeech_multilingual | DAC | out_16kbps | 100.0000 | 100.0000 | 8.00 |
| afrispeech_multilingual | DAC | out_24kbps | 100.0000 | 100.0000 | 8.00 |
| afrispeech_multilingual | DAC | out_8kbps | 100.0000 | 100.0000 | 8.00 |
| afrispeech_multilingual | Encodec | out_12kbps | 100.0000 | 100.0000 | 100.00 |
| afrispeech_multilingual | Encodec | out_24kbps | 100.0000 | 100.0000 | 100.00 |
| afrispeech_multilingual | Encodec | out_3kbps | 100.0000 | 100.0000 | 100.00 |
| afrispeech_multilingual | Encodec | out_6kbps | 100.0000 | 100.0000 | 100.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_12_5hz | 4.0000 | 11.9661 | 100.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_25hz | 9.0000 | 22.8256 | 100.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_50hz | 25.0000 | 41.4836 | 100.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_50hz_2k_causal | 7.0000 | 15.6516 | 100.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_50hz_4k_causal | 5.0000 | 16.5825 | 100.00 |
| afrispeech_multilingual | FocalCodec | focalcodec_50hz_65k_causal | 9.0000 | 19.8056 | 100.00 |
| afrispeech_multilingual | LanguageCodec | bandwidth_0 | 100.0000 | 100.0000 | 100.00 |
| afrispeech_multilingual | LanguageCodec | bandwidth_1 | 100.0000 | 100.0000 | 100.00 |
| afrispeech_multilingual | LanguageCodec | bandwidth_2 | 100.0000 | 100.0000 | 100.00 |
| afrispeech_multilingual | LanguageCodec | bandwidth_3 | 100.0000 | 100.0000 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 0.31kbps | 97.0000 | 98.5000 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 0.33kbps | 99.0000 | 99.5000 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 0.63kbps | 98.0000 | 98.8333 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 0.68kbps | 96.0000 | 97.6667 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 1.25kbps | 99.0000 | 99.5000 | 100.00 |
| afrispeech_multilingual | SemantiCodec | 1.40kbps | 100.0000 | 100.0000 | 100.00 |
| afrispeech_multilingual | UniCodec | out_6.6kbps | 100.0000 | 100.0000 | 100.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_large-speech-75token | 100.0000 | 100.0000 | 100.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_large-unify-40token | 90.0000 | 93.2667 | 100.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_medium-music-audio-75token | 99.0000 | 99.3333 | 100.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_medium-speech-75token | 100.0000 | 100.0000 | 100.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_small-320-24k-4096 | 96.0000 | 97.6667 | 100.00 |
| afrispeech_multilingual | WavTokenizer | WavTokenizer_small-600-24k-4096 | 93.0000 | 95.8095 | 100.00 |

## WER_B200 (application-level)

### Dataset × Codec Means

| Dataset | Codec | Variants | wer | cer | mean_num_scored_files |
|---|---|---|---|---|---|
| afrispeech_dialog | DAC | 3 | 0.2908 | 0.2622 | 49.00 |
| afrispeech_dialog | Encodec | 4 | 0.3237 | 0.2911 | 49.00 |
| afrispeech_dialog | FocalCodec | 6 | 0.4343 | 0.3782 | 49.00 |
| afrispeech_dialog | LanguageCodec | 4 | 0.2944 | 0.2646 | 49.00 |
| afrispeech_dialog | SemantiCodec | 6 | 0.5281 | 0.4578 | 49.00 |
| afrispeech_dialog | UniCodec | 1 | 0.5817 | 0.4898 | 49.00 |
| afrispeech_dialog | WavTokenizer | 6 | 0.5053 | 0.4332 | 49.00 |

### Variant-Level Results

| Dataset | Codec | Variant | wer | cer | num_scored_files |
|---|---|---|---|---|---|
| afrispeech_dialog | DAC | out_16kbps | 0.2826 | 0.2554 | 49.00 |
| afrispeech_dialog | DAC | out_24kbps | 0.2923 | 0.2625 | 49.00 |
| afrispeech_dialog | DAC | out_8kbps | 0.2974 | 0.2686 | 49.00 |
| afrispeech_dialog | Encodec | out_12kbps | 0.3034 | 0.2729 | 49.00 |
| afrispeech_dialog | Encodec | out_24kbps | 0.2936 | 0.2635 | 49.00 |
| afrispeech_dialog | Encodec | out_3kbps | 0.3727 | 0.3348 | 49.00 |
| afrispeech_dialog | Encodec | out_6kbps | 0.3252 | 0.2933 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_12_5hz | 0.5702 | 0.4818 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_25hz | 0.3951 | 0.3449 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_50hz | 0.3490 | 0.3061 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_50hz_2k_causal | 0.4610 | 0.4056 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_50hz_4k_causal | 0.4359 | 0.3823 | 49.00 |
| afrispeech_dialog | FocalCodec | focalcodec_50hz_65k_causal | 0.3948 | 0.3484 | 49.00 |
| afrispeech_dialog | LanguageCodec | bandwidth_0 | 0.2947 | 0.2650 | 49.00 |
| afrispeech_dialog | LanguageCodec | bandwidth_1 | 0.2929 | 0.2630 | 49.00 |
| afrispeech_dialog | LanguageCodec | bandwidth_2 | 0.2928 | 0.2625 | 49.00 |
| afrispeech_dialog | LanguageCodec | bandwidth_3 | 0.2973 | 0.2678 | 49.00 |
| afrispeech_dialog | SemantiCodec | 0.31kbps | 0.7622 | 0.6496 | 49.00 |
| afrispeech_dialog | SemantiCodec | 0.33kbps | 0.7452 | 0.6324 | 49.00 |
| afrispeech_dialog | SemantiCodec | 0.63kbps | 0.4714 | 0.4112 | 49.00 |
| afrispeech_dialog | SemantiCodec | 0.68kbps | 0.4526 | 0.3955 | 49.00 |
| afrispeech_dialog | SemantiCodec | 1.25kbps | 0.3777 | 0.3387 | 49.00 |
| afrispeech_dialog | SemantiCodec | 1.40kbps | 0.3591 | 0.3191 | 49.00 |
| afrispeech_dialog | UniCodec | out_6.6kbps | 0.5817 | 0.4898 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_large-speech-75token | 0.4204 | 0.3663 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_large-unify-40token | 0.5347 | 0.4555 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_medium-music-audio-75token | 0.5314 | 0.4678 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_medium-speech-75token | 0.4152 | 0.3597 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_small-320-24k-4096 | 0.4454 | 0.3827 | 49.00 |
| afrispeech_dialog | WavTokenizer | WavTokenizer_small-600-24k-4096 | 0.6846 | 0.5671 | 49.00 |

## Interpretation Notes

- Lower is better: `wer`, `cer`, `eer_percent`, `min_dcf`, `f0_rmse`.
- Higher is better: NISQA/UTMOS/ViSQOL/STOI/SPK_SIM, `acc_percent`, `map_percent`, and `overall_prosody` (as used in this repo).
- `WER_B200` currently covers `afrispeech_dialog` only; the other metrics include all three datasets.

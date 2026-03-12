# ASV Summary

Variants processed: 90

## Overall codec ranking (lower is better)

| Rank | Codec | Mean EER% | Mean minDCF | Mean EER CI upper | Overlap variants |
|---:|---|---:|---:|---:|---:|
| 1 | DAC | 0.009 | 0.0068 | 0.019 | 3 |
| 2 | Encodec | 0.147 | 0.0528 | 0.286 | 8 |
| 3 | LanguageCodec | 0.185 | 0.0710 | 0.354 | 8 |
| 4 | WavTokenizer | 1.097 | 0.2909 | 1.909 | 18 |
| 5 | SemantiCodec | 1.644 | 0.4560 | 2.685 | 18 |
| 6 | UniCodec | 7.409 | 0.5070 | 9.892 | 3 |
| 7 | FocalCodec | 20.953 | 0.9783 | 24.252 | 18 |

## Notes

- `quality_flag=zero_eer_fragile` means EER is zero but CI upper bound is not tight.
- `has_overlap=true` indicates target/impostor score overlap was observed in that variant.

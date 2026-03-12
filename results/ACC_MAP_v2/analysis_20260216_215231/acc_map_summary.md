# ACC/mAP Summary

Variants processed: 90

## Overall codec ranking (higher is better)

| Rank | Codec | Mean ACC% | Mean mAP% | Mean ACC CI upper | Mean mAP CI upper | Weak variants |
|---:|---|---:|---:|---:|---:|---:|
| 1 | DAC | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| 2 | LanguageCodec | 100.000 | 100.000 | 100.000 | 100.000 | 0 |
| 3 | Encodec | 99.833 | 99.917 | 100.000 | 100.000 | 0 |
| 4 | WavTokenizer | 96.989 | 98.294 | 98.944 | 99.397 | 0 |
| 5 | SemantiCodec | 94.295 | 96.863 | 98.005 | 98.897 | 0 |
| 6 | UniCodec | 68.707 | 73.953 | 72.109 | 76.789 | 1 |
| 7 | FocalCodec | 19.151 | 31.907 | 27.467 | 39.175 | 18 |

## Notes

- This implementation is retrieval-based: each degraded file queries all references by cosine score.
- `ACC` is top-1 identification accuracy.
- `mAP` here is equivalent to mean reciprocal rank because each query has one relevant item.

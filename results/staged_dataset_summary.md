# Staged Dataset Summary

The staged fall-anticipation dataset combines three sub-datasets: GMDCSA24,
LE2I, and CAUCAFall. Samples are fixed-length observation windows. The canonical
staged metadata used by the V-JEPA models contains 4,211 windows from 218 unique
videos.

| Sub-dataset | Train | Val | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| GMDCSA24 | 525 | 75 | 724 | 1,324 |
| CAUCAFall | 328 | 141 | 117 | 586 |
| LE2I | 1,516 | 437 | 348 | 2,301 |
| **Total** | **2,369** | **653** | **1,189** | **4,211** |

Class balance by split:

| Split | Negative | Positive | Total |
| --- | ---: | ---: | ---: |
| Train | 1,998 | 371 | 2,369 |
| Val | 567 | 86 | 653 |
| Test | 1,028 | 161 | 1,189 |
| **Total** | **3,593** | **618** | **4,211** |

For the pose-feature experiments, RTMPose feature extraction produced 4,206
usable windows. The only difference from the canonical staged metadata is that
five CAUCAFall validation negatives are missing after pose feature extraction.

| Pose-feature sub-dataset | Train | Val | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| GMDCSA24 | 525 | 75 | 724 | 1,324 |
| CAUCAFall | 328 | 136 | 117 | 581 |
| LE2I | 1,516 | 437 | 348 | 2,301 |
| **Total** | **2,369** | **648** | **1,189** | **4,206** |

Pose-feature class balance by split:

| Split | Negative | Positive | Total |
| --- | ---: | ---: | ---: |
| Train | 1,998 | 371 | 2,369 |
| Val | 562 | 86 | 648 |
| Test | 1,028 | 161 | 1,189 |
| **Total** | **3,588** | **618** | **4,206** |

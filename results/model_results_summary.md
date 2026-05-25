# Model Results Summary

Current fixed-threshold results use classification threshold `0.5`.

| Model | Variant | Best Val Epoch | Val Acc | Test Acc | Test TN | Test FP | Test FN | Test TP | Pos Precision | Pos Recall | Pos F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Video CNN + Transformer | raw frames | 5 | 0.8135 | 0.7494 | 325 | 36 | 76 | 10 | 0.2174 | 0.1163 | 0.1515 |
| Pose Transformer | raw RTMPose | 2 | 0.8316 | 0.8114 | 370 | 0 | 86 | 0 | 0.0000 | 0.0000 | 0.0000 |
| Pose Transformer | normalized pose + velocity | 5 | 0.4767 | 0.5503 | 186 | 175 | 26 | 60 | 0.2553 | 0.6977 | 0.3738 |
| V-JEPA Baseline | latent + classification loss only | 2 | 0.8808 | 0.7603 | 284 | 89 | 21 | 65 | 0.4221 | 0.7558 | 0.5417 |
| V-JEPA Predictive | lambda 0.1 | 4 | 0.8731 | 0.7952 | 301 | 72 | 22 | 64 | 0.4706 | 0.7442 | 0.5766 |
| V-JEPA Predictive | latent + predictive loss | 4 | 0.8886 | 0.7843 | 295 | 78 | 21 | 65 | 0.4545 | 0.7558 | 0.5677 |
| V-JEPA Predictive | lambda 0.5 | 5 | 0.8290 | 0.7429 | 268 | 105 | 13 | 73 | 0.4101 | 0.8488 | 0.5530 |

## Recall-Prioritized Threshold Tuning

Thresholds below were selected by validation positive-class F2.

| Model | Threshold | Test TN | Test FP | Test FN | Test TP | Pos Precision | Pos Recall | Pos F1 | Test Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Video CNN + Transformer | 0.3405 | 60 | 301 | 15 | 71 | 0.1909 | 0.8256 | 0.3100 | 0.2931 |
| Pose Transformer normalized | 0.3890 | 55 | 306 | 4 | 82 | 0.2113 | 0.9535 | 0.3460 | 0.3065 |
| V-JEPA Predictive | 0.4300 | 292 | 81 | 18 | 68 | 0.4564 | 0.7907 | 0.5787 | 0.7843 |

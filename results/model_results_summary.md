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

## Fallen-State Prediction

Results below use the staged fallen-state dataset with a 2.0 second prediction
horizon. Positive label is `fallen`; LE2I fallen state is derived from the
fall-end frame. Fixed classification threshold is `0.5`.

| Model | Variant | Best Val Epoch | Val Acc | Test Acc | Test TN | Test FP | Test FN | Test TP | Pos Precision | Pos Recall | Pos F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pose Transformer | RTMPose keypoints | 5 | 0.7868 | 0.7325 | 698 | 214 | 119 | 214 | 0.5000 | 0.6426 | 0.5624 |
| V-JEPA Baseline | latent + classification loss only | 1 | 0.8231 | 0.8145 | 746 | 166 | 65 | 268 | 0.6175 | 0.8048 | 0.6988 |
| V-JEPA Predictive | latent + predictive loss, lambda 0.2 | 3 | 0.8413 | 0.8185 | 744 | 168 | 58 | 275 | 0.6208 | 0.8258 | 0.7088 |

## Three-Staged Dataset F2-Tuned Results

Results below use GMDCSA24, LE2I, and CAUCAFall. Thresholds were selected on the
validation split by maximizing positive-class F2, then evaluated on the test
split. Balanced accuracy is the average of positive recall and non-fall
specificity.

| Task | Model | Threshold | Test Acc | Balanced Acc | Pos Precision | Pos Recall | Pos F1 | Pos F2 | Test TN | Test FP | Test FN | Test TP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fall anticipation | Pose Transformer | 0.333 | 0.257 | 0.544 | 0.147 | 0.938 | 0.255 | 0.453 | 155 | 873 | 10 | 151 |
| Fall anticipation | Pose Predictive | 0.464 | 0.218 | 0.521 | 0.141 | 0.938 | 0.245 | 0.440 | 108 | 920 | 10 | 151 |
| Fall anticipation | V-JEPA Baseline | 0.270 | 0.521 | 0.615 | 0.185 | 0.745 | 0.296 | 0.464 | 499 | 529 | 41 | 120 |
| Fall anticipation | V-JEPA Predictive | 0.578 | 0.671 | 0.679 | 0.246 | 0.689 | 0.362 | 0.506 | 687 | 341 | 50 | 111 |
| Fallen-state | Pose Transformer | 0.250 | 0.339 | 0.526 | 0.302 | 0.971 | 0.460 | 0.673 | 75 | 866 | 11 | 374 |
| Fallen-state | Pose Predictive | 0.297 | 0.291 | 0.501 | 0.291 | 1.000 | 0.450 | 0.672 | 1 | 940 | 0 | 385 |
| Fallen-state | V-JEPA Baseline | 0.402 | 0.827 | 0.830 | 0.660 | 0.836 | 0.738 | 0.794 | 775 | 166 | 63 | 322 |
| Fallen-state | V-JEPA Predictive | 0.162 | 0.810 | 0.835 | 0.620 | 0.894 | 0.732 | 0.821 | 730 | 211 | 41 | 344 |

## Three-Staged Dataset Balanced-Accuracy Thresholds

Thresholds below were selected on the validation split by maximizing balanced
accuracy, then evaluated on the test split.

| Task | Model | Threshold | Test Acc | Balanced Acc | Pos Precision | Pos Recall | Pos F1 | Pos F2 | Test TN | Test FP | Test FN | Test TP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fall anticipation | Pose Transformer | 0.344 | 0.431 | 0.569 | 0.161 | 0.758 | 0.265 | 0.435 | 391 | 637 | 39 | 122 |
| Fall anticipation | Pose Predictive | 0.464 | 0.218 | 0.521 | 0.141 | 0.938 | 0.245 | 0.440 | 108 | 920 | 10 | 151 |
| Fall anticipation | V-JEPA Baseline | 0.580 | 0.765 | 0.657 | 0.291 | 0.509 | 0.370 | 0.443 | 828 | 200 | 79 | 82 |
| Fall anticipation | V-JEPA Predictive | 0.578 | 0.671 | 0.679 | 0.246 | 0.689 | 0.362 | 0.506 | 687 | 341 | 50 | 111 |
| Fallen-state | Pose Transformer | 0.439 | 0.742 | 0.680 | 0.559 | 0.532 | 0.545 | 0.537 | 779 | 162 | 180 | 205 |
| Fallen-state | Pose Predictive | 0.550 | 0.725 | 0.670 | 0.527 | 0.538 | 0.532 | 0.535 | 755 | 186 | 178 | 207 |
| Fallen-state | V-JEPA Baseline | 0.402 | 0.827 | 0.830 | 0.660 | 0.836 | 0.738 | 0.794 | 775 | 166 | 63 | 322 |
| Fallen-state | V-JEPA Predictive | 0.648 | 0.875 | 0.859 | 0.765 | 0.821 | 0.792 | 0.809 | 844 | 97 | 69 | 316 |

## Expanded Staged + OOPs Fall-Anticipation Results

Results below use GMDCSA24, LE2I, CAUCAFall, and OOPs with OOPs negatives
sampled to match OOPs positives. Fixed classification threshold is `0.5`.

| Model | Best Val Epoch | Val Acc | Test Acc | Pos Precision | Pos Recall | Pos F1 | Test TN | Test FP | Test FN | Test TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pose Transformer | 9 | 0.603 | 0.589 | 0.395 | 0.632 | 0.486 | 889 | 671 | 256 | 439 |
| V-JEPA Baseline | 1 | 0.730 | 0.755 | 0.576 | 0.663 | 0.617 | 1307 | 339 | 234 | 461 |
| V-JEPA Predictive | 1 | 0.727 | 0.762 | 0.589 | 0.665 | 0.624 | 1323 | 323 | 233 | 462 |

## Expanded Staged + OOPs F2-Tuned Thresholds

Results below use GMDCSA24, LE2I, CAUCAFall, and OOPs. Thresholds were
selected on the validation split by maximizing positive-class F2, then
evaluated on the test split.

| Model | Threshold | Test Acc | Balanced Acc | Pos Precision | Pos Recall | Pos F1 | Pos F2 | Test TN | Test FP | Test FN | Test TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pose Transformer | 0.146 | 0.333 | 0.512 | 0.313 | 0.977 | 0.474 | 0.686 | 72 | 1488 | 16 | 679 |
| V-JEPA Baseline | 0.123 | 0.592 | 0.684 | 0.415 | 0.911 | 0.570 | 0.735 | 753 | 893 | 62 | 633 |
| V-JEPA Predictive | 0.182 | 0.607 | 0.695 | 0.425 | 0.911 | 0.579 | 0.741 | 789 | 857 | 62 | 633 |

## Expanded Staged + OOPs Balanced-Accuracy Thresholds

Thresholds below were selected on the validation split by maximizing balanced
accuracy, then evaluated on the test split.

| Model | Threshold | Test Acc | Balanced Acc | Pos Precision | Pos Recall | Pos F1 | Pos F2 | Test TN | Test FP | Test FN | Test TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pose Transformer | 0.597 | 0.678 | 0.589 | 0.470 | 0.358 | 0.407 | 0.376 | 1279 | 281 | 446 | 249 |
| V-JEPA Baseline | 0.287 | 0.712 | 0.747 | 0.509 | 0.833 | 0.632 | 0.739 | 1088 | 558 | 116 | 579 |
| V-JEPA Predictive | 0.249 | 0.671 | 0.727 | 0.470 | 0.865 | 0.609 | 0.741 | 969 | 677 | 94 | 601 |

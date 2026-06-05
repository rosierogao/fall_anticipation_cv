# Expanded Temporal Ablation Summary

Model set: staged+unstaged fall anticipation (GMDCSA24 + LE2I + CAUCAFall + OOPs).

## Delete-One-Step Ablation

| Model | Obs. steps | Most important step | Relative position | Mean delta prob | Mean abs delta prob |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pose Transformer | 16 | 10 | 0.67 | 0.00786 | 0.01505 |
| V-JEPA Baseline | 8 | 7 | 1.00 | -0.01351 | 0.04910 |
| V-JEPA Predictive | 8 | 7 | 1.00 | 0.05344 | 0.06431 |

## Mask-In-Place Ablation, Zero Mask

| Model | Obs. steps | Most important step | Relative position | Mean delta prob | Mean abs delta prob |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pose Transformer | 16 | 15 | 1.00 | 0.00629 | 0.01689 |
| V-JEPA Baseline | 8 | 7 | 1.00 | 0.08874 | 0.11461 |
| V-JEPA Predictive | 8 | 7 | 1.00 | 0.01852 | 0.05982 |

## Top Predictive V-JEPA Mask-In-Place Steps

| Rank | Token | Approx raw frames | Relative position | Mean delta prob | Mean abs delta prob |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 7 | frames 14-15 | 1.00 | 0.01852 | 0.05982 |
| 2 | 2 | frames 4-5 | 0.29 | -0.00465 | 0.00585 |
| 3 | 3 | frames 6-7 | 0.43 | -0.00765 | 0.00918 |
| 4 | 5 | frames 10-11 | 0.71 | -0.00783 | 0.01094 |
| 5 | 4 | frames 8-9 | 0.57 | -0.00830 | 0.01003 |

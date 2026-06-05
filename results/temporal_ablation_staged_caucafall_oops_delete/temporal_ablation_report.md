# Temporal Ablation Results

Ablation method: delete. The reported value is the change in positive-class probability on the held-out test split.

## pose_transformer

Validation F2 threshold: 0.146311

Top time steps by mean positive-probability drop:

| rank | time_index | mean_relative_position | mean_delta_prob | support |
| --- | ---: | ---: | ---: | ---: |
| 1 | 10 | 0.667 | 0.007864 | 2255 |
| 2 | 8 | 0.533 | 0.007775 | 2255 |
| 3 | 5 | 0.333 | 0.007686 | 2255 |
| 4 | 9 | 0.600 | 0.007674 | 2255 |
| 5 | 4 | 0.267 | 0.007630 | 2255 |

## vjepa_baseline

Validation F2 threshold: 0.122600

Top time steps by mean positive-probability drop:

| rank | time_index | mean_relative_position | mean_delta_prob | support |
| --- | ---: | ---: | ---: | ---: |
| 1 | 7 | 1.000 | -0.013508 | 2341 |
| 2 | 6 | 0.857 | -0.025831 | 2341 |
| 3 | 3 | 0.429 | -0.030470 | 2341 |
| 4 | 5 | 0.714 | -0.031894 | 2341 |
| 5 | 4 | 0.571 | -0.032554 | 2341 |

## vjepa_predictive

Validation F2 threshold: 0.181602

Top time steps by mean positive-probability drop:

| rank | time_index | mean_relative_position | mean_delta_prob | support |
| --- | ---: | ---: | ---: | ---: |
| 1 | 7 | 1.000 | 0.053437 | 2341 |
| 2 | 5 | 0.714 | 0.005673 | 2341 |
| 3 | 2 | 0.286 | 0.005305 | 2341 |
| 4 | 4 | 0.571 | 0.002893 | 2341 |
| 5 | 6 | 0.857 | 0.002623 | 2341 |


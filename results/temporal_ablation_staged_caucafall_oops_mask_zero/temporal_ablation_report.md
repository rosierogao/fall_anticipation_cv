# Temporal Ablation Results

Ablation method: mask. The reported value is the change in positive-class probability on the held-out test split.

## pose_transformer

Validation F2 threshold: 0.146311

Top time steps by mean positive-probability drop:

| rank | time_index | mean_relative_position | mean_delta_prob | support |
| --- | ---: | ---: | ---: | ---: |
| 1 | 15 | 1.000 | 0.006289 | 2255 |
| 2 | 11 | 0.733 | 0.006064 | 2255 |
| 3 | 8 | 0.533 | 0.005669 | 2255 |
| 4 | 7 | 0.467 | 0.005200 | 2255 |
| 5 | 12 | 0.800 | 0.004322 | 2255 |

## vjepa_baseline

Validation F2 threshold: 0.122600

Top time steps by mean positive-probability drop:

| rank | time_index | mean_relative_position | mean_delta_prob | support |
| --- | ---: | ---: | ---: | ---: |
| 1 | 7 | 1.000 | 0.088736 | 2341 |
| 2 | 6 | 0.857 | 0.038902 | 2341 |
| 3 | 5 | 0.714 | 0.003525 | 2341 |
| 4 | 3 | 0.429 | -0.000083 | 2341 |
| 5 | 0 | 0.000 | -0.002215 | 2341 |

## vjepa_predictive

Validation F2 threshold: 0.181602

Top time steps by mean positive-probability drop:

| rank | time_index | mean_relative_position | mean_delta_prob | support |
| --- | ---: | ---: | ---: | ---: |
| 1 | 7 | 1.000 | 0.018519 | 2341 |
| 2 | 2 | 0.286 | -0.004650 | 2341 |
| 3 | 3 | 0.429 | -0.007647 | 2341 |
| 4 | 5 | 0.714 | -0.007828 | 2341 |
| 5 | 4 | 0.571 | -0.008297 | 2341 |


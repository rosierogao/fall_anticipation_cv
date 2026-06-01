# Fallen-State Horizon Statistics

This note summarizes the measured time between fall onset and the fallen state
for the staged datasets. The secondary fallen-state prediction task uses a
2.0 second prediction horizon.

## Definition

| Dataset | Transition Duration Definition |
|---|---|
| GMDCSA24 | `fallen.start - nearest prior fall.start` within the same video |
| LE2I | `(fall_end_frame - fall_start_frame) / video_fps` |
| CAUCAFall | `fallen.start - nearest prior fall.start` within the same video |

## Duration Statistics

| Dataset | Events | Median | Mean | P75 | P90 | Max | % <= 2s |
|---|---:|---:|---:|---:|---:|---:|---:|
| GMDCSA24 | 77 | 1.76s | 1.97s | 2.34s | 2.91s | 3.52s | 54.5% |
| LE2I | 107 | 0.62s | 0.70s | 1.10s | 1.28s | 1.80s | 100.0% |
| CAUCAFall | 46 | 2.39s | 2.47s | 2.81s | 3.34s | 4.14s | 26.1% |
| Combined | 230 | 1.38s | 1.48s | 2.14s | 2.77s | 4.14s | 70.0% |

## Interpretation

A 2.0 second horizon is a reasonable initial setting for the fallen-state
prediction experiment. It covers all LE2I measured transitions, about half of
GMDCSA24 transitions, and a smaller fraction of CAUCAFall transitions.

This makes the task meaningfully predictive rather than simply detecting an
already-completed fall. The tradeoff is that CAUCAFall has slower transitions,
so some positive fallen-state events may occur beyond the 2.0 second horizon.
For the project, keep 2.0 seconds as the secondary-task horizon and report this
dataset-dependent timing caveat.

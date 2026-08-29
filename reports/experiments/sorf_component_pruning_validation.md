# SORF component pruning validation

## Question and controlled setting

This experiment tests whether observation dropping plus sparse-view consistency learning is necessary. The full and pruned models use the same user-disjoint split, five-rate data, matched B0 checkpoints, training budget, validation-only calibration, seeds, and test sets. The pruned model changes only `drop_probability`, `drop_weight`, and `consistency_weight` from their full-model values to zero.

## Five-rate result (seed 10086, %)

| Rate | Full Acc | Pruned Acc | Delta | Full F1 | Pruned F1 | Delta |
|---:|---:|---:|---:|---:|---:|---:|
| 5 s | 81.22 | 81.79 | +0.57 | 77.04 | 77.54 | +0.50 |
| 10 s | 78.69 | 79.84 | +1.15 | 73.93 | 74.72 | +0.80 |
| 20 s | 79.01 | 77.94 | -1.07 | 72.71 | 71.94 | -0.76 |
| 30 s | 76.58 | 75.91 | -0.67 | 70.74 | 70.30 | -0.44 |
| 60 s | 72.43 | 72.69 | +0.26 | 66.38 | 66.72 | +0.34 |
| Mean | - | - | +0.05 | - | - | +0.09 |

## Low-rate three-seed result (mean ± sample std, %)

| Rate | Full Acc | Pruned Acc | Delta | Full F1 | Pruned F1 | Delta |
|---:|---:|---:|---:|---:|---:|---:|
| 30 s | 75.90 ± 0.77 | 75.76 ± 0.53 | -0.14 | 70.30 ± 0.59 | 70.09 ± 0.22 | -0.21 |
| 60 s | 70.79 ± 1.64 | 70.75 ± 2.62 | -0.03 | 64.86 ± 1.49 | 64.97 ± 2.34 | +0.11 |

## Decision

The component is removed from the final model. Its removal preserves the target 30 s/60 s three-seed performance and does not reduce the five-rate mean, while simplifying training and eliminating an unsupported component claim.

The isolated 20 s score decreases, so the result is not described as uniform improvement. The pruning decision is based on the declared low-rate target and aggregate five-rate behavior, not cherry-picking.

Final model: real-observation relational expert + validation-constrained calibrated probability fusion.

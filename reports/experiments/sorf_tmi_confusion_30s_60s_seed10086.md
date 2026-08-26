# SORF-TMI low-rate confusion-matrix audit

> Seed: 10086
> Checkpoints, temperatures and fusion weights are identical to the audited five-rate main experiment.

## 30 seconds

| Model | Accuracy | Macro-F1 | Walk | Bike | Bus | Car | Train |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 72.89 | 67.27 | 86.37 | 75.17 | 55.45 | 74.56 | 44.77 |
| SORF-TMI | 76.58 | 70.74 | 87.51 | 76.71 | 59.40 | 79.44 | 50.65 |

Class recall changes (SORF-TMI − B0): Walk +1.53 pp, Bike +2.62 pp, Bus +0.47 pp, Car +6.75 pp, Train +2.05 pp.

## 60 seconds

| Model | Accuracy | Macro-F1 | Walk | Bike | Bus | Car | Train |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 71.55 | 63.73 | 84.74 | 67.44 | 52.28 | 74.66 | 39.55 |
| SORF-TMI | 72.43 | 66.38 | 85.69 | 69.92 | 53.10 | 74.65 | 48.54 |

Class recall changes (SORF-TMI − B0): Walk +2.51 pp, Bike -3.68 pp, Bus +0.47 pp, Car -2.03 pp, Train +20.12 pp.

## Audit status

- Recomputed B0 Accuracy/Macro-F1 exactly match the stored main result.
- Recomputed SORF-TMI Accuracy/Macro-F1 exactly match the stored main result.
- B0 and relation-expert labels are identical for each rate.
- Raw counts and row-normalized matrices are stored in the JSON artifact.

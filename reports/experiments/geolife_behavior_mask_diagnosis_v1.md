# GeoLife Behavior Mask degradation diagnosis v1

- Split: `train`
- Motion feature channels: `[3, 4, 5, 8]`
- Mask convention: `0 = masked`, `1 = retained`
- Physical duration: positive `delta_t` support corrected from N interpolated values to N-1 intervals

## Five-rate summary

| Rate | Samples | Avg points | Median points | Avg segment (s) | TS mask ratio | FS mask ratio | TS duration (s) | Fully masked TS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 s | 61142 | 47.86 | 54.00 | 275.31 | 80.74% | 76.61% | 219.61 | 15.02% |
| 10 s | 63618 | 26.34 | 30.00 | 276.47 | 96.12% | 74.63% | 265.26 | 55.34% |
| 20 s | 64055 | 14.28 | 15.00 | 279.63 | 99.54% | 70.20% | 278.34 | 96.41% |
| 30 s | 63857 | 10.07 | 10.00 | 282.26 | 99.82% | 64.52% | 281.77 | 98.90% |
| 60 s | 62786 | 5.44 | 5.00 | 269.98 | 99.97% | 52.98% | 269.91 | 99.86% |

## Per-class summary

| Rate | Class | Samples | Avg points | TS mask ratio | FS mask ratio | TS duration (s) |
|---:|---|---:|---:|---:|---:|---:|
| 5 s | Walk | 13068 | 49.87 | 81.48% | 83.84% | 225.84 |
| 5 s | Bike | 13084 | 46.90 | 83.94% | 83.08% | 214.30 |
| 5 s | Bus | 12853 | 53.91 | 75.90% | 72.74% | 222.99 |
| 5 s | Car | 13097 | 44.54 | 82.19% | 73.79% | 222.76 |
| 5 s | Train | 9040 | 42.57 | 79.81% | 66.42% | 208.94 |
| 10 s | Walk | 13059 | 26.27 | 96.32% | 81.16% | 269.18 |
| 10 s | Bike | 13085 | 25.53 | 97.13% | 78.78% | 256.84 |
| 10 s | Bus | 13011 | 28.23 | 95.62% | 72.84% | 279.55 |
| 10 s | Car | 13095 | 26.28 | 96.10% | 73.75% | 263.88 |
| 10 s | Train | 11368 | 25.29 | 95.32% | 65.40% | 255.67 |
| 20 s | Walk | 13038 | 14.18 | 99.57% | 75.68% | 279.36 |
| 20 s | Bike | 13080 | 14.16 | 99.82% | 72.91% | 274.81 |
| 20 s | Bus | 13109 | 14.79 | 99.54% | 72.35% | 287.84 |
| 20 s | Car | 13095 | 14.09 | 99.38% | 69.28% | 274.60 |
| 20 s | Train | 11733 | 14.15 | 99.39% | 59.70% | 274.72 |
| 30 s | Walk | 13002 | 9.95 | 99.87% | 69.06% | 279.81 |
| 30 s | Bike | 13081 | 10.22 | 99.93% | 66.45% | 284.79 |
| 30 s | Bus | 13108 | 10.21 | 99.87% | 67.86% | 285.48 |
| 30 s | Car | 13095 | 9.87 | 99.71% | 62.76% | 275.89 |
| 30 s | Train | 11571 | 10.11 | 99.69% | 55.45% | 283.02 |
| 60 s | Walk | 12749 | 5.39 | 99.99% | 53.34% | 267.35 |
| 60 s | Bike | 12983 | 5.58 | 100.00% | 52.80% | 277.86 |
| 60 s | Bus | 13074 | 5.53 | 99.99% | 53.72% | 275.14 |
| 60 s | Car | 13058 | 5.22 | 99.96% | 53.18% | 256.62 |
| 60 s | Train | 10922 | 5.49 | 99.91% | 51.67% | 273.08 |

## Hypothesis decision

**GO**

- 30/60 s mean mask ratio exceeds 5/10 s by at least 5 percentage points
- 60 s mean TS masked duration covers at least half of the 300 s window
- Bus or Train TS mask ratio increases by at least 5 percentage points at 30/60 s

## Diagnostic limitations

- Saved masks do not preserve KDE peak counts, duplicate mappings, EP counts, or fallback events.
- Those events require optional instrumentation during future feature regeneration.
- This diagnosis uses the training split only; the test split was not inspected for method design.

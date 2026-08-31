# Representative-method comparison under the shared five-rate protocol

Protocol SHA-256: `04cd4e719c34a035c79e4eabeca9d28301908e3f32dbab30ef269c268048a1ac`

All values below are newly evaluated on the same user-disjoint test sets. They are not copied from the cited papers. Each cell is Accuracy/Macro-F1 (%).

| Method | 5 s | 10 s | 20 s | 30 s | 60 s | Mean Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Takahashi GPS-only RF (2026) | 78.81/74.38 | 77.60/72.61 | 75.26/70.29 | 74.37/69.43 | 68.93/64.42 | 70.23 |
| XGBoost | 79.64/75.35 | 77.86/73.52 | 75.97/71.70 | 75.79/70.94 | 69.60/64.88 | 71.28 |
| Dabiri CNN (2018) | 78.18/73.72 | 76.93/72.75 | 76.46/71.14 | 72.04/66.50 | 67.89/62.51 | 69.32 |
| DeepInsight-ViT (2025, adapted) | 75.26/71.10 | 76.30/71.26 | 74.37/69.47 | 68.81/63.98 | 69.36/64.30 | 68.02 |
| MASO-MSF (2023, adapted) | 76.78/72.08 | 77.39/73.56 | 74.62/70.17 | 74.27/67.90 | 68.71/63.20 | 69.38 |
| B0 dual-branch Transformer | 78.61/74.36 | 74.03/68.92 | 73.16/66.56 | 72.89/67.27 | 71.55/63.73 | 68.17 |
| SORF-TMI (final, pruned) | 81.79/77.54 | 79.84/74.72 | 77.94/71.94 | 75.91/70.30 | 72.69/66.72 | 72.25 |

## Interpretation boundary

The MASO-MSF and DeepInsight-ViT rows are controlled architecture/input adaptations because their original data products and random splits are incompatible with the shared protocol. Takahashi is GPS-only: GIS/POI features are excluded. These labels must be retained in the paper. The SORF-TMI row uses the final pruned model with observation-drop augmentation and sparse-view consistency disabled; external-method rows are unchanged.

# Representative-method fair comparison protocol

All methods in this benchmark use the existing user-disjoint GeoLife protocol.
The train, validation and test user sets contain 44, 6 and 12 users. Five
matched views (5, 10, 20, 30 and 60 seconds) are evaluated independently.

## Non-negotiable controls

- Every method loads data only through `scripts/fair_comparison_protocol.py`.
- The loader asserts that user sets are disjoint and identical across rates.
- A label/user SHA-256 fingerprint fixes sample membership and ordering.
- Deterministic feature transforms are allowed; learned preprocessing is fitted
  on the training set only.
- Test probabilities must have shape `(N_test, 5)` in saved sample order.
- Accuracy and Macro-F1 are calculated only by `evaluate_probabilities`.
- Hyperparameters and checkpoints are selected using validation Macro-F1; the
  test set is not used for model selection.

## Implemented methods

- `takahashi_rf`: GPS-only sparse-trajectory RF; no GIS/POI information.
- `xgboost`: gradient-boosted trees on the same training-only statistics.
- `dabiri_cnn`: controlled 1-D CNN reimplementation using speed,
  acceleration, jerk and bearing-change rate.
- `deepinsight_vit`: controlled DeepInsight-style feature-image plus ViT
  reimplementation. Feature placement and normalization use training data only.
- `maso_msf`: official MSF channel/scale-fusion architecture adapted to the
  common motion tensor. The original random split and incompatible MASO image
  dataset are not reused.

These are protocol-controlled adaptations, not claims that the authors'
published numbers were exactly reproduced. Published numbers use different
splits and must not be copied into the fair-comparison result table.

## Commands

Validate data and build deterministic caches:

```bash
python scripts/run_fair_representative_comparison.py --validate_only
```

Run all methods and rates, resuming completed cells:

```bash
python scripts/run_fair_representative_comparison.py --resume
```

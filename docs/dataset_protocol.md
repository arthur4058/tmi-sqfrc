# GeoLife data protocol

## Scope

This document records the GeoLife artifacts used to reproduce the
upstream baseline before implementing the user-disjoint variable-sampling
paper protocol.

Protocol identifier:

```text
geolife-upstream-random-v1
```

This protocol reproduces the upstream repository behavior. It is not the
final variable-sampling paper protocol.

## Source data

Dataset:

```text
Microsoft Research GeoLife GPS Trajectories 1.3
```

Archive:

```text
Geolife Trajectories 1.3.zip
```

Archive SHA-256:

```text
1107c5ac064d0a23c8d021a8736a77e53abc75b227062e6260342c6a8d86bdb6
```

Raw dataset statistics:

- User directories: 182
- Users with `labels.txt`: 69
- Trajectory `.plt` files: 18,670

Raw and generated datasets are stored under `data/` and are not committed
to Git.

## Transportation-mode mapping

The upstream extraction script uses the following five output classes:

| Class ID | Transportation mode |
|---:|---|
| 0 | walk |
| 1 | bike |
| 2 | bus |
| 3 | car or taxi |
| 4 | subway or train |

Other GeoLife transportation modes are excluded.

## Preprocessing pipeline

### S1: trajectory extraction

Script:

```text
tmi/data_preprocess/s1_trajectory_extraction_geolife.py
```

Output directory:

```text
data/geolife_extracted/
```

Output statistics:

- Trajectories: 14,686
- Labels: 14,686
- Class distribution: `[6460, 2089, 2853, 2172, 1112]`
- Empty trajectories: 5,097

The upstream script shuffles extracted trajectories using random seed
`10086`.

### S2: train/test split

Script:

```text
tmi/data_preprocess/s2_dataset_split.py
```

Configuration:

- Test ratio: 0.2
- Split type: stratified trajectory-level random split
- Random state: 42

Training split:

- Samples: 11,748
- Class distribution: `[5168, 1671, 2282, 1737, 890]`
- Empty trajectories: 4,043

Test split:

- Samples: 2,938
- Class distribution: `[1292, 418, 571, 435, 222]`
- Empty trajectories: 1,054

### S3: training-data augmentation

Script:

```text
tmi/data_preprocess/s3_data_augmentation.py
```

Configuration:

- Random seed: 42
- Only the training split is augmented
- The test split is copied without augmentation

Output statistics:

- Augmented training samples: 25,840
- Training class distribution: `[5168, 5168, 5168, 5168, 5168]`
- Test samples: 2,938

The upstream augmentation-count bug was fixed so the number of selected
augmentation methods cannot exceed the number of available methods.

Fix commit:

```text
e68051d
```

### S4: trajectory segmentation and feature calculation

Script:

```text
tmi/data_preprocess/s4_trajectory_feature_calculation_with_CPD.py
```

Configuration:

```text
n_class=5
kde_bw=1
kde_kernel=epa
mask_mode=ep
trj_mask_mode=kde
mask_ratio=0.3
```

Training feature output:

- Segments: 43,546
- Class distribution: `[7273, 9761, 10426, 8184, 7902]`
- `clean_trj_segs.npy`: `(43546, 2)`
- `noise_trj_segs.npy`: `(43546, 2)`
- `trj_seg_masks.npy`: `(43546,)`
- `clean_multi_feature_segs.npy`: `(43546, 10)`
- `noise_multi_feature_segs.npy`: `(43546, 10)`
- `fs_seg_masks.npy`: `(43546, 9)`

Test feature output:

- Segments: 5,321
- Class distribution: `[1856, 908, 1241, 836, 480]`
- `clean_trj_segs.npy`: `(5321, 2)`
- `noise_trj_segs.npy`: `(5321, 2)`
- `trj_seg_masks.npy`: `(5321,)`
- `clean_multi_feature_segs.npy`: `(5321, 10)`
- `noise_multi_feature_segs.npy`: `(5321, 10)`
- `fs_seg_masks.npy`: `(5321, 9)`

Feature consistency checks passed for both splits.

## Smoke-training validation

A balanced subset of the S4 outputs was used to validate that the complete
training pipeline can start, train, save checkpoints and evaluate.

Smoke dataset:

- Training pool: 320 segments
- Actual training samples: 256
- Validation samples: 64
- Independent test samples: 160
- Five balanced classes

Validation configuration:

- Epochs: 2
- Batch size: 16
- GPU: NVIDIA GeForce RTX 5070
- PyTorch: 2.7.1+cu128
- CUDA capability: 12.0

Results:

- Training completed successfully
- Best and last checkpoints were saved
- Independent smoke-test accuracy: `99 / 160 = 0.61875`

This accuracy is a pipeline diagnostic result, not a reported baseline
performance result.

Training-pipeline compatibility fix commit:

```text
dbe3a80
```

## Reproducibility records

Preprocessing code commit:

```text
af66560016e5a2c62ff51ab8ecd5624f1024fa06
```

Validation code commit:

```text
8e9cd41f98816862c5b9724721310e80f1d021c9
```

Artifact checksums:

```text
reports/manifests/v0.2.0_geolife_sha256.txt
```

Artifact statistics:

```text
reports/manifests/v0.2.0_geolife_stats.json
```

## Known limitations

- The upstream S2 split is trajectory-level random splitting, not
  user-disjoint splitting.
- The S1 artifacts do not preserve user IDs, so cross-user leakage cannot
  be checked from the generated arrays.
- Empty trajectories are retained by the upstream S1 and S2 behavior.
- S4 does not expose a complete fixed random-seed interface. The checksum
  manifest records the exact generated artifact snapshot used here.
- Augmentation may occasionally return the original trajectory when a
  transformed coordinate is invalid.
- The smoke-test subset and its accuracy must not be treated as the final
  baseline experiment.
- The variable-sampling paper experiments must use a new user-disjoint
  protocol created before sampling-rate views are generated.

#!/usr/bin/env python3
"""Create an isolated feature tree with TS-SABM trajectory masks."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Support both ``python -m scripts.generate_sabm_masks`` and direct execution.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tmi.data_preprocess.utils import generate_mask_for_trj_using_KDE_SABM


def _link_non_mask_arrays(source_dir, target_dir):
    for source in sorted(source_dir.glob('*.npy')):
        if source.name == 'trj_seg_masks.npy':
            continue
        target = target_dir / source.name
        if target.exists() or target.is_symlink():
            continue
        relative_source = os.path.relpath(source, target.parent)
        target.symlink_to(relative_source)


def _load_rows(split_dir):
    trajectories = np.load(
        split_dir / 'clean_trj_segs.npy', allow_pickle=True)
    features = np.load(
        split_dir / 'clean_multi_feature_segs.npy', allow_pickle=True)
    if trajectories.ndim != 2 or trajectories.shape[1] != 2:
        raise ValueError(
            f'unexpected trajectory layout: {trajectories.shape}')
    if features.ndim != 2 or features.shape[1] < 1:
        raise ValueError(f'unexpected feature layout: {features.shape}')
    if len(trajectories) != len(features):
        raise ValueError('trajectory and feature sample counts differ')
    return trajectories, features


def generate_split(
        source_dir,
        target_dir,
        target_mask_ratio,
        mask_duration_seconds,
        kde_bw,
        kde_kernel):
    target_dir.mkdir(parents=True, exist_ok=False)
    _link_non_mask_arrays(source_dir, target_dir)
    trajectories, features = _load_rows(source_dir)

    masks = []
    metadata_rows = []
    for sample_index in range(len(trajectories)):
        trajectory = np.column_stack((
            np.asarray(trajectories[sample_index, 0], dtype=float),
            np.asarray(trajectories[sample_index, 1], dtype=float),
        ))
        delta_times = np.asarray(features[sample_index, 0], dtype=float)
        mask, metadata = generate_mask_for_trj_using_KDE_SABM(
            trajectory,
            delta_times,
            target_mask_ratio=target_mask_ratio,
            mask_duration_seconds=mask_duration_seconds,
            bw=kde_bw,
            kernel=kde_kernel,
            return_metadata=True,
        )
        masks.append(mask)
        metadata_rows.append(metadata)

    mask_array = np.empty(len(masks), dtype=object)
    mask_array[:] = masks
    np.save(target_dir / 'trj_seg_masks.npy', mask_array, allow_pickle=True)

    ratios = np.asarray([
        row['actual_mask_ratio'] for row in metadata_rows], dtype=float)
    block_lengths = np.asarray([
        row['block_length_points'] for row in metadata_rows], dtype=float)
    effective_dts = np.asarray([
        row['effective_dt'] for row in metadata_rows], dtype=float)
    fully_masked = np.asarray([
        bool(np.all(mask == 0)) for mask in masks], dtype=float)
    return {
        'samples': int(len(masks)),
        'average_mask_ratio': float(np.mean(ratios)),
        'median_mask_ratio': float(np.median(ratios)),
        'average_block_length_points': float(np.mean(block_lengths)),
        'median_effective_dt_seconds': float(np.median(effective_dts)),
        'fully_masked_sample_ratio': float(np.mean(fully_masked)),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-data-name', required=True)
    parser.add_argument('--target-data-name', required=True)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--target-mask-ratio', type=float, default=0.30)
    parser.add_argument('--mask-duration-seconds', type=float, default=30.0)
    parser.add_argument('--kde-bw', type=float, default=1.0)
    parser.add_argument('--kde-kernel', default='epa')
    parser.add_argument('--report', type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    source_root = (
        args.data_root / f'{args.source_data_name}_features')
    target_root = (
        args.data_root / f'{args.target_data_name}_features')
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if target_root.exists():
        raise FileExistsError(
            f'{target_root} already exists; refusing to overwrite data')

    report = {
        'protocol': 'ts-sabm-v1',
        'source_data_name': args.source_data_name,
        'target_data_name': args.target_data_name,
        'parameters': {
            'target_mask_ratio': args.target_mask_ratio,
            'mask_duration_seconds': args.mask_duration_seconds,
            'kde_bw': args.kde_bw,
            'kde_kernel': args.kde_kernel,
        },
        'splits': {},
    }
    try:
        for split in args.splits:
            source_dir = source_root / split
            if not source_dir.is_dir():
                raise FileNotFoundError(source_dir)
            report['splits'][split] = generate_split(
                source_dir,
                target_root / split,
                args.target_mask_ratio,
                args.mask_duration_seconds,
                args.kde_bw,
                args.kde_kernel,
            )
    except Exception:
        # Keep any partial directory for inspection.  A rerun remains explicit
        # because silently overwriting scientific artifacts is unsafe.
        raise

    report_path = args.report or (
        target_root / 'ts_sabm_manifest.json')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

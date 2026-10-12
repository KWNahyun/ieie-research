"""Verified base-only data access; geometry and EMF labels stay separate.

Target exposes encoded B/G channels, not unverified occupancy. No transforms,
resizing, label-dependent inputs, or full-array RAM copies are performed.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


RESEARCH = Path(__file__).resolve().parent
TARGET_ROOT = Path('/home/ailab/Desktop/data/preprocessing')
TARGET_FILES = {
    'building': 'x_image.npy',
    'numeric': 'x_numeric.npy',
    'horizontal': 'x_pattern_hori.npy',
    'vertical': 'x_pattern_vert.npy',
    'target': 'y.npy',
}


def read_rows(path: Path) -> list[dict]:
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def tx_map(x: float, y: float, height: int, width: int) -> np.ndarray:
    """Floor interior raster coordinates; reject out-of-range, including endpoints."""
    if not (np.isfinite(x) and np.isfinite(y) and 0 <= x < width and 0 <= y < height):
        raise ValueError(f'Tx outside raster: ({x}, {y}) for {height}x{width}')
    result = np.zeros((1, height, width), dtype=np.float32)
    result[0, int(np.floor(y)), int(np.floor(x))] = 1
    return result


def condition_vector(numeric: np.ndarray) -> np.ndarray:
    """Preserve five scalars; represent azimuth by sine/cosine, before train scaling."""
    numeric = np.asarray(numeric, dtype=np.float64)
    angle = np.deg2rad(numeric[..., 5])
    return np.concatenate((numeric[..., :5], np.sin(angle)[..., None],
                           np.cos(angle)[..., None]), axis=-1)


def image_geometry_proxy(building_encoded: np.ndarray) -> np.ndarray:
    """Operational black-pixel geometry proxy; no label-dependent thresholds.

    Both encoded building channels must be exactly zero. This reproduces an
    image convention, not a verified simulator footprint or physical height.
    """
    building_encoded = np.asarray(building_encoded)
    if building_encoded.ndim != 3 or building_encoded.shape[-1] != 2:
        raise ValueError('Expected [H,W,2] encoded building image')
    return np.all(building_encoded == 0, axis=-1)[None].astype(np.float32)


class TargetDataset(Dataset):
    def __init__(self, split: str, *, manifest: Path | str = RESEARCH / 'target_manifest.csv',
                 data_root: Path | str = TARGET_ROOT, subset: dict | None = None,
                 include_target: bool = True, normalization: Path | str | None = None,
                 evaluation_mode: str = 'whole_raster', include_building_encoded: bool = True):
        if split not in {'train', 'validation', 'test'}:
            raise ValueError('Unknown split')
        if evaluation_mode not in {'whole_raster', 'legacy_label_positive'}:
            raise ValueError('Independent outdoor mask is unresolved; choose an explicit evaluation mode')
        if not include_target and evaluation_mode != 'whole_raster':
            raise ValueError('Label-positive evaluation requires labels')
        self.split = split
        self.data_root = Path(data_root)
        self.include_target = include_target
        self.evaluation_mode = evaluation_mode
        self.include_building_encoded = include_building_encoded
        self._proxy_cache = {}
        self.rows = [r for r in read_rows(Path(manifest))
                     if r['split'] == split and r['transform'] == 'identity']
        if split == 'train' and any(r['use_for_training'] != 'True' for r in self.rows):
            raise ValueError('Manifest disallows a training sample')
        if any(r['base_id'] != r['entry_index'] for r in self.rows):
            raise ValueError('Identity sample must reference the original entry')
        if subset is not None:
            if split != 'train':
                raise ValueError('Budget subsets apply only to train')
            requested = set(map(int, subset['base_ids']))
            available = {int(r['base_id']) for r in self.rows}
            if not requested or not requested <= available:
                raise ValueError('Empty subset or IDs outside train')
            if len(requested) != len(subset['base_ids']):
                raise ValueError('Duplicate subset IDs')
            chosen_groups = {int(r['geometry_group_id']) for r in self.rows
                             if int(r['base_id']) in requested}
            if set(map(int, subset['group_ids'])) != chosen_groups:
                raise ValueError('Subset group IDs do not match samples')
            group_base_ids = {int(r['base_id']) for r in self.rows
                              if int(r['geometry_group_id']) in chosen_groups}
            if requested != group_base_ids:
                raise ValueError('Group budget must include each selected group in full')
            self.rows = [r for r in self.rows if int(r['base_id']) in requested]
        if not self.rows:
            raise ValueError('No samples selected')
        self._arrays = {}
        self.normalization = None
        if normalization is not None:
            self.normalization = json.loads(Path(normalization).read_text())
            # The training IDs are checked against the current manifest even for evaluation.
            train_ids = {int(r['base_id']) for r in read_rows(Path(manifest))
                         if r['split'] == 'train' and r['transform'] == 'identity'}
            fitted_ids = set(self.normalization['fitted_base_ids'])
            if self.normalization['fit_split'] != 'train' or not fitted_ids <= train_ids:
                raise ValueError('Normalization was not fitted solely on training samples')
            if split == 'train' and fitted_ids != {int(r['base_id']) for r in self.rows}:
                raise ValueError('Normalization belongs to a different training budget')

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_arrays'] = {}  # Spawn workers reopen read-only memmaps instead of serializing them.
        state['_proxy_cache'] = {}
        return state

    def array(self, key: str):
        if key == 'target' and not self.include_target:
            raise ValueError('Labels disabled for this dataset')
        if key not in TARGET_FILES:
            raise ValueError('Unknown modality; masks and beam are excluded from inputs')
        if key not in self._arrays:
            self._arrays[key] = np.load(self.data_root / TARGET_FILES[key], mmap_mode='r')
        return self._arrays[key]

    def __getitem__(self, item):
        row = self.rows[item]
        index = int(row['base_id'])
        building = self.array('building')[index]
        height, width, channels = building.shape
        if channels != 2:
            raise ValueError('Expected encoded B/G building channels')
        numeric_raw = np.asarray(self.array('numeric')[index], dtype=np.float64)
        numeric = np.array(numeric_raw, dtype=np.float32, copy=True)
        pattern = np.stack((self.array('horizontal')[index], self.array('vertical')[index])).astype(np.float32)
        condition = condition_vector(numeric_raw).astype(np.float32)
        if self.normalization:
            stats = self.normalization
            condition = ((condition - np.asarray(stats['condition_mean'], np.float32)) /
                         np.asarray(stats['condition_std'], np.float32))
            pattern = ((pattern - np.asarray(stats['pattern_mean'], np.float32)[:, None]) /
                       np.asarray(stats['pattern_std'], np.float32)[:, None])
        sample = {
            'inputs': {
                'tx': torch.from_numpy(tx_map(float(numeric_raw[0]), float(numeric_raw[1]), height, width)),
                'numeric': torch.from_numpy(numeric),
                'condition': torch.from_numpy(np.ascontiguousarray(condition)),
                'pattern': torch.from_numpy(np.ascontiguousarray(pattern)),
            },
            'metadata': {'base_id': index, 'group_id': int(row['geometry_group_id']),
                         'building_id': row['building_id'], 'split': self.split},
        }
        if row['building_id'] not in self._proxy_cache:
            self._proxy_cache[row['building_id']] = image_geometry_proxy(building)
        sample['inputs']['occupancy_proxy'] = torch.from_numpy(self._proxy_cache[row['building_id']].copy())
        if self.include_building_encoded:
            sample['inputs']['building_encoded'] = torch.from_numpy(np.array(building.transpose(2, 0, 1), dtype=np.float32, copy=True))
        if self.include_target:
            target = np.array(self.array('target')[index][None], dtype=np.float32, copy=True)
            sample['target'] = torch.from_numpy(target)
            # Never passed to inputs. Mode name makes label-dependent evaluation explicit.
            mask = np.ones_like(target, dtype=np.bool_) if self.evaluation_mode == 'whole_raster' else target > 0
            sample['evaluation'] = {'mask': torch.from_numpy(mask), 'mode': self.evaluation_mode}
        return sample


class SourceDataset(Dataset):
    """Read candidate binary occupancy and verified Tx maps from DPM triples."""
    def __init__(self, split: str, *, manifest: Path | str = RESEARCH / 'source_manifest.csv'):
        if split not in {'train', 'validation', 'test'}:
            raise ValueError('Unknown split')
        self.rows = [r for r in read_rows(Path(manifest)) if r['split'] == split]
        if not self.rows:
            raise ValueError('No source samples')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        row = self.rows[item]
        building = cv2.imread(row['building_path'], cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(row['gain_path'], cv2.IMREAD_GRAYSCALE)
        if building is None or label is None or building.shape != label.shape:
            raise ValueError('Missing or misaligned source files')
        if not np.isin(building, [0, 255]).all():
            raise ValueError('Source building must be binary')
        height, width = building.shape
        return {
            'inputs': {
                'occupancy_proxy': torch.from_numpy((building > 0)[None].astype(np.float32)),
                'tx': torch.from_numpy(tx_map(float(row['tx_column']), float(row['tx_row']), height, width)),
            },
            'target': torch.from_numpy(label[None].astype(np.float32) / 255),
            'metadata': {'map_id': int(row['map_id']), 'tx_id': int(row['tx_id']), 'split': row['split']},
        }


def common_spatial_input(inputs: dict, *, geometry_mode: str | None = None) -> torch.Tensor:
    """Require an explicit geometry convention; never silently claim verification."""
    if geometry_mode == 'image_proxy':
        key = 'occupancy_proxy'
    elif geometry_mode == 'verified':
        key = 'occupancy'
    else:
        raise ValueError('Choose image_proxy or verified geometry explicitly')
    if key not in inputs:
        raise ValueError(f'Missing {key}; encoded images and labels cannot substitute implicitly')
    return torch.cat((inputs[key], inputs['tx']), dim=-3)


def fit_normalization(dataset: TargetDataset) -> dict:
    """Fit only condition/pattern statistics, using the exact selected training budget."""
    if dataset.split != 'train' or dataset.normalization is not None:
        raise ValueError('Fit normalization on raw training subset only')
    indices = [int(r['base_id']) for r in dataset.rows]
    numeric = np.asarray(dataset.array('numeric')[indices], dtype=np.float64)
    conditions = condition_vector(numeric)
    pattern = np.stack((dataset.array('horizontal')[indices], dataset.array('vertical')[indices]), axis=1)
    condition_std = conditions.std(axis=0)
    pattern_std = pattern.std(axis=(0, 2))
    return {
        'fit_split': 'train', 'fitted_base_ids': indices,
        'condition_order': ['x', 'y', 'height', 'downtilt', 'power', 'sin_azimuth', 'cos_azimuth'],
        'condition_mean': conditions.mean(axis=0).tolist(),
        'condition_std': np.where(condition_std > 1e-8, condition_std, 1).tolist(),
        'pattern_mean': pattern.mean(axis=(0, 2)).tolist(),
        'pattern_std': np.where(pattern_std > 1e-8, pattern_std, 1).tolist(),
        'label_scaling': 'fixed PNG/255; no label-fitted statistics',
        'building_scaling': 'fixed PNG/255; encoded channels, not physical height',
    }

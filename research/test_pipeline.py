"""Checks for leakage, geometry conventions, checkpoint pairing and native resolution."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from baseline_model import RadioUNetBaseline
from data_pipeline import (RESEARCH, TargetDataset, SourceDataset, condition_vector,
                           common_spatial_input, fit_normalization, image_geometry_proxy, tx_map)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_tx_axes_floor_and_boundaries(self):
        tx = tx_map(3.9, 1.2, 5, 7)
        self.assertEqual(np.argwhere(tx[0] > 0).tolist(), [[1, 3]])
        self.assertEqual(float(tx.sum()), 1)
        for x, y in [(7, 1), (1, 5), (-1, 1), (np.nan, 0)]:
            with self.assertRaises(ValueError):
                tx_map(x, y, 5, 7)

    def test_proxy_rule_and_cyclic_azimuth(self):
        building = np.array([[[0, 0], [0, 1]], [[1, 0], [1, 1]]], dtype=np.float64)
        np.testing.assert_array_equal(image_geometry_proxy(building), [[[1, 0], [0, 0]]])
        numeric = np.array([[1, 2, 3, 4, 5, 0], [1, 2, 3, 4, 5, 360]])
        vectors = condition_vector(numeric)
        np.testing.assert_allclose(vectors[0], vectors[1], atol=1e-7)

    def test_target_no_label_dependent_inputs(self):
        unlabelled = TargetDataset('train', include_target=False)
        labelled = TargetDataset('train')
        a, b = unlabelled[0], labelled[0]
        self.assertNotIn('target', a)
        self.assertNotIn('target', unlabelled._arrays)
        self.assertNotIn('mask', a['inputs'])
        for key in a['inputs']:
            torch.testing.assert_close(a['inputs'][key], b['inputs'][key], rtol=0, atol=0)
        self.assertTrue(all(isinstance(v, np.memmap) and v.mode == 'r' for v in unlabelled._arrays.values()))
        self.assertEqual(tuple(b['target'].shape), (1, 500, 500))
        self.assertTrue(b['evaluation']['mask'].all())
        self.assertEqual(b['inputs']['building_encoded'].dtype, torch.float32)
        numeric = unlabelled.array('numeric')[a['metadata']['base_id']]
        self.assertEqual(torch.nonzero(a['inputs']['tx'][0]).tolist(), [[int(numeric[1]), int(numeric[0])]])
        for dataset in [unlabelled, TargetDataset('validation'), TargetDataset('test')]:
            self.assertTrue(all(r['transform'] == 'identity' for r in dataset.rows))
        # Only manifest IDs are inspected for test; no test sample/image/label is opened.
        with self.assertRaises(ValueError):
            common_spatial_input(a['inputs'])
        with self.assertRaises(ValueError):
            common_spatial_input(a['inputs'], geometry_mode='verified')
        self.assertEqual(tuple(common_spatial_input(a['inputs'], geometry_mode='image_proxy').shape), (2, 500, 500))

    def test_training_statistics_and_subset_isolation(self):
        subsets = json.loads((RESEARCH / 'splits/target_subsets.json').read_text())['seeds']['0']
        train = TargetDataset('train', subset=subsets['25'])
        stats = fit_normalization(train)
        indices = stats['fitted_base_ids']
        expected = condition_vector(train.array('numeric')[indices]).mean(axis=0)
        np.testing.assert_allclose(stats['condition_mean'], expected)
        with self.assertRaises(ValueError):
            fit_normalization(TargetDataset('validation'))
        bad_subset = {'base_ids': [int(TargetDataset('validation').rows[0]['base_id'])], 'group_ids': []}
        with self.assertRaises(ValueError):
            TargetDataset('train', subset=bad_subset)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'stats.json'
            path.write_text(json.dumps(stats))
            normalized = TargetDataset('train', subset=subsets['25'], normalization=path)
            batch = next(iter(DataLoader(normalized, batch_size=2, num_workers=0)))
            self.assertEqual(tuple(batch['inputs']['pattern'].shape), (2, 2, 360))
            self.assertEqual(tuple(batch['inputs']['condition'].shape), (2, 7))
            with self.assertRaises(ValueError):
                TargetDataset('train', subset=subsets['10'], normalization=path)
            stats['fitted_base_ids'].append(int(TargetDataset('validation').rows[0]['base_id']))
            path.write_text(json.dumps(stats))
            with self.assertRaises(ValueError):
                TargetDataset('validation', normalization=path)

    def test_source_coordinates_and_pair(self):
        source = SourceDataset('train')[0]
        self.assertEqual(tuple(source['target'].shape), (1, 256, 256))
        self.assertEqual(float(source['inputs']['tx'].sum()), 1)
        self.assertTrue(torch.isfinite(source['target']).all())

    def test_model_native_shapes_paired_initialization_and_gradients(self):
        torch.manual_seed(0)
        a = RadioUNetBaseline(base_channels=4)
        torch.manual_seed(0)
        b = RadioUNetBaseline(conditioned=True, base_channels=4)
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)
        spatial = torch.zeros(1, 2, 500, 500)
        scalar, pattern = torch.randn(1, 7), torch.randn(1, 2, 360)
        pred = b(spatial, scalar, pattern)
        self.assertEqual(tuple(pred.shape), (1, 1, 500, 500))
        pred.mean().backward()
        self.assertGreater(float(b.conditioning.film.weight.grad.abs().sum()), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in b.parameters() if p.grad is not None))
        with torch.no_grad():
            self.assertEqual(tuple(a(torch.zeros(1, 2, 256, 256)).shape), (1, 1, 256, 256))


if __name__ == '__main__':
    unittest.main(verbosity=2)

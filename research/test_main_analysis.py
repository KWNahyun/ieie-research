"""Scientific metric checks on synthetic inputs only; no source/target datasets."""
import unittest
import numpy as np
import torch
from unittest.mock import patch
from evaluate_main import evaluate_samples
from analyze_main import paired_bootstrap


class MainMetricTests(unittest.TestCase):
    def test_equal_group_weight_despite_unequal_sample_counts(self):
        batch = {'target': torch.tensor([1., 3., 3., 3.]).reshape(4, 1, 1, 1),
                 'metadata': {'base_id': torch.arange(4), 'group_id': torch.tensor([10, 20, 20, 20])}}
        with patch('evaluate_main.forward', return_value=torch.zeros(4, 1, 1, 1)):
            samples, groups = evaluate_samples(torch.nn.Identity(), [batch], 'cpu')
        self.assertEqual(len(samples), 4)
        self.assertEqual([g['sample_count'] for g in groups], [1, 3])
        self.assertEqual(sum(g['mae'] for g in groups) / len(groups), 2.)
        self.assertEqual(sum(s['mae'] for s in samples) / len(samples), 2.5)

    def test_bootstrap_preserves_pairing_and_sign(self):
        b = np.linspace(.01, .9, 14)
        d = b + .005
        interval = paired_bootstrap(b, d, np.random.default_rng(20261010))
        np.testing.assert_allclose(interval, [-.005, -.005], atol=1e-12)
        np.testing.assert_array_equal(paired_bootstrap(b, b, np.random.default_rng(3)), [0., 0.])


if __name__ == '__main__':
    unittest.main()

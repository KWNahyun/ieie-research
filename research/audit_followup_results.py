"""Independent follow-up audit of topology, configs and saved test metrics."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
import torch
from data_pipeline import RESEARCH
from run_convergence import digest
from run_required_followup import verify
from train_baseline import write_json


def main():
    root = RESEARCH / 'runs/followup_v1'
    status = json.loads((root / 'status.json').read_text())
    assert status['status'] == 'experiments_complete'
    protocol = json.loads((root / 'protocol.json').read_text())
    for name, value in protocol['input_sha256'].items():
        assert digest(RESEARCH / name) == digest(root / 'inputs_snapshot' / name) == value
    manifest_path = root / 'frozen_test_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest_hash = digest(manifest_path)
    assert manifest['protocol_sha256'] == digest(root / 'protocol.json')
    assert manifest_path.stat().st_mtime <= (root / 'test_access.json').stat().st_mtime
    records = manifest['runs']
    assert len(records) == len({r['id'] for r in records}) == 152
    target = [r for r in csv.DictReader((RESEARCH / 'target_manifest.csv').open()) if r['transform'] == 'identity']
    test = {int(r['base_id']): int(r['geometry_group_id']) for r in target if r['split'] == 'test'}
    train_ids = {int(r['base_id']) for r in target if r['split'] == 'train'}
    valid_ids = {int(r['base_id']) for r in target if r['split'] == 'validation'}
    group_subsets = json.loads((RESEARCH / 'splits/target_subsets.json').read_text())['seeds']
    setting_subsets = json.loads((RESEARCH / 'splits/setting_subsets_v1.json').read_text())['seeds']
    torch.set_num_threads(2)
    count = 0
    for r in records:
        verified = verify(r)
        assert verified['checkpoint_sha256'] == r['checkpoint_sha256']
        assert verified['config_sha256'] == r['config_sha256'] and verified['normalization_sha256'] == r['normalization_sha256']
        subset = setting_subsets[str(r['seed'])][str(r['settings'])] if r['task'] == 'settings' else group_subsets[str(r['seed'])][str(r['budget'])]
        assert set(r['train_ids']) == set(subset['base_ids']) <= train_ids
        assert set(r['validation_ids']) == valid_ids
        assert not (set(r['train_ids']) | set(r['validation_ids'])) & set(test)
        folder = Path(r['folder'])
        stats = json.loads((folder / 'normalization.json').read_text())
        assert stats['fit_split'] == 'train' and set(stats['fitted_base_ids']) == set(r['train_ids'])
        for key in ['condition_mean', 'condition_std', 'pattern_mean', 'pattern_std']: assert np.isfinite(stats[key]).all()
        metric = json.loads((root / 'test_metrics' / f"{r['id']}.json").read_text())
        assert metric['manifest_sha256'] == manifest_hash and metric['checkpoint_sha256'] == r['checkpoint_sha256']
        assert len(metric['samples']) == 126 and len(metric['groups']) == 14
        assert {s['base_id'] for s in metric['samples']} == set(test)
        by_group = defaultdict(list)
        for sample in metric['samples']:
            assert sample['group_id'] == test[sample['base_id']]
            assert np.isfinite([sample[k] for k in ['mae', 'rmse', 'ssim']]).all()
            by_group[sample['group_id']].append(sample)
        for group in metric['groups']:
            samples = by_group[group['group_id']]
            assert len(samples) == group['sample_count']
            for key in ['mae', 'rmse', 'ssim']:
                assert abs(np.mean([s[key] for s in samples]) - group[key]) < 1e-12
        for key in ['mae', 'rmse', 'ssim']:
            assert abs(np.mean([g[key] for g in metric['groups']]) - metric[f'macro_{key}']) < 1e-12
        count += 1
    assert sum(r['task'] == 'scalar' for r in records) == 80
    assert sum(r['task'] == 'settings' for r in records) == 60
    assert sum(r['task'] == 'reference' for r in records) == 12
    write_json(root / 'final_audit.json', {'status': 'PASS', 'cells_verified': count, 'sample_metrics_verified': count * 126,
               'raw_dataset_arrays_opened': False, 'scope': 'supporting follow-up; original test already known',
               'checks': 'immutable snapshots/current code; exact topology/checkpoint/config/normalization; finite strict inference; fixed subsets/train-only stats; validation-selected checkpoint; frozen evaluation manifest; sample/group identities; recalculated MAE/RMSE/SSIM'})
    print('Follow-up independent audit PASS:', count, 'cells;', count * 126, 'sample metrics')


if __name__ == '__main__':
    main()

"""Independent artifact audit; never loads source or target dataset arrays."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
import torch
from data_pipeline import RESEARCH
from run_convergence import digest
from analyze_source_sensitivity import verify_checkpoint
from train_baseline import write_json


def close(a, b):
    assert np.isfinite(a) and np.isfinite(b) and abs(a - b) < 1e-12, (a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['main_v1', 'spatial_ablation_v1'], required=True)
    args = parser.parse_args()
    root = RESEARCH / 'runs' / args.stage
    status = json.loads((root / 'status.json').read_text())
    assert status['status'] == 'complete' and status['test_accessed']
    protocol = json.loads((root / 'protocol.json').read_text())
    for name, expected in protocol['input_sha256'].items():
        assert digest(root / 'inputs_snapshot' / name) == expected
        assert digest(RESEARCH / name) == expected, name
    manifest_path = root / 'frozen_test_manifest.json'
    manifest_hash = digest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    assert manifest['protocol_sha256'] == digest(root / 'protocol.json')
    assert manifest_path.stat().st_mtime <= (root / 'test_access.json').stat().st_mtime
    with (RESEARCH / 'target_manifest.csv').open(newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['transform'] == 'identity']
    ids = {split: {int(r['base_id']) for r in rows if r['split'] == split} for split in ['train', 'validation', 'test']}
    groups = {split: {int(r['geometry_group_id']) for r in rows if r['split'] == split} for split in ids}
    test_groups = {int(r['base_id']): int(r['geometry_group_id']) for r in rows if r['split'] == 'test'}
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        assert not ids[a] & ids[b] and not groups[a] & groups[b]
    assert len(ids['test']) == 126 and len(groups['test']) == 14
    subsets = json.loads((RESEARCH / 'splits/target_subsets.json').read_text())['seeds']
    records = manifest['runs']
    expected_count = 80 if args.stage == 'main_v1' else 40
    assert len(records) == expected_count
    keys = {(r['model'], r.get('source_seed'), r['budget'], r['seed']) for r in records}
    if args.stage == 'main_v1':
        expected_keys = {(model, source, budget, seed) for seed in range(5) for budget in [10, 25, 50, 100]
                         for model, source in [('B', None), ('D', 0), ('D', 1), ('D', 2)]}
    else:
        expected_keys = {(model, None, budget, seed) for seed in range(5) for budget in [10, 25, 50, 100] for model in ['A', 'C']}
    assert keys == expected_keys
    metric_files = list((root / 'test_metrics').glob('*.json'))
    assert len(metric_files) == expected_count
    metrics = [json.loads(f.read_text()) for f in metric_files]
    torch.set_num_threads(2)
    verified = []
    for r in records:
        folder = Path(r['folder'])
        for name, key in [('best.pt', 'checkpoint_sha256'), ('config.json', 'config_sha256'), ('normalization.json', 'normalization_sha256')]:
            assert digest(folder / name) == r[key]
        config, state = verify_checkpoint(folder, r['model'] in ['B', 'D'])
        assert config['model'] == r['model'] and config['seed'] == r['seed'] and config['budget'] == r['budget']
        for name in ['data_pipeline.py', 'baseline_model.py', 'train_baseline.py']:
            assert config['code_sha256'][name] == protocol['input_sha256'][name]
        assert not config['test_accessed'] and not config['smoke']
        actual_ids = set(config['actual_train_base_ids'])
        subset = subsets[str(r['seed'])][str(r['budget'])]
        assert actual_ids == set(subset['base_ids']) and len(actual_ids) == config['actual_train_count']
        assert set(config['actual_validation_base_ids']) == ids['validation']
        assert actual_ids <= ids['train'] and not actual_ids & ids['test']
        assert {int(row['geometry_group_id']) for row in rows if int(row['base_id']) in actual_ids} == set(subset['group_ids'])
        normalization = json.loads((folder / 'normalization.json').read_text())
        assert normalization['fit_split'] == 'train' and set(normalization['fitted_base_ids']) == actual_ids
        for key in ['condition_mean', 'condition_std', 'pattern_mean', 'pattern_std']:
            assert np.isfinite(normalization[key]).all()
        assert min(normalization['condition_std']) > 0 and min(normalization['pattern_std']) > 0
        history = json.loads((folder / 'history.json').read_text())
        summary = json.loads((folder / 'summary.json').read_text())
        assert len(history) == summary['epochs'] and not summary['test_accessed']
        selected = min(history, key=lambda h: h['whole_raster_group_macro_mae'])
        assert selected['epoch'] == state['epoch']
        close(selected['whole_raster_group_macro_mae'], summary['best_validation_mae'])
        assert config['epochs'] in [150, 300] and config['patience'] == 25
        if r['model'] in ['C', 'D']:
            assert digest(Path(config['encoder_checkpoint'])) == config['source_checkpoint_sha256']
            source_config = json.loads((Path(config['encoder_checkpoint']).parent / 'config.json').read_text())
            assert source_config['seed'] == (r['source_seed'] if r['model'] == 'D' else 0)
            assert source_config['max_updates'] == 1500 and source_config['domain'] == 'RadioMapSeer_DPM'
        else:
            assert config['encoder_checkpoint'] is None and config['source_checkpoint_sha256'] is None
        match = [m for m in metrics if m['model'] == r['model'] and m['budget'] == r['budget'] and m['target_seed'] == r['seed']
                 and (args.stage != 'main_v1' or m['source_seed'] == r['source_seed'])]
        assert len(match) == 1
        m = match[0]
        assert m['manifest_sha256'] == manifest_hash and m['checkpoint_sha256'] == r['checkpoint_sha256']
        assert len(m['samples']) == 126 and len(m['groups']) == 14
        assert {s['base_id'] for s in m['samples']} == ids['test']
        sample_groups = defaultdict(list)
        for s in m['samples']:
            assert s['group_id'] == test_groups[s['base_id']]
            assert np.isfinite([s['mae'], s['rmse']]).all() and s['mae'] >= 0 and s['rmse'] >= 0
            sample_groups[s['group_id']].append(s)
        assert {g['group_id'] for g in m['groups']} == groups['test']
        for g in m['groups']:
            samples = sample_groups[g['group_id']]
            assert g['sample_count'] == len(samples)
            close(g['mae'], np.mean([s['mae'] for s in samples]))
            close(g['rmse'], np.mean([s['rmse'] for s in samples]))
        close(m['macro_mae'], np.mean([g['mae'] for g in m['groups']]))
        close(m['macro_rmse'], np.mean([g['rmse'] for g in m['groups']]))
        verified.append({'model': r['model'], 'source_seed': r.get('source_seed'), 'budget': r['budget'],
                         'target_seed': r['seed'], 'sample_count': 126, 'group_count': 14})
    analysis = json.loads((root / 'test_analysis.json').read_text())
    assert analysis['test_accessed'] and len(analysis['budgets']) == 4
    if args.stage == 'main_v1':
        for result in analysis['budgets']:
            for model in ['B', 'D']:
                selected = [m['macro_mae'] for m in metrics if m['budget'] == result['budget'] and m['model'] == model]
                close(result[f'{model}_mae'], np.mean(selected))
            close(result['B_minus_D'], result['B_mae'] - result['D_mae'])
    else:
        for result in analysis['budgets']:
            close(result['interaction'], (result['B_mae'] - result['D_mae']) - (result['A_mae'] - result['C_mae']))
    report = {'stage': args.stage, 'status': 'PASS', 'runs_verified': expected_count,
              'test_predictions_sample_metrics_verified': expected_count * 126,
              'checks': 'input/snapshot/code hashes; split isolation; exact frozen checkpoint/config/normalization; strict finite checkpoint inference; train-only normalization and fixed subsets; best validation selection; test sample/group identity; recalculated group/macro errors; analysis aggregates; manifest precedes test access',
              'test_arrays_opened_by_audit': False, 'runs': verified}
    write_json(root / 'final_audit.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'runs'}, indent=2))


if __name__ == '__main__':
    main()

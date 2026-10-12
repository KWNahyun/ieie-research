"""Supporting scalar/setting/reference comparisons; shared controls not replicas."""
import argparse
import json
from pathlib import Path
import numpy as np
from data_pipeline import RESEARCH
from run_convergence import digest
from train_baseline import write_json


def group_vector(record, ids):
    return np.asarray([next(g['mae'] for g in record['groups'] if g['group_id'] == gid) for gid in ids])


def summarize(selected, ids, rng):
    models = {mode: [r for r in selected if r['model'] == mode] for mode in ['B', 'D']}
    means = {mode: np.mean([group_vector(r, ids) for r in values], axis=0) for mode, values in models.items()}
    difference = means['B'] - means['D']
    indices = rng.integers(0, len(ids), size=(10000, len(ids)))
    return {'B_mae': float(means['B'].mean()), 'D_mae': float(means['D'].mean()),
            'B_minus_D': float(difference.mean()), 'relative_reduction_percent': float(100 * difference.mean() / means['B'].mean()),
            'descriptive_geometry_bootstrap95': np.quantile(difference[indices].mean(axis=1), [.025, .975]).tolist(),
            'B_models': len(models['B']), 'D_models': len(models['D']),
            **{f'{mode}_{key}': float(np.mean([r[f'macro_{key}'] for r in models[mode]])) for mode in ['B', 'D'] for key in ['rmse', 'ssim']}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=RESEARCH / 'runs/followup_v1')
    args = parser.parse_args()
    manifest = json.loads((args.root / 'frozen_test_manifest.json').read_text())
    manifest_hash = digest(args.root / 'frozen_test_manifest.json')
    metrics = [json.loads(p.read_text()) for p in (args.root / 'test_metrics').glob('*.json')]
    assert len(metrics) == 152 and len({r['id'] for r in metrics}) == 152
    expected = {r['id']: r for r in manifest['runs']}
    for r in metrics:
        assert r['checkpoint_sha256'] == expected[r['id']]['checkpoint_sha256'] and r['manifest_sha256'] == manifest_hash
    ids = sorted(g['group_id'] for g in metrics[0]['groups'])
    assert len(ids) == 14
    rng = np.random.default_rng(20261012)
    results = {'scalar': [], 'settings': [], 'reference': []}
    original = [json.loads(p.read_text()) for p in (RESEARCH / 'runs/main_v1/test_metrics').glob('*.json')]
    for budget in [10, 25, 50, 100]:
        selected = [r for r in metrics if r['task'] == 'scalar' and r['budget'] == budget]
        assert len(selected) == 20
        result = {'budget': budget, **summarize(selected, ids, rng)}
        for mode in ['B', 'D']:
            full = [r for r in original if r['budget'] == budget and r['model'] == mode]
            scalar = [r for r in selected if r['model'] == mode]
            information_gain = np.mean([group_vector(r, ids) for r in scalar], axis=0) - np.mean([group_vector(r, ids) for r in full], axis=0)
            indices = rng.integers(0, 14, size=(10000, 14))
            result[f'{mode}_scalar_minus_full_pattern_mae'] = float(information_gain.mean())
            result[f'{mode}_pattern_information_descriptive_bootstrap95'] = np.quantile(information_gain[indices].mean(axis=1), [.025, .975]).tolist()
        results['scalar'].append(result)
    for settings in [1, 2, 3]:
        selected = [r for r in metrics if r['task'] == 'settings' and r['settings'] == settings]
        assert len(selected) == 20
        results['settings'].append({'settings_per_spreadsheet_row': settings, 'training_samples': 178 * settings,
                                    'train_geometry_groups': 60, 'spreadsheet_rows': 178, 'fixed_tx': False,
                                    **summarize(selected, ids, rng)})
    for budget in [25, 100]:
        selected = [r for r in metrics if r['task'] == 'reference' and r['budget'] == budget]
        assert len(selected) == 6
        results['reference'].append({'budget': budget, 'source_seeds': [0], 'target_seeds': [0, 1, 2],
                                     'scope': 'adapted HRFormer-S+multistage FiLM+CA; shared audited inputs/loss/split; not published outdoor-score reproduction',
                                     **summarize(selected, ids, rng)})
    write_json(args.root / 'analysis.json', {'results': results, 'runs': 152, 'manifest_sha256': manifest_hash,
               'bootstrap_unit': '14 image geometry groups; average model seeds first', 'bootstrap_seed': 20261012,
               'scope': 'supporting/exploratory after original test was seen; no new blind-test claim',
               'test_based_model_selection': False})
    lines = ['# Required follow-up results', '',
             'Supporting/exploratory evaluations after original main test was known. Models selected only by validation; geometry bootstrap intervals are descriptive.', '',
             '## Parameter-count-matched scalar-only information controls', '',
             '| Group budget | Scalar-only B MAE | Scalar-only D MAE | D reduction | B scalar−full | D scalar−full |',
             '| --- | --- | --- | --- | --- | --- |']
    lines += [f"| {r['budget']}% | {r['B_mae']:.6f} | {r['D_mae']:.6f} | {r['relative_reduction_percent']:.2f}% | {r['B_scalar_minus_full_pattern_mae']:.6f} | {r['D_scalar_minus_full_pattern_mae']:.6f} |" for r in results['scalar']]
    lines += ['', 'Positive scalar−full values mean lower error with pattern information. The pattern encoder remains present with zero normalized pattern inputs; this is an information ablation, not a smaller-architecture comparison.', '',
              '## Configuration-density axis', '', '| Configurations per spreadsheet row | Train samples | B MAE | D MAE | D reduction |',
              '| --- | --- | --- | --- | --- |']
    lines += [f"| {r['settings_per_spreadsheet_row']} | {r['training_samples']} | {r['B_mae']:.6f} | {r['D_mae']:.6f} | {r['relative_reduction_percent']:.2f}% |" for r in results['settings']]
    lines += ['', 'All60 geometry groups and178 spreadsheet rows are retained. Antenna-number entries can have different Tx coordinates/heights, so this is not fixed-Tx angular variation. The534-sample references are frozen main100% results, not new independent replicates.', '',
              '## Adapted prior architecture', '', '| Group budget | HRT scratch MAE | HRT source-encoder MAE | Reduction |',
              '| --- | --- | --- | --- |']
    lines += [f"| {r['budget']}% | {r['B_mae']:.6f} | {r['D_mae']:.6f} | {r['relative_reduction_percent']:.2f}% |" for r in results['reference']]
    lines += ['', 'HRFormer-S+multistage scalar FiLM+pattern cross-attention uses audited B/T inputs, cyclic scalars, full-raster L1 and group split. Published7-channel/beam/outdoor/random-split/TTA scores are not directly compared. Source replication is limited to seed0 for this expensive reference study. Geometry/renderer/absolute-unit/geographic-overlap limitations remain.']
    (args.root / 'results.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()

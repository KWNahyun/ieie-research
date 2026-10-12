"""Seed-averaged paired geometry bootstrap and standalone main-result plot."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from data_pipeline import RESEARCH
from train_baseline import write_json
from run_convergence import digest


def paired_bootstrap(b, d, rng, draws=10000):
    indices = rng.integers(0, len(b), size=(draws, len(b)))
    samples = (b - d)[indices].mean(axis=1)
    return np.quantile(samples, [0.025, 0.975]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=RESEARCH / 'runs/main_v1')
    args = parser.parse_args()
    manifest_hash = digest(args.root / 'frozen_test_manifest.json')
    manifest = json.loads((args.root / 'frozen_test_manifest.json').read_text())
    records = [json.loads(f.read_text()) for f in (args.root / 'test_metrics').glob('*.json')]
    assert len(records) == 80 and all(r['manifest_sha256'] == manifest_hash for r in records)
    assert len({(r['model'], r['source_seed'], r['budget'], r['target_seed']) for r in records}) == 80
    expected = {(r['model'], r['source_seed'], r['budget'], r['seed']): r['checkpoint_sha256']
                for r in manifest['runs']}
    assert len(expected) == 80
    for r in records:
        assert r['checkpoint_sha256'] == expected[(r['model'], r['source_seed'], r['budget'], r['target_seed'])]
    groups = sorted(g['group_id'] for g in records[0]['groups'])
    assert len(groups) == 14
    for r in records:
        assert sorted(g['group_id'] for g in r['groups']) == groups
    rng = np.random.default_rng(20261010)
    summary = []
    seed_rows = []
    for budget in [10, 25, 50, 100]:
        subset = [r for r in records if r['budget'] == budget]
        matrices = {}
        for model in ['B', 'D']:
            selected = [r for r in subset if r['model'] == model]
            assert len(selected) == (5 if model == 'B' else 15)
            matrices[model] = np.asarray([[next(g for g in r['groups'] if g['group_id'] == gid)['mae']
                                           for gid in groups] for r in selected])
        b, d = [matrices[m].mean(axis=0) for m in ['B', 'D']]
        target_values = {m: [] for m in ['B', 'D']}
        for seed in range(5):
            row = {'budget': budget, 'target_seed': seed}
            for model in ['B', 'D']:
                selected = [r for r in subset if r['model'] == model and r['target_seed'] == seed]
                assert len(selected) == (1 if model == 'B' else 3)
                value = float(np.mean([r['macro_mae'] for r in selected]))
                row[f'{model}_mae'] = value
                target_values[model].append(value)
            row['B_minus_D'] = row['B_mae'] - row['D_mae']
            seed_rows.append(row)
        source_means = [float(np.mean([r['macro_mae'] for r in subset if r['model'] == 'D' and r['source_seed'] == seed]))
                        for seed in range(3)]
        summary.append({'budget': budget, 'primary': budget == 25, 'B_mae': float(b.mean()),
                        'D_mae': float(d.mean()), 'B_minus_D': float((b - d).mean()),
                        'relative_reduction_percent': float(100 * (b.mean() - d.mean()) / b.mean()),
                        'paired_geometry_bootstrap95': paired_bootstrap(b, d, rng),
                        'B_target_seed_sd': float(np.std(target_values['B'], ddof=1)),
                        'D_target_seed_sd_after_source_averaging': float(np.std(target_values['D'], ddof=1)),
                        'D_source_seed_means': source_means, 'D_source_seed_sd_after_target_averaging': float(np.std(source_means, ddof=1)),
                        'B_rmse': float(np.mean([r['macro_rmse'] for r in subset if r['model'] == 'B'])),
                        'D_rmse': float(np.mean([r['macro_rmse'] for r in subset if r['model'] == 'D']))})
    write_json(args.root / 'test_analysis.json', {'budgets': summary, 'target_seed_pairs': seed_rows,
               'test_accessed': True, 'test_groups': groups, 'bootstrap_draws': 10000,
               'bootstrap_seed': 20261010, 'manifest_sha256': manifest_hash})
    with (args.root / 'test_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    lines = ['# Main v1 independent held-out image-group results', '',
             'B: scratch conditioned U-Net; D: source encoder transfer with identical target inputs/architecture. Five target seeds and three1500-update source seeds; test checkpoints frozen before evaluation. Primary endpoint:25% target data.', '',
             '| Target group budget | B test MAE | D test MAE | Relative reduction | B−D paired geometry95% interval |',
             '| --- | --- | --- | --- | --- |']
    for r in summary:
        lo, hi = r['paired_geometry_bootstrap95']
        lines.append(f"| {r['budget']}% | {r['B_mae']:.6f} | {r['D_mae']:.6f} | {r['relative_reduction_percent']:.2f}% | [{lo:.6f}, {hi:.6f}] |")
    lines += ['', 'Intervals resample14 image geometry groups, after averaging model-seed errors. They condition on this split and trained models; pixel counts and shared controls are not independent replicates. Source and target-seed variation are reported separately in test_analysis.json. Secondary-budget intervals are descriptive, with no multiple-comparison significance claim.', '',
              f"Verified training checkpoints:80. Unresolved final epoch-cap runs:{sum(r['epoch_cap'] for r in manifest['runs'])}.", '',
              'The percentages are fractions of60 train geometry groups, not exact sample fractions. Actual per-seed base counts are in training_records.json. This is normalized-image whole-raster evaluation with image geometry proxies. Absolute EMF, independent outdoor masks, physical-scene provenance and geographic/source-target overlap remain unresolved; no physical/geographic generalization or exact data-saving claim.', '',
              'All80 test metric files preserve per-sample/group errors and checkpoint hashes. Test results were not used to retrain or choose checkpoints. Source budget sensitivity is reported separately.']
    (args.root / 'results.md').write_text('\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, label in [('B', 'Scratch B'), ('D', 'Transfer D (3 source seeds)')]:
        ax.errorbar([r['budget'] for r in summary], [r[f'{m}_mae'] for r in summary],
                    yerr=[r['B_target_seed_sd'] if m == 'B' else r['D_target_seed_sd_after_source_averaging'] for r in summary],
                    marker='o', capsize=3, label=label)
    ax.set(xlabel='Train geometry-group budget (%)', ylabel='Test group-macro normalized-image MAE',
           title='Main v1: means and target-seed SD')
    ax.set_xticks([10, 25, 50, 100])
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.root / 'test_budget_curve.png', dpi=180)
    plt.close(fig)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()

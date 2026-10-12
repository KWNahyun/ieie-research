"""Post-hoc descriptive equal-error interpolation; no dataset arrays opened."""
import json
from pathlib import Path
import numpy as np
from data_pipeline import RESEARCH
from run_convergence import digest
from train_baseline import write_json


def required_groups(errors, threshold, counts):
    errors = np.minimum.accumulate(np.asarray(errors, dtype=float))
    if threshold >= errors[0]:
        return None, 'left_censored_at_smallest_budget'
    if threshold < errors[-1]:
        return None, 'right_censored_beyond_largest_budget'
    i = int(np.flatnonzero(errors <= threshold)[0])
    if errors[i - 1] == errors[i]:
        return float(counts[i]), 'identified'
    weight = (errors[i - 1] - threshold) / (errors[i - 1] - errors[i])
    return float(counts[i - 1] + weight * (counts[i] - counts[i - 1])), 'identified'


def estimate(curves, threshold, counts, resampled_curves):
    point = {}
    for model in ['B', 'D']:
        n, support = required_groups(curves[model], threshold, counts)
        point[model] = {'required_groups_interpolated': n, 'support': support}
    point['relative_group_savings_percent'] = (100 * (1 - point['D']['required_groups_interpolated'] / point['B']['required_groups_interpolated'])
                                                if all(point[m]['support'] == 'identified' for m in ['B', 'D']) else None)
    savings = []
    censoring = {'B_left': 0, 'B_right': 0, 'D_left': 0, 'D_right': 0}
    for b, d in zip(resampled_curves['B'], resampled_curves['D']):
        nb, sb = required_groups(b, threshold, counts)
        nd, sd = required_groups(d, threshold, counts)
        for m, support in [('B', sb), ('D', sd)]:
            if support.startswith('left'): censoring[f'{m}_left'] += 1
            if support.startswith('right'): censoring[f'{m}_right'] += 1
        if sb == sd == 'identified':
            savings.append(100 * (1 - nd / nb))
    point.update(threshold=threshold, bootstrap_draws=len(resampled_curves['B']),
                 bootstrap_joint_identifiable_fraction=len(savings) / len(resampled_curves['B']),
                 bootstrap_censoring_counts=censoring,
                 conditional_identifiable_bootstrap95_percent=np.quantile(savings, [.025, .975]).tolist() if savings else None,
                 interval_scope='Conditional on both interpolated group requirements lying inside observed budgets; not an unconditional confidence interval')
    return point


def main():
    root = RESEARCH / 'runs/main_v1'
    output = RESEARCH / 'runs/followup_v1/data_efficiency'
    output.mkdir(parents=True, exist_ok=True)
    manifest_hash = digest(root / 'frozen_test_manifest.json')
    metrics = [json.loads(p.read_text()) for p in (root / 'test_metrics').glob('*.json')]
    assert len(metrics) == 80 and all(m['manifest_sha256'] == manifest_hash for m in metrics)
    ids = sorted(g['group_id'] for g in metrics[0]['groups'])
    budgets = [10, 25, 50, 100]
    subsets = json.loads((RESEARCH / 'splits/target_subsets.json').read_text())['seeds']
    counts = [len(subsets['0'][str(b)]['group_ids']) for b in budgets]
    assert all([len(subsets[str(s)][str(b)]['group_ids']) for b in budgets] == counts for s in range(5))
    matrices = {}
    for model in ['B', 'D']:
        matrices[model] = np.asarray([[np.mean([next(g['mae'] for g in m['groups'] if g['group_id'] == gid)
                                              for m in metrics if m['model'] == model and m['budget'] == b])
                                      for gid in ids] for b in budgets])
    curves = {m: a.mean(axis=1) for m, a in matrices.items()}
    rng = np.random.default_rng(20261011)
    indices = rng.integers(0, len(ids), size=(10000, len(ids)))
    resampled = {m: a[:, indices].mean(axis=-1).T for m, a in matrices.items()}
    illustrative = [estimate(curves, t, counts, resampled) for t in [.080, .085]]
    common = [max(curves[m][-1] for m in curves), min(curves[m][0] for m in curves)]
    grid = []
    for threshold in np.linspace(common[0] + 1e-7, common[1] - 1e-7, 64):
        point = {m: required_groups(curves[m], float(threshold), counts)[0] for m in ['B', 'D']}
        grid.append({'threshold': float(threshold), 'B_required_groups': point['B'], 'D_required_groups': point['D'],
                     'relative_group_savings_percent': 100 * (1 - point['D'] / point['B'])})
    report = {'scope': 'post-hoc descriptive equal-error interpolation, not newly confirmed data savings',
              'manifest_sha256': manifest_hash, 'budgets': budgets, 'train_group_counts': counts,
              'raw_error_curves': {m: c.tolist() for m, c in curves.items()},
              'monotonic_envelope_curves': {m: np.minimum.accumulate(c).tolist() for m, c in curves.items()},
              'common_observed_error_range': common, 'illustrative_thresholds': illustrative, 'common_range_grid': grid,
              'bootstrap_seed': 20261011, 'bootstrap_unit': '14 image geometry groups; trained-seed metrics averaged first',
              'data_axis': 'train geometry groups, not exact simulation sample fraction',
              'limitations': 'four sampled budgets; linear interpolation; post-hoc thresholds; censoring must be retained; renderer/geographic uncertainty',
              'dataset_arrays_opened': False}
    write_json(output / 'analysis.json', report)
    lines = ['# Descriptive equal-error data-efficiency analysis', '',
             'Post-hoc analysis of frozen main_v1 test metrics. Values interpolate train geometry-group counts within four observed budgets; no extrapolation or new empirical data-saving claim.', '',
             '| Target normalized MAE | B interpolated groups | D interpolated groups | Estimated group savings | Joint bootstrap identifiable fraction | Conditional95% savings interval |',
             '| --- | --- | --- | --- | --- | --- |']
    for p in illustrative:
        nb, nd = [p[m]['required_groups_interpolated'] for m in ['B', 'D']]
        ci = p['conditional_identifiable_bootstrap95_percent']
        lines.append(f"| {p['threshold']:.3f} | {nb:.2f} | {nd:.2f} | {p['relative_group_savings_percent']:.2f}% | {p['bootstrap_joint_identifiable_fraction']:.2%} | [{ci[0]:.2f}%, {ci[1]:.2f}%] |")
    lines += ['', 'Bootstrap intervals condition on both requirements being identifiable inside observed budgets. Censored draws are counted and explicitly reported in analysis.json; the interval is not an unconditional confidence interval. Fractional group counts are interpolation estimates, not actually trained experiments. A prespecified fresh evaluation is needed for a confirmatory equal-error data-saving claim.', '',
              'Group count axis:6/15/30/60. Exact sample counts vary across target seeds. Source pretraining cost is additional and is reported separately.']
    (output / 'results.md').write_text('\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for model in ['B', 'D']:
        axes[0].plot(counts, curves[model], marker='o', label=model)
    axes[0].set(xlabel='Train geometry groups', ylabel='Test group-macro normalized-image MAE', title='Four actual budget points')
    axes[0].legend()
    axes[1].plot([p['threshold'] for p in grid], [p['relative_group_savings_percent'] for p in grid])
    axes[1].set(xlabel='Common reachable normalized MAE', ylabel='Interpolated group savings (%)', title='Post-hoc descriptive estimates only')
    fig.tight_layout()
    fig.savefig(output / 'equal_error_efficiency.png', dpi=180)
    plt.close(fig)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()

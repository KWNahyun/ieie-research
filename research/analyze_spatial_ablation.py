"""Supporting A/C transfer and matched source0 conditioning interaction."""
import argparse
import json
from pathlib import Path
import numpy as np
from data_pipeline import RESEARCH
from train_baseline import write_json
from run_convergence import digest
from analyze_main import paired_bootstrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=RESEARCH / 'runs/spatial_ablation_v1')
    parser.add_argument('--main-root', type=Path, default=RESEARCH / 'runs/main_v1')
    args = parser.parse_args()
    root = args.root
    manifest = json.loads((root / 'frozen_test_manifest.json').read_text())
    manifest_hash = digest(root / 'frozen_test_manifest.json')
    main_root = args.main_root
    main_hash = digest(main_root / 'frozen_test_manifest.json')
    assert main_hash == manifest['main_manifest_sha256']
    records = [json.loads(f.read_text()) for f in (root / 'test_metrics').glob('*.json')]
    assert len(records) == 40 and len({(r['model'], r['budget'], r['target_seed']) for r in records}) == 40
    expected = {(r['model'], r['budget'], r['seed']): r['checkpoint_sha256'] for r in manifest['runs']}
    for r in records:
        assert r['manifest_sha256'] == manifest_hash
        assert r['checkpoint_sha256'] == expected[(r['model'], r['budget'], r['target_seed'])]
    groups = sorted(g['group_id'] for g in records[0]['groups'])
    assert len(groups) == 14
    rng = np.random.default_rng(20261011)
    results = []
    seed_results = []
    for budget in [10, 25, 50, 100]:
        matrices = {model: [] for model in ['A', 'B', 'C', 'D']}
        for seed in range(5):
            per_seed = {'budget': budget, 'target_seed': seed}
            for model in ['A', 'B', 'C', 'D']:
                if model in ['A', 'C']:
                    metric = next(r for r in records if r['model'] == model and r['budget'] == budget and r['target_seed'] == seed)
                else:
                    source = 'None' if model == 'B' else '0'
                    metric = json.loads((main_root / 'test_metrics' / f'{model}_src{source}_budget{budget}_seed{seed}.json').read_text())
                    assert metric['manifest_sha256'] == main_hash
                assert sorted(g['group_id'] for g in metric['groups']) == groups
                values = [next(g['mae'] for g in metric['groups'] if g['group_id'] == gid) for gid in groups]
                matrices[model].append(values)
                per_seed[f'{model}_mae'] = float(np.mean(values))
            per_seed['interaction'] = (per_seed['B_mae'] - per_seed['D_mae']) - (per_seed['A_mae'] - per_seed['C_mae'])
            seed_results.append(per_seed)
        means = {m: np.mean(values, axis=0) for m, values in matrices.items()}
        spatial = means['A'] - means['C']
        conditioned = means['B'] - means['D']
        results.append({'budget': budget, **{f'{m}_mae': float(v.mean()) for m, v in means.items()},
                        'A_minus_C': float(spatial.mean()), 'B_minus_D_source0': float(conditioned.mean()),
                        'interaction': float((conditioned - spatial).mean()),
                        'A_minus_C_bootstrap95': paired_bootstrap(means['A'], means['C'], rng),
                        'interaction_bootstrap95': paired_bootstrap(conditioned, spatial, rng)})
    write_json(root / 'test_analysis.json', {'budgets': results, 'target_seed_results': seed_results,
               'source_seed_for_C_and_D': 0, 'test_accessed': True, 'bootstrap_draws': 10000, 'bootstrap_seed': 20261011})
    lines = ['# Spatial ablation v1 held-out image-group results', '',
             'Supporting preregistered comparison; C and D both use source seed0. Five target seeds. A/C spatial-only; B/D scalar/pattern-conditioned.', '',
             '| Budget | A MAE | C MAE | A−C | B−D_source0 | Interaction | Interaction geometry95% interval |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for r in results:
        lo, hi = r['interaction_bootstrap95']
        lines.append(f"| {r['budget']}% | {r['A_mae']:.6f} | {r['C_mae']:.6f} | {r['A_minus_C']:.6f} | {r['B_minus_D_source0']:.6f} | {r['interaction']:.6f} | [{lo:.6f}, {hi:.6f}] |")
    lines += ['', 'Interaction=(B−D_source0)−(A−C). Positive values indicate larger conditioned transfer improvement for these trained models; this is not proof of mechanism or significance. Intervals are descriptive paired14-geometry-group bootstrap intervals conditional on this split and trained models. C was not replicated across source seeds. Geometry/rendering/absolute-unit/geographic-overlap limitations remain. No test-based setting or checkpoint changes were made.']
    (root / 'results.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()

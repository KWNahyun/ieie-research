"""Balanced nine-condition validation grid with full checkpoint/pair checks."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from data_pipeline import RESEARCH
from train_baseline import write_json
from run_convergence import digest, SOURCE
from analyze_source_sensitivity import verify_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=RESEARCH / 'runs/source_crossed_v1')
    args = parser.parse_args()
    root = args.root
    initial = RESEARCH / 'runs/source_sensitivity_v1'
    control = RESEARCH / 'runs/convergence_v1'
    protocol = json.loads((root / 'protocol.json').read_text())
    for name, expected in protocol['input_sha256'].items():
        assert digest(root / 'inputs_snapshot' / name) == expected
        assert digest(RESEARCH / name) == expected
    original_conditions = {(500, 0), (1500, 0), (4500, 0), (1500, 1), (1500, 2)}
    torch.set_num_threads(2)
    sources, rows = [], []
    for steps in [500, 1500, 4500]:
        for source_seed in range(3):
            base = initial if (steps, source_seed) in original_conditions else root
            source = SOURCE.parent if (steps, source_seed) == (1500, 0) else base / f'source_steps{steps}_seed{source_seed}'
            config, state = verify_checkpoint(source, False)
            assert config['seed'] == source_seed and config['max_updates'] == steps
            assert config['domain'] == 'RadioMapSeer_DPM'
            history = json.loads((source / 'history.json').read_text())
            summary = json.loads((source / 'summary.json').read_text())
            best = min(history, key=lambda h: h['validation_map_macro_mae'])
            assert state['updates'] == best['updates'] and summary['updates'] == steps
            assert summary['best_source_validation_mae'] == best['validation_map_macro_mae']
            assert not summary['test_accessed']
            source_hash = digest(source / 'best.pt')
            sources.append({'steps': steps, 'source_seed': source_seed, 'selected_updates': state['updates'],
                            'source_validation_mae': summary['best_source_validation_mae'],
                            'checkpoint_sha256': source_hash, 'reused': (steps, source_seed) in original_conditions})
            for budget in [25, 100]:
                for seed in range(3):
                    target = control / f'D_budget{budget}_seed{seed}' if (steps, source_seed) == (1500, 0) else base / f'D_steps{steps}_srcseed{source_seed}_budget{budget}_seed{seed}'
                    tc, checkpoint = verify_checkpoint(target, True)
                    baseline = control / f'B_budget{budget}_seed{seed}'
                    bc = json.loads((baseline / 'config.json').read_text())
                    for name in ['actual_train_base_ids', 'actual_validation_base_ids', 'learning_rate', 'batch_size', 'base_channels', 'epochs', 'patience', 'seed', 'budget']:
                        assert tc[name] == bc[name]
                    for name in ['data_pipeline.py', 'baseline_model.py', 'train_baseline.py']:
                        assert tc['code_sha256'][name] == protocol['input_sha256'][name]
                    assert tc['source_checkpoint_sha256'] == source_hash
                    assert digest(target / 'normalization.json') == digest(baseline / 'normalization.json')
                    h = json.loads((target / 'history.json').read_text())
                    s = json.loads((target / 'summary.json').read_text())
                    selected = min(h, key=lambda r: r['whole_raster_group_macro_mae'])
                    assert selected['epoch'] == checkpoint['epoch'] and selected['whole_raster_group_macro_mae'] == s['best_validation_mae']
                    assert len(h) == s['epochs'] and not s['test_accessed']
                    bmae = json.loads((baseline / 'summary.json').read_text())['best_validation_mae']
                    rows.append({'steps': steps, 'source_seed': source_seed, 'budget': budget,
                                 'target_seed': seed, 'mae': s['best_validation_mae'],
                                 'rmse': selected['whole_raster_group_macro_rmse'], 'B_mae': bmae,
                                 'B_minus_D': bmae - s['best_validation_mae'], 'epochs': len(h),
                                 'epoch_cap': len(h) == 150, 'reused': (steps, source_seed) in original_conditions})
    assert len(sources) == 9 and len(rows) == 54
    aggregates = []
    for steps in [500, 1500, 4500]:
        for budget in [25, 100]:
            selected = [r for r in rows if r['steps'] == steps and r['budget'] == budget]
            d = float(np.mean([r['mae'] for r in selected]))
            b = float(np.mean([r['B_mae'] for r in selected]))
            source_means = [float(np.mean([r['mae'] for r in selected if r['source_seed'] == seed])) for seed in range(3)]
            target_means = [float(np.mean([r['mae'] for r in selected if r['target_seed'] == seed])) for seed in range(3)]
            aggregates.append({'steps': steps, 'budget': budget, 'D_mean_mae': d, 'B_mean_mae': b,
                               'relative_reduction_percent': 100 * (b - d) / b,
                               'source_seed_sd_after_target_averaging': float(np.std(source_means, ddof=1)),
                               'target_seed_sd_after_source_averaging': float(np.std(target_means, ddof=1)),
                               'positive_pairs': sum(r['B_minus_D'] > 0 for r in selected)})
    write_json(root / 'analysis.json', {'sources': sources, 'runs': rows, 'aggregates': aggregates,
               'test_accessed': False, 'source_conditions_verified': 9, 'target_checkpoints_verified': 54,
               'verification': 'strict finite source/target checkpoints, selected checkpoint, paired IDs/normalization/source hashes, input/code snapshots'})
    lines = ['# Fully crossed source grid: validation-only results', '',
             'Three maximum source-update budgets × three source seeds; three target seeds at25/100% target data. This is a preregistered supporting validation study with shared scratch controls; no new test evaluation.', '',
             '| Source max updates | Target data | D mean MAE | Source-seed SD of target-averaged MAE | Target-seed SD of source-averaged MAE | Reduction vs B | Positive paired cases |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for r in aggregates:
        lines.append(f"| {r['steps']} | {r['budget']}% | {r['D_mean_mae']:.6f} | {r['source_seed_sd_after_target_averaging']:.6f} | {r['target_seed_sd_after_source_averaging']:.6f} | {r['relative_reduction_percent']:.2f}% | {r['positive_pairs']}/9 |")
    lines += ['', f"Verified nine source and54 target checkpoints. Epoch-cap runs:{sum(r['epoch_cap'] for r in rows)}/54.", '',
              'Maximum source budgets differ from actual source-validation-selected updates, reported in analysis.json. Shared controls and source/target seed combinations are not independent group replicates. No significance, independent-test improvement or data-saving rate is established by this grid. Float32/TF32/nondeterministic-kernel and geometry/units/geographic-overlap limitations remain. Main and spatial test selections remain unchanged.']
    (root / 'results.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()

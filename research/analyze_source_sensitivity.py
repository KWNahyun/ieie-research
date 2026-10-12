"""Verify sensitivity artifacts and compare against shared scratch controls."""
import argparse
import json
from pathlib import Path
import statistics
import torch
from baseline_model import RadioUNetBaseline
from data_pipeline import RESEARCH
from train_baseline import write_json
from run_convergence import digest


def verify_checkpoint(folder, conditioned):
    config = json.loads((folder / 'config.json').read_text())
    state = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
    assert state['config'] == config
    model = RadioUNetBaseline(conditioned=conditioned, base_channels=16)
    model.load_state_dict(state['model'], strict=True)
    assert all(torch.isfinite(v).all() for v in state['model'].values())
    model.eval()
    with torch.no_grad():
        pred = model(torch.zeros(1, 2, 32, 32), torch.zeros(1, 7), torch.zeros(1, 2, 360))
    assert pred.shape == (1, 1, 32, 32) and torch.isfinite(pred).all()
    assert not config['test_accessed']
    return config, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=RESEARCH / 'runs/source_sensitivity_v1')
    args = parser.parse_args()
    root = args.root
    protocol = json.loads((root / 'protocol.json').read_text())
    control = Path(protocol['reused_control_root'])
    for name, value in protocol['control_artifact_sha256'].items():
        assert digest(control / name) == value
    for name, value in protocol['input_sha256'].items():
        assert digest(root / 'inputs_snapshot' / name) == value
    torch.set_num_threads(2)
    conditions = [{'steps': 1500, 'source_seed': 0}] + protocol['source_conditions']
    sources, rows = [], []
    for condition in conditions:
        steps, seed = condition['steps'], condition['source_seed']
        source = Path(protocol['reused_source_checkpoint']).parent if (steps, seed) == (1500, 0) else root / f'source_steps{steps}_seed{seed}'
        source_config, state = verify_checkpoint(source, False)
        assert source_config['seed'] == seed and source_config['max_updates'] == steps
        summary = json.loads((source / 'summary.json').read_text())
        assert summary['updates'] == steps and not summary['test_accessed']
        source_history = json.loads((source / 'history.json').read_text())
        selected = min(source_history, key=lambda r: r['validation_map_macro_mae'])
        assert state['updates'] == selected['updates']
        assert summary['best_source_validation_mae'] == selected['validation_map_macro_mae']
        source_hash = digest(source / 'best.pt')
        sources.append({**condition, 'selected_updates': state['updates'],
                        'checkpoint_sha256': source_hash, **summary})
        for job in protocol['target_jobs']:
            budget, target_seed = job['budget'], job['seed']
            suffix = f'budget{budget}_seed{target_seed}'
            target = control / f'D_{suffix}' if (steps, seed) == (1500, 0) else root / f'D_steps{steps}_srcseed{seed}_{suffix}'
            config, checkpoint = verify_checkpoint(target, True)
            bconfig, _ = verify_checkpoint(control / f'B_{suffix}', True)
            assert config['source_checkpoint_sha256'] == source_hash
            for key in ['actual_train_base_ids', 'actual_validation_base_ids', 'learning_rate',
                        'batch_size', 'base_channels', 'epochs', 'patience', 'seed', 'budget']:
                assert config[key] == bconfig[key], key
            assert digest(target / 'normalization.json') == digest(control / f'B_{suffix}/normalization.json')
            history = json.loads((target / 'history.json').read_text())
            best = min(history, key=lambda r: r['whole_raster_group_macro_mae'])
            assert checkpoint['epoch'] == best['epoch']
            ts = json.loads((target / 'summary.json').read_text())
            assert not ts['test_accessed'] and len(history) == ts['epochs']
            assert ts['best_validation_mae'] == best['whole_raster_group_macro_mae']
            bmae = json.loads((control / f'B_{suffix}/summary.json').read_text())['best_validation_mae']
            rows.append({**condition, 'budget': budget, 'target_seed': target_seed,
                         'mae': ts['best_validation_mae'], 'rmse': best['whole_raster_group_macro_rmse'],
                         'B_mae': bmae, 'B_minus_D': bmae - ts['best_validation_mae'],
                         'epochs': len(history), 'selected_epoch': best['epoch'] + 1,
                         'epoch_cap': len(history) == 150, 'reused': (steps, seed) == (1500, 0)})
    aggregates = []
    for c in conditions:
        for budget in [25, 100]:
            group = [r for r in rows if r['steps'] == c['steps'] and r['source_seed'] == c['source_seed'] and r['budget'] == budget]
            mae = statistics.mean(r['mae'] for r in group)
            b = statistics.mean(r['B_mae'] for r in group)
            aggregates.append({**c, 'budget': budget, 'mean_mae': mae,
                               'seed_sd': statistics.stdev(r['mae'] for r in group),
                               'B_reduction_percent': 100 * (b - mae) / b,
                               'positive_pairs': sum(r['B_minus_D'] > 0 for r in group)})
    write_json(root / 'analysis.json', {'sources': sources, 'target_runs': rows,
               'aggregates': aggregates, 'test_accessed': False,
               'verification': 'source/target strict loads, finite synthetic inference, exact configs, selected checkpoint, source hashes and paired controls; control/snapshot hashes verified'})
    lines = ['# Source sensitivity v1 validation results', '',
             'Separate source-budget and source-seed sensitivity axes; shared scratch controls. Validation-selected results; test untouched.', '',
             '| Source maximum updates | Source seed | Selected updates | Source MAE |',
             '| --- | --- | --- | --- |']
    lines += [f"| {s['steps']} | {s['source_seed']} | {s['selected_updates']} | {s['best_source_validation_mae']:.6f} |" for s in sources]
    lines += ['', '| Source updates | Source seed | Target data | D MAE ± target-seed SD | Reduction vs B | D better pairs |',
              '| --- | --- | --- | --- | --- | --- |']
    lines += [f"| {a['steps']} | {a['source_seed']} | {a['budget']}% | {a['mean_mae']:.6f} ± {a['seed_sd']:.6f} | {a['B_reduction_percent']:.2f}% | {a['positive_pairs']}/3 |" for a in aggregates]
    lines += ['', f"Target epoch-cap runs: {sum(r['epoch_cap'] for r in rows)}/30 (includes six reused fits).", '',
              'Source seed0/1500 and scratch controls are reused from prior runs. Shared controls and source conditions are not independent statistical replicates. Source-budget selection uses a fixed validation probe; actual selected updates can be lower than the maximum. No significance or data-saving claim is established. Geometry/rendering/geographic-overlap limitations remain.', '',
              'All checkpoint and pairing checks passed. Per-target results are in analysis.json; histories and configs remain in individual run folders.']
    (root / 'results.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()

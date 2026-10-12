"""Verify and summarize convergence artifacts without opening target datasets."""
import argparse
import json
import statistics
import torch
from baseline_model import RadioUNetBaseline
from data_pipeline import RESEARCH
from train_baseline import write_json
from run_convergence import digest, PILOT, SOURCE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=__import__('pathlib').Path,
                        default=RESEARCH / 'runs/convergence_v1')
    args = parser.parse_args()
    protocol = json.loads((args.root / 'protocol.json').read_text())
    torch.set_num_threads(2)
    rows = []
    for job in protocol['jobs']:
        name = f"{job['model']}_budget{job['budget']}_seed{job['seed']}"
        folder = args.root / name
        config = json.loads((folder / 'config.json').read_text())
        history = json.loads((folder / 'history.json').read_text())
        summary = json.loads((folder / 'summary.json').read_text())
        pilot_config = json.loads((PILOT / name / 'config.json').read_text())
        assert config['actual_train_base_ids'] == pilot_config['actual_train_base_ids']
        assert config['actual_validation_base_ids'] == pilot_config['actual_validation_base_ids']
        assert config['epochs'] == 150 and config['patience'] == 25
        assert config['learning_rate'] == job['learning_rate']
        assert config['source_checkpoint_sha256'] == (digest(SOURCE) if job['model'] == 'D' else None)
        assert config['data_contract_sha256'] == protocol['input_sha256']['data_contract.json']
        assert not config['test_accessed'] and not summary['test_accessed']
        assert digest(folder / 'normalization.json') == digest(PILOT / name / 'normalization.json')
        state = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
        assert state['config'] == config
        model = RadioUNetBaseline(conditioned=True, base_channels=16)
        model.load_state_dict(state['model'], strict=True)
        assert all(torch.isfinite(v).all() for v in state['model'].values())
        model.eval()
        with torch.no_grad():
            prediction = model(torch.zeros(1, 2, 32, 32), torch.zeros(1, 7), torch.zeros(1, 2, 360))
        assert prediction.shape == (1, 1, 32, 32) and torch.isfinite(prediction).all()
        best = min(history, key=lambda r: r['whole_raster_group_macro_mae'])
        assert best['epoch'] == state['epoch']
        assert best['whole_raster_group_macro_mae'] == summary['best_validation_mae']
        assert len(history) == summary['epochs']
        old = json.loads((PILOT / name / 'summary.json').read_text())['best_validation_mae']
        rows.append({**job, 'name': name, 'best_epoch': best['epoch'] + 1,
                     'epochs': len(history), 'mae': summary['best_validation_mae'],
                     'rmse': best['whole_raster_group_macro_rmse'],
                     'epoch_cap': len(history) == 150,
                     'best_in_last25': best['epoch'] >= len(history) - 25,
                     'pilot_mae': old, 'pilot_reduction_percent': 100 * (old - summary['best_validation_mae']) / old,
                     'runtime_seconds': summary['runtime_seconds']})
    pairs = []
    for budget in [25, 100]:
        for seed in range(3):
            b, d = [next(r for r in rows if r['budget'] == budget and r['seed'] == seed and r['model'] == m)
                    for m in ['B', 'D']]
            pairs.append({'budget': budget, 'seed': seed, 'B_minus_D': b['mae'] - d['mae']})
    write_json(args.root / 'analysis.json', {'runs': rows, 'pairs': pairs,
               'checkpoint_verification': '12 strict loads, finite weights and synthetic forwards; paired IDs, normalization and source hashes checked',
               'test_accessed': False})
    lines = ['# Convergence v1 validation results', '',
             'Longer fresh B/D fits with fixed pilot LR; validation-selected results. Test data was not opened.', '',
             '| Budget | Model | MAE mean ± seed SD | RMSE mean | Epoch-cap runs |',
             '| --- | --- | --- | --- | --- |']
    for budget in [25, 100]:
        for m in ['B', 'D']:
            group = [r for r in rows if r['budget'] == budget and r['model'] == m]
            lines.append(f"| {budget}% | {m} | {statistics.mean(r['mae'] for r in group):.6f} ± {statistics.stdev(r['mae'] for r in group):.6f} | {statistics.mean(r['rmse'] for r in group):.6f} | {sum(r['epoch_cap'] for r in group)}/3 |")
    lines += ['', '| Run | Selected/total epoch | MAE | Reduction from pilot |', '| --- | --- | --- | --- |']
    lines += [f"| {r['name']} | {r['best_epoch']}/{r['epochs']} | {r['mae']:.6f} | {r['pilot_reduction_percent']:.2f}% |" for r in rows]
    lines += ['', '| Budget | Seed | B−D |', '| --- | --- | --- |']
    lines += [f"| {p['budget']}% | {p['seed']} | {p['B_minus_D']:.6f} |" for p in pairs]
    lines += ['', f"Epoch cap: {sum(r['epoch_cap'] for r in rows)}/12. Best in final25 epochs: {sum(r['best_in_last25'] for r in rows)}/12.", '',
              'These flags are conservative diagnostics; early stopping does not prove convergence. Preserve source seed/budget, geometry proxy and geographic-overlap limitations. No significance or data-saving claim is established.', '',
              'Checkpoint verification passed for all12 runs. Full histories and analysis.json accompany this report.']
    (args.root / 'results.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()

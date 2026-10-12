"""One test evaluation per frozen checkpoint; metrics only, read-only arrays."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from baseline_model import RadioUNetBaseline
from data_pipeline import RESEARCH, TargetDataset
from train_baseline import forward, write_json
from run_convergence import digest


def evaluate_samples(model, loader, device):
    samples = []
    groups = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            error = forward(model, batch, device) - batch['target'].to(device)
            assert torch.isfinite(error).all()
            maes = error.abs().mean((1, 2, 3)).cpu().tolist()
            rmses = error.square().mean((1, 2, 3)).sqrt().cpu().tolist()
            for base, group, mae, rmse in zip(batch['metadata']['base_id'].tolist(),
                                             batch['metadata']['group_id'].tolist(), maes, rmses):
                samples.append({'base_id': base, 'group_id': group, 'mae': mae, 'rmse': rmse})
                groups[group].append((mae, rmse))
    group_metrics = [{'group_id': group, 'sample_count': len(values),
                      'mae': sum(v[0] for v in values) / len(values),
                      'rmse': sum(v[1] for v in values) / len(values)} for group, values in sorted(groups.items())]
    return samples, group_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=RESEARCH / 'runs/main_v1')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    manifest_path = args.root / 'frozen_test_manifest.json'
    manifest_hash = digest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    protocol = json.loads((args.root / 'protocol.json').read_text())
    assert digest(args.root / 'protocol.json') == manifest['protocol_sha256']
    assert len(manifest['runs']) == 80
    for name in ['data_pipeline.py', 'baseline_model.py', 'train_baseline.py', 'evaluate_main.py']:
        assert digest(RESEARCH / name) == protocol['input_sha256'][name]
    # Validate all80 artifact hashes before opening any test arrays.
    for r in manifest['runs']:
        folder = Path(r['folder'])
        for name, key in [('best.pt', 'checkpoint_sha256'), ('config.json', 'config_sha256'),
                          ('normalization.json', 'normalization_sha256')]:
            assert digest(folder / name) == r[key]
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dest = args.root / 'test_metrics'
    dest.mkdir(exist_ok=True)
    write_json(args.root / 'test_access.json', {'test_accessed': True,
               'frozen_manifest_sha256': manifest_hash, 'policy': 'fixed checkpoints; no retraining or selection after test'})
    for index, r in enumerate(manifest['runs']):
        label = f"{r['model']}_src{r['source_seed']}_budget{r['budget']}_seed{r['seed']}"
        path = dest / f'{label}.json'
        if path.exists():
            saved = json.loads(path.read_text())
            assert saved['manifest_sha256'] == manifest_hash and saved['checkpoint_sha256'] == r['checkpoint_sha256']
            assert len(saved['samples']) == 126 and len(saved['groups']) == 14
            continue
        folder = Path(r['folder'])
        state = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
        model = RadioUNetBaseline(conditioned=True, base_channels=16).to(args.device)
        model.load_state_dict(state['model'], strict=True)
        test = TargetDataset('test', normalization=folder / 'normalization.json', include_building_encoded=False)
        test_ids = {int(row['base_id']) for row in test.rows}
        assert not test_ids & set(r['train_base_ids']) and not test_ids & set(r['validation_base_ids'])
        loader = DataLoader(test, batch_size=4, num_workers=2)
        samples, groups = evaluate_samples(model, loader, args.device)
        assert len(samples) == 126 and len(groups) == 14
        write_json(path, {'model': r['model'], 'source_seed': r['source_seed'], 'budget': r['budget'],
                         'target_seed': r['seed'], 'manifest_sha256': manifest_hash,
                         'checkpoint_sha256': r['checkpoint_sha256'], 'samples': samples, 'groups': groups,
                         'macro_mae': sum(g['mae'] for g in groups) / len(groups),
                         'macro_rmse': sum(g['rmse'] for g in groups) / len(groups)})
        del model, loader, test
        if args.device == 'cuda':
            torch.cuda.empty_cache()
        print(f'Test evaluated {label}: {index + 1}/80', flush=True)


if __name__ == '__main__':
    main()

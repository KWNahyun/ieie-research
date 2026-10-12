"""Evaluate frozen spatial A/C checkpoints without changing main selection."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from baseline_model import RadioUNetBaseline
from data_pipeline import RESEARCH, TargetDataset
from evaluate_main import evaluate_samples
from train_baseline import write_json
from run_convergence import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=RESEARCH / 'runs/spatial_ablation_v1')
    args = parser.parse_args()
    manifest_path = args.root / 'frozen_test_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest_hash = digest(manifest_path)
    assert len(manifest['runs']) == 40
    protocol = json.loads((args.root / 'protocol.json').read_text())
    assert digest(args.root / 'protocol.json') == manifest['protocol_sha256']
    for name in ['evaluate_spatial_ablation.py', 'evaluate_main.py', 'data_pipeline.py', 'baseline_model.py', 'train_baseline.py', 'target_manifest.csv', 'splits/target_groups.json']:
        assert digest(RESEARCH / name) == protocol['input_sha256'][name]
    for r in manifest['runs']:
        folder = Path(r['folder'])
        for name, key in [('best.pt', 'checkpoint_sha256'), ('config.json', 'config_sha256'), ('normalization.json', 'normalization_sha256')]:
            assert digest(folder / name) == r[key]
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    assert torch.cuda.is_available()
    metrics = args.root / 'test_metrics'
    metrics.mkdir(exist_ok=True)
    write_json(args.root / 'test_access.json', {'test_accessed': True, 'manifest_sha256': manifest_hash})
    for index, r in enumerate(manifest['runs']):
        name = f"{r['model']}_budget{r['budget']}_seed{r['seed']}"
        path = metrics / f'{name}.json'
        if path.exists():
            prior = json.loads(path.read_text())
            assert prior['manifest_sha256'] == manifest_hash and prior['checkpoint_sha256'] == r['checkpoint_sha256']
            continue
        folder = Path(r['folder'])
        state = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
        model = RadioUNetBaseline(base_channels=16).cuda()
        model.load_state_dict(state['model'], strict=True)
        dataset = TargetDataset('test', normalization=folder / 'normalization.json', include_building_encoded=False)
        assert not {int(row['base_id']) for row in dataset.rows} & (set(r['train_base_ids']) | set(r['validation_base_ids']))
        samples, groups = evaluate_samples(model, DataLoader(dataset, batch_size=4, num_workers=2), 'cuda')
        assert len(samples) == 126 and len(groups) == 14
        write_json(path, {'model': r['model'], 'budget': r['budget'], 'target_seed': r['seed'],
                         'manifest_sha256': manifest_hash, 'checkpoint_sha256': r['checkpoint_sha256'],
                         'samples': samples, 'groups': groups,
                         'macro_mae': sum(g['mae'] for g in groups) / len(groups),
                         'macro_rmse': sum(g['rmse'] for g in groups) / len(groups)})
        del model, dataset
        torch.cuda.empty_cache()
        print(f'Test evaluated {name}: {index + 1}/40', flush=True)


if __name__ == '__main__':
    main()

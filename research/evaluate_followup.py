"""Supporting evaluation of a newly frozen follow-up manifest; no selection."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity
from data_pipeline import RESEARCH, TargetDataset
from followup_train import make_model, predict
from run_convergence import digest
from train_baseline import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=RESEARCH / 'runs/followup_v1')
    args = parser.parse_args()
    manifest_path = args.root / 'frozen_test_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest_hash = digest(manifest_path)
    assert len(manifest['runs']) == 152
    for r in manifest['runs']:
        folder = Path(r['folder'])
        for filename, key in [('best.pt', 'checkpoint_sha256'), ('config.json', 'config_sha256'), ('normalization.json', 'normalization_sha256')]:
            assert digest(folder / filename) == r[key]
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    output = args.root / 'test_metrics'
    output.mkdir(exist_ok=True)
    write_json(args.root / 'test_access.json', {'frozen_manifest_sha256': manifest_hash, 'scope': 'supporting follow-up after original test seen', 'test_accessed': True})
    for index, r in enumerate(manifest['runs']):
        path = output / f"{r['id']}.json"
        if path.exists():
            record = json.loads(path.read_text())
            assert record['manifest_sha256'] == manifest_hash and record['checkpoint_sha256'] == r['checkpoint_sha256']
            continue
        if r['reused']:
            src = RESEARCH / 'runs/main_v1/test_metrics' / f"{r['model']}_src{r['source_seed']}_budget100_seed{r['seed']}.json"
            original = json.loads(src.read_text())
            assert original['checkpoint_sha256'] == r['checkpoint_sha256']
            supplemental = json.loads((args.root / 'supplementary/metrics' / f"{r['model']}_src{r['source_seed']}_budget100_seed{r['seed']}.json").read_text())
            samples = [{**s, 'ssim': next(q['ssim'] for q in supplemental['samples'] if q['base_id'] == s['base_id'])} for s in original['samples']]
            copied = True
        else:
            model = make_model(r['family']).cuda()
            state = torch.load(Path(r['folder']) / 'best.pt', map_location='cpu', weights_only=True)
            model.load_state_dict(state['model'], strict=True)
            model.eval()
            dataset = TargetDataset('test', normalization=Path(r['folder']) / 'normalization.json', include_building_encoded=False)
            assert not {int(x['base_id']) for x in dataset.rows} & (set(r['train_ids']) | set(r['validation_ids']))
            samples = []
            with torch.no_grad():
                for batch in DataLoader(dataset, batch_size=4, num_workers=2):
                    preds = predict(model, batch, 'cuda', r['scalar_only']).cpu().numpy()[:, 0]
                    for i, (pred, target) in enumerate(zip(preds, batch['target'].numpy()[:, 0])):
                        error = pred - target
                        assert np.isfinite(error).all()
                        samples.append({'base_id': int(batch['metadata']['base_id'][i]), 'group_id': int(batch['metadata']['group_id'][i]),
                                        'mae': float(np.abs(error).mean()), 'rmse': float(np.sqrt(np.square(error).mean())),
                                        'ssim': float(structural_similarity(target, pred, data_range=1., gaussian_weights=True,
                                                                          sigma=1.5, win_size=11, use_sample_covariance=False))})
            del model, state, dataset
            torch.cuda.empty_cache()
            copied = False
        groups = defaultdict(list)
        for s in samples: groups[s['group_id']].append(s)
        group_metrics = [{'group_id': group, 'sample_count': len(values),
                          **{key: float(np.mean([v[key] for v in values])) for key in ['mae', 'rmse', 'ssim']}}
                         for group, values in sorted(groups.items())]
        assert len(samples) == 126 and len(group_metrics) == 14
        write_json(path, {**r, 'manifest_sha256': manifest_hash, 'copied_frozen_reference_metrics': copied,
                         'samples': samples, 'groups': group_metrics,
                         **{f'macro_{key}': float(np.mean([g[key] for g in group_metrics])) for key in ['mae', 'rmse', 'ssim']}})
        print(f"Supporting evaluation {r['id']}: {index + 1}/152", flush=True)


if __name__ == '__main__':
    main()

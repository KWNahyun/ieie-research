"""Frozen follow-up scalar/setting/reference training; validation selection only."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from baseline_model import RadioUNetBaseline
from data_pipeline import RESEARCH, TargetDataset, SourceDataset, fit_normalization, common_spatial_input
from followup_data import SettingDataset
from train_baseline import write_json


def make_model(family, source=False):
    if family == 'cnn':
        return RadioUNetBaseline(conditioned=not source, base_channels=16)
    from followup_hrformer import HRFormerConditioned
    return HRFormerConditioned(conditioned=not source)


def predict(model, batch, device, scalar_only=False, source=False):
    inputs = batch['inputs']
    spatial = common_spatial_input(inputs, geometry_mode='image_proxy').to(device)
    if source:
        return model(spatial)
    pattern = torch.zeros_like(inputs['pattern']) if scalar_only else inputs['pattern']
    return model(spatial, inputs['condition'].to(device), pattern.to(device))


@torch.no_grad()
def evaluate(model, loader, device, scalar_only=False, source=False):
    model.eval()
    groups = defaultdict(list)
    for batch in loader:
        error = predict(model, batch, device, scalar_only, source) - batch['target'].to(device)
        assert torch.isfinite(error).all()
        mae = error.abs().mean((1, 2, 3)).cpu().tolist()
        rmse = error.square().mean((1, 2, 3)).sqrt().cpu().tolist()
        key = 'map_id' if source else 'group_id'
        for group, m, r in zip(batch['metadata'][key].tolist(), mae, rmse):
            groups[group].append((m, r))
    mean = np.mean([np.mean(v, axis=0) for v in groups.values()], axis=0)
    return {'whole_raster_group_macro_mae': float(mean[0]),
            'whole_raster_group_macro_rmse': float(mean[1]), 'group_count': len(groups)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['scalar', 'settings', 'reference', 'source'], required=True)
    parser.add_argument('--model', choices=['B', 'D'], default='B')
    parser.add_argument('--family', choices=['cnn', 'hrt'], default='cnn')
    parser.add_argument('--budget', type=int, choices=[10, 25, 50, 100], default=25)
    parser.add_argument('--settings', type=int, choices=[1, 2, 3], default=1)
    parser.add_argument('--seed', type=int, choices=range(5), default=0)
    parser.add_argument('--encoder-checkpoint', type=Path)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.task == 'source'
    if not source and ((args.model == 'D') != bool(args.encoder_checkpoint)):
        parser.error('D requires encoder checkpoint, B requires scratch')
    if source and args.encoder_checkpoint:
        parser.error('Source runs initialize from scratch')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Refusing to overwrite existing fit')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    if min(args.epochs, args.patience, args.steps, args.batch_size) < 1:
        parser.error('positive budgets required')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if source:
        train = SourceDataset('train')
        raw_validation = SourceDataset('validation')
        probes = [i for i, r in enumerate(raw_validation.rows) if int(r['tx_id']) in [0, 40]]
        validation = Subset(raw_validation, probes)
        train_ids = sorted({int(r['map_id']) for r in train.rows})
        valid_ids = sorted({int(raw_validation.rows[i]['map_id']) for i in probes})
        assert not set(train_ids) & set(valid_ids)
    else:
        if args.task == 'settings':
            subset = json.loads((RESEARCH / 'splits/setting_subsets_v1.json').read_text())['seeds'][str(args.seed)][str(args.settings)]
            raw_train = SettingDataset(subset)
        else:
            subset = json.loads((RESEARCH / 'splits/target_subsets.json').read_text())['seeds'][str(args.seed)][str(args.budget)]
            raw_train = TargetDataset('train', subset=subset, include_building_encoded=False)
        stats = fit_normalization(raw_train)
        write_json(args.output / 'normalization.json', stats)
        train = SettingDataset(subset, normalization=args.output / 'normalization.json') if args.task == 'settings' else TargetDataset(
            'train', subset=subset, normalization=args.output / 'normalization.json', include_building_encoded=False)
        validation = TargetDataset('validation', normalization=args.output / 'normalization.json', include_building_encoded=False)
        train_ids = stats['fitted_base_ids']
        valid_ids = [int(r['base_id']) for r in validation.rows]
        assert not set(train_ids) & set(valid_ids)
    if args.smoke:
        train = Subset(train, range(min(2, len(train))))
        validation = Subset(validation, range(min(2, len(validation))))
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=generator,
                        num_workers=args.workers, persistent_workers=args.workers > 0)
    valid_loader = DataLoader(validation, batch_size=args.batch_size, num_workers=args.workers,
                              persistent_workers=args.workers > 0)
    model = make_model(args.family, source)
    source_hash = None
    if args.encoder_checkpoint:
        state = torch.load(args.encoder_checkpoint, map_location='cpu', weights_only=True)
        assert state['config']['domain'] == 'RadioMapSeer_DPM'
        if args.family == 'hrt': assert state['config']['family'] == 'hrt'
        before = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith('encoder.')}
        model.encoder.load_state_dict({k.removeprefix('encoder.'): v for k, v in state['model'].items() if k.startswith('encoder.')}, strict=True)
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
        source_hash = hashlib.sha256(args.encoder_checkpoint.read_bytes()).hexdigest()
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    files = ['baseline_model.py', 'data_pipeline.py', 'followup_data.py', 'followup_train.py', 'followup_protocol.md']
    if args.family == 'hrt': files += ['followup_hrformer.py', 'vendor/hrformer/backbone.py', 'vendor/hrformer/config.json']
    config.update(domain='RadioMapSeer_DPM' if source else 'normalized_EMF_image', scalar_only=args.task == 'scalar',
                  loss='whole_raster_MAE', geometry_mode='image_proxy', test_accessed=False,
                  scope='engineering_smoke' if args.smoke else 'post_main_supporting_followup',
                  actual_train_count=len(train), validation_count=len(validation), train_ids=train_ids, validation_ids=valid_ids,
                  source_checkpoint_sha256=source_hash, parameter_count=sum(p.numel() for p in model.parameters()),
                  code_sha256={name: hashlib.sha256((RESEARCH / name).read_bytes()).hexdigest() for name in files},
                  source_or_target_split_sha256=hashlib.sha256((RESEARCH / ('splits/source_maps.json' if source else 'splits/target_groups.json')).read_bytes()).hexdigest())
    write_json(args.output / 'config.json', config)
    start = time.perf_counter()
    history = []
    best = float('inf')
    stale = updates = 0

    def record(epoch, train_loss):
        nonlocal best, stale
        metrics = evaluate(model, valid_loader, args.device, args.task == 'scalar', source)
        history.append({'epoch': epoch, 'updates': updates, 'train_mae': train_loss, **metrics})
        write_json(args.output / 'history.json', history)
        if metrics['whole_raster_group_macro_mae'] < best:
            best = metrics['whole_raster_group_macro_mae']
            stale = 0
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': config,
                        'epoch': epoch, 'updates': updates}, args.output / 'best.pt')
        else: stale += 1
        print(json.dumps(history[-1]), flush=True)

    if source:
        iterator = iter(loader)
        window = []
        target_steps = 2 if args.smoke else args.steps
        while updates < target_steps:
            try: batch = next(iterator)
            except StopIteration: iterator = iter(loader); batch = next(iterator)
            model.train()
            loss = (predict(model, batch, args.device, source=True) - batch['target'].to(args.device)).abs().mean()
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
            window.append(loss.item())
            updates += 1
            if updates % 250 == 0 or updates == target_steps:
                record(len(history), float(np.mean(window)))
                window = []
    else:
        for epoch in range(2 if args.smoke else args.epochs):
            model.train()
            total = count = 0
            for batch in loader:
                loss = (predict(model, batch, args.device, args.task == 'scalar') - batch['target'].to(args.device)).abs().mean()
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                optimizer.step()
                total += loss.item() * len(batch['target'])
                count += len(batch['target'])
                updates += 1
            record(epoch, total / count)
            if stale >= args.patience: break
    summary = {'epochs': len(history), 'updates': updates, 'best_validation_mae': best,
               'runtime_seconds': time.perf_counter() - start, 'test_accessed': False,
               'epoch_cap': not source and not args.smoke and len(history) == args.epochs}
    write_json(args.output / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()

"""Scratch A/B training with explicit image proxy and whole-raster evaluation.

Smoke example: python IEIE/research/train_baseline.py --model A --smoke
Full run example: python IEIE/research/train_baseline.py --model B --budget 25
Test split is never opened here. Smoke metrics are engineering checks only.
"""
import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from baseline_model import RadioUNetBaseline
from data_pipeline import RESEARCH, TargetDataset, common_spatial_input, fit_normalization


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def forward(model, batch, device):
    inputs = batch['inputs']
    spatial = common_spatial_input(inputs, geometry_mode='image_proxy').to(device)
    return model(spatial, inputs['condition'].to(device), inputs['pattern'].to(device))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    groups = defaultdict(list)
    for batch in loader:
        prediction = forward(model, batch, device)
        error = prediction - batch['target'].to(device)
        mae = error.abs().mean(dim=(1, 2, 3)).cpu().tolist()
        rmse = error.square().mean(dim=(1, 2, 3)).sqrt().cpu().tolist()
        for group, m, r in zip(batch['metadata']['group_id'].tolist(), mae, rmse):
            groups[group].append((m, r))
    macro = np.mean([np.mean(values, axis=0) for values in groups.values()], axis=0)
    return {'whole_raster_group_macro_mae': float(macro[0]),
            'whole_raster_group_macro_rmse': float(macro[1]), 'group_count': len(groups)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['A', 'B', 'C', 'D'], required=True)
    parser.add_argument('--encoder-checkpoint', type=Path)
    parser.add_argument('--budget', type=int, choices=[10, 25, 50, 100], default=25)
    parser.add_argument('--seed', type=int, choices=range(5), default=0)
    parser.add_argument('--base-channels', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--steps', type=int, default=8, help='Smoke update count')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if (args.model in {'C', 'D'}) != (args.encoder_checkpoint is not None):
        parser.error('C/D require source encoder checkpoint; A/B must use scratch initialization')
    if min(args.epochs, args.patience, args.batch_size, args.steps, args.threads) < 1:
        parser.error('epochs/patience/batch-size/steps/threads must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; choose CPU or use an environment with a visible GPU')
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    output = args.output or RESEARCH / 'runs' / (
        f'{"smoke" if args.smoke else "scratch"}_{args.model}_budget{args.budget}_seed{args.seed}')
    if output.exists() and any(output.iterdir()):
        parser.error(f'Refusing to overwrite existing run: {output}')
    output.mkdir(parents=True, exist_ok=True)
    subset = json.loads((RESEARCH / 'splits/target_subsets.json').read_text())['seeds'][str(args.seed)][str(args.budget)]
    raw_train = TargetDataset('train', subset=subset)
    stats = fit_normalization(raw_train)
    write_json(output / 'normalization.json', stats)
    train = TargetDataset('train', subset=subset, normalization=output / 'normalization.json', include_building_encoded=False)
    validation = TargetDataset('validation', normalization=output / 'normalization.json', include_building_encoded=False)
    if args.smoke:
        train = Subset(train, range(min(2, len(train))))
        validation = Subset(validation, range(min(2, len(validation))))
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=not args.smoke,
                        generator=generator, num_workers=args.workers, persistent_workers=args.workers > 0)
    valid_loader = DataLoader(validation, batch_size=args.batch_size, num_workers=args.workers, persistent_workers=args.workers > 0)
    model = RadioUNetBaseline(conditioned=args.model in {'B', 'D'}, base_channels=args.base_channels)
    source_hash = None
    if args.encoder_checkpoint:
        state = torch.load(args.encoder_checkpoint, map_location='cpu', weights_only=True)
        if state['config'].get('domain') != 'RadioMapSeer_DPM':
            raise ValueError('Pretraining checkpoint must identify DPM source domain')
        decoder_before = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith('encoder.')}
        encoder = {k.removeprefix('encoder.'): v for k, v in state['model'].items() if k.startswith('encoder.')}
        model.encoder.load_state_dict(encoder, strict=True)
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in decoder_before.items())
        source_hash = hashlib.sha256(args.encoder_checkpoint.read_bytes()).hexdigest()
    model = model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    config = vars(args).copy()
    config['output'] = str(output)
    config['encoder_checkpoint'] = str(args.encoder_checkpoint) if args.encoder_checkpoint else None
    config['source_checkpoint_sha256'] = source_hash
    config.update(geometry_mode='image_proxy', loss='whole_raster_MAE',
                  evaluation='whole_raster_group_macro; geometry groups, not physical scenes',
                  train_base_ids=stats['fitted_base_ids'], actual_train_count=len(train),
                  actual_train_base_ids=stats['fitted_base_ids'][:2] if args.smoke else stats['fitted_base_ids'],
                  actual_validation_base_ids=[int(r['base_id']) for r in
                                              (validation.dataset.rows[:2] if args.smoke else validation.rows)],
                  validation_count=len(validation), test_accessed=False,
                  scope='engineering smoke only' if args.smoke else 'normalized-image empirical pilot',
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  data_contract_sha256=hashlib.sha256((RESEARCH / 'data_contract.json').read_bytes()).hexdigest(),
                  code_sha256={name: hashlib.sha256((RESEARCH / name).read_bytes()).hexdigest()
                               for name in ['data_pipeline.py', 'baseline_model.py', 'train_baseline.py']})
    write_json(output / 'config.json', config)
    with (RESEARCH / '작업로그.md').open('a') as log:
        log.write(f'\n## 2026-10-08 — {args.model} {"smoke" if args.smoke else "scratch"} 실행 시작\n\n'
                  f'- Output: {output}. Base-only, image-proxy 입력, whole-raster MAE, test 미접근.\n')
    start = time.perf_counter()
    history = []
    best = float('inf')
    stale = 0
    update = 0
    for epoch in range(args.steps if args.smoke else args.epochs):
        model.train()
        total = 0.
        count = 0
        for batch in loader:
            prediction = forward(model, batch, args.device)
            loss = (prediction - batch['target'].to(args.device)).abs().mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError('Nonfinite gradient')
            optimizer.step()
            total += loss.item() * len(batch['target'])
            count += len(batch['target'])
            update += 1
            if args.smoke and update >= args.steps:
                break
        metrics = evaluate(model, valid_loader, args.device)
        history.append({'epoch': epoch, 'updates': update, 'train_mae': total / count, **metrics})
        write_json(output / 'history.json', history)
        value = metrics['whole_raster_group_macro_mae']
        if value < best:
            best = value
            stale = 0
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'config': config, 'epoch': epoch, 'updates': update}, output / 'best.pt')
        else:
            stale += 1
        print(json.dumps(history[-1]), flush=True)
        if (args.smoke and update >= args.steps) or stale >= args.patience:
            break
    result = {'scope': config['scope'], 'updates': update, 'epochs': len(history),
              'runtime_seconds': time.perf_counter() - start, 'best_validation_mae': best,
              'train_loss_first_epoch': history[0]['train_mae'],
              'train_loss_last_epoch': history[-1]['train_mae'], 'test_accessed': False}
    write_json(output / 'summary.json', result)
    with (RESEARCH / '작업로그.md').open('a') as log:
        log.write(f'\n## 2026-10-08 — {args.model} {"smoke" if args.smoke else "scratch"} 실행 완료\n\n'
                  f'- {update} updates, {result["runtime_seconds"]:.1f} seconds. '
                  f'결과: {output / "summary.json"}. Test는 읽지 않았다.\n'
                  '- Smoke 결과는 최종 성능이나 transfer 효과의 증거로 사용하지 않는다.\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

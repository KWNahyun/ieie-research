"""SSIM/figures/compute for frozen main and spatial checkpoints; no retraining."""
import json
import os
from pathlib import Path
import time
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ieie_matplotlib_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from baseline_model import RadioUNetBaseline
from data_pipeline import RESEARCH, TargetDataset
from train_baseline import forward, write_json
from run_convergence import digest


def benchmark(model):
    model.eval()
    inputs = (torch.zeros(1, 2, 500, 500, device='cuda'), torch.zeros(1, 7, device='cuda'), torch.zeros(1, 2, 360, device='cuda'))
    with torch.no_grad():
        for _ in range(50): model(*inputs)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(200):
            start.record(); model(*inputs); end.record(); end.synchronize(); times.append(start.elapsed_time(end))
        allocated = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], with_flops=True) as prof:
            model(*inputs)
            torch.cuda.synchronize()
    flops = sum(e.flops for e in prof.key_averages())
    model.train()
    batch = (inputs[0].expand(4, -1, -1, -1).contiguous(), inputs[1].expand(4, -1).contiguous(), inputs[2].expand(4, -1, -1).contiguous())
    torch.cuda.reset_peak_memory_stats()
    prediction = model(*batch)
    prediction.abs().mean().backward()
    torch.cuda.synchronize()
    training_peak = torch.cuda.max_memory_allocated()
    model.zero_grad(set_to_none=True)
    del prediction, batch
    return {'parameters': sum(p.numel() for p in model.parameters()), 'batch1_median_latency_ms': float(np.median(times)),
            'batch1_p95_latency_ms': float(np.quantile(times, .95)), 'warmup': 50, 'timed_iterations': 200,
            'conv_matmul_flops': flops, 'estimated_conv_matmul_macs': flops / 2,
            'inference_peak_allocated_bytes': allocated, 'inference_peak_reserved_bytes': reserved,
            'synthetic_batch4_forward_backward_peak_allocated_bytes': training_peak,
            'flops_scope': 'Torch profiler supported conv/matmul operations only; norm/activation/interpolation and other unsupported operations excluded',
            'timing_scope': 'model-only CUDA events, inputs already on GPU; excludes preprocessing/I/O; current shared GPU load; not historical training peak'}


def main():
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for GPU benchmark/evaluation')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    root = RESEARCH / 'runs/followup_v1/supplementary'
    root.mkdir(parents=True, exist_ok=True)
    raw = root / 'metrics'
    raw.mkdir(exist_ok=True)
    records = []
    representative = {}
    compute = {}
    for stage in ['main_v1', 'spatial_ablation_v1']:
        folder = RESEARCH / 'runs' / stage
        manifest_path = folder / 'frozen_test_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest_hash = digest(manifest_path)
        for r in manifest['runs']:
            source_seed = r.get('source_seed', 0 if r['model'] == 'C' else None)
            label = f"{r['model']}_src{source_seed}_budget{r['budget']}_seed{r['seed']}"
            path = raw / f'{label}.json'
            is_representative = r['budget'] == 25 and r['seed'] == 0 and source_seed in [None, 0]
            if path.exists() and not is_representative:
                saved = json.loads(path.read_text())
                assert saved['checkpoint_sha256'] == r['checkpoint_sha256'] and saved['manifest_sha256'] == manifest_hash
                records.append(saved)
                continue
            model = RadioUNetBaseline(conditioned=r['model'] in ['B', 'D'], base_channels=16).cuda()
            assert digest(Path(r['folder']) / 'best.pt') == r['checkpoint_sha256']
            state = torch.load(Path(r['folder']) / 'best.pt', map_location='cpu', weights_only=True)
            model.load_state_dict(state['model'], strict=True)
            model.eval()
            dataset = TargetDataset('test', normalization=Path(r['folder']) / 'normalization.json', include_building_encoded=False)
            # Fixed qualitative selection depends only on metadata, never on errors.
            selected_groups = sorted({int(x['geometry_group_id']) for x in dataset.rows})[:4]
            selected_ids = {min(int(x['base_id']) for x in dataset.rows if int(x['geometry_group_id']) == gid) for gid in selected_groups}
            metrics = []
            with torch.no_grad():
                for batch in DataLoader(dataset, batch_size=4, num_workers=2):
                    predictions = forward(model, batch, 'cuda').cpu().numpy()[:, 0]
                    targets = batch['target'].numpy()[:, 0]
                    for index, (pred, target) in enumerate(zip(predictions, targets)):
                        base = int(batch['metadata']['base_id'][index]); group = int(batch['metadata']['group_id'][index])
                        error = pred - target
                        assert np.isfinite(pred).all()
                        ssim = structural_similarity(target, pred, data_range=1., gaussian_weights=True, sigma=1.5,
                                                     use_sample_covariance=False, win_size=11)
                        metrics.append({'base_id': base, 'group_id': group, 'mae': float(np.abs(error).mean()),
                                        'rmse': float(np.sqrt(np.square(error).mean())), 'ssim': float(ssim)})
                        if is_representative and base in selected_ids:
                            representative.setdefault(base, {})[r['model']] = pred.copy()
                            representative[base]['truth'] = target.copy()
                            representative[base]['proxy'] = batch['inputs']['occupancy_proxy'][index, 0].numpy().copy()
                            representative[base]['tx'] = batch['inputs']['numeric'][index, :2].numpy().tolist()
            original_label = f"{r['model']}_src{source_seed}_budget{r['budget']}_seed{r['seed']}" if stage == 'main_v1' else f"{r['model']}_budget{r['budget']}_seed{r['seed']}"
            original = json.loads((folder / 'test_metrics' / f'{original_label}.json').read_text())
            original_samples = {s['base_id']: s for s in original['samples']}
            assert len(metrics) == 126
            for m in metrics:
                for key in ['mae', 'rmse']:
                    assert abs(m[key] - original_samples[m['base_id']][key]) < 1e-6
            groups = defaultdict(list)
            for m in metrics: groups[m['group_id']].append(m)
            group_metrics = [{'group_id': gid, 'ssim': float(np.mean([m['ssim'] for m in values])), 'sample_count': len(values)}
                             for gid, values in sorted(groups.items())]
            saved = {'model': r['model'], 'source_seed': source_seed, 'budget': r['budget'], 'target_seed': r['seed'],
                     'manifest_sha256': manifest_hash, 'checkpoint_sha256': r['checkpoint_sha256'],
                     'ssim_policy': 'whole raster, Gaussian sigma1.5/window11/data_range1/sample_covarianceFalse',
                     'samples': metrics, 'groups': group_metrics, 'macro_ssim': float(np.mean([g['ssim'] for g in group_metrics])),
                     'original_mae_rmse_reproduction_verified_atol': 1e-6}
            write_json(path, saved)
            records.append(saved)
            if r['model'] not in compute:
                compute[r['model']] = benchmark(model)
            del model, state, dataset
            torch.cuda.empty_cache()
            print(f'SSIM/reproduction verified {label}: {len(records)}/120', flush=True)
    assert len(records) == 120 and len(representative) == 4
    aggregate = []
    for budget in [10, 25, 50, 100]:
        for model in ['A', 'B', 'C', 'D']:
            selected = [r for r in records if r['budget'] == budget and r['model'] == model]
            aggregate.append({'budget': budget, 'model': model, 'model_count': len(selected),
                              'test_geometry_macro_ssim': float(np.mean([r['macro_ssim'] for r in selected]))})
    write_json(root / 'analysis.json', {'runs': 120, 'sample_metrics': 15120, 'aggregates': aggregate,
               'compute': compute, 'gpu': torch.cuda.get_device_name(), 'ssim_scope': 'supporting frozen-checkpoint metric, no retraining/selection',
               'figure_base_ids': sorted(representative), 'figure_selection': 'lowest base ID of four lowest test geometry groups; budget25 target/source seed0'})
    for base, examples in representative.items():
        assert all(m in examples for m in ['A', 'B', 'C', 'D'])
        fig, axes = plt.subplots(2, 5, figsize=(13, 5.5))
        axes[0, 0].imshow(examples['truth'], vmin=0, vmax=1, cmap='viridis');axes[0, 0].set_title(f'Truth; base{base}')
        axes[1, 0].imshow(examples['proxy'], cmap='gray', vmin=0, vmax=1);axes[1, 0].scatter(*examples['tx'], c='red', s=10);axes[1, 0].set_title('Image proxy and Tx')
        for column, model in enumerate(['A', 'B', 'C', 'D'], 1):
            axes[0, column].imshow(examples[model], vmin=0, vmax=1, cmap='viridis');axes[0, column].set_title(model)
            axes[1, column].imshow(np.abs(examples[model] - examples['truth']), vmin=0, vmax=1, cmap='magma');axes[1, column].set_title('Absolute error [0,1]')
        for ax in axes.flat:ax.axis('off')
        fig.suptitle('Fixed representative sample; budget25%, target/source seed0; common scales')
        fig.tight_layout();fig.savefig(root / f'prediction_error_base{base}.png', dpi=180);plt.close(fig)
    lines = ['# Supplementary frozen-checkpoint evaluation', '',
             'All120 main/spatial checkpoints evaluated with fixed SSIM policy; repeated MAE/RMSE matched prior outputs at1e-6. No model selection/retraining.', '',
             '| Budget | Model | Test group-macro SSIM |', '| --- | --- | --- |']
    lines += [f"| {r['budget']}% | {r['model']} | {r['test_geometry_macro_ssim']:.6f} |" for r in aggregate]
    lines += ['', 'Compute uses model-only CUDA events under current shared-GPU load. FLOPs include supported conv/matmul only; synthetic batch4 memory is not historical training peak VRAM. Parameter counts and timing/memory details are in analysis.json. Representative figures use metadata-only selection and identical color scales.']
    (root / 'results.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()

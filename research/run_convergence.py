"""Paired longer B/D fits; dry-run preflight never opens target arrays."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

from data_pipeline import RESEARCH, TARGET_ROOT, TARGET_FILES
from train_baseline import write_json

PILOT = RESEARCH / 'runs/pilot_v1'
SOURCE = PILOT / 'source_DPM_seed0/best.pt'
INPUTS = ['data_contract.json', 'target_manifest.csv', 'splits/target_groups.json',
          'splits/target_subsets.json', 'data_pipeline.py', 'baseline_model.py',
          'train_baseline.py', 'run_convergence.py', 'analyze_convergence.py', 'convergence_protocol.md']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight():
    snapshot = PILOT / 'inputs_snapshot/data_contract.json'
    old_contract = json.loads(snapshot.read_text())
    current_contract = json.loads((RESEARCH / 'data_contract.json').read_text())
    current_contract.pop('pilot_v1', None)
    for contract in [old_contract, current_contract]:
        contract['implementation'].pop('training_scope', None)
    assert old_contract == current_contract, 'Data contract semantics changed'
    subsets = json.loads((RESEARCH / 'splits/target_subsets.json').read_text())['seeds']
    lrs = json.loads((PILOT / 'selected_learning_rates.json').read_text())
    source_hash = digest(SOURCE)
    jobs = []
    for seed in range(3):
        for budget in [25, 100]:
            configs = [json.loads((PILOT / f'{m}_budget{budget}_seed{seed}/config.json').read_text())
                       for m in ['B', 'D']]
            assert configs[0]['actual_train_base_ids'] == configs[1]['actual_train_base_ids']
            assert configs[0]['actual_validation_base_ids'] == configs[1]['actual_validation_base_ids']
            requested = subsets[str(seed)][str(budget)]['base_ids']
            assert len(requested) == len(set(requested))
            assert sorted(configs[0]['actual_train_base_ids']) == sorted(requested)
            assert not set(configs[0]['actual_train_base_ids']) & set(configs[0]['actual_validation_base_ids'])
            for model, config in zip(['B', 'D'], configs):
                assert config['source_checkpoint_sha256'] == (source_hash if model == 'D' else None)
                assert config['test_accessed'] is False
                assert config['learning_rate'] == lrs[model]
                for name in ['data_pipeline.py', 'baseline_model.py']:
                    assert digest(RESEARCH / name) == config['code_sha256'][name], name
                assert digest(snapshot) == config['data_contract_sha256']
                assert digest(PILOT / f'B_budget{budget}_seed{seed}/normalization.json') == digest(
                    PILOT / f'D_budget{budget}_seed{seed}/normalization.json')
                jobs.append({'model': model, 'budget': budget, 'seed': seed, 'learning_rate': lrs[model]})
    missing = [str(TARGET_ROOT / name) for name in TARGET_FILES.values()
               if not (TARGET_ROOT / name).is_file()]
    return {'name': 'convergence_v1', 'jobs': jobs, 'maximum_epochs': 150,
            'patience': 25, 'batch_size': 4, 'base_channels': 16,
            'initialization': 'fresh paired target initialization; encoder-only source transfer for D',
            'source_checkpoint_sha256': source_hash,
            'input_sha256': {name: digest(RESEARCH / name) for name in INPUTS},
            'missing_target_files': missing, 'test_accessed': False}


def command(job, output):
    cmd = [sys.executable, str(RESEARCH / 'train_baseline.py'), '--model', job['model'],
           '--budget', str(job['budget']), '--seed', str(job['seed']), '--learning-rate',
           str(job['learning_rate']), '--epochs', '150', '--patience', '25',
           '--batch-size', '4', '--base-channels', '16', '--workers', '2',
           '--device', 'cuda', '--output', str(output)]
    if job['model'] == 'D':
        cmd += ['--encoder-checkpoint', str(SOURCE)]
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--gpus', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--output', type=Path, default=RESEARCH / 'runs/convergence_v1')
    args = parser.parse_args()
    if 0 in args.gpus or len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 1:
        parser.error('Use distinct positive GPU IDs; GPU0 is reserved')
    protocol = preflight()
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return
    if protocol['missing_target_files']:
        parser.error('Missing target files: ' + ', '.join(protocol['missing_target_files']))
    for gpu in args.gpus:
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)}
        subprocess.run([sys.executable, '-c',
                        'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'],
                       env=env, check=True)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Refusing to overwrite existing convergence run')
    args.output.mkdir(parents=True, exist_ok=True)
    protocol['gpu_ids'] = args.gpus
    write_json(args.output / 'protocol.json', protocol)
    for name in INPUTS:
        destination = args.output / 'inputs_snapshot' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(RESEARCH / name, destination)
    logs = args.output / 'console_logs'
    logs.mkdir()
    lock = threading.Lock()
    queue = iter(protocol['jobs'])
    results = []

    def worker(gpu):
        while True:
            with lock:
                job = next(queue, None)
            if job is None:
                return
            name = f"{job['model']}_budget{job['budget']}_seed{job['seed']}"
            output = args.output / name
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '2'}
            with (logs / f'{name}.log').open('w') as stream:
                subprocess.run(command(job, output), env=env, stdout=stream,
                               stderr=subprocess.STDOUT, check=True)
            summary = json.loads((output / 'summary.json').read_text())
            assert summary['test_accessed'] is False
            with lock:
                results.append({**job, **summary})
                write_json(args.output / 'results.json', results)
                print(f'Completed {name}: {len(results)}/12', flush=True)
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in args.gpus]
        for future in futures:
            future.result()
    subprocess.run([sys.executable, str(RESEARCH / 'analyze_convergence.py'),
                    '--root', str(args.output)], check=True)


if __name__ == '__main__':
    main()

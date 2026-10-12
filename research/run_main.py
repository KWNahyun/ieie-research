"""Frozen80-fit main B/D study with validated reuse and one test evaluation."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from data_pipeline import RESEARCH
from train_baseline import write_json
from run_convergence import command, digest, SOURCE
from analyze_source_sensitivity import verify_checkpoint

SENSITIVITY = RESEARCH / 'runs/source_sensitivity_v1'
CONTROL = RESEARCH / 'runs/convergence_v1'
DEADLINE = 1791629194  # 2026-10-10 10:46:34 UTC
INPUTS = ['main_protocol.md', 'run_main.py', 'evaluate_main.py', 'analyze_main.py',
          'train_baseline.py', 'data_pipeline.py', 'baseline_model.py',
          'data_contract.json', 'target_manifest.csv', 'splits/target_groups.json',
          'splits/target_subsets.json']


def plan(root):
    jobs = []
    for seed in range(5):
        for budget in [10, 25, 50, 100]:
            base = {'budget': budget, 'seed': seed, 'learning_rate': 3e-4}
            reuse = seed < 3 and budget in [25, 100]
            jobs.append({**base, 'model': 'B', 'source_seed': None, 'source_checkpoint': None,
                         'reused': reuse, 'folder': str((CONTROL if reuse else root) / f'B_budget{budget}_seed{seed}')})
            for source_seed in range(3):
                source = SOURCE if source_seed == 0 else SENSITIVITY / f'source_steps1500_seed{source_seed}/best.pt'
                name = f'D_srcseed{source_seed}_budget{budget}_seed{seed}'
                folder = root / name
                if reuse:
                    folder = CONTROL / f'D_budget{budget}_seed{seed}' if source_seed == 0 else SENSITIVITY / f'D_steps1500_srcseed{source_seed}_budget{budget}_seed{seed}'
                jobs.append({**base, 'model': 'D', 'source_seed': source_seed,
                             'source_checkpoint': str(source), 'reused': reuse, 'folder': str(folder)})
    return jobs


def check_split():
    with (RESEARCH / 'target_manifest.csv').open(newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['transform'] == 'identity']
    ids = {split: {int(r['base_id']) for r in rows if r['split'] == split} for split in ['train', 'validation', 'test']}
    groups = {split: {int(r['geometry_group_id']) for r in rows if r['split'] == split} for split in ids}
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        assert not ids[a] & ids[b] and not groups[a] & groups[b]
    assert len(ids['test']) == 126 and len(groups['test']) == 14
    return {s: {'base_count': len(ids[s]), 'group_count': len(groups[s])} for s in ids}


def verify(job, hashes):
    folder = Path(job['folder'])
    config, state = verify_checkpoint(folder, True)
    assert config['model'] == job['model'] and config['budget'] == job['budget'] and config['seed'] == job['seed']
    for key, value in [('learning_rate', 3e-4), ('batch_size', 4), ('base_channels', 16), ('patience', 25)]:
        assert config[key] == value
    assert config['epochs'] in [150, 300] and not config['smoke']
    for name in ['train_baseline.py', 'data_pipeline.py', 'baseline_model.py']:
        assert config['code_sha256'][name] == hashes[name]
    expected = digest(Path(job['source_checkpoint'])) if job['source_checkpoint'] else None
    assert config['source_checkpoint_sha256'] == expected
    history = json.loads((folder / 'history.json').read_text())
    summary = json.loads((folder / 'summary.json').read_text())
    assert not summary['test_accessed'] and len(history) == summary['epochs']
    best = min(history, key=lambda h: h['whole_raster_group_macro_mae'])
    assert best['epoch'] == state['epoch'] and summary['best_validation_mae'] == best['whole_raster_group_macro_mae']
    return config, summary, best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--wait-sensitivity', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', type=Path, default=RESEARCH / 'runs/main_v1')
    args = parser.parse_args()
    if args.gpu < 1:
        parser.error('GPU0 is reserved')
    jobs = plan(args.output)
    split = check_split()
    if args.dry_run:
        print(json.dumps({'total_fits': len(jobs), 'reused': sum(j['reused'] for j in jobs),
                          'new_fits': sum(not j['reused'] for j in jobs), 'split': split}, indent=2))
        return
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        parser.error('Use --resume for an existing main run; artifacts are never overwritten')
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = {name: digest(RESEARCH / name) for name in INPUTS}
    protocol_path = args.output / 'protocol.json'
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        assert protocol['input_sha256'] == hashes
    else:
        write_json(protocol_path, {'name': 'main_v1', 'jobs': jobs, 'gpu': args.gpu,
                                  'input_sha256': hashes, 'split': split, 'deadline_utc_epoch': DEADLINE,
                                  'test_accessed': False})
        for name in INPUTS:
            dest = args.output / 'inputs_snapshot' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(RESEARCH / name, dest)
    gate = SENSITIVITY / 'analysis.json'
    while not gate.exists():
        if not args.wait_sensitivity:
            parser.error('Source sensitivity verification is not complete')
        if time.time() >= DEADLINE:
            write_json(args.output / 'status.json', {'status': 'deadline_before_sensitivity', 'test_accessed': False})
            return
        time.sleep(30)
    sensitivity = json.loads(gate.read_text())
    assert not sensitivity['test_accessed'] and len(sensitivity['target_runs']) == 30
    write_json(args.output / 'sensitivity_gate.json', {'analysis_sha256': digest(gate),
               'aggregates': sensitivity['aggregates'], 'source_records': sensitivity['sources'],
               'target_cap_runs': sum(r['epoch_cap'] for r in sensitivity['target_runs']),
               'decision': 'Proceed with preregistered1500-update three-source-seed main comparison, preserving negative outcomes'})
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(args.gpu), 'OMP_NUM_THREADS': '2'}
    subprocess.run([sys.executable, '-c', 'import torch; assert torch.cuda.is_available()'], env=env, check=True)
    logs = args.output / 'console_logs'
    logs.mkdir(exist_ok=True)
    records = []
    for job in jobs:
        folder = Path(job['folder'])
        if not (folder / 'summary.json').exists():
            if time.time() >= DEADLINE:
                write_json(args.output / 'status.json', {'status': 'deadline_partial_training', 'completed': len(records), 'total': 80, 'test_accessed': False})
                return
            if folder.exists() and any(folder.iterdir()):
                raise RuntimeError(f'Incomplete run requires recovery without overwrite: {folder}')
            cmd = command(job, folder)
            if job['model'] == 'D':
                cmd[cmd.index('--encoder-checkpoint') + 1] = job['source_checkpoint']
            with (logs / f'{folder.name}.log').open('w') as stream:
                subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        config, summary, best = verify(job, hashes)
        if summary['epochs'] == 150 and config['epochs'] == 150:
            extension = args.output / (folder.name + '_cap300')
            if not (extension / 'summary.json').exists():
                if time.time() >= DEADLINE:
                    write_json(args.output / 'status.json', {'status': 'deadline_before_cap_extension', 'completed': len(records), 'total': 80, 'test_accessed': False})
                    return
                cmd = command(job, extension)
                cmd[cmd.index('--epochs') + 1] = '300'
                if job['model'] == 'D':
                    cmd[cmd.index('--encoder-checkpoint') + 1] = job['source_checkpoint']
                with (logs / f'{extension.name}.log').open('w') as stream:
                    subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
            job = {**job, 'folder': str(extension), 'cap150_original': str(folder), 'reused': False}
            folder = extension
            config, summary, best = verify(job, hashes)
        records.append({**job, 'checkpoint_sha256': digest(folder / 'best.pt'),
                        'config_sha256': digest(folder / 'config.json'),
                        'normalization_sha256': digest(folder / 'normalization.json'),
                        'train_base_ids': config['actual_train_base_ids'],
                        'validation_base_ids': config['actual_validation_base_ids'],
                        'best_epoch': best['epoch'] + 1, 'epoch_cap': summary['epochs'] == config['epochs'],
                        'train_count': config['actual_train_count'], **summary})
        write_json(args.output / 'training_records.json', records)
        print(f"Verified {folder.name}: {len(records)}/80", flush=True)
    for seed in range(5):
        for budget in [10, 25, 50, 100]:
            paired = [r for r in records if r['seed'] == seed and r['budget'] == budget]
            assert len(paired) == 4
            for r in paired[1:]:
                for key in ['train_base_ids', 'validation_base_ids', 'normalization_sha256']:
                    assert r[key] == paired[0][key]
    manifest = args.output / 'frozen_test_manifest.json'
    if manifest.exists():
        assert json.loads(manifest.read_text())['runs'] == records
    else:
        write_json(manifest, {'runs': records, 'protocol_sha256': digest(protocol_path),
                             'test_group_count': 14, 'test_sample_count': 126,
                             'status': 'frozen_before_test_arrays_opened'})
    write_json(args.output / 'status.json', {'status': 'training_verified_test_manifest_frozen', 'completed': 80, 'test_accessed': False, 'test_manifest_sha256': digest(manifest)})
    subprocess.run([sys.executable, str(RESEARCH / 'evaluate_main.py'), '--root', str(args.output)], env=env, check=True)
    subprocess.run([sys.executable, str(RESEARCH / 'analyze_main.py'), '--root', str(args.output)], check=True)
    write_json(args.output / 'status.json', {'status': 'complete', 'completed': 80, 'test_accessed': True, 'test_manifest_sha256': digest(manifest)})


if __name__ == '__main__':
    main()

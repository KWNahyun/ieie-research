"""Optional40-fit A/C study, queued behind the full primary main study."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from data_pipeline import RESEARCH
from train_baseline import write_json
from run_convergence import SOURCE, command, digest
from run_main import DEADLINE, check_split
from analyze_source_sensitivity import verify_checkpoint

MAIN = RESEARCH / 'runs/main_v1'
INPUTS = ['spatial_ablation_protocol.md', 'run_spatial_ablation.py', 'evaluate_spatial_ablation.py',
          'analyze_spatial_ablation.py', 'evaluate_main.py', 'analyze_main.py',
          'train_baseline.py', 'data_pipeline.py', 'baseline_model.py', 'data_contract.json',
          'target_manifest.csv', 'splits/target_groups.json', 'splits/target_subsets.json']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', type=Path, default=RESEARCH / 'runs/spatial_ablation_v1')
    args = parser.parse_args()
    if args.gpu < 1:
        parser.error('GPU0 is reserved')
    jobs = [{'model': model, 'budget': budget, 'seed': seed, 'learning_rate': 3e-4,
             'folder': str(args.output / f'{model}_budget{budget}_seed{seed}')}
            for seed in range(5) for budget in [10, 25, 50, 100] for model in ['A', 'C']]
    split = check_split()
    if args.dry_run:
        print(json.dumps({'fits': len(jobs), 'split': split, 'source_sha256': digest(SOURCE)}, indent=2))
        return
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        parser.error('Existing study requires --resume; never overwrite artifacts')
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = {name: digest(RESEARCH / name) for name in INPUTS}
    protocol = {'name': 'spatial_ablation_v1', 'jobs': jobs, 'input_sha256': hashes,
                'source_sha256': digest(SOURCE), 'split': split, 'gpu': args.gpu,
                'deadline_utc_epoch': DEADLINE, 'test_accessed': False}
    if (args.output / 'protocol.json').exists():
        assert json.loads((args.output / 'protocol.json').read_text()) == protocol
    else:
        write_json(args.output / 'protocol.json', protocol)
        for name in INPUTS:
            dest = args.output / 'inputs_snapshot' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(RESEARCH / name, dest)
    while True:
        status = MAIN / 'status.json'
        if status.exists() and json.loads(status.read_text())['status'] == 'complete':
            break
        if time.time() >= DEADLINE:
            write_json(args.output / 'status.json', {'status': 'deadline_before_main_completion', 'test_accessed': False})
            return
        time.sleep(30)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(args.gpu), 'OMP_NUM_THREADS': '2'}
    subprocess.run([sys.executable, '-c', 'import torch; assert torch.cuda.is_available()'], env=env, check=True)
    logs = args.output / 'console_logs'
    logs.mkdir(exist_ok=True)
    records = []

    def fit(job, folder, epochs):
        if (folder / 'summary.json').exists():
            return
        if folder.exists() and any(folder.iterdir()):
            raise RuntimeError(f'Incomplete run requires recovery: {folder}')
        cmd = command(job, folder)
        cmd[cmd.index('--epochs') + 1] = str(epochs)
        if job['model'] == 'C':
            cmd += ['--encoder-checkpoint', str(SOURCE)]
        with (logs / f'{folder.name}.log').open('w') as stream:
            subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)

    for job in jobs:
        if time.time() >= DEADLINE:
            write_json(args.output / 'status.json', {'status': 'deadline_partial_training', 'completed': len(records), 'total': 40, 'test_accessed': False})
            return
        folder = Path(job['folder'])
        fit(job, folder, 150)
        summary = json.loads((folder / 'summary.json').read_text())
        if summary['epochs'] == 150:
            original = folder
            folder = args.output / (folder.name + '_cap300')
            if time.time() >= DEADLINE:
                write_json(args.output / 'status.json', {'status': 'deadline_before_extension', 'completed': len(records), 'test_accessed': False})
                return
            fit(job, folder, 300)
            job = {**job, 'folder': str(folder), 'cap150_original': str(original)}
            summary = json.loads((folder / 'summary.json').read_text())
        config, state = verify_checkpoint(folder, False)
        assert config['model'] == job['model'] and config['seed'] == job['seed'] and config['budget'] == job['budget']
        assert config['learning_rate'] == 3e-4 and config['batch_size'] == 4 and config['patience'] == 25
        assert config['source_checkpoint_sha256'] == (digest(SOURCE) if job['model'] == 'C' else None)
        for name in ['data_pipeline.py', 'baseline_model.py', 'train_baseline.py']:
            assert config['code_sha256'][name] == hashes[name]
        history = json.loads((folder / 'history.json').read_text())
        best = min(history, key=lambda h: h['whole_raster_group_macro_mae'])
        assert best['epoch'] == state['epoch'] and best['whole_raster_group_macro_mae'] == summary['best_validation_mae']
        assert len(history) == summary['epochs'] and not summary['test_accessed']
        records.append({**job, 'checkpoint_sha256': digest(folder / 'best.pt'),
                        'config_sha256': digest(folder / 'config.json'),
                        'normalization_sha256': digest(folder / 'normalization.json'),
                        'train_base_ids': config['actual_train_base_ids'],
                        'validation_base_ids': config['actual_validation_base_ids'],
                        'epoch_cap': summary['epochs'] == config['epochs'], **summary})
        write_json(args.output / 'training_records.json', records)
        print(f'Verified {folder.name}: {len(records)}/40', flush=True)
    main_manifest = MAIN / 'frozen_test_manifest.json'
    frozen_main = json.loads(main_manifest.read_text())
    for r in records:
        match = next(m for m in frozen_main['runs'] if m['model'] == 'B' and m['budget'] == r['budget'] and m['seed'] == r['seed'])
        for key in ['train_base_ids', 'validation_base_ids', 'normalization_sha256']:
            assert r[key] == match[key]
    manifest = {'runs': records, 'protocol_sha256': digest(args.output / 'protocol.json'),
                'main_manifest_sha256': digest(main_manifest), 'status': 'frozen_before_ablation_test'}
    path = args.output / 'frozen_test_manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == manifest
    else:
        write_json(path, manifest)
    subprocess.run([sys.executable, str(RESEARCH / 'evaluate_spatial_ablation.py'), '--root', str(args.output)], env=env, check=True)
    subprocess.run([sys.executable, str(RESEARCH / 'analyze_spatial_ablation.py'), '--root', str(args.output)], check=True)
    write_json(args.output / 'status.json', {'status': 'complete', 'completed': 40, 'test_accessed': True})


if __name__ == '__main__':
    main()

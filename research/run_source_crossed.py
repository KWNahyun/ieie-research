"""Optional fully crossed source grid, after main and spatial studies."""
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
from run_convergence import command, digest
from run_main import DEADLINE, check_split

CONDITIONS = [(500, 1), (500, 2), (4500, 1), (4500, 2)]
INPUTS = ['source_crossed_protocol.md', 'run_source_crossed.py', 'analyze_source_crossed.py',
          'train_source.py', 'train_baseline.py', 'baseline_model.py', 'data_pipeline.py',
          'source_manifest.csv', 'target_manifest.csv', 'data_contract.json',
          'splits/source_maps.json', 'splits/target_groups.json', 'splits/target_subsets.json']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', type=Path, default=RESEARCH / 'runs/source_crossed_v1')
    args = parser.parse_args()
    if args.gpu < 1:
        parser.error('GPU0 is reserved')
    check_split()
    jobs = [{'model': 'D', 'budget': b, 'seed': s, 'learning_rate': 3e-4} for s in range(3) for b in [25, 100]]
    if args.dry_run:
        print(json.dumps({'new_source_conditions': CONDITIONS, 'new_source_fits': 4,
                          'new_target_fits': 24, 'full_grid_source_conditions': 9,
                          'full_grid_target_fits': 54}, indent=2))
        return
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        parser.error('Existing grid requires --resume; never overwrite artifacts')
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = {name: digest(RESEARCH / name) for name in INPUTS}
    protocol = {'name': 'source_crossed_v1', 'new_source_conditions': CONDITIONS,
                'target_jobs': jobs, 'gpu': args.gpu, 'input_sha256': hashes,
                'deadline_utc_epoch': DEADLINE, 'test_accessed': False}
    path = args.output / 'protocol.json'
    if path.exists():
        stored = json.loads(path.read_text())
        assert stored['input_sha256'] == hashes and stored['gpu'] == args.gpu
    else:
        write_json(path, protocol)
        for name in INPUTS:
            dest = args.output / 'inputs_snapshot' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(RESEARCH / name, dest)
    gate = RESEARCH / 'runs/spatial_ablation_v1/status.json'
    while True:
        if gate.exists() and json.loads(gate.read_text())['status'] == 'complete':
            break
        if time.time() >= DEADLINE:
            write_json(args.output / 'status.json', {'status': 'deadline_before_core_completion', 'test_accessed': False})
            return
        time.sleep(30)
    write_json(args.output / 'core_gate.json', {'spatial_status_sha256': digest(gate),
               'main_manifest_sha256': digest(RESEARCH / 'runs/main_v1/frozen_test_manifest.json'),
               'spatial_manifest_sha256': digest(RESEARCH / 'runs/spatial_ablation_v1/frozen_test_manifest.json'),
               'settings_selected_from_test': False})
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(args.gpu), 'OMP_NUM_THREADS': '2'}
    logs = args.output / 'console_logs'
    logs.mkdir(exist_ok=True)
    progress = []

    def run(cmd, folder):
        if not (folder / 'summary.json').exists():
            if time.time() >= DEADLINE:
                write_json(args.output / 'status.json', {'status': 'deadline_partial_grid', 'completed_new_stages': len(progress), 'total_new_stages': 28, 'test_accessed': False})
                return False
            if folder.exists() and any(folder.iterdir()):
                raise RuntimeError(f'Incomplete run requires recovery: {folder}')
            with (logs / f'{folder.name}.log').open('w') as stream:
                subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        summary = json.loads((folder / 'summary.json').read_text())
        assert not summary['test_accessed']
        progress.append({'name': folder.name, **summary})
        write_json(args.output / 'progress.json', progress)
        print(f'Completed {folder.name}: {len(progress)}/28 new stages', flush=True)
        return True
    for steps, source_seed in CONDITIONS:
        folder = args.output / f'source_steps{steps}_seed{source_seed}'
        if not run([sys.executable, str(RESEARCH / 'train_source.py'), '--steps', str(steps),
                    '--seed', str(source_seed), '--output', str(folder), '--device', 'cuda'], folder):
            return
        for job in jobs:
            out = args.output / f"D_steps{steps}_srcseed{source_seed}_budget{job['budget']}_seed{job['seed']}"
            cmd = command(job, out)
            cmd[cmd.index('--encoder-checkpoint') + 1] = str(folder / 'best.pt')
            if not run(cmd, out):
                return
    subprocess.run([sys.executable, str(RESEARCH / 'analyze_source_crossed.py'), '--root', str(args.output)], check=True)
    write_json(args.output / 'status.json', {'status': 'complete', 'completed_new_stages': 28,
               'source_conditions': 9, 'target_fits': 54, 'test_accessed': False})


if __name__ == '__main__':
    main()

"""Four new source conditions and24 paired D fits; reuse convergence controls."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from data_pipeline import RESEARCH
from train_baseline import write_json
from run_convergence import preflight as convergence_preflight, digest, command, SOURCE

CONTROL = RESEARCH / 'runs/convergence_v1'
CONDITIONS = [(500, 0), (4500, 0), (1500, 1), (1500, 2)]
INPUTS = ['data_contract.json', 'source_manifest.csv', 'target_manifest.csv',
          'splits/source_maps.json', 'splits/target_groups.json', 'splits/target_subsets.json',
          'data_pipeline.py', 'baseline_model.py', 'train_source.py', 'train_baseline.py',
          'run_convergence.py', 'analyze_convergence.py', 'convergence_protocol.md',
          'run_source_sensitivity.py', 'analyze_source_sensitivity.py', 'source_sensitivity_protocol.md']


def preflight():
    prior = convergence_preflight()
    assert not prior['missing_target_files'], prior['missing_target_files']
    verification = json.loads((CONTROL / 'analysis.json').read_text())
    assert len(verification['runs']) == 12 and not verification['test_accessed']
    with (RESEARCH / 'source_manifest.csv').open(newline='') as stream:
        source_rows = [row for row in csv.DictReader(stream) if row['split'] in ['train', 'validation']]
    missing = {row[key] for row in source_rows for key in ['building_path', 'antenna_path', 'gain_path']
               if not Path(row[key]).is_file()}
    assert not missing, f'Missing source files: {len(missing)}'
    for job in prior['jobs']:
        folder = CONTROL / f"{job['model']}_budget{job['budget']}_seed{job['seed']}"
        config = json.loads((folder / 'config.json').read_text())
        assert config['epochs'] == 150 and config['patience'] == 25
        assert config['learning_rate'] == job['learning_rate'] == 3e-4
        assert config['batch_size'] == 4 and config['base_channels'] == 16
        assert not config['test_accessed']
        for name, value in config['code_sha256'].items():
            assert digest(RESEARCH / name) == value
    return {'name': 'source_sensitivity_v1', 'source_conditions':
            [{'steps': steps, 'source_seed': seed} for steps, seed in CONDITIONS],
            'target_jobs': [j for j in prior['jobs'] if j['model'] == 'D'],
            'reused_control_root': str(CONTROL), 'reused_source_checkpoint': str(SOURCE),
            'reused_source_sha256': digest(SOURCE), 'target_maximum_epochs': 150,
            'target_patience': 25, 'test_accessed': False,
            'input_sha256': {name: digest(RESEARCH / name) for name in INPUTS},
            'control_artifact_sha256': {str(f.relative_to(CONTROL)): digest(f)
                                       for f in CONTROL.glob('*_budget*_seed*/*')
                                       if f.name in ['config.json', 'summary.json', 'normalization.json', 'best.pt']}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--output', type=Path, default=RESEARCH / 'runs/source_sensitivity_v1')
    args = parser.parse_args()
    if args.gpu < 1:
        parser.error('GPU0 is reserved; choose a positive GPU ID')
    protocol = preflight()
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(args.gpu), 'OMP_NUM_THREADS': '2'}
    subprocess.run([sys.executable, '-c', 'import torch; assert torch.cuda.is_available()'],
                   env=env, check=True)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Refusing to overwrite existing sensitivity run')
    args.output.mkdir(parents=True, exist_ok=True)
    protocol['gpu'] = args.gpu
    write_json(args.output / 'protocol.json', protocol)
    for name in INPUTS:
        dest = args.output / 'inputs_snapshot' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(RESEARCH / name, dest)
    logs = args.output / 'console_logs'
    logs.mkdir()
    progress = []

    def run(cmd, folder, name):
        with (logs / f'{name}.log').open('w') as stream:
            subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        summary = json.loads((folder / 'summary.json').read_text())
        assert not summary['test_accessed']
        progress.append({'name': name, **summary})
        write_json(args.output / 'progress.json', progress)
        print(f'Completed {name}: {len(progress)}/28 stages', flush=True)

    for condition in protocol['source_conditions']:
        steps, seed = condition['steps'], condition['source_seed']
        prefix = f'source_steps{steps}_seed{seed}'
        folder = args.output / prefix
        run([sys.executable, str(RESEARCH / 'train_source.py'), '--steps', str(steps),
             '--seed', str(seed), '--output', str(folder), '--device', 'cuda'], folder, prefix)
        for job in protocol['target_jobs']:
            name = f"D_steps{steps}_srcseed{seed}_budget{job['budget']}_seed{job['seed']}"
            output = args.output / name
            cmd = command(job, output)
            cmd[cmd.index('--encoder-checkpoint') + 1] = str(folder / 'best.pt')
            run(cmd, output, name)
    subprocess.run([sys.executable, str(RESEARCH / 'analyze_source_sensitivity.py'),
                    '--root', str(args.output)], check=True)


if __name__ == '__main__':
    main()

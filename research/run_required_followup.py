"""Complete scalar, configuration-density and reference-backbone experiments."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import torch
from data_pipeline import RESEARCH
from followup_train import make_model
from run_convergence import digest, SOURCE
from train_baseline import write_json

INPUTS = ['followup_protocol.md', 'followup_data.py', 'followup_train.py', 'followup_hrformer.py',
          'run_required_followup.py', 'evaluate_followup.py', 'analyze_followup.py',
          'baseline_model.py', 'data_pipeline.py', 'data_contract.json', 'target_manifest.csv',
          'source_manifest.csv', 'splits/target_groups.json', 'splits/target_subsets.json',
          'splits/setting_subsets_v1.json', 'splits/source_maps.json', 'vendor/hrformer/backbone.py',
          'vendor/hrformer/config.json', 'vendor/hrformer/LICENSE', 'vendor/hrformer/provenance.json']


def source_path(seed):
    return SOURCE if seed == 0 else RESEARCH / 'runs/source_sensitivity_v1' / f'source_steps1500_seed{seed}/best.pt'


def jobs(root):
    output = []
    for task in ['scalar', 'settings']:
        levels = [10, 25, 50, 100] if task == 'scalar' else [1, 2, 3]
        for seed in range(5):
            for level in levels:
                for model, src in [('B', None), ('D', 0), ('D', 1), ('D', 2)]:
                    reuse = task == 'settings' and level == 3
                    identity = f'{task}_{model}_src{src}_level{level}_seed{seed}'
                    folder = root / identity
                    if reuse:
                        main = json.loads((RESEARCH / 'runs/main_v1/frozen_test_manifest.json').read_text())
                        folder = Path(next(r['folder'] for r in main['runs'] if r['model'] == model and r['source_seed'] == src and r['seed'] == seed and r['budget'] == 100))
                    output.append({'id': identity, 'task': task, 'model': model, 'source_seed': src, 'seed': seed,
                                   'budget': level if task == 'scalar' else 100, 'settings': level if task == 'settings' else None,
                                   'family': 'cnn', 'scalar_only': task == 'scalar', 'folder': str(folder), 'reused': reuse,
                                   'encoder_checkpoint': str(source_path(src)) if src is not None else None, 'learning_rate': 3e-4})
    for seed in range(3):
        for budget in [25, 100]:
            for model in ['B', 'D']:
                identity = f'reference_{model}_budget{budget}_seed{seed}'
                output.append({'id': identity, 'task': 'reference', 'model': model, 'source_seed': 0 if model == 'D' else None,
                               'seed': seed, 'budget': budget, 'settings': None, 'family': 'hrt', 'scalar_only': False,
                               'folder': str(root / identity), 'reused': False,
                               'encoder_checkpoint': str(root / 'source_hrt_seed0/best.pt') if model == 'D' else None,
                               'learning_rate': None})
    assert len(output) == 152
    return output


def command(job, folder, epochs=150, patience=25):
    cmd = [sys.executable, str(RESEARCH / 'followup_train.py'), '--task', job['task'], '--family', job['family'],
           '--model', job['model'], '--seed', str(job['seed']), '--budget', str(job['budget']), '--epochs', str(epochs),
           '--patience', str(patience), '--learning-rate', str(job['learning_rate']), '--output', str(folder), '--device', 'cuda']
    if job['settings'] is not None: cmd += ['--settings', str(job['settings'])]
    if job['encoder_checkpoint']: cmd += ['--encoder-checkpoint', job['encoder_checkpoint']]
    return cmd


def verify(job):
    folder = Path(job['folder'])
    config = json.loads((folder / 'config.json').read_text())
    state = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
    assert state['config'] == config
    model = make_model(job['family'])
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    assert all(torch.isfinite(v).all() for v in state['model'].values())
    with torch.no_grad():
        out = model(torch.zeros(1, 2, 64, 64), torch.zeros(1, 7), torch.zeros(1, 2, 360))
    assert out.shape == (1, 1, 64, 64) and torch.isfinite(out).all()
    summary = json.loads((folder / 'summary.json').read_text())
    history = json.loads((folder / 'history.json').read_text())
    best = min(history, key=lambda h: h['whole_raster_group_macro_mae'])
    assert state['epoch'] == best['epoch'] and summary['best_validation_mae'] == best['whole_raster_group_macro_mae']
    assert config['seed'] == job['seed'] and config['model'] == job['model'] and not config['smoke']
    assert not summary['test_accessed'] and not config['test_accessed']
    if job['encoder_checkpoint']: assert config['source_checkpoint_sha256'] == digest(Path(job['encoder_checkpoint']))
    if not job['reused']: assert config['scalar_only'] == job['scalar_only'] and config['task'] == job['task']
    stats = json.loads((folder / 'normalization.json').read_text())
    train_ids = config.get('train_ids', config.get('actual_train_base_ids'))
    validation_ids = config.get('validation_ids', config.get('actual_validation_base_ids'))
    assert set(stats['fitted_base_ids']) == set(train_ids) and not set(train_ids) & set(validation_ids)
    return {**job, **summary, 'train_ids': train_ids, 'validation_ids': validation_ids,
            'checkpoint_sha256': digest(folder / 'best.pt'), 'config_sha256': digest(folder / 'config.json'),
            'normalization_sha256': digest(folder / 'normalization.json'), 'selected_epoch': state['epoch'] + 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--output', type=Path, default=RESEARCH / 'runs/followup_v1')
    args = parser.parse_args()
    if args.gpu < 1: parser.error('GPU0 remains reserved')
    planned = jobs(args.output)
    if args.dry_run:
        print(json.dumps({'evaluation_cells': len(planned), 'scalar_new': 80, 'setting_new': 40,
                          'setting_reused': 20, 'reference_final': 12, 'reference_screen': 4, 'reference_source': 1}, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {'jobs': planned, 'input_sha256': {name: digest(RESEARCH / name) for name in INPUTS}, 'gpu': args.gpu,
                'scope': 'new authorized required follow-up; no expired deadline reused; original test already seen'}
    path = args.output / 'protocol.json'
    if path.exists(): assert json.loads(path.read_text()) == protocol
    else:
        write_json(path, protocol)
        for name in INPUTS:
            dst = args.output / 'inputs_snapshot' / name
            dst.parent.mkdir(parents=True, exist_ok=True);shutil.copy2(RESEARCH / name, dst)
    torch.set_num_threads(2)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(args.gpu), 'OMP_NUM_THREADS': '2', 'MPLCONFIGDIR': '/tmp/ieie_matplotlib_cache'}
    env.pop('drop_path_rate', None)
    logs = args.output / 'console_logs';logs.mkdir(exist_ok=True)
    records = []

    def run(cmd, folder):
        if (folder / 'summary.json').exists(): return
        if folder.exists() and any(folder.iterdir()): raise RuntimeError(f'Incomplete fit requires preserved recovery: {folder}')
        with (logs / f'{folder.name}.log').open('w') as stream:
            subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)

    # Supplementary evaluation is independently running; wait to avoid GPU contention.
    supplement = args.output / 'supplementary/analysis.json'
    while not supplement.exists(): time.sleep(15)
    selected_file = args.output / 'reference_selected_lrs.json'
    for job in planned:
        if job['task'] == 'reference' and not selected_file.exists():
            source = args.output / 'source_hrt_seed0'
            run([sys.executable, str(RESEARCH / 'followup_train.py'), '--task', 'source', '--family', 'hrt', '--seed', '0',
                 '--steps', '1500', '--batch-size', '16', '--output', str(source), '--device', 'cuda'], source)
            selected = {}
            for mode in ['B', 'D']:
                candidates = {}
                for lr in [1e-4, 3e-4]:
                    screen = {**job, 'model': mode, 'seed': 0, 'budget': 25, 'learning_rate': lr,
                              'encoder_checkpoint': str(source / 'best.pt') if mode == 'D' else None}
                    out = args.output / f'screen_reference_{mode}_lr{lr}'
                    run(command(screen, out, 15, 5), out)
                    candidates[lr] = json.loads((out / 'summary.json').read_text())['best_validation_mae']
                selected[mode] = min(candidates, key=candidates.get)
            write_json(selected_file, selected)
        if job['task'] == 'reference':
            job = {**job, 'learning_rate': json.loads(selected_file.read_text())[job['model']]}
        folder = Path(job['folder'])
        if not job['reused']: run(command(job, folder), folder)
        result = verify(job)
        if not job['reused'] and result['epoch_cap']:
            extension = args.output / (folder.name + '_cap300')
            run(command(job, extension, 300), extension)
            job = {**job, 'folder': str(extension), 'cap150_original': str(folder)}
            result = verify(job)
        records.append(result)
        write_json(args.output / 'training_records.json', records)
        print(f"Verified {job['id']}: {len(records)}/152", flush=True)
    for task in ['scalar', 'settings']:
        for seed in range(5):
            for level in ([10, 25, 50, 100] if task == 'scalar' else [1, 2, 3]):
                subset = [r for r in records if r['task'] == task and r['seed'] == seed and (r['budget'] if task == 'scalar' else r['settings']) == level]
                assert len(subset) == 4
                assert len({r['normalization_sha256'] for r in subset}) == 1
    frozen = {'runs': records, 'protocol_sha256': digest(path), 'scope': 'supporting after original main test; validation selection only'}
    frozen_path = args.output / 'frozen_test_manifest.json'
    if frozen_path.exists(): assert json.loads(frozen_path.read_text()) == frozen
    else: write_json(frozen_path, frozen)
    subprocess.run([sys.executable, str(RESEARCH / 'evaluate_followup.py'), '--root', str(args.output)], env=env, check=True)
    subprocess.run([sys.executable, str(RESEARCH / 'analyze_followup.py'), '--root', str(args.output)], check=True)
    write_json(args.output / 'status.json', {'status': 'experiments_complete', 'cells': 152, 'test_accessed': True})


if __name__ == '__main__':
    main()

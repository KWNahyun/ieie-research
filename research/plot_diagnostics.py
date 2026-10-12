"""Standalone validation diagnostics from saved histories; no datasets opened."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ieie_matplotlib_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from data_pipeline import RESEARCH


def convergence(root):
    analysis = json.loads((root / 'analysis.json').read_text())
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=False, sharey='row')
    for row, budget in enumerate([25, 100]):
        for seed in range(3):
            ax = axes[row, seed]
            for model in ['B', 'D']:
                history = json.loads((root / f'{model}_budget{budget}_seed{seed}/history.json').read_text())
                x = [h['epoch'] + 1 for h in history]
                y = [h['whole_raster_group_macro_mae'] for h in history]
                line, = ax.plot(x, y, label=f'{model}: {"scratch" if model == "B" else "transfer"}')
                best = int(np.argmin(y))
                ax.plot(x[best], y[best], 'o', color=line.get_color(), markersize=4)
            ax.set_title(f'{budget}% groups, target seed {seed}')
            ax.set_xlabel('Target epoch')
            if seed == 0:
                ax.set_ylabel('Validation group-macro MAE')
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Convergence v1: paired validation curves (selected epochs marked)')
    fig.tight_layout()
    fig.savefig(root / 'validation_learning_curves.png', dpi=180)
    plt.close(fig)
    budgets = [25, 100]
    fig, ax = plt.subplots(figsize=(6, 4))
    for model in ['B', 'D']:
        groups = [[r['mae'] for r in analysis['runs'] if r['budget'] == b and r['model'] == model] for b in budgets]
        ax.errorbar(budgets, [np.mean(g) for g in groups], yerr=[np.std(g, ddof=1) for g in groups],
                    marker='o', capsize=3, label=model)
    ax.set(xlabel='Train geometry-group budget (%)', ylabel='Selected validation group-macro MAE',
           title='Convergence v1: means and target-seed SD')
    ax.set_xticks(budgets)
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / 'validation_budget_curve.png', dpi=180)
    plt.close(fig)


def sensitivity(root):
    analysis = json.loads((root / 'analysis.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    conditions = [(1500, 0), (500, 0), (4500, 0), (1500, 1), (1500, 2)]
    labels = [f'{steps} updates\nsource seed {seed}' for steps, seed in conditions]
    for ax, budget in zip(axes, [25, 100]):
        groups = [[r['mae'] for r in analysis['target_runs'] if r['steps'] == steps and r['source_seed'] == seed and r['budget'] == budget]
                  for steps, seed in conditions]
        ax.errorbar(range(5), [np.mean(g) for g in groups], yerr=[np.std(g, ddof=1) for g in groups],
                    fmt='o', capsize=3, label='D, target-seed mean ± SD')
        controls = [r['B_mae'] for r in analysis['target_runs'] if r['steps'] == 1500 and r['source_seed'] == 0 and r['budget'] == budget]
        ax.axhline(np.mean(controls), color='tab:orange', label='Shared scratch B mean')
        ax.set_xticks(range(5), labels, fontsize=8)
        ax.set(title=f'{budget}% target geometry groups', ylabel='Selected validation group-macro MAE')
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Source sensitivity: separate budget and seed axes (shared controls)')
    fig.tight_layout()
    fig.savefig(root / 'validation_source_sensitivity.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for steps, seed in conditions:
        source = RESEARCH / 'runs/pilot_v1/source_DPM_seed0' if (steps, seed) == (1500, 0) else root / f'source_steps{steps}_seed{seed}'
        history = json.loads((source / 'history.json').read_text())
        x = [h['updates'] for h in history]
        y = [h['validation_map_macro_mae'] for h in history]
        line, = ax.plot(x, y, marker='.', label=f'{steps} max updates, seed{seed}')
        selected = int(np.argmin(y))
        ax.plot(x[selected], y[selected], 'o', color=line.get_color())
    ax.set(xlabel='Source updates', ylabel='Source validation map-macro MAE',
           title='DPM source histories: selected checkpoints marked')
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(root / 'source_learning_curves.png', dpi=180)
    plt.close(fig)


def spatial(root):
    analysis = json.loads((root / 'test_analysis.json').read_text())
    results = analysis['budgets']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for model, label in [('A', 'A: scratch spatial'), ('C', 'C: source0 spatial'),
                         ('B', 'B: scratch conditioned'), ('D', 'D: source0 conditioned')]:
        axes[0].plot([r['budget'] for r in results], [r[f'{model}_mae'] for r in results],
                     marker='o', label=label)
    axes[0].set(xlabel='Train geometry-group budget (%)', ylabel='Test group-macro normalized-image MAE',
                title='Means across five target seeds; source0 only')
    axes[0].set_xticks([10, 25, 50, 100])
    axes[0].legend(fontsize=8)
    values = [r['interaction'] for r in results]
    bounds = [r['interaction_bootstrap95'] for r in results]
    axes[1].errorbar([r['budget'] for r in results], values,
                     yerr=[[v - b[0] for v, b in zip(values, bounds)],
                           [b[1] - v for v, b in zip(values, bounds)]],
                     fmt='o', capsize=4)
    axes[1].axhline(0, color='gray', linewidth=1)
    axes[1].set(xlabel='Train geometry-group budget (%)',
                ylabel='(B-D source0) - (A-C), MAE',
                title='Descriptive paired geometry bootstrap95% intervals')
    axes[1].set_xticks([10, 25, 50, 100])
    fig.tight_layout()
    fig.savefig(root / 'test_spatial_and_interaction.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['convergence', 'sensitivity', 'spatial'], required=True)
    parser.add_argument('--root', type=Path)
    args = parser.parse_args()
    defaults = {'convergence': 'convergence_v1', 'sensitivity': 'source_sensitivity_v1', 'spatial': 'spatial_ablation_v1'}
    root = args.root or RESEARCH / 'runs' / defaults[args.stage]
    {'convergence': convergence, 'sensitivity': sensitivity, 'spatial': spatial}[args.stage](root)
    print(f'Generated standalone figures in {root}; no dataset arrays opened')


if __name__ == '__main__':
    main()

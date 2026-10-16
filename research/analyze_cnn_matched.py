"""Analyze the matched CNN using the colleague metrics without changing scope."""
import json
from pathlib import Path
import numpy as np

RUN=Path(__file__).resolve().parent/'runs/cnn_matched_v1'


def main():
    records={(m,s):json.loads((RUN/f'{m}_seed{s}'/'test_metrics.json').read_text()) for m in ['base','encoder'] for s in [42,43,44]}
    groups=sorted(records['base',42]['metrics']['group_mae']);assert len(groups)==9
    assert all(sorted(r['metrics']['group_mae'])==groups for r in records.values())
    rows=[]
    for mode in ['base','encoder']:
        metrics=[records[mode,s]['metrics'] for s in [42,43,44]]
        rows.append({'mode':mode,'mae_mean':float(np.mean([r['mae'] for r in metrics])),
            'mae_seed_sd':float(np.std([r['mae'] for r in metrics],ddof=1)),
            'rmse_mean':float(np.mean([r['rmse'] for r in metrics])),'ssim_mean':float(np.mean([r['ssim'] for r in metrics]))})
    diff=np.mean([[records['base',s]['metrics']['group_mae'][g]-records['encoder',s]['metrics']['group_mae'][g] for g in groups] for s in [42,43,44]],axis=0)
    draws=np.random.default_rng(20261013).integers(0,9,size=(10000,9))
    result={'rows':rows,'paired_pixel_global_differences':[records['base',s]['metrics']['mae']-records['encoder',s]['metrics']['mae'] for s in [42,43,44]],
        'group_macro_difference':float(diff.mean()),'group_macro_bootstrap95':np.quantile(diff[draws].mean(1),[.025,.975]).tolist(),
        'scope':'existing test known; single-source conditional; same data/SQP/loss/evaluation as DINO; architecture and initial pretraining differ'}
    (RUN/'analysis.json').write_text(json.dumps(result,indent=2))
    lines=['# Matched CNN results','','Same colleague data/SQP/masked-L1/evaluation; single-source conditional.','','| Condition | MAE mean | Seed SD | RMSE mean | SSIM mean |','| --- | --- | --- | --- | --- |']
    lines+=[f"| {r['mode']} | {r['mae_mean']:.6f} | {r['mae_seed_sd']:.6f} | {r['rmse_mean']:.6f} | {r['ssim_mean']:.6f} |" for r in rows]
    lines+=['','Pixel-global paired differences and group-macro bootstrap estimate different quantities; see analysis.json.']
    (RUN/'results.md').write_text('\n'.join(lines)+'\n')
    print('All6 matched CNN target records analyzed')


if __name__=='__main__':main()

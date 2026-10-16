"""Paired single-source-conditional comparison of twelve target runs."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent/'runs/dino_replication_v1'


def main():
    modes=['base','encoder','decoder_lora','decoder_full']
    records={(m,s):json.loads((ROOT/f'{m}_seed{s}'/'test_metrics.json').read_text()) for m in modes for s in [42,43,44]}
    groups=sorted(records['base',42]['metrics']['group_mae'])
    assert len(groups)==9
    assert all(sorted(r['metrics']['group_mae'])==groups for r in records.values())
    rng=np.random.default_rng(20261013)
    draws=rng.integers(0,len(groups),size=(10000,len(groups)))
    rows=[]
    for mode in modes:
        metrics=[records[mode,s]['metrics'] for s in [42,43,44]]
        rows.append({'mode':mode,'mae_mean':float(np.mean([r['mae'] for r in metrics])),
            'mae_seed_sd':float(np.std([r['mae'] for r in metrics],ddof=1)),
            'rmse_mean':float(np.mean([r['rmse'] for r in metrics])),
            'ssim_mean':float(np.mean([r['ssim'] for r in metrics]))})
    comparisons=[]
    for a,b in [('base','encoder'),('base','decoder_lora'),('encoder','decoder_lora'),('encoder','decoder_full'),('decoder_full','decoder_lora')]:
        diff=np.mean([[records[a,s]['metrics']['group_mae'][g]-records[b,s]['metrics']['group_mae'][g] for g in groups] for s in [42,43,44]],axis=0)
        ci=np.quantile(diff[draws].mean(1),[.025,.975]).tolist()
        comparisons.append({'A':a,'B':b,'positive_means_B_better':True,
            'pixel_global_paired_seed_differences':[records[a,s]['metrics']['mae']-records[b,s]['metrics']['mae'] for s in [42,43,44]],
            'group_macro_difference':float(diff.mean()),'group_macro_bootstrap95':ci,'improved_groups':int((diff>0).sum())})
    (ROOT/'analysis.json').write_text(json.dumps({'rows':rows,'comparisons':comparisons,
        'scope':'descriptive single-source-conditional group bootstrap; existing test known; no multiplicity claim'},indent=2))
    lines=['# DINOv2 replication results','','Existing test known; one source seed, three target seeds.','','| Condition | MAE mean | Seed SD | RMSE mean | SSIM mean |','| --- | --- | --- | --- | --- |']
    lines += [f"| {r['mode']} | {r['mae_mean']:.6f} | {r['mae_seed_sd']:.6f} | {r['rmse_mean']:.6f} | {r['ssim_mean']:.6f} |" for r in rows]
    lines += ['','Paired group-macro intervals and pixel-global differences are distinct estimands; see analysis.json. No geographic/physical-unit guarantee.']
    (ROOT/'results.md').write_text('\n'.join(lines)+'\n')
    print('All12 target records analyzed')


if __name__=='__main__': main()

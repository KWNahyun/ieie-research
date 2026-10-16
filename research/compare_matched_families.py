"""Wait for both independent pipelines and compare their shared estimands."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parent/'runs'


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def complete(path):
    state=path/'status.json'
    return state.exists() and json.loads(state.read_text()).get('status')=='experiments_complete'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    cnn=ROOT/'cnn_matched_v1';dino=ROOT/'dino_replication_v1';out=ROOT/'matched_family_comparison_v1';out.mkdir(exist_ok=True)
    while not (complete(cnn) and complete(dino)):
        if not args.wait:raise RuntimeError('Both pipelines must finish before comparison')
        (out/'status.json').write_text(json.dumps({'status':'waiting_for_both_pipelines'}))
        time.sleep(30)
    for stage in ['source','target']:
        assert digest(cnn/f'{stage}_manifest.json')==digest(dino/f'{stage}_manifest.json')
    data={}
    rows=[]
    for family,path,modes in [('cnn',cnn,['base','encoder']),('dino',dino,['base','encoder','decoder_lora','decoder_full'])]:
        frozen=json.loads((path/'frozen_test_manifest.json').read_text())
        assert len(frozen['runs'])==3*len(modes)
        for record in frozen['runs']:
            r=json.loads((Path(record['folder'])/'test_metrics.json').read_text())
            assert r['checkpoint_sha256']==record['checkpoint_sha256']==digest(Path(record['folder'])/'best.pt')
            assert r['metrics']['valid_pixels']==14956749
            assert len(r['metrics']['group_mae'])==9
            assert abs(r['metrics']['macro_group_mae']-np.mean(list(r['metrics']['group_mae'].values())))<1e-12
            data[family,record['mode'],record['seed']]=r['metrics']
        for mode in modes:
            scores=[data[family,mode,s] for s in [42,43,44]]
            rows.append({'family':family,'mode':mode,'mean_pixel_global_mae':float(np.mean([r['mae'] for r in scores])),
                'pixel_global_mae_seed_sd':float(np.std([r['mae'] for r in scores],ddof=1)),
                'mean_group_macro_mae':float(np.mean([r['macro_group_mae'] for r in scores])),
                'rmse_mean':float(np.mean([r['rmse'] for r in scores])),'ssim_mean':float(np.mean([r['ssim'] for r in scores]))})
    effects=[]
    for family,comparisons in [('cnn',[('base','encoder')]),('dino',[('base','encoder'),('base','decoder_lora'),('encoder','decoder_lora'),('encoder','decoder_full'),('decoder_full','decoder_lora')])]:
        for a,b in comparisons:
            x=next(r for r in rows if r['family']==family and r['mode']==a)['mean_pixel_global_mae']
            y=next(r for r in rows if r['family']==family and r['mode']==b)['mean_pixel_global_mae']
            effects.append({'family':family,'A':a,'B':b,'A_minus_B':x-y,'relative_reduction_percent':100*(x-y)/x})
    result={'rows':rows,'within_family_effects':effects,'verification':'PASS',
        'scope':'identical data/SQP/mask/L1/global metrics; architecture, adaptation and general-image pretraining differ; single source per family; existing test known'}
    (out/'analysis.json').write_text(json.dumps(result,indent=2))
    lines=['# Common-condition CNN and DINOv2 comparison','','Both pipelines completed;6 CNN and12 DINO target runs, three target seeds each.','','| Family | Condition | Pixel MAE mean | Seed SD | Group macro MAE | RMSE mean | SSIM mean |','| --- | --- | --- | --- | --- | --- | --- |']
    lines += [f"| {r['family']} | {r['mode']} | {r['mean_pixel_global_mae']:.6f} | {r['pixel_global_mae_seed_sd']:.6f} | {r['mean_group_macro_mae']:.6f} | {r['rmse_mean']:.6f} | {r['ssim_mean']:.6f} |" for r in rows]
    lines += ['',result['scope'],'','This table does not isolate a causal backbone advantage or geographic/physical exposure accuracy. Paired within-family intervals are in each pipeline analysis.json.']
    (out/'results.md').write_text('\n'.join(lines)+'\n')
    (out/'status.json').write_text(json.dumps({'status':'comparison_complete','verification':'PASS','target_runs':18}))
    print('Verified common-condition comparison generated')


if __name__=='__main__':main()

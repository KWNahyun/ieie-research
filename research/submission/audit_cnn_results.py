"""Independent final artifact checks for the conference CNN section."""
import hashlib,json,math,csv
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]/'runs/cnn_matched_v1'
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert read(R/'status.json')['status']=='experiments_complete'
f=read(R/'frozen_test_manifest.json');assert len(f['runs'])==6
assert f['protocol_sha256']==digest(R/'protocol.json')
for name in ['source','target']:
 assert digest(R/f'{name}_manifest.json')==digest(R.parent/'dino_replication_v1'/f'{name}_manifest.json')
records=[]
for row in f['runs']:
 p=Path(row['folder']);s=read(p/'summary.json');t=read(p/'test_metrics.json')
 assert s['validation_reload_verified']
 assert digest(p/'best.pt')==row['checkpoint_sha256']==s['checkpoint_sha256']==t['checkpoint_sha256']
 history=[json.loads(x) for x in (p/'metrics.jsonl').read_text().splitlines()]
 assert [x['epoch'] for x in history]==list(range(1,s['epochs']+1))
 assert abs(min(x['val']['mae'] for x in history)-s['best_validation_mae'])<1e-10
 assert len(t['metrics']['group_mae'])==9 and t['metrics']['ssim_images']==81
 assert all(math.isfinite(t['metrics'][k]) for k in ['mae','rmse','ssim','psnr'])
 records.append({'mode':s['mode'],'seed':s['seed'],'best_epoch':s['selected_epoch'],'epochs':s['epochs'],'epoch_cap':s['epoch_cap'],'validation_mae':s['best_validation_mae'],**{k:t['metrics'][k] for k in ['mae','rmse','ssim','psnr','valid_pixels']}})
a=read(R/'analysis.json')
for row in a['rows']:
 m=[x for x in records if x['mode']==row['mode']]
 assert len(m)==3
 assert abs(np.mean([x['mae'] for x in m])-row['mae_mean'])<1e-12
 assert abs(np.std([x['mae'] for x in m],ddof=1)-row['mae_seed_sd'])<1e-12
assert len({r['valid_pixels'] for r in records})==1
assert all(r['valid_pixels']==14956749 for r in records)
metrics={(m,s):read(R/f'{m}_seed{s}'/'test_metrics.json')['metrics'] for m in ['base','encoder'] for s in [42,43,44]}
groups=sorted(metrics['base',42]['group_mae'])
differences=np.mean([[metrics['base',s]['group_mae'][g]-metrics['encoder',s]['group_mae'][g] for g in groups] for s in [42,43,44]],axis=0)
draws=np.random.default_rng(20261013).integers(0,9,size=(10000,9))
assert abs(differences.mean()-a['group_macro_difference'])<1e-12
assert np.allclose(np.quantile(differences[draws].mean(1),[.025,.975]),a['group_macro_bootstrap95'],rtol=0,atol=1e-12)
out={'status':'PASS','scope':'matched CNN 1 source + 6 targets; final test and manuscript inputs','checks':['same DINO manifests','all six selections frozen','checkpoint hashes match','continuous histories including resumed epoch','validation minima match','81 samples and 9 test groups','same valid pixel counts','independent MAE mean and SD'],'records':records,'analysis':a}
(R/'submission_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
with (R/'submission_results.csv').open('w',newline='') as stream:
 writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
print(json.dumps(out,ensure_ascii=False,indent=2))

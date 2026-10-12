"""Summarize completed validation-only pilot; no dataset/test access."""
from collections import defaultdict
from pathlib import Path
import csv,json,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/ieie-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_pipeline import RESEARCH
from train_baseline import write_json
ROOT=RESEARCH/'runs/pilot_v1'

def main():
 records=json.loads((ROOT/'results.json').read_text());assert len(records)==24
 expected={(m,b,s) for m in 'ABCD' for b in (25,100) for s in (0,1,2)}
 assert {(r['model'],r['budget'],r['seed']) for r in records}==expected
 grouped=defaultdict(list);rows=[];pairs=[];source_hashes=set()
 for r in records:
  folder=ROOT/f"{r['model']}_budget{r['budget']}_seed{r['seed']}";history=json.loads((folder/'history.json').read_text());config=json.loads((folder/'config.json').read_text())
  assert r['test_accessed'] is False and config['test_accessed'] is False
  assert config['actual_validation_base_ids']==json.loads((ROOT/'A_budget25_seed0/config.json').read_text())['actual_validation_base_ids']
  selected=min(history,key=lambda e:e['whole_raster_group_macro_mae'])
  assert abs(selected['whole_raster_group_macro_mae']-r['best_validation_mae'])<1e-10
  row={**r,'best_epoch':selected['epoch']+1,'selected_validation_rmse':selected['whole_raster_group_macro_rmse'],'train_base_count':len(config['actual_train_base_ids']),'group_count':selected['group_count'],'epoch_limit_reached':r['epochs']==30}
  grouped[(r['model'],r['budget'])].append(row);rows.append(row)
  if r['model'] in 'CD':source_hashes.add(config['source_checkpoint_sha256'])
 for budget in (25,100):
  for seed in (0,1,2):
   values={r['model']:r['best_validation_mae'] for r in records if r['budget']==budget and r['seed']==seed}
   configs={m:json.loads((ROOT/f'{m}_budget{budget}_seed{seed}/config.json').read_text()) for m in 'ABCD'}
   assert all(configs[m]['actual_train_base_ids']==configs['A']['actual_train_base_ids'] for m in 'BCD')
   norms=[json.loads((ROOT/f'{m}_budget{budget}_seed{seed}/normalization.json').read_text()) for m in 'ABCD'];assert all(n==norms[0] for n in norms[1:])
   pairs.append({'budget':budget,'seed':seed,'A_minus_C':values['A']-values['C'],'B_minus_D':values['B']-values['D'],'interaction':values['B']-values['D']-values['A']+values['C']})
 assert len(source_hashes)==1
 summary=[]
 for (model,budget),values in sorted(grouped.items(),key=lambda t:(t[0][1],t[0][0])):
  scores=[v['best_validation_mae'] for v in values];summary.append({'model':model,'budget':budget,'validation_mae_mean':float(np.mean(scores)),'seed_std':float(np.std(scores,ddof=1)),'validation_rmse_mean':float(np.mean([v['selected_validation_rmse'] for v in values])),'seeds':len(values),'epoch_limit_runs':sum(v['epoch_limit_reached'] for v in values)})
 analysis={'scope':'validation-selected pilot; no independent test inference','summary':summary,'paired_seed_differences':pairs,'source_checkpoint_count':len(source_hashes),'source_checkpoint_sha256':list(source_hashes)[0],'max_epoch_runs':sum(r['epoch_limit_reached'] for r in rows),'limitations':['3 target seeds and 1 source seed','LR and checkpoint selected on the same validation split','Source 1500-update budget, target max30epochs may not converge','Image proxy and unresolved geographical overlap; no outdoor-only/absolute EMF claim']}
 write_json(ROOT/'analysis.json',analysis)
 with (ROOT/'selected_metrics.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
 for ax,budget in zip(axes,(25,100)):
  vals=[next(r for r in summary if r['model']==m and r['budget']==budget) for m in 'ABCD'];ax.bar(list('ABCD'),[r['validation_mae_mean'] for r in vals],yerr=[r['seed_std'] for r in vals],capsize=4);ax.set_title(f'{budget}% train geometry groups');ax.set_xlabel('Model');ax.grid(axis='y',alpha=.2)
 axes[0].set_ylabel('Whole-raster normalized validation MAE');fig.suptitle('Validation-selected pilot; error bars = target seed SD');fig.tight_layout();fig.savefig(ROOT/'validation_mae.png',dpi=180);plt.close(fig)
 text=['# Pilot v1 validation 결과','', 'Source pretraining, 동일 LR screen, 24 target fits를 완료했다. 이 수치는 validation에서 checkpoint와 LR를 선택한 pilot 결과이며 독립 test 결과가 아니다.','', '| Budget | Model | Validation MAE mean ± seed SD | Selected RMSE mean | 30epoch 도달 runs |','| --- | --- | --- | --- | --- |']
 for r in summary:text.append(f"| {r['budget']}% | {r['model']} | {r['validation_mae_mean']:.6f} ± {r['seed_std']:.6f} | {r['validation_rmse_mean']:.6f} | {r['epoch_limit_runs']}/3 |")
 text+=['','동일 seed/subset에서 positive B−D는 conditioned transfer의 낮은 validation 오차를 의미한다. 아래 차이는 이미 validation 선택을 거친 수치이므로 유의성 또는 test 개선의 근거로 사용하지 않는다.','','| Budget | Seed | A−C | B−D | Interaction |','| --- | --- | --- | --- | --- |']
 for r in pairs:text.append(f"| {r['budget']}% | {r['seed']} | {r['A_minus_C']:.6f} | {r['B_minus_D']:.6f} | {r['interaction']:.6f} |")
 text+=['',f"최대30epoch에 도달한 run은 {analysis['max_epoch_runs']}/24이다. Source는 1500update, target은 max30epoch 제한이므로 convergence를 확인하기 전에는 최종 정확도 또는 negative transfer의 원인을 단정하지 않는다.",'','재현 파일은 runs/pilot_v1의 protocol.json, selected_learning_rates.json, source checkpoint, 24개 config/history/normalization/best.pt/summary와 analysis.json/selected_metrics.csv다. [Validation plot](runs/pilot_v1/validation_mae.png).','','다음 단계는 동일 validation protocol에서 convergence/LR screen의 민감도를 검토하고 필요한 경우 학습을 연장하는 것이다. 본 실험의 예산/5개 seed와 source seed 민감도 protocol을 확정한 뒤 독립 test를 한 번 평가한다. 현재 pilot에서는 test image/label을 열지 않았다.']
 (RESEARCH/'pilot_v1_results.md').write_text('\n'.join(text)+'\n')
 with (RESEARCH/'작업로그.md').open('a') as f:f.write('\n## 2026-10-08 — Pilot v1 분석 완료\n\n- 24개 target fit의 selected metric, paired subset/statistics, source checkpoint hash 일관성, test_accessed=false를 검증했다.\n- 생성: pilot_v1_results.md, runs/pilot_v1/analysis.json, selected_metrics.csv, validation_mae.png. Validation-selected 결과이며 test 성능/유의성으로 해석하지 않는다.\n')
 print(json.dumps(analysis,indent=2))
if __name__=='__main__':main()

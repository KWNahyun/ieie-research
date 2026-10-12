"""Run source pretraining, equal LR screening, and 24 paired target pilot fits.
GPU0 is deliberately excluded. Jobs execute on GPUs1/2, never test data.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import csv,json,os,subprocess,sys,threading,time
from data_pipeline import RESEARCH
from train_baseline import write_json
ROOT=RESEARCH/'runs/pilot_v1'
LOCK=threading.Lock()
SOURCE=ROOT/'source_DPM_seed0'

def log(message):
 with LOCK:
  with (RESEARCH/'작업로그.md').open('a') as f:f.write(f'\n### Pilot 진행 — {time.strftime("%Y-%m-%d %H:%M:%S")}\n\n- {message}\n')

def run(command,out,gpu):
 out.mkdir(parents=True,exist_ok=True);env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);env['OMP_NUM_THREADS']='2'
 console=ROOT/'console_logs';console.mkdir(exist_ok=True)
 with (console/(out.name+'.log')).open('w') as stream:
  process=subprocess.run([sys.executable,*command],env=env,stdout=stream,stderr=subprocess.STDOUT)
 if process.returncode:
  log(f'실패: {out}; exit={process.returncode}. console.log 확인.');raise RuntimeError(str(out))
 log(f'완료: {out}; GPU{gpu}.')
 return json.loads((out/'summary.json').read_text())

def target(model,budget,seed,lr,epochs,patience,out,gpu):
 cmd=[str(RESEARCH/'train_baseline.py'),'--model',model,'--budget',str(budget),'--seed',str(seed),'--base-channels','16','--batch-size','4','--epochs',str(epochs),'--patience',str(patience),'--learning-rate',str(lr),'--workers','2','--device','cuda','--output',str(out)]
 if model in ['C','D']:cmd+=['--encoder-checkpoint',str(SOURCE/'best.pt')]
 return run(cmd,out,gpu)

def screen(models,gpu):
 selected={}
 for model in models:
  scores={}
  for lr in [1e-4,3e-4]:
   out=ROOT/f'screen_{model}_lr{lr}';scores[lr]=target(model,25,0,lr,10,3,out,gpu)['best_validation_mae']
  selected[model]=min(scores,key=scores.get);log(f'LR screen {model}: {scores}; 선택 {selected[model]}.')
 return selected

def main():
 if ROOT.exists() and any(ROOT.iterdir()):raise ValueError('Pilot already exists; inspect previous run instead of overwriting')
 ROOT.mkdir(parents=True)
 protocol={'name':'pilot_v1','source_updates':1500,'source_seed':0,'source_batch_size':16,'target_models':['A','B','C','D'],'budgets':[25,100],'target_seeds':[0,1,2],'base_channels':16,'target_batch_size':4,'lr_candidates':[1e-4,3e-4],'lr_screen':{'budget':25,'seed':0,'maximum_epochs':10,'patience':3,'selection':'minimum validation group macro MAE, independent per model'},'final_maximum_epochs':30,'final_patience':8,'source_validation':'fixed Tx IDs 0 and 40 on all held-out validation maps','transfer':'encoder only, full fine-tuning from update 1; target decoder/head/condition freshly initialized','metrics':'whole-raster normalized-image group macro MAE/RMSE','test_accessed':False,'gpu_ids':[1,2],'geometry':'image proxy; geographic overlap unresolved','limits':'source fixed-update pilot and target epoch-limited fits; inspect convergence before final claims'}
 write_json(ROOT/'protocol.json',protocol);log('Pilot v1 시작: source1500 updates; 동일 LR screen 8회; A–D×2budget×3seed 24회. GPU1/2만 사용.')
 with ThreadPoolExecutor(max_workers=2) as pool:
  scratch=pool.submit(screen,['A','B'],2)
  run([str(RESEARCH/'train_source.py'),'--output',str(SOURCE),'--steps','1500','--device','cuda'],SOURCE,1)
  transferred=pool.submit(screen,['C','D'],1)
  selected={**scratch.result(),**transferred.result()}
 write_json(ROOT/'selected_learning_rates.json',selected)
 jobs=[(m,b,s) for s in [0,1,2] for b in [25,100] for m in ['A','B','C','D']];queue=iter(jobs);results=[]
 def worker(gpu):
  while True:
   with LOCK:
    job=next(queue,None)
   if job is None:return
   model,budget,seed=job;out=ROOT/f'{model}_budget{budget}_seed{seed}'
   result=target(model,budget,seed,selected[model],30,8,out,gpu)
   row={'model':model,'budget':budget,'seed':seed,'learning_rate':selected[model],**result}
   with LOCK:
    results.append(row);write_json(ROOT/'results.json',results)
 with ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(worker,gpu) for gpu in [1,2]]
  for future in futures:future.result()
 with (ROOT/'results.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(sorted(results,key=lambda r:(r['budget'],r['seed'],r['model'])))
 log('Pilot v1 전체 완료. Source와 LR-screen/24개 target 결과 저장. Test는 평가하지 않았다.')
if __name__=='__main__':main()

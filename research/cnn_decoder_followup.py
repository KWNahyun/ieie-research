"""Single-seed CNN encoder+decoder control using the frozen matched-CNN trainer."""
import argparse,copy,json,os,subprocess,sys,time,traceback
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/cnn_decoder_seed42_v1'
PYTHON='/home/ailab/anaconda3/envs/emf_dinov2/bin/python'
MODE='encoder_decoder';SEED=42

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');parser.add_argument('--gpu',type=int,default=2);args=parser.parse_args()
 if args.launch:
  RUN.mkdir(exist_ok=True)
  if (RUN/'pid.json').exists():
   previous=json.loads((RUN/'pid.json').read_text())
   try:os.kill(previous['pid'],0)
   except ProcessLookupError:pass
   else:raise RuntimeError('Existing runner is alive; do not duplicate')
  env={**os.environ,'CUDA_VISIBLE_DEVICES':str(args.gpu),'OMP_NUM_THREADS':'2','HF_HUB_OFFLINE':'1','MPLCONFIGDIR':'/tmp/ieie_matplotlib_cache'}
  with (RUN/'runner.log').open('a') as log:
   p=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),'--gpu',str(args.gpu)],cwd=ROOT.parent,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  (RUN/'pid.json').write_text(json.dumps({'pid':p.pid,'gpu':args.gpu,'seed':SEED,'launched_at':time.time()},indent=2));print('Launched',p.pid,flush=True);return
 import torch
 import cnn_matched_train as trainer
 from cnn_matched_model import MatchedCNN
 from train import set_seed
 shared=trainer.shared;original_run=trainer.RUN
 source=original_run/'source/best.pt';encoder_cfg=trainer.config('encoder',SEED)
 RUN.mkdir(exist_ok=True)
 def write(name,data):shared.write(RUN/name,data)
 def cfg_for(mode,seed):
  assert mode==MODE and seed==SEED
  cfg=copy.deepcopy(encoder_cfg);cfg.save_dir=str(RUN);cfg.decoder_adaptation='source_full'
  return cfg
 def make(mode,seed,initialize=True):
  assert mode==MODE and seed==SEED
  set_seed(seed,'cuda' if torch.cuda.is_available() else 'cpu')
  model=MatchedCNN(source=False)
  if initialize:
   initial={k:v.clone() for k,v in model.state_dict().items()}
   state=torch.load(source,map_location='cpu',weights_only=False)
   assert state['mode']=='source'
   supplied=state['model']
   transferred=[k for k in initial if k.startswith(('encoder.','decoder.','head.'))]
   assert set(supplied)==set(transferred)
   model.load_state_dict(supplied,strict=False)
   actual=model.state_dict()
   assert all(torch.equal(actual[k],supplied[k]) for k in transferred)
   fresh=[k for k in initial if k.startswith(('side_embed.','film.'))]
   assert set(initial)==set(transferred+fresh)
   assert all(torch.equal(actual[k],initial[k]) for k in fresh)
   assert all(p.requires_grad for p in model.parameters())
  return model
 trainer.RUN=RUN;trainer.config=cfg_for;trainer.make_model=make
 write('status.json',{'status':'verifying','seed':SEED})
 try:
  protocol={'seed':SEED,'source_seed':142,'gpu':args.gpu,'transfer_scope':['encoder','decoder','head'],'fresh_target_modules':['side_embed','film'],'target_adaptation':'full','existing_test_known':True,
   'configuration':asdict(cfg_for(MODE,SEED)),
   'input_hashes':{str(p):shared.digest(p) for p in [Path(__file__),ROOT/'cnn_matched_train.py',ROOT/'cnn_matched_model.py',ROOT/'dino_replication.py',source,original_run/'target_manifest.json',original_run/'encoder_seed42/config.json',original_run/'base_seed42/best.pt',original_run/'encoder_seed42/best.pt']}}
  if (RUN/'protocol.json').exists():assert json.loads((RUN/'protocol.json').read_text())==protocol
  else:write('protocol.json',protocol)
  # Same target initialization and frozen schedule except transferred decoder/head.
  base=trainer.make_model(MODE,SEED,False);transferred=make(MODE,SEED)
  state=torch.load(source,map_location='cpu',weights_only=False)['model']
  assert all(torch.equal(transferred.state_dict()[k],state[k]) for k in state)
  fresh=[k for k in base.state_dict() if k.startswith(('side_embed.','film.'))]
  assert all(torch.equal(base.state_dict()[k],transferred.state_dict()[k]) for k in fresh)
  old=json.loads((original_run/'encoder_seed42/config.json').read_text())['training'];new=asdict(cfg_for(MODE,SEED))
  delta={k:[old[k],new[k]] for k in old if json.loads(json.dumps(old[k]))!=json.loads(json.dumps(new[k]))}
  assert set(delta)=={'save_dir','decoder_adaptation'},delta
  write('transfer_verification.json',{'status':'PASS','transferred_tensors':len(state),'fresh_tensors':len(fresh),'all_parameters_trainable':True,'config_differences':delta})
  del state,base,transferred
  if not (RUN/'smoke/verification.json').exists():trainer.fit(MODE,SEED,smoke=True)
  folder=RUN/f'{MODE}_seed{SEED}'
  write('status.json',{'status':'training','mode':MODE,'seed':SEED,'planned_target_runs':1})
  if not (folder/'summary.json').exists():trainer.fit(MODE,SEED,resume=(folder/'last.pt').exists())
  summary=json.loads((folder/'summary.json').read_text())
  assert summary['validation_reload_verified'] and shared.digest(folder/'best.pt')==summary['checkpoint_sha256']
  write('frozen_test_manifest.json',{'run':summary,'protocol_sha256':shared.digest(RUN/'protocol.json'),'source_checkpoint_sha256':shared.digest(source),'frozen_at':time.time()})
  write('status.json',{'status':'evaluating','mode':MODE,'seed':SEED})
  if not (folder/'test_metrics.json').exists():trainer.evaluate(MODE,SEED)
  observed=json.loads((folder/'test_metrics.json').read_text())
  rows=[]
  for label,path in [('BASE',original_run/'base_seed42/test_metrics.json'),('ENC',original_run/'encoder_seed42/test_metrics.json'),('ENC+DEC',folder/'test_metrics.json')]:
   r=json.loads(path.read_text());rows.append({'condition':label,'seed':SEED,'metrics':r['metrics'],'checkpoint_sha256':r['checkpoint_sha256']})
  assert observed['checkpoint_sha256']==summary['checkpoint_sha256']
  assert len({r['metrics']['valid_pixels'] for r in rows})==1
  write('comparison.json',{'rows':rows,'scope':'single source seed142 and target seed42; known test; within-CNN transfer-range comparison'})
  lines=['# CNN transfer-range comparison (target seed 42)','', '| Condition | Test MAE | Test RMSE | Test SSIM | Test PSNR |','| --- | --- | --- | --- | --- |']
  for r in rows:
   m=r['metrics'];lines.append(f"| {r['condition']} | {m['mae']:.8f} | {m['rmse']:.8f} | {m['ssim']:.8f} | {m['psnr']:.8f} |")
  (RUN/'results.md').write_text('\n'.join(lines)+'\n')
  write('status.json',{'status':'experiments_complete','target_runs':1,'seed':SEED,'test_evaluated':1,'manuscript_auto_updated':False})
 except Exception as exc:
  write('status.json',{'status':'failed','error':repr(exc),'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()

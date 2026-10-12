"""Fixed-budget DPM pretraining. Only train/validation map manifests are opened."""
import argparse,hashlib,json,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from baseline_model import RadioUNetBaseline
from data_pipeline import RESEARCH,SourceDataset,common_spatial_input
from train_baseline import write_json

@torch.no_grad()
def validate(model,loader,device):
 model.eval();groups=defaultdict(list)
 for batch in loader:
  pred=model(common_spatial_input(batch['inputs'],geometry_mode='image_proxy').to(device))
  values=(pred-batch['target'].to(device)).abs().mean((1,2,3)).cpu().tolist()
  for mid,v in zip(batch['metadata']['map_id'].tolist(),values):groups[mid].append(v)
 return float(np.mean([np.mean(v) for v in groups.values()]))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--steps',type=int,default=1500);parser.add_argument('--seed',type=int,default=0);parser.add_argument('--device',choices=['cuda','cpu'],default='cuda');args=parser.parse_args()
 if args.steps < 1 or args.seed < 0:parser.error('steps must be positive and seed nonnegative')
 if args.device=='cuda' and not torch.cuda.is_available():parser.error('CUDA unavailable')
 if args.output.exists() and any(args.output.iterdir()):raise ValueError('Run already exists')
 args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);torch.manual_seed(args.seed);np.random.seed(args.seed);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 train=SourceDataset('train');validation=SourceDataset('validation');selected=[i for i,r in enumerate(validation.rows) if int(r['tx_id']) in (0,40)]
 loader=DataLoader(train,batch_size=16,shuffle=True,generator=torch.Generator().manual_seed(args.seed),num_workers=2,persistent_workers=True)
 vloader=DataLoader(Subset(validation,selected),batch_size=16,num_workers=2,persistent_workers=True)
 model=RadioUNetBaseline(base_channels=16).to(args.device);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
 config={'domain':'RadioMapSeer_DPM','seed':args.seed,'base_channels':16,'batch_size':16,'max_updates':args.steps,'learning_rate':3e-4,'train_pairs':len(train),'validation_pairs':len(selected),'validation_policy':'Tx IDs 0 and 40 from every validation map; fixed source probe','validation_map_ids':sorted({int(validation.rows[i]['map_id']) for i in selected}),'loss':'whole_raster normalized grayscale MAE','test_accessed':False,'source_scope':'fixed-update propagation-supervised sensitivity; not fully converged source benchmark','code_sha256':{name:hashlib.sha256((RESEARCH/name).read_bytes()).hexdigest() for name in ['train_source.py','data_pipeline.py','baseline_model.py']},'input_sha256':{name:hashlib.sha256((RESEARCH/name).read_bytes()).hexdigest() for name in ['source_manifest.csv','splits/source_maps.json']}}
 write_json(args.output/'config.json',config);history=[];best=float('inf');step=0;iterator=iter(loader);start=time.perf_counter();losses=[]
 with (RESEARCH/'작업로그.md').open('a') as f:f.write(f'\n## {time.strftime("%Y-%m-%d")} — source pretraining 시작\n\n- Run: {args.output}; {args.steps} updates, seed{args.seed}, source train/validation only.\n')
 while step<args.steps:
  try:batch=next(iterator)
  except StopIteration:iterator=iter(loader);batch=next(iterator)
  model.train();pred=model(common_spatial_input(batch['inputs'],geometry_mode='image_proxy').to(args.device));loss=(pred-batch['target'].to(args.device)).abs().mean()
  if not torch.isfinite(loss):raise ValueError('Nonfinite source loss')
  optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();losses.append(loss.item());step+=1
  if step%250==0 or step==args.steps:
   metric=validate(model,vloader,args.device);record={'updates':step,'train_mae_window':float(np.mean(losses)),'validation_map_macro_mae':metric};losses=[];history.append(record);write_json(args.output/'history.json',history)
   if metric<best:
    best=metric;torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'config':config,'updates':step},args.output/'best.pt')
   print(json.dumps(record),flush=True)
 summary={'updates':step,'runtime_seconds':time.perf_counter()-start,'best_source_validation_mae':best,'test_accessed':False};write_json(args.output/'summary.json',summary)
 with (RESEARCH/'작업로그.md').open('a') as f:f.write(f'\n## {time.strftime("%Y-%m-%d")} — source pretraining 완료\n\n- Run: {args.output}; summary/checkpoint/history 저장. {step} updates; test 미접근.\n')
if __name__=='__main__':main()

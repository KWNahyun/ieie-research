"""Train one matched CNN source and six target runs, freeze and evaluate."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import hashlib

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/cnn_matched_v1'
PYTHON='/home/ailab/anaconda3/envs/emf_dinov2/bin/python'


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(name,value):
    path=RUN/name;tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,default=2);args=parser.parse_args()
    assert args.gpu in [1,2]
    inputs=[ROOT/n for n in ['cnn_matched_model.py','cnn_matched_train.py','cnn_matched_protocol.md','run_cnn_matched.py','analyze_cnn_matched.py','baseline_model.py','dino_replication.py']]
    inputs+=list((ROOT.parent/'EMF-DINOv2').rglob('*.py'))
    protocol={'input_hashes':{str(p):digest(p) for p in inputs},'seeds':[42,43,44],'source_seed':142,'gpu':args.gpu,
        'manifest_hashes':{s:digest(RUN/f'{s}_manifest.json') for s in ['source','target']}}
    for stage in ['source','target']:assert protocol['manifest_hashes'][stage]==digest(ROOT/'runs/dino_replication_v1'/f'{stage}_manifest.json')
    if (RUN/'protocol.json').exists():assert json.loads((RUN/'protocol.json').read_text())==protocol
    else:write('protocol.json',protocol)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(args.gpu),'OMP_NUM_THREADS':'2','MPLCONFIGDIR':'/tmp/ieie_matplotlib_cache','HF_HUB_OFFLINE':'1'}
    logs=RUN/'console_logs';logs.mkdir(exist_ok=True)
    records=[]
    for mode,seed in [('source',142)]+[(m,s) for s in [42,43,44] for m in ['base','encoder']]:
        folder=RUN/('source' if mode=='source' else f'{mode}_seed{seed}')
        if not (folder/'summary.json').exists():
            write('status.json',{'status':'training','mode':mode,'seed':seed,'completed':len(records),'planned':7})
            cmd=[PYTHON,str(ROOT/'cnn_matched_train.py'),'--action','fit','--mode',mode,'--seed',str(seed)]
            if (folder/'last.pt').exists():cmd+=['--resume']
            with (logs/f'{folder.name}.log').open('a') as stream:subprocess.run(cmd,cwd=ROOT.parent,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        record=json.loads((folder/'summary.json').read_text())
        assert record['mode']==mode and record['seed']==seed and record['validation_reload_verified']
        assert digest(folder/'best.pt')==record['checkpoint_sha256']
        record['folder']=str(folder);records.append(record);write('training_records.json',records)
        print('Verified',folder.name,len(records),'/7',flush=True)
    frozen={'runs':records[1:],'protocol_sha256':digest(RUN/'protocol.json'),'frozen_at':time.time()}
    if (RUN/'frozen_test_manifest.json').exists():
        prior=json.loads((RUN/'frozen_test_manifest.json').read_text());assert prior['runs']==frozen['runs'] and prior['protocol_sha256']==frozen['protocol_sha256']
    else:write('frozen_test_manifest.json',frozen)
    for r in records[1:]:
        folder=Path(r['folder'])
        if not (folder/'test_metrics.json').exists():subprocess.run([PYTHON,str(ROOT/'cnn_matched_train.py'),'--action','evaluate','--mode',r['mode'],'--seed',str(r['seed'])],cwd=ROOT.parent,env=env,check=True)
        assert json.loads((folder/'test_metrics.json').read_text())['checkpoint_sha256']==r['checkpoint_sha256']
    subprocess.run([PYTHON,str(ROOT/'analyze_cnn_matched.py')],cwd=ROOT.parent,check=True)
    write('status.json',{'status':'experiments_complete','source_runs':1,'target_runs':6,'test_evaluated':6})


if __name__=='__main__':main()

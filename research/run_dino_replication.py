"""Run a frozen source and twelve target fits before test evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/dino_replication_v1'
PYTHON='/home/ailab/anaconda3/envs/emf_dinov2/bin/python'
MODES=['base','encoder','decoder_lora','decoder_full']


def digest(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def write(name,value):
    path=RUN/name
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2))
    temp.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',type=int,default=1)
    args=parser.parse_args()
    assert args.gpu in [1,2], 'GPU0 reserved'
    RUN.mkdir(parents=True,exist_ok=True)
    for stage in ['source','target']: assert (RUN/f'{stage}_manifest.json').exists()
    paths=[ROOT/n for n in ['dino_replication.py','run_dino_replication.py','analyze_dino_replication.py','dino_replication_protocol.md']]
    paths+=list((ROOT.parent/'EMF-DINOv2').rglob('*.py'))
    protocol={'sha256':{str(p):digest(p) for p in paths},'target_seeds':[42,43,44],
        'modes':MODES,'source_seed':142,'gpu':args.gpu,
        'manifest_sha256':{s:digest(RUN/f'{s}_manifest.json') for s in ['source','target']},
        'scope':'supporting after existing test known; target replication conditional on one source'}
    if (RUN/'protocol.json').exists(): assert json.loads((RUN/'protocol.json').read_text())==protocol
    else: write('protocol.json',protocol)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(args.gpu),'OMP_NUM_THREADS':'2','HF_HUB_OFFLINE':'1','MPLCONFIGDIR':'/tmp/ieie_matplotlib_cache'}
    logs=RUN/'console_logs'; logs.mkdir(exist_ok=True)
    records=[]
    for mode,seed in [('source',142)]+[(m,s) for s in [42,43,44] for m in MODES]:
        folder=RUN/('source' if mode=='source' else f'{mode}_seed{seed}')
        if not (folder/'summary.json').exists():
            cmd=[PYTHON,str(ROOT/'dino_replication.py'),'--action','fit','--mode',mode,'--seed',str(seed)]
            if (folder/'last.pt').exists(): cmd+=['--resume']
            write('status.json',{'status':'training','mode':mode,'seed':seed,'completed':len(records),'planned':13})
            with (logs/f'{folder.name}.log').open('a') as stream:
                subprocess.run(cmd,cwd=ROOT.parent,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        r=json.loads((folder/'summary.json').read_text())
        assert r['mode']==mode and r['seed']==seed and r['validation_reload_verified']
        assert digest(folder/'best.pt')==r['checkpoint_sha256']
        r['folder']=str(folder)
        records.append(r)
        write('training_records.json',records)
        print('Verified',folder.name,len(records),'/13',flush=True)
    manifest={'runs':records[1:],'protocol_sha256':digest(RUN/'protocol.json'),'frozen_at_unix':time.time()}
    frozen=RUN/'frozen_test_manifest.json'
    if frozen.exists():
        prior=json.loads(frozen.read_text())
        assert prior['runs']==manifest['runs'] and prior['protocol_sha256']==manifest['protocol_sha256']
    else: write(frozen.name,manifest)
    for r in records[1:]:
        path=Path(r['folder'])/'test_metrics.json'
        if not path.exists():
            subprocess.run([PYTHON,str(ROOT/'dino_replication.py'),'--action','evaluate','--mode',r['mode'],'--seed',str(r['seed'])],env=env,cwd=ROOT.parent,check=True)
        assert json.loads(path.read_text())['checkpoint_sha256']==r['checkpoint_sha256']
    subprocess.run([PYTHON,str(ROOT/'analyze_dino_replication.py')],cwd=ROOT.parent,check=True)
    write('status.json',{'status':'experiments_complete','source_runs':1,'target_runs':12,'test_evaluated':12})


if __name__=='__main__': main()

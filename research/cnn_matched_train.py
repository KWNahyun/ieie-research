"""Same colleague data, masked loss, optimizer schedule and metrics for CNN."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from cnn_matched_model import MatchedCNN
import dino_replication as shared
from data.radio_maps import create_dataloaders,load_manifest
from losses.combined import build_loss
from train import set_seed,get_scheduler,train_one_epoch,validate

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/cnn_matched_v1'


def config(mode,seed):
    cfg=shared.config('source' if mode=='source' else 'base',seed)
    cfg.backbone='matched_conditioned_unet'
    cfg.init_source='cnn_source' if mode=='encoder' else 'scratch'
    cfg.decoder_type='unet'
    cfg.film_location='encoder_bottleneck'
    cfg.encoder_size=256 if mode=='source' else 504
    cfg.padding=0
    cfg.layers=(0,1,2,3)
    cfg.adaptation='full'
    cfg.source_ckpt=str(RUN/'source/best.pt') if mode=='encoder' else None
    if mode=='encoder':cfg.source_loss_type='masked_l1'
    cfg.manifest=str(RUN/('source_manifest.json' if mode=='source' else 'target_manifest.json'))
    cfg.save_dir=str(RUN)
    cfg.drop_rate=cfg.drop_path_rate=0.
    return cfg


def make_model(mode,seed,initialize=True):
    set_seed(seed,'cuda' if torch.cuda.is_available() else 'cpu')
    model=MatchedCNN(source=mode=='source')
    if mode=='encoder' and initialize:
        before={k:v.clone() for k,v in model.state_dict().items() if not k.startswith('encoder.')}
        state=torch.load(RUN/'source/best.pt',map_location='cpu',weights_only=False)
        assert state['mode']=='source'
        encoder={k.removeprefix('encoder.'):v for k,v in state['model'].items() if k.startswith('encoder.')}
        model.encoder.load_state_dict(encoder,strict=True)
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    return model


def save(path,model,optimizer,scheduler,generator,epoch,step,mode,cfg,best,patience):
    state={'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'generator':generator.get_state(),
        'epoch':epoch,'step':step,'mode':mode,'config':asdict(cfg),'best':best,'patience':patience,
        'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()}}
    temp=path.with_suffix('.pt.tmp')
    torch.save(state,temp);temp.replace(path)


def fit(mode,seed,resume=False,smoke=False):
    cfg=config(mode,seed)
    folder=RUN/('smoke' if smoke else 'source' if mode=='source' else f'{mode}_seed{seed}')
    if folder.exists() and any(folder.iterdir()) and not resume: raise RuntimeError('Preserve existing fit; explicit resume required')
    folder.mkdir(parents=True,exist_ok=resume)
    torch.set_num_threads(2)
    model=make_model(mode,seed,not resume).cuda()
    generator=torch.Generator().manual_seed(seed)
    train,val=create_dataloaders(cfg,generator)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    scheduler=get_scheduler(optimizer,cfg)
    criterion=build_loss(cfg).cuda()
    manifest=load_manifest(cfg.manifest)
    shared.write(folder/'config.json',{'training':asdict(cfg),'mode':mode,'smoke':smoke,'architecture':'matched CNN, linear output',
        'source_checkpoint_sha256':shared.digest(RUN/'source/best.pt') if mode=='encoder' else None})
    shared.write(folder/'manifest.json',manifest)
    if smoke:
        batch=next(iter(train)).to('cuda')
        prediction=model(batch)
        loss,_=criterion(prediction,batch.target,batch.valid_mask)
        assert prediction.shape==batch.target.shape and torch.isfinite(loss)
        loss.backward()
        assert model.side_embed.weight.grad.abs().sum()>0
        assert model.film[-1].weight.grad.abs().sum()>0
        shared.write(folder/'verification.json',{'status':'PASS','native_shape':list(prediction.shape),'loss':loss.item(),
            'pattern_gradient':True,'film_gradient':True,'same_dino_manifest':shared.digest(cfg.manifest)==shared.digest(shared.RUN/'target_manifest.json')})
        print('CNN native500 masked-loss backward PASS',flush=True)
        return
    start=step=patience=0;best=float('inf')
    if resume:
        state=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        assert state['config']==asdict(cfg) and state['mode']==mode
        model.load_state_dict(state['model'],strict=True)
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        generator.set_state(state['generator'])
        random.setstate(state['rng']['python']);np.random.set_state(state['rng']['numpy']);torch.set_rng_state(state['rng']['torch']);torch.cuda.set_rng_state(state['rng']['cuda'])
        start,step,best,patience=state['epoch'],state['step'],state['best'],state['patience']
        del state;gc.collect()
    for epoch in range(start,cfg.epochs):
        began=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        training=train_one_epoch(model,train,criterion,optimizer,cfg,epoch,torch.device('cuda'))
        metrics=validate(model,val,criterion,torch.device('cuda'))
        assert np.isfinite(metrics['mae'])
        improved=metrics['mae']<best;best=min(best,metrics['mae']);patience=0 if improved else patience+1
        step+=training['optimizer_steps'];scheduler.step()
        save(folder/'last.pt',model,optimizer,scheduler,generator,epoch+1,step,mode,cfg,best,patience)
        if improved: save(folder/'best.pt',model,optimizer,scheduler,generator,epoch+1,step,mode,cfg,best,patience)
        with (folder/'metrics.jsonl').open('a') as stream: stream.write(json.dumps({'epoch':epoch+1,'step':step,'train':training,'val':metrics,'seconds':time.perf_counter()-began,'peak_cuda_bytes':torch.cuda.max_memory_allocated()})+'\n')
        print(mode,seed,'epoch',epoch+1,'val_mae',metrics['mae'],flush=True)
        if epoch+1>=cfg.es_min_epoch and patience>=cfg.early_stopping_patience: break
    selected=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(selected['model'],strict=True)
    reloaded=validate(model,val,criterion,torch.device('cuda'))
    assert abs(reloaded['mae']-best)<1e-7
    shared.write(folder/'summary.json',{'mode':mode,'seed':seed,'selected_epoch':selected['epoch'],'best_validation_mae':best,
        'epochs':epoch+1,'epoch_cap':epoch+1==cfg.epochs,'validation_reload_verified':True,'checkpoint_sha256':shared.digest(folder/'best.pt'),'test_accessed':False})


def evaluate(mode,seed):
    folder=RUN/f'{mode}_seed{seed}'
    cfg=config(mode,seed)
    state=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    assert state['config']==asdict(cfg)
    model=make_model(mode,seed,False).cuda();model.load_state_dict(state['model'],strict=True)
    loader,=create_dataloaders(cfg,splits=('test',))
    metrics=validate(model,loader,build_loss(cfg).cuda(),torch.device('cuda'))
    shared.write(folder/'test_metrics.json',{'mode':mode,'seed':seed,'metrics':metrics,'units':'PNG/255',
        'checkpoint_sha256':shared.digest(folder/'best.pt'),'scope':'supporting single-source conditional'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--action',choices=['fit','smoke','evaluate'],required=True)
    parser.add_argument('--mode',choices=['source','base','encoder'],default='base');parser.add_argument('--seed',type=int,default=42);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.action=='evaluate':evaluate(args.mode,args.seed)
    else:fit(args.mode,args.seed,args.resume,args.action=='smoke')


if __name__=='__main__':main()

"""Independent run adapter; colleague code is imported without modification."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
TEAM = ROOT.parent / 'EMF-DINOv2'
sys.path.insert(0, str(TEAM))
import numpy as np
import torch
import timm
import cv2
from configs.presets import get_preset, TrainingConfig
from data.radio_maps import prepare_manifest, load_manifest, create_dataloaders
from models.factory import build_model
from losses.combined import build_loss
from train import train_one_epoch, validate, get_scheduler, set_seed
from utils.checkpoint import save_checkpoint, load_checkpoint, export_encoder, load_source_model

RUN = ROOT / 'runs/dino_replication_v1'
TARGET = Path('/home/ailab/Desktop/data/emf_data')
RAW_SOURCE = ROOT.parent / 'data/RadioMapSeer'
SOURCE = RUN / 'source_view'
WEIGHTS = Path('/home/ailab/.cache/huggingface/hub/models--timm--vit_base_patch14_reg4_dinov2.lvd142m/snapshots/3b06466a5548c52b8b98822e1390987695fcbf82/model.safetensors')
original_create = timm.create_model


def local_create(*args, **kwargs):
    if kwargs.get('pretrained'):
        kwargs['pretrained_cfg_overlay'] = {'file':str(WEIGHTS), 'hf_hub_id':None}
    return original_create(*args, **kwargs)


timm.create_model = local_create


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temp.replace(path)


def config(mode, seed):
    preset = {'source':'SRC_L1','base':'CT_BASE_T_L1','encoder':'CT_RMS_L1_T_L1',
              'decoder_full':'CT_RMS_L1_T_L1','decoder_lora':'CT_RMS_L1_D_LORA_T_L1'}[mode]
    source = mode == 'source'
    checkpoint = None if mode in ['source','base'] else str(RUN/'source'/('best.pt' if mode=='decoder_lora' else 'best_encoder.pt'))
    return get_preset(preset, seed=seed, batch_size=8, num_workers=2,
        data_dir=str(SOURCE if source else TARGET), manifest=str(RUN/('source_manifest.json' if source else 'target_manifest.json')),
        save_dir=str(RUN), source_ckpt=checkpoint)


def model_for(cfg, mode, initialize=True):
    model = build_model(cfg, pretrained=initialize, load_source=initialize)
    if mode == 'decoder_full' and initialize:
        # The encoder was already strictly loaded BEFORE LoRA wrapping by the
        # colleague constructor. Reloading it after wrapping changes key names.
        path=RUN/'source/best.pt'
        state=load_checkpoint(path)
        source_cfg=TrainingConfig(**state['config']).validate()
        assert source_cfg.stage=='source' and source_cfg.run_name=='SRC_L1'
        assert state['evaluation_kind']=='raw' and state['ema'] is None
        exported=torch.load(RUN/'source/best_encoder.pt',map_location='cpu',weights_only=False)
        assert all(torch.equal(value,state['model']['encoder.'+key]) for key,value in exported['encoder'].items())
        decoder={key.removeprefix('decoder.'):value for key,value in state['model'].items() if key.startswith('decoder.')}
        model.decoder.load_state_dict(decoder,strict=True)
        model.source_lineage.update(decoder_adaptation='source_full',transfer_scope='encoder+decoder',
            full_source_path=str(path),full_source_sha256=digest(path),source_epoch=state['epoch'])
    if mode == 'decoder_full':
        model.decoder.requires_grad_(True)
        assert all(p.requires_grad for p in model.decoder.parameters())
    return model


def prepare():
    RUN.mkdir(parents=True, exist_ok=True)
    # The unsuffixed directories contain an older transformed subset. The
    # previously audited *_orig assets match JSON positions and DPM labels.
    SOURCE.mkdir(parents=True, exist_ok=True)
    for relative, original in [('png/buildings_complete','png/buildings_complete_orig'),
                               ('png/antennas','png/antennas_orig'),('gain','gain')]:
        link = SOURCE/relative
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists(): link.symlink_to(RAW_SOURCE/original, target_is_directory=True)
        assert link.resolve()==(RAW_SOURCE/original).resolve()
    checked = 0
    with (ROOT/'source_manifest.csv').open() as stream:
        for row in csv.DictReader(stream):
            if int(row['map_id']) >= 700: continue
            filename = f"{row['map_id']}_{row['tx_id']}.png"
            marker = np.zeros((256,256), dtype=np.uint8)
            marker[int(row['tx_row']), int(row['tx_column'])] = 255
            original = RAW_SOURCE/'png/antennas_orig'/filename
            assert np.array_equal(cv2.imread(str(original),0), marker)
            checked += 1
    write(RUN/'source_view_provenance.json',{'coordinate_manifest_sha256':digest(ROOT/'source_manifest.csv'),
        'existing_png_matches':checked,'generated_pngs':0,
        'rule':'unaltered *_orig PNGs; all 56000 positions match audited row=255-json_y,col=json_x; raw source untouched'})
    for stage, root in [('target',TARGET),('source',SOURCE)]:
        path = RUN/f'{stage}_manifest.json'
        if not path.exists():
            prepare_manifest(root,stage,path)
        manifest = load_manifest(path)
        print(stage, {k:len(v) for k,v in manifest['splits'].items()}, flush=True)


def fit(mode, seed, smoke=False, resume=False):
    cfg = config(mode,seed)
    folder = RUN/('smoke' if smoke else 'source' if mode=='source' else f'{mode}_seed{seed}')
    if folder.exists() and any(folder.iterdir()) and not resume:
        raise RuntimeError(f'Existing run requires explicit recovery: {folder}')
    folder.mkdir(parents=True, exist_ok=resume)
    set_seed(seed,'cuda')
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(seed)
    train_loader, val_loader = create_dataloaders(cfg,generator)
    model = model_for(cfg,mode,initialize=not resume).cuda()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg.lr,weight_decay=cfg.weight_decay)
    scheduler = get_scheduler(optimizer,cfg)
    criterion = build_loss(cfg).cuda()
    manifest = load_manifest(cfg.manifest)
    metadata = {k:manifest[k] for k in ['manifest_id','inventory_id','normalization_id']}
    metadata.update(comparison_mode=mode,source_lineage=model.source_lineage,
        pretrained_weights_sha256=digest(WEIGHTS),scope='supporting after existing test was known',
        parameter_count=sum(p.numel() for p in model.parameters()),
        trainable_count=sum(p.numel() for p in model.parameters() if p.requires_grad),test_accessed=False)
    write(folder/'config.json',{'training_config':asdict(cfg),'comparison_mode':mode,'smoke':smoke})
    write(folder/'manifest.json',manifest)
    if smoke:
        batch = next(iter(train_loader)).to('cuda')
        pred = model(batch)
        loss,_ = criterion(pred,batch.target,batch.valid_mask)
        assert pred.shape == batch.target.shape and torch.isfinite(loss)
        loss.backward()
        assert model.side_embed.weight.grad.abs().sum() > 0
        write(folder/'verification.json',{'finite_loss':loss.item(),'native_shape':list(pred.shape),'pattern_gradient':True})
        print('Native target GPU forward/backward smoke PASS',flush=True)
        return
    best = float('inf')
    patience = step = 0
    start=0
    if resume:
        state=load_checkpoint(folder/'last.pt',model,optimizer,scheduler,None,generator,
            restore_rng=True,expected_config=asdict(cfg))
        assert state['metadata']['comparison_mode']==mode
        metadata=state['metadata']
        start,step,best,patience=state['epoch'],state['step'],metadata['best_mae'],metadata['patience']
    for epoch in range(start,cfg.epochs):
        began=time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        training=train_one_epoch(model,train_loader,criterion,optimizer,cfg,epoch,torch.device('cuda'))
        validation=validate(model,val_loader,criterion,torch.device('cuda'))
        score=validation['mae']
        assert score is not None and np.isfinite(score)
        improved=score < best
        best=min(best,score)
        patience=0 if improved else patience+1
        step+=training['optimizer_steps']
        scheduler.step()
        metadata.update(best_mae=best,patience=patience,epoch_seconds=time.perf_counter()-began,
            peak_cuda_bytes=torch.cuda.max_memory_allocated())
        save_checkpoint(folder/'last.pt',model,optimizer,scheduler,None,asdict(cfg),epoch+1,step,metadata,generator)
        if improved:
            save_checkpoint(folder/'best.pt',model,optimizer,scheduler,None,asdict(cfg),epoch+1,step,metadata,generator)
            if mode=='source': export_encoder(model,folder/'best_encoder.pt',metadata)
        with (folder/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps({'epoch':epoch+1,'step':step,'train':training,'val':validation,'resources':metadata})+'\n')
        print(mode,seed,'epoch',epoch+1,'val_mae',score,'seconds',metadata['epoch_seconds'],flush=True)
        if epoch+1>=cfg.es_min_epoch and patience>=cfg.early_stopping_patience: break
    state=load_checkpoint(folder/'best.pt')
    model.load_state_dict(state['model'],strict=True)
    reloaded=validate(model,val_loader,criterion,torch.device('cuda'))
    assert abs(reloaded['mae']-best)<1e-7
    write(folder/'summary.json',{'mode':mode,'seed':seed,'epochs':epoch+1,'selected_epoch':state['epoch'],
        'best_validation_mae':best,'validation_reload_verified':True,'epoch_cap':epoch+1==cfg.epochs,
        'checkpoint_sha256':digest(folder/'best.pt'),'test_accessed':False})


def evaluate_folder(folder):
    state=load_checkpoint(folder/'best.pt')
    cfg=TrainingConfig(**state['config']).validate()
    mode=state['metadata']['comparison_mode']
    model=model_for(cfg,mode,initialize=False).cuda()
    model.load_state_dict(state['model'],strict=True)
    loader,=create_dataloaders(cfg,splits=('test',))
    metrics=validate(model,loader,build_loss(cfg).cuda(),torch.device('cuda'))
    write(folder/'test_metrics.json',{'metrics':metrics,'checkpoint_sha256':digest(folder/'best.pt'),
        'comparison_mode':mode,'seed':cfg.seed,'units':'PNG/255','scope':'supporting/exploratory'})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--action',choices=['prepare','fit','smoke','evaluate'],required=True)
    parser.add_argument('--mode',choices=['source','base','encoder','decoder_lora','decoder_full'],default='base')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.action=='prepare': prepare()
    elif args.action in ['fit','smoke']: fit(args.mode,args.seed,args.action=='smoke',args.resume)
    else: evaluate_folder(RUN/f'{args.mode}_seed{args.seed}')


if __name__=='__main__': main()

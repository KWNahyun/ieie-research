"""Check all saved pilot checkpoints/configs on CPU; never open datasets."""
import hashlib,json
from pathlib import Path
import torch
from baseline_model import RadioUNetBaseline
from data_pipeline import RESEARCH
from train_baseline import write_json

def main():
 root=RESEARCH/'runs/pilot_v1';records=json.loads((root/'results.json').read_text());assert len(records)==24
 source=root/'source_DPM_seed0/best.pt';source_hash=hashlib.sha256(source.read_bytes()).hexdigest();contract_hash=hashlib.sha256((root/'inputs_snapshot/data_contract.json').read_bytes()).hexdigest();verified=[];torch.set_num_threads(2)
 for r in records:
  folder=root/f"{r['model']}_budget{r['budget']}_seed{r['seed']}";config=json.loads((folder/'config.json').read_text());state=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
  assert state['config']==config
  assert config['data_contract_sha256']==contract_hash
  assert config['actual_train_count']==len(config['actual_train_base_ids'])
  assert config['actual_validation_base_ids'] and config['validation_count']==108
  assert config['source_checkpoint_sha256']==(source_hash if r['model'] in 'CD' else None)
  model=RadioUNetBaseline(base_channels=16,conditioned=r['model'] in 'BD');model.load_state_dict(state['model'],strict=True)
  assert all(torch.isfinite(v).all() for v in state['model'].values())
  assert config['parameter_count']==sum(p.numel() for p in model.parameters())
  with torch.no_grad():prediction=model(torch.zeros(1,2,32,32),torch.zeros(1,7),torch.zeros(1,2,360))
  assert prediction.shape==(1,1,32,32) and torch.isfinite(prediction).all()
  verified.append(folder.name)
 source_state=torch.load(source,map_location='cpu',weights_only=True);assert source_state['config']['domain']=='RadioMapSeer_DPM'
 model=RadioUNetBaseline(base_channels=16);model.load_state_dict(source_state['model'],strict=True)
 report={'target_checkpoints_verified':len(verified),'runs':verified,'source_checkpoint_verified':True,'source_checkpoint_sha256':source_hash,'snapshot_contract_hash_verified':True,'checks':'strict load, finite weights/synthetic inference, exact config, parameter counts, subset count, source hash','dataset_test_accessed':False}
 write_json(root/'artifact_verification.json',report)
 with (RESEARCH/'작업로그.md').open('a') as f:f.write('\n## 2026-10-08 — Pilot checkpoint 산출물 검증\n\n- Source와 24개 target best.pt를 CPU에서 weights_only/strict load하고 finite weights/synthetic prediction, config, parameter count, subset count, snapshot contract 및 source hash를 검사하여 통과했다.\n- 생성: runs/pilot_v1/artifact_verification.json. Dataset/test를 읽지 않았다.\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()

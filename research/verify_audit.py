"""Verify completed audits and absence of base/building/D4 split leakage."""
from pathlib import Path
import csv,json
P=Path(__file__).resolve().parent
def read(name):return json.loads((P/name).read_text())
def main():
 a=read('augmentation_audit.json');c=read('conditioning_audit.json');s=read('source_audit.json');split=read('splits/target_groups.json');subsets=read('splits/target_subsets.json')
 assert all(not v['nonfinite_entries'] for v in a['global_statistics'].values())
 assert all(not v['failed_indices'] for v in a['raw_correspondence'].values())
 for k,v in a['checks'].items():
  if k!='numeric_to_img_flip':assert not v['failed_indices'],k
 assert not c['mask_equals_y_positive']['failed_indices']
 assert not c['legacy_beam_regeneration']['failed_indices']
 assert not s['failed_pairs'] and s['pairs']==56080 and set(s['pairs_per_map'].values())=={80}
 manifest=list(csv.DictReader((P/'target_manifest.csv').open()));assert len(manifest)==3072
 assert len({int(r['entry_index']) for r in manifest})==3072
 bybase={}
 for r in manifest:bybase.setdefault(r['base_id'],[]).append(r)
 assert len(bybase)==768
 for entries in bybase.values():
  assert len(entries)==4 and len({r['split'] for r in entries})==1
  assert {r['transform'] for r in entries}=={'identity','horizontal','vertical','combined'}
  assert sum(r['use_for_evaluation']=='True' for r in entries)==1
  for r in entries:
   assert (r['use_for_training']=='True')==(r['transform']=='identity' and r['split']=='train')
 for key in ['building_id','geometry_group_id']:
  seen={}
  for r in manifest:
   assert r[key] not in seen or seen[r[key]]==r['split'];seen[r[key]]=r['split']
 for seed,budgets in subsets['seeds'].items():
  last=set()
  for pct in ['10','25','50','100']:
   current=set(budgets[pct]['base_ids']);assert last<=current<=set(split['base_ids']['train']);last=current
  assert last==set(split['base_ids']['train'])
 source_split=read('splits/source_maps.json')['map_ids'];sets=[set(v) for v in source_split.values()]
 assert len(set.union(*sets))==701 and sum(map(len,sets))==701
 print('PASS: provenance/finite checks, beam regeneration, base/group split isolation, nested subsets, source map isolation.')
 with (P/'작업로그.md').open('a') as f:f.write('\n## 2026-10-08 — 감사 산출물 검증\n\n- 실행: verify_audit.py. Target 전체 provenance/finite, beam 재생성, base의 4개 변형과 split 일관성, building/group 분리, 5개 seed 중첩 subset, source map 분리를 검사했다.\n- 결과: PASS. 이 검사는 지리 tile 중복 부재나 augmentation의 물리 타당성을 증명하지 않는다.\n')
if __name__=='__main__':main()

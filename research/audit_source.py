"""Audit local DPM triples without changing source data. Produces candidate manifest."""
from pathlib import Path
from collections import defaultdict
import csv,hashlib,json,struct
import numpy as np
import cv2
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent/'data/RadioMapSeer'
def sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def canonical(a):return min(sha(np.rot90(b,k)) for b in (a,np.fliplr(a)) for k in range(4))
def png_shape(p):
 with p.open('rb') as f: h=f.read(24)
 if h[:8]!=b'\x89PNG\r\n\x1a\n': raise ValueError(str(p))
 w,h=struct.unpack('>II',h[16:24]); return h,w

def main():
 gains=list((ROOT/'gain/DPM').glob('*.png')); failures=[]; positions=[]; maps={}; rows=[]; target=np.load('/home/ailab/Desktop/data/preprocessing/building_mask.npy',mmap_mode='r'); target_keys={canonical((1-target[i]).astype(np.uint8)) for i in range(768)}
 for f in sorted(gains,key=lambda p:tuple(map(int,p.stem.split('_')))):
  mid,tid=map(int,f.stem.split('_')); bp=ROOT/'png/buildings_complete_orig'/f'{mid}.png'; ap=ROOT/'png/antennas_orig'/f.name; jp=ROOT/'antenna'/f'{mid}.json'
  if mid not in maps:
   b=cv2.imread(str(bp),0);coords=json.loads(jp.read_text());maps[mid]={'shape':list(b.shape),'binary':bool(np.isin(b,[0,255]).all()),'coords':coords,'geometry':canonical((b>0).astype(np.uint8))}
  m=maps[mid];x,y=m['coords'][tid]; a=cv2.imread(str(ap),0)
  checks={'building_binary':m['binary'],'gain_shape':png_shape(f)==tuple(m['shape']),'antenna_shape':a is not None and a.shape==tuple(m['shape']),'antenna_position':a is not None and np.count_nonzero(a)==1 and a[255-y,x]==255,'coordinate_bounds':0<=x<256 and 0<=y<256}
  if not all(checks.values()):failures.append({'file':f.name,'checks':checks})
  rows.append({'domain':'RadioMapSeer','simulator':'DPM','map_id':mid,'tx_id':tid,'building_path':str(bp),'antenna_path':str(ap),'gain_path':str(f),'antenna_json_path':str(jp),'tx_column':x,'tx_row':255-y,'alignment':'column=x,row=255-y; exact antenna PNG verified' if checks['antenna_position'] else 'FAILED','status':'candidate; polygon/building meaning and cross-domain overlap unresolved'})
  if len(rows)%10000==0:print('source pairs',len(rows),flush=True)
 # Keep exact-equivalent map geometries in a common split.
 bygeom=defaultdict(list)
 for mid,m in maps.items():bygeom[m['geometry']].append(mid)
 groups=sorted(bygeom.values(),key=min); perm=np.random.default_rng(42).permutation(len(groups)).tolist();a=int(.7*len(groups));b=int(.15*len(groups)); splits={'train':perm[:a],'validation':perm[a:a+b],'test':perm[a+b:]}; assign={mid:s for s,gs in splits.items() for g in gs for mid in groups[g]}
 for r in rows:r['split']=assign[r['map_id']]
 with (OUT/'source_manifest.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 report={'simulator':'DPM','pairs':len(rows),'map_count':len(maps),'map_ids':[min(maps),max(maps)],'pairs_per_map':{str(mid):sum(r['map_id']==mid for r in rows) for mid in maps},'failed_pairs':failures,'geometry_groups':len(groups),'cross_domain_overlap_check':{'method':'exact D4 binary occupancy hash; no resizing or geographic registration','matches':[mid for mid,m in maps.items() if m['geometry'] in target_keys],'limitation':'different 256/500 raster sizes prevent exact matches; geographic overlap NOT resolved'},'gain_values':'PNG grayscale only; simulator-to-physical-unit conversion unverified','building_occupancy':'nonzero PNG candidate; polygon alignment remains to be checked'}
 (OUT/'source_audit.json').write_text(json.dumps(report,indent=2)+'\n')
 (OUT/'splits/source_maps.json').write_text(json.dumps({'seed':42,'status':'provisional pending physical provenance and geographic overlap checks','geometry_groups':groups,'split_group_ids':splits,'map_ids':{s:[mid for g in gs for mid in groups[g]] for s,gs in splits.items()}},indent=2)+'\n')
 print({k:report[k] for k in ('pairs','map_count','map_ids','geometry_groups')},'failed',len(failures),flush=True)
 with (OUT/'작업로그.md').open('a') as f:f.write('\n## 2026-10-08 — source DPM 감사 완료\n\n- 실행: audit_source.py. 모든 DPM 파일의 대응 건물·antenna JSON/PNG·gain header를 검사했다.\n- 생성: source_manifest.csv, source_audit.json, splits/source_maps.json.\n- 제한: grayscale의 물리값 변환, polygon 의미, source/target 지리 중복은 미검증이며 manifest/split은 candidate/provisional이다.\n')
if __name__=='__main__':main()

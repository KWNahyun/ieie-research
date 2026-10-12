"""Check mask dependence, regenerate legacy beam, and inspect reflection symmetry."""
from pathlib import Path
import math,json
import numpy as np
import cv2
from shapely.geometry import LineString,Point
from audit_target import excel_rows,RAW,DATA,OUT

def beam(n,width,legacy_marker=True):
 origin=(int(n[0]),int(n[1]));tmp=np.zeros((500,500,1),np.float64)
 if legacy_marker:tmp[origin[0],origin[1],0]=255
 else:tmp[origin[1],origin[0],0]=255
 angle=360-((450-n[5])%360); angles=[angle,angle-width/2,angle+width/2]
 boundaries=[LineString(pair) for pair in [[(0,0),(0,500)],[(0,500),(500,500)],[(500,500),(500,0)],[(500,0),(0,0)]]]
 intersections=[]
 for a in angles:
  a=math.radians(a);line=LineString([origin,(math.cos(a)*1000+origin[0],math.sin(a)*1000+origin[1])])
  for idx,b in enumerate(boundaries):
   point=line.intersection(b)
   if not point.is_empty:intersections.append((idx,list(point.coords)[0]));break
 center,p1,p2=intersections;points=[origin,p1[1],p2[1]]
 for side in (p1,p2):
  if center[0]!=side[0]:
   start=min(center[0],side[0]); diff=abs(center[0]-side[0])
   for i in range(diff):
    pt=boundaries[start+i].intersection(boundaries[start+i+1])
    if not pt.is_empty:points.append(list(pt.coords)[0])
 for i in range(len(points)-1):
  distances=[Point(points[i]).distance(Point(points[j])) for j in range(i+1,len(points))]
  selected=i+1+np.argsort(distances)[0];points[i+1],points[selected]=points[selected],points[i+1]
 return cv2.fillPoly(tmp,np.array([points],np.int32),255)[:,:,0]/255

def main():
 a={n:np.load(DATA/(n+'.npy'),mmap_mode='r') for n in ['building_mask','y','x_numeric','x_pattern_hori','numeric_to_img']}
 rows=excel_rows(RAW/'antenna_configuration.xlsx');maskfail=[];nonbinary=[];beamfail=[];marker_effect=[]
 for i in range(3072):
  m=a['building_mask'][i]
  if not np.array_equal(m,a['y'][i]>0):maskfail.append(i)
  if not np.isin(m,[0,1]).all():nonbinary.append(i)
  base=i if i<768 else (i-768)//3; width=float(rows[base%256]['antenna_pattern'].split('_')[2])
  expected=beam(a['x_numeric'][i],width);stored=a['numeric_to_img'][i,:,:,0]
  if not np.array_equal(expected,stored):beamfail.append(i)
  if not np.array_equal(expected,beam(a['x_numeric'][i],width,False)):marker_effect.append(i)
  if i%768==0:print('conditioning entries',i,flush=True)
 symmetric=[];asymmetric=[]
 for i in range(768):
  h=a['x_pattern_hori'][i];mirror=h[(-np.arange(360))%360]
  (symmetric if np.allclose(h,mirror,rtol=0,atol=1e-10) else asymmetric).append(i)
 # Inspect whether fixed geometry has setting-dependent mask pixels.
 import csv
 manifest=list(csv.DictReader((OUT/'target_manifest.csv').open()));groups={}
 for rec in manifest[:768]:groups.setdefault(rec['building_id'],[]).append(int(rec['base_id']))
 varying=[]
 for bid,ids in groups.items():
  if any(not np.array_equal(a['building_mask'][ids[0]],a['building_mask'][i]) for i in ids[1:]):varying.append(bid)
 report={'mask_equals_y_positive':{'tested':3072,'failed_indices':maskfail},'binary_mask_failures':nonbinary,'buildings_with_setting_dependent_masks':varying,'legacy_beam_regeneration':{'tested':3072,'failed_indices':beamfail,'formula_source':'emf_project/ViT/utils/data.py: origin, angle2azimuth, intersections, nearest-neighbor ordering, cv2.fillPoly'},'transposed_marker_changes_beam_entries':marker_effect,'horizontal_pattern_reflection':{'symmetric_base_count':len(symmetric),'asymmetric_base_count':len(asymmetric),'asymmetric_base_ids':asymmetric,'implication':'Repeating an asymmetric local pattern while mirroring scene/azimuth is not a verified physical reflection. Use base-only train until reflection convention is resolved.'}}
 (OUT/'conditioning_audit.json').write_text(json.dumps(report,indent=2)+'\n')
 print('mask mismatch',len(maskfail),'beam mismatch',len(beamfail),'marker changes',len(marker_effect),'asymmetric',len(asymmetric),'varying masks',len(varying),flush=True)
 with (OUT/'작업로그.md').open('a') as f:f.write('\n## 2026-10-08 — conditioning 감사 완료\n\n- 실행: audit_conditioning.py. 전체 3072개의 mask와 y>0 관계, legacy beam 재생성, transpose marker 영향, 768개 수평 pattern의 반사 대칭을 검사했다.\n- 생성: conditioning_audit.json. 동일 building 내 mask 변동도 기록했다.\n- 주의: label 기반 mask를 독립 occupancy 입력으로 사용하지 않는다. Pattern augmentation의 물리 정합 확인 전에는 base-only 학습을 기본으로 둔다.\n')
if __name__=='__main__':main()

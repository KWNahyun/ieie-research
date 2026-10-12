"""Read-only full target audit; writes reports/manifests beside this script.
Run: python IEIE/research/audit_target.py
No training, no execution of legacy preprocessing code.
"""
from pathlib import Path
from collections import defaultdict
import csv, hashlib, json, zipfile, xml.etree.ElementTree as ET
import numpy as np
import cv2

OUT=Path(__file__).resolve().parent
DATA=Path('/home/ailab/Desktop/data/preprocessing')
RAW=Path('/home/ailab/Desktop/data/emf_data')
NS={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}

def excel_rows(path):
    with zipfile.ZipFile(path) as z:
        strings=[''.join(si.itertext()) for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',NS)]
        table=[]
        for row in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('s:sheetData/s:row',NS):
            cells={}
            for c in row:
                v=c.find('s:v',NS)
                value=v.text if v is not None else ''.join(c.itertext())
                cells[''.join(filter(str.isalpha,c.attrib['r']))]=strings[int(value)] if c.attrib.get('t')=='s' else value
            table.append(cells)
        header=table[0]
        return [{header[k]:v for k,v in row.items()} for row in table[1:]]

def pattern(path):
    sections={}; current=None; gain=None
    for line in path.read_text().splitlines():
        parts=line.split()
        if not parts: continue
        if parts[0]=='GAIN': gain=float(parts[1])
        elif parts[0] in ('HORIZONTAL','VERTICAL'): current=parts[0]; sections[current]=[]
        elif current:
            try: sections[current].append([float(parts[0]),float(parts[-1])])
            except ValueError: current=None
    result={}
    for key,values in sections.items():
        values=np.asarray(values)
        assert values.shape==(360,2) and np.array_equal(values[:,0],np.arange(360)), (path,key)
        result[key]=gain-values[:,1]+40
    return result

def digest(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def d4(a): return min(digest(np.rot90(b,k)) for b in (a,np.fliplr(a)) for k in range(4))
def save(name,value): (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def main():
    rows=excel_rows(RAW/'antenna_configuration.xlsx'); assert len(rows)==256
    arrays={p.stem:np.load(p,mmap_mode='r') for p in DATA.glob('*.npy')}
    assert all(len(a)==3072 for a in arrays.values())
    audit={'entry_count':3072,'base_count':768,'spreadsheet_rows':len(rows),'checks':{},'global_statistics':{}}
    for name,a in arrays.items():
        lo=np.inf; hi=-np.inf; bad=[]
        for i in range(len(a)):
            s=a[i]; lo=min(lo,float(s.min())); hi=max(hi,float(s.max()))
            if not np.isfinite(s).all(): bad.append(i)
        audit['global_statistics'][name]={'min':lo,'max':hi,'nonfinite_entries':bad,'shape':list(a.shape),'dtype':str(a.dtype)}
        print('scanned',name,flush=True)
    # Exact spatial transforms; float64 serialization, so no rounding.
    for name in ('x_image','y','building_mask','numeric_to_img','1_1_x_image'):
        a=arrays[name]; failures=[]
        for i in range(768):
            for j,axes in enumerate(((1,),(0,),(0,1))):
                idx=768+3*i+j
                if not np.array_equal(np.flip(a[i],axis=axes),a[idx]): failures.append(idx)
        audit['checks'][name+'_flip']={'tested':2304,'failed_indices':failures}
        print('flip',name,len(failures),flush=True)
    for name in ('x_pattern_hori','x_pattern_vert'):
        a=arrays[name]; failures=[]
        for i in range(768):
            for j in range(3):
                idx=768+3*i+j
                if not np.array_equal(a[i],a[idx]): failures.append(idx)
        audit['checks'][name+'_repeat']={'tested':2304,'failed_indices':failures}
    failures=[]
    for i in range(768):
        for j in range(3):
            expected=arrays['x_numeric'][i].copy()
            if j in (0,2): expected[0]=500-expected[0]; expected[5]=360-expected[5]
            if j in (1,2): expected[1]=500-expected[1]; expected[5]=(540-expected[5])%360
            idx=768+3*i+j
            if not np.allclose(expected,arrays['x_numeric'][idx],rtol=0,atol=1e-10): failures.append(idx)
    audit['checks']['numeric_flip']={'tested':2304,'failed_indices':failures,'atol':1e-10}
    raw_fail={k:[] for k in ('building','y','numeric','hori','vert')}; cache={}; records=[]
    geometry={}; mask_fail={k:[] for k in ('binary','channel0_nonzero','channel0_zero','channel1_zero','both_zero')}
    for i in range(768):
        r=rows[i%256]; ant=i//256+1; bid=r['building_file'].split('.')[0]
        bp=RAW/'input_image_resize'/f'{bid}.png'
        yp=RAW/'output_image_resize'/f"{r['분류']}-building_num{bid}_Antenna {ant} Field Strength.png"
        building=cv2.imread(str(bp),cv2.IMREAD_COLOR); label=cv2.imread(str(yp),cv2.IMREAD_GRAYSCALE)
        if building is None or not np.array_equal(building[:,:,:2]/255,arrays['x_image'][i]): raw_fail['building'].append(i)
        if label is None or not np.array_equal(label/255,arrays['y'][i]): raw_fail['y'].append(i)
        coord=list(map(float,r[f'ant{ant}-coord'].split(',')))
        expected=[coord[1],500-coord[0],coord[2]]+[float(r[f'ant{ant}-{k}']) for k in ('downtilt','power','azimuth')]
        if not np.allclose(expected,arrays['x_numeric'][i],rtol=0,atol=1e-10): raw_fail['numeric'].append(i)
        pn=r['antenna_pattern']
        if pn not in cache: cache[pn]=pattern(RAW/'radi_pattern'/pn)
        for k,field in [('hori','HORIZONTAL'),('vert','VERTICAL')]:
            if not np.allclose(cache[pn][field],arrays['x_pattern_'+k][i],rtol=0,atol=1e-10):raw_fail[k].append(i)
        x=arrays['x_image'][i]; m=arrays['building_mask'][i]
        predicates={'binary':np.isin(m,[0,1]),'channel0_nonzero':m==(x[:,:,0]!=0),'channel0_zero':m==(x[:,:,0]==0),'channel1_zero':m==(x[:,:,1]==0),'both_zero':m==np.all(x==0,axis=-1)}
        for k,v in predicates.items():
            if not v.all():mask_fail[k].append(i)
        geometry[i]=d4(np.rint(x*255).astype(np.uint8))
        records.append({'base_id':i,'configuration_row':i%256,'run_id':r['분류'],'building_id':bid,'antenna_number':ant,'pattern_file':pn,'frequency_mhz':r['freq.'],'raw_building_path':str(bp),'raw_y_path':str(yp)})
    audit['raw_correspondence']={k:{'tested':768,'failed_indices':v} for k,v in raw_fail.items()}
    audit['mask_hypotheses']={k:{'tested':768,'failed_indices':v} for k,v in mask_fail.items()}
    # Conservative union: file identity or exact two-channel D4 equality.
    parent=list(range(768))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):parent[find(b)]=find(a)
    for key in ('building_id','geometry'):
        seen={}
        for rec in records:
            i=rec['base_id']; value=geometry[i] if key=='geometry' else rec[key]
            if value in seen:union(seen[value],i)
            else:seen[value]=i
    grouped=defaultdict(list)
    for i in range(768):grouped[find(i)].append(i)
    groups=sorted(grouped.values(),key=lambda g:min(g)); mapping={i:g for g,items in enumerate(groups) for i in items}
    rng=np.random.default_rng(42); ids=rng.permutation(len(groups)).tolist(); ntrain=int(.7*len(groups));nval=int(.15*len(groups))
    split_groups={'train':ids[:ntrain],'validation':ids[ntrain:ntrain+nval],'test':ids[ntrain+nval:]}
    group_split={g:s for s,gs in split_groups.items() for g in gs}
    manifest=[]
    for idx in range(3072):
        base=idx if idx<768 else (idx-768)//3
        transform='identity' if idx<768 else ('horizontal','vertical','combined')[(idx-768)%3]
        rec=records[base].copy();rec.update(entry_index=idx,transform=transform,geometry_group_id=mapping[base],split=group_split[mapping[base]],use_for_evaluation=transform=='identity',use_for_training=transform=='identity' and group_split[mapping[base]]=='train',data_root=str(DATA),physical_scene_id='unknown')
        manifest.append(rec)
    with (OUT/'target_manifest.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(manifest[0]));w.writeheader();w.writerows(manifest)
    (OUT/'splits').mkdir(exist_ok=True)
    split={'seed':42,'status':'provisional; tile overlap and physical scene identity remain unverified','grouping':'union of raw building file identity and exact two-channel D4 geometry equality','groups':groups,'split_group_ids':split_groups,'base_ids':{s:[i for g in gs for i in groups[g]] for s,gs in split_groups.items()},'augmentation_rule':'base-only train, validation and test; stored flips disabled until asymmetric pattern reflection is validated'}
    save('splits/target_groups.json',split)
    subsets={}
    for seed in range(5):
        order=np.random.default_rng(seed).permutation(split_groups['train']).tolist(); budgets={}
        for pct in (10,25,50,100):
            chosen=order[:max(1,int(np.ceil(len(order)*pct/100)))]; bases=[i for g in chosen for i in groups[g]]
            budgets[str(pct)]={'group_ids':chosen,'base_ids':bases,'group_count':len(chosen),'base_count':len(bases)}
        subsets[str(seed)]=budgets
    save('splits/target_subsets.json',{'axis':'number of geometry groups; percentages refer to train group count','seeds':subsets})
    audit['grouping']={'unique_building_ids':len(set(r['building_id'] for r in records)),'d4_geometry_groups':len(set(geometry.values())),'union_groups':len(groups),'split_groups':{s:len(gs) for s,gs in split_groups.items()},'split_base_counts':{s:len(v) for s,v in split['base_ids'].items()}}
    # Leakage verification is an explicit check, not inferred from row counts.
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        aa=set(split['base_ids'][a]);bb=set(split['base_ids'][b]);assert not aa&bb
        assert not {records[i]['building_id'] for i in aa}&{records[i]['building_id'] for i in bb}
        assert not {geometry[i] for i in aa}&{geometry[i] for i in bb}
    audit['split_checks']='passed: base IDs, building IDs, exact D4 geometry disjoint; does not establish tile-overlap absence'
    save('augmentation_audit.json',audit)
    print(json.dumps({'raw_correspondence':{k:len(v) for k,v in raw_fail.items()},'mask_hypotheses':{k:len(v) for k,v in mask_fail.items()},'grouping':audit['grouping']},indent=2),flush=True)
    with (OUT/'작업로그.md').open('a') as f:f.write('\n## 2026-10-08 — 전체 target 감사 실행\n\n- 실행: `python IEIE/research/audit_target.py`. 원본을 읽기 전용으로 검사했다.\n- 생성: augmentation_audit.json, target_manifest.csv, splits/target_groups.json, splits/target_subsets.json.\n- 검증 결과와 실패 index는 augmentation_audit.json에 기록했다. Split은 원본 building ID와 두 채널의 D4 동등성 합집합을 사용하며 물리 scene/겹치는 tile는 미검증인 provisional split이다.\n- 원본, preprocessing, 기존 연구 문서는 수정하지 않았다.\n')
if __name__=='__main__':main()

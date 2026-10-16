"""Organize around transfer questions and normalize conference body formatting."""
import re,copy
from docx.shared import Pt,Twips
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def refine(d):
 def find(s):return next(p for p in d.paragraphs if p.text==s)
 def replace(p,s):
  assert not p._p.xpath('.//m:oMath | .//w:drawing')
  p.runs[0].text=s
  for r in p.runs[1:]:r.text=''
 def remove(p):p._p.getparent().remove(p._p)
 # Relocate CNN structure and its figure into the shared architecture subsection.
 architecture=find('2.2 DINOv2 기반 예측 모델');replace(architecture,'2.2 모델 아키텍처')
 transfer=find('2.3 전이 전략과 손실 함수')
 start=find('2.4 CNN 기반 조건부 인코더 전이');end=find('3. 구현')
 nodes=list(d._element.body); block=nodes[nodes.index(start._p)+1:nodes.index(end._p)]
 structure=[];strategy=[]
 for e in block:
  t=''.join(e.xpath('.//w:t/text()'))
  if t=='대응 초기화 및 소스 전이':e.getparent().remove(e);continue
  if t.startswith(('Source는 일반 영상','Source와 Target 시드는')):strategy.append(e)
  else:structure.append(e)
 for e in structure:transfer._p.addprevious(e)
 remove(start)
 first_cnn=next(p for p in d.paragraphs if p.text.startswith('DINOv2의 전이 범위 비교에 추가하여 CNN'))
 replace(first_cnn,'전이 효과의 보조 비교에는 2.1절과 동일한 S·P·Q를 입력받는 CNN 기반 예측기를 사용했다. 두 모델의 구조는 기존 아키텍처를 바탕으로 구성했다.')
 # Explicit model context belongs in prose, not subsection headings.
 src=next(p for p in d.paragraphs if p.text.startswith('Source는 공개 DINOv2'))
 replace(src,'DINOv2 기반 모델의 '+src.text)
 anchor=find('3. 구현')
 for e in strategy:anchor._p.addprevious(e)
 strategy_p=next(p for p in d.paragraphs if p.text.startswith('Source는 일반 영상'))
 replace(strategy_p,'CNN 기반 모델의 '+strategy_p.text)
 replace(find('3.2 DINOv2 사전학습 및 전이 성능 비교'),'3.2 사전학습 및 전이 성능 비교')
 remove(find('3.3 CNN 인코더 전이 성능 비교'))
 replace(find('3.4 실험 결과 종합 및 논의'),'3.3 실험 결과 종합 및 논의')
 # Move model-setting comparison from results into the shared setup subsection.
 settings=next(p for p in d.paragraphs if p.text.startswith('CNN은 3.1절과 동일한 데이터'))
 replace(settings,'두 모델은 동일한 데이터 분할과 평가 함수를 사용했다. CNN의 Source와 Target 시드는 각각 142·42로 설정했다. CNN의 BASE는 무작위 초기화 모델이며, 모델별 설정 차이는 표 3과 같다.')
 caption=find('표 3. 모델별 실험 구성 차이');table=d.tables[4]
 optim=next(p for p in d.paragraphs if p.text.startswith('최적화는 AdamW와 cosine schedule'))
 replace(optim,'CNN 기반 모델도 AdamW와 cosine schedule을 사용했다. 학습률, warmup, 최대 epoch 및 조기 종료 대기 기간은 위와 동일하게 설정했다. 증강·EMA·출력 clipping은 적용하지 않았다. 검증 MAE가 가장 낮은 모델을 선택하여 평가용 데이터에서 성능을 측정했다.')
 result=find('3.2 사전학습 및 전이 성능 비교')
 for e in [settings._p,caption._p,table._tbl,optim._p]:result._p.addprevious(e)
 for p in d.paragraphs:
  if p.text.startswith('DINOv2 실험에서 '):replace(p,p.text.replace('DINOv2 실험에서 ','DINOv2 기반 모델의 ',1))
  # Renumber tables in document order after moving setup table.
  if '표 ' in p.text and not p._p.xpath('.//m:oMath | .//w:drawing'):
   replace(p,re.sub(r'표 ([123])',lambda m:'표 '+{'1':'2','2':'3','3':'1'}[m[1]],p.text))
  if p.text=='표 4. CNN 성능 비교 (Target 시드 42)':replace(p,'표 4. 인코더 전이 효과의 보조 비교 (CNN, Target 시드 42)')
 # Explain the CNN control as a transfer question, without prospective experiments.
 intro=next(p for p in d.paragraphs if p.text.startswith('  본 논문은 DINOv2'))
 added=d.add_paragraph('또한 소스 인코더 초기화가 일반 영상 사전학습 없이 시작하는 예측기에서도 도움이 되는지 확인하기 위해 CNN을 보조 비교에 사용하였다. 동일한 데이터 분할, 입력 정보, L1 손실 및 평가 기준에서 무작위 초기화와 소스 인코더 초기화를 비교하였다. 이는 두 모델의 우열을 판단하기 위한 비교가 아니라, CNN 내부에서 소스 인코더 전이의 효과를 확인하기 위한 비교이다.')
 intro._p.addnext(added._p)
 replace(first_cnn,'일반 영상 사전학습 없이 시작하는 모델에서 소스 인코더 초기화의 효과를 확인하기 위해, 2.1절과 동일한 S·P·Q를 입력받는 CNN 기반 예측기를 보조 비교에 사용했다. 두 모델의 구조는 기존 아키텍처를 바탕으로 구성했다.')
 cnn_result=next(p for p in d.paragraphs if p.text.startswith('DINOv2와 동일한 Target 시드 42에서 BASE'))
 replace(cnn_result,'동일한 Target 시드 42에서 CNN의 BASE와 ENC의 test MAE는 각각 0.079636, 0.082767였다. 소스 인코더 초기화 후 MAE는 3.93% 증가했다. RMSE와 PSNR은 BASE가 좋았고 SSIM은 ENC가 좋았다. 따라서 이 조건에서 소스 인코더 전이는 지도의 구조적 유사도를 높였으나, 픽셀별 예측 오차를 줄이지는 못했다.')
 discussion=next(p for p in d.paragraphs if p.text.startswith('DINOv2의 L1→L1 조건과 CNN 모두'))
 replace(discussion,'DINOv2의 L1→L1 조건과 CNN의 보조 비교 모두에서 인코더 전이만으로 test MAE가 개선되지 않았다. CNN 결과는 일반 영상 사전학습 없이 시작하는 모델에서도 소스 인코더 초기화가 오차 감소를 보장하지 않음을 보여준다. 반면 DINOv2에서는 인코더·디코더 전이와 LoRA를 결합한 조건에서 MAE가 낮아졌다. 이번 결과는 추가 사전학습의 유무뿐 아니라 전이 범위와 적응 방식을 함께 고려할 필요가 있음을 시사한다.')
 limit=next(p for p in d.paragraphs if p.text.startswith('DINOv2는 공개 사전학습 가중치와 LoRA'))
 replace(limit,'DINOv2는 공개 사전학습 가중치와 LoRA를 사용하고 CNN은 무작위 초기화와 전체 미세조정을 사용했다. 따라서 두 모델의 성능 차이를 아키텍처 또는 전이 범위 하나의 효과로 해석할 수는 없다. 본 비교는 각 모델 내부의 전이 전후 결과를 분석하며, 결론은 단일 소스 초기화와 Target 시드 42 조건에서 관찰한 결과로 한정한다.')
 future=next(p for p in d.paragraphs if p.text.startswith('이후 연구에서는 추가 사전학습'))
 extra=d.add_paragraph('한편 CNN의 보조 비교에서는 소스 인코더 초기화 후 test MAE가 3.93% 증가하였다. DINOv2의 L1→L1 인코더 단독 전이에서도 MAE가 증가하였다. 따라서 RadioMapSeer에서 학습한 인코더를 가져오는 것만으로 성능 개선을 기대하기는 어려우며, 이번에 관찰된 개선은 전이 범위와 적응 방식에 따른 조건부 효과로 해석해야 한다.')
 future._p.addprevious(extra._p)
 abstract=next(p for p in d.paragraphs if 'Directional electromagnetic field (EMF) dataset' in p.text)
 replace(abstract,abstract.text+' In the CNN comparison, source encoder initialization increases test MAE by 3.93% relative to random initialization. Encoder-only transfer also increases MAE in the DINOv2 L1-to-L1 setting, indicating that the benefit of source pretraining depends on the transfer scope and adaptation strategy.')
 # Remove a redundant section break between introduction and body (both two columns).
 sections=d._element.xpath('.//w:sectPr')
 for a,b in zip(sections,sections[1:]):
  ac=a.find(qn('w:cols'));bc=b.find(qn('w:cols'))
  if ac is not None and bc is not None and ac.get(qn('w:num'))==bc.get(qn('w:num'))=='2' and a.getparent().tag==qn('w:pPr'):
   a.getparent().remove(a)
 last=d._element.body.find(qn('w:sectPr'))
 typ=last.find(qn('w:type'))
 if typ is None:typ=OxmlElement('w:type');last.insert(0,typ)
 typ.set(qn('w:val'),'continuous')
 # Conference body: preserve title block, math runs, figures and original section margins.
 active=False
 for p in d.paragraphs:
  if p.text=='Abstract':active=True
  if not active:continue
  if p._p.xpath('.//w:sectPr'):continue
  fmt=p.paragraph_format
  fmt.page_break_before=False;fmt.widow_control=True
  fmt.line_spacing=1.2;fmt.space_before=Pt(0);fmt.space_after=Pt(2)
  fmt.keep_with_next=False;fmt.keep_together=False
  txt=p.text.strip()
  heading=bool(re.match(r'^[1-4](?:\.\d)?\.?(?: |$)',txt)) or txt in ['Abstract','참고문헌']
  caption=bool(re.match(r'^(표|그림) \d+\.',txt))
  drawing=bool(p._p.xpath('.//w:drawing | .//w:pict'))
  equation=bool(p._p.xpath('.//m:oMath | .//m:oMathPara'))
  if heading:
   fmt.keep_with_next=True;fmt.space_before=Pt(5);fmt.space_after=Pt(2);fmt.first_line_indent=Pt(0)
   p.alignment=WD_ALIGN_PARAGRAPH.CENTER if re.match(r'^[1-4]\. ',txt) or txt in ['Abstract','참고문헌'] else WD_ALIGN_PARAGRAPH.LEFT
  elif caption or drawing or equation:
   p.alignment=WD_ALIGN_PARAGRAPH.CENTER;fmt.first_line_indent=Pt(0)
   fmt.keep_with_next=txt.startswith('표 ') or drawing
   fmt.keep_together=True
  elif txt:
   p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY;fmt.first_line_indent=Pt(9)
   if txt.startswith('['):fmt.first_line_indent=Pt(-9);fmt.left_indent=Pt(9)
  for r in p.runs:
   if r._element.xpath('.//m:oMath | .//w:drawing | .//w:pict | .//w:object'):continue
   r.font.name='HY신명조';r._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'HY신명조')
   r.font.size=Pt(10 if heading else 8 if caption else 9)
   r.bold=heading
 # All body tables share a column width, rules, header treatment and cell typography.
 widths={3:[1560,1470,1470],6:[700,760,760,760,760,760],7:[640,640,644,644,644,644,644]}
 for t in d.tables[2:]:
  n=len(t.columns);ww=widths[n]
  t.autofit=False;t.alignment=WD_TABLE_ALIGNMENT.CENTER
  pr=t._tbl.tblPr
  for tag in ['tblW','tblInd','tblBorders','tblCellMar','tblLook']:
   for el in list(pr.findall(qn('w:'+tag))):pr.remove(el)
  w=OxmlElement('w:tblW');w.set(qn('w:type'),'dxa');w.set(qn('w:w'),'4500');pr.append(w)
  borders=OxmlElement('w:tblBorders')
  for side in ['top','bottom','left','right','insideH','insideV']:
   b=OxmlElement('w:'+side);b.set(qn('w:val'),'single' if side in ['top','bottom'] else 'nil');b.set(qn('w:sz'),'6');b.set(qn('w:color'),'000000');borders.append(b)
  pr.append(borders)
  margins=OxmlElement('w:tblCellMar')
  for side,num in [('top',45),('bottom',45),('left',18),('right',18)]:
   el=OxmlElement('w:'+side);el.set(qn('w:w'),str(num));el.set(qn('w:type'),'dxa');margins.append(el)
  pr.append(margins)
  for g,w in zip(t._tbl.tblGrid.gridCol_lst,ww):g.set(qn('w:w'),str(w))
  for ri,row in enumerate(t.rows):
   rp=row._tr.get_or_add_trPr();ns=OxmlElement('w:cantSplit');rp.append(ns)
   if ri==0:rp.append(OxmlElement('w:tblHeader'))
   ci=0
   for tc in row._tr.tc_lst:
    span=tc.tcPr.find(qn('w:gridSpan'));span=int(span.get(qn('w:val'))) if span is not None else 1
    tc.tcPr.find(qn('w:tcW')).set(qn('w:w'),str(sum(ww[ci:ci+span])));ci+=span
   seen=set()
   for c in row.cells:
    if id(c._tc) in seen:continue
    seen.add(id(c._tc));c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
    cp=c._tc.get_or_add_tcPr()
    for x in list(cp.findall(qn('w:tcBorders'))):cp.remove(x)
    if ri==0:
     cb=OxmlElement('w:tcBorders');b=OxmlElement('w:bottom');b.set(qn('w:val'),'single');b.set(qn('w:sz'),'4');cb.append(b);cp.append(cb)
    for p in c.paragraphs:
     p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.space_before=Pt(0);p.paragraph_format.space_after=Pt(0)
     p.paragraph_format.first_line_indent=Pt(0);p.paragraph_format.line_spacing=1.0
     p.paragraph_format.keep_with_next=ri<len(t.rows)-1;p.paragraph_format.keep_together=True
     for r in p.runs:
      r.font.name='HY신명조';r._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'HY신명조');r.font.size=Pt(7.5 if n==7 else 8);r.bold=ri==0
 # Use compact Latin glyphs and prohibit breaking numbers inside cells.
 for t in d.tables[2:]:
  for row in t.rows:
   for c in row.cells:
    for p in c.paragraphs:
     wp=p._p.get_or_add_pPr().find(qn('w:wordWrap'))
     if wp is None:wp=OxmlElement('w:wordWrap');p._p.get_or_add_pPr().append(wp)
     wp.set(qn('w:val'),'0')
     for r in p.runs:
      rf=r._element.get_or_add_rPr().get_or_add_rFonts()
      rf.set(qn('w:ascii'),'Times New Roman');rf.set(qn('w:hAnsi'),'Times New Roman')
 # Keep metric headers short enough for a conference column.
 for t in d.tables[2:]:
  for c in t.rows[0].cells:
   for p in c.paragraphs:
    for r in p.runs:r.text=r.text.replace('Validation','Val.')
 # Standardize CNN result header to the existing metric-table convention.
 cnn=d.tables[-1]
 for c,s in zip(cnn.rows[0].cells,['조건','Val.\nMAE','Test\nMAE','Test\nRMSE','Test\nSSIM','Test\nPSNR']):
  p=c.paragraphs[0];p.runs[0].text=s
 d.core_properties.author=''

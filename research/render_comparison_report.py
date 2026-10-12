"""Render the locally sourced comparison report as editable Word and PDF."""
import json
from pathlib import Path
import re
import write_korean_manuscript as renderer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'reports'
STEM = 'experiment_comparison_integration_ko'


def main():
    text = (OUT / (STEM + '.md')).read_text()
    def plain(value):
        return re.sub(r'\[([^]]+)\]\(([^)]+)\)', r'\1 (\2)', value).replace('**', '')
    lines = text.splitlines()
    sections = []
    current = None
    i = 0
    table_count = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('## '):
            current = {'title':line[3:], 'content':[]}
            sections.append(current)
            i += 1
        elif current is None or not line.strip():
            i += 1
        elif line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                rows.append([plain(x.strip()) for x in lines[i].strip('|').split('|')])
                i += 1
            assert all(len(r) == len(rows[0]) for r in rows)
            table_count += 1
            current['content'].append({'type':'table','headers':rows[0],'rows':rows[2:], 'caption':f'표 {table_count}'})
        else:
            paragraphs = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(('## ', '|', '- ')):
                paragraphs.append(lines[i])
                i += 1
            current['content'].append({'type':'paragraph', 'text':plain(' '.join(paragraphs))})
    doc = {'title':lines[0][2:], 'authors':'작성일: 2026년 10월 13일',
           'status':'실험 비교 및 논문 통합 계획 보고서 · 추가 연결 실험은 제안 단계',
           'abstract_label':'요약',
           'abstract':'IEIE는 무작위 초기화 대비 인코더 전이 효과를 반복 시드와 데이터 규모로 확인했고, EMF-DINOv2는 일반 영상 사전학습 이후 전이 범위와 적응 조합을 탐색했다. 분할과 평가 규칙이 달라 절대 점수의 직접 순위 비교는 불가능하다. 공통 프로토콜의 연결 실험을 추가하여 통합하거나, 별도 프로토콜의 사례 연구로 구분하여 결합할 수 있다.',
           'keywords':'인코더 전이, DINOv2, 디코더 적응, 실험 비교, 논문 통합',
           'footer':'IEIE와 EMF-DINOv2 실험 비교 및 논문 통합 방안',
           'sections':sections, 'references':[]}
    renderer.OUT = OUT
    renderer.word(doc)
    renderer.pdf(doc)
    (OUT / 'manuscript_ko.docx').rename(OUT / (STEM + '.docx'))
    (OUT / 'manuscript_ko_review.pdf').rename(OUT / (STEM + '.pdf'))
    (OUT / (STEM + '_content.json')).write_text(json.dumps(doc, ensure_ascii=False, indent=2))
    assert table_count == 8
    print(f'Report generated: {len(sections)} sections, {table_count} tables')


if __name__ == '__main__':
    main()

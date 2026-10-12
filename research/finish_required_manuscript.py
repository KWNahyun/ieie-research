"""Wait for the frozen experiment pipeline, audit it, then generate the paper."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/followup_v1'


def main():
    state = RUN / 'manuscript_completion.json'
    state.write_text(json.dumps({'status': 'waiting_for_experiments'}, indent=2))
    while True:
        path = RUN / 'status.json'
        if path.exists() and json.loads(path.read_text()).get('status') == 'experiments_complete':
            break
        time.sleep(30)
    try:
        for script in ['audit_followup_results.py', 'compare_reference_matched.py', 'write_manuscript.py']:
            state.write_text(json.dumps({'status': 'running', 'script': script}, indent=2))
            subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT.parent, check=True)
        content = json.loads((ROOT / 'manuscript/manuscript_content.json').read_text())
        tables = [b for s in content['sections'] for b in s['content'] if b['type'] == 'table']
        assert len(content['references']) == 8 and len(tables) >= 7
        for name in ['manuscript.md', 'manuscript.docx', 'manuscript_review.pdf', 'manuscript.tex']:
            assert (ROOT / 'manuscript' / name).stat().st_size > 1000
        state.write_text(json.dumps({'status': 'draft_generated', 'audit': 'PASS',
            'directory': str(ROOT / 'manuscript'), 'tables': len(tables),
            'note': 'Research draft; author and affiliation details pending. PDF is a review copy; IEEE LaTeX source is not compiled.'}, indent=2))
    except Exception as exc:
        state.write_text(json.dumps({'status': 'failed', 'error': repr(exc)}, indent=2))
        raise


if __name__ == '__main__':
    main()

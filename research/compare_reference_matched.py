"""Cross-backbone descriptive table using identical target/source seed scope."""
import json
import numpy as np
from data_pipeline import RESEARCH
from train_baseline import write_json


def main():
    root = RESEARCH / 'runs/followup_v1'
    new = [json.loads(p.read_text()) for p in (root / 'test_metrics').glob('reference_*.json')]
    assert len(new) == 12
    old = [json.loads(p.read_text()) for p in (RESEARCH / 'runs/main_v1/test_metrics').glob('*.json')]
    rows = []
    for budget in [25, 100]:
        for family, data in [('cnn', old), ('hrt_adapter', new)]:
            means = {}
            for mode in ['B', 'D']:
                selected = [r for r in data if r['budget'] == budget and r['model'] == mode
                            and r.get('target_seed', r.get('seed')) in [0, 1, 2]
                            and r['source_seed'] == (0 if mode == 'D' else None)]
                assert len(selected) == 3
                means[mode] = float(np.mean([r['macro_mae'] for r in selected]))
            rows.append({'budget': budget, 'family': family, 'B_mae': means['B'], 'D_mae': means['D'],
                         'relative_reduction_percent': 100 * (means['B'] - means['D']) / means['B'],
                         'target_seeds': [0, 1, 2], 'source_seed_D': 0})
    write_json(root / 'reference_matched_comparison.json', {'rows': rows,
               'scope': 'same geometry subsets/target seeds0–2/source0;descriptivecross-family comparison;capacity/conditioning/training LR differ',
               'published_outdoor_scores_compared': False})
    print('MatchedCNN/HRT source0/target0–2 comparison saved')


if __name__ == '__main__':
    main()

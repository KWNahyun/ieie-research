"""Explicit setting subsets; original geometry-group subset contract stays intact."""
import json
from pathlib import Path
from data_pipeline import TargetDataset


class SettingDataset(TargetDataset):
    def __init__(self, subset, *, normalization=None):
        super().__init__('train', include_building_encoded=False)
        ids = set(map(int, subset['base_ids']))
        assert ids and len(ids) == len(subset['base_ids'])
        all_ids = {int(r['base_id']) for r in self.rows}
        all_groups = {int(r['geometry_group_id']) for r in self.rows}
        all_configs = {int(r['configuration_row']) for r in self.rows}
        assert ids <= all_ids
        self.rows = [r for r in self.rows if int(r['base_id']) in ids]
        assert {int(r['geometry_group_id']) for r in self.rows} == all_groups == set(subset['group_ids'])
        assert {int(r['configuration_row']) for r in self.rows} == all_configs == set(subset['configuration_rows'])
        if normalization is not None:
            self.normalization = json.loads(Path(normalization).read_text())
            assert self.normalization['fit_split'] == 'train'
            assert set(self.normalization['fitted_base_ids']) == ids

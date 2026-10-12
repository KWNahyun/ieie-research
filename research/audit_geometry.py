"""Group target geometry by exact 8-bit equality up to quarter-turn rotation.

Run: python IEIE/research/audit_geometry.py
These are candidate groups, not verified physical scene IDs.
"""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import numpy as np

root = Path(__file__).resolve().parents[1]
array = np.load(root / 'data/OurData/x_image.npy', mmap_mode='r')
groups = defaultdict(list)
for index, sample in enumerate(array):
    pixels = np.rint(sample * 255).astype(np.uint8)
    hashes = [hashlib.sha256(np.ascontiguousarray(np.rot90(pixels, k)).tobytes()).hexdigest()
              for k in range(4)]
    groups[min(hashes)].append(index)
result = {
    'samples': len(array),
    'rotation_equivalent_geometry_groups': len(groups),
    'group_size_histogram': dict(Counter(map(len, groups.values()))),
    'groups': list(groups.values()),
    'note': 'Pixel equality up to quarter-turn rotation only; not a verified physical scene ID. Reflections and near duplicates not merged.',
}
output = root / 'research/geometry_groups.json'
output.write_text(json.dumps(result, indent=2))
print({key: value for key, value in result.items() if key != 'groups'})

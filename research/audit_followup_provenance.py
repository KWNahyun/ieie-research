"""Available metadata and exhaustive whole-frame proxy correspondence audit."""
import csv
import json
from pathlib import Path
import re
import cv2
import numpy as np
from PIL import Image
from zipfile import ZipFile
import xml.etree.ElementTree as ET
from data_pipeline import RESEARCH
from train_baseline import write_json


def main():
    out = RESEARCH / 'runs/followup_v1/provenance'
    out.mkdir(parents=True, exist_ok=True)
    target_rows = [r for r in csv.DictReader((RESEARCH / 'target_manifest.csv').open()) if r['transform'] == 'identity']
    target_images = {int(r['geometry_group_id']): Path(r['raw_building_path']) for r in target_rows}
    source_rows = list(csv.DictReader((RESEARCH / 'source_manifest.csv').open()))
    source_images = {int(r['map_id']): Path(r['building_path']) for r in source_rows}
    geographic_key = re.compile(r'latitude|longitude|\blat\b|\blon\b|epsg|crs|geotransform|georef|utm|bbox|extent|origin', re.I)
    png_metadata = []
    for domain, images in [('target', target_images), ('source', source_images)]:
        for identity, path in images.items():
            with Image.open(path) as image:
                fields = {str(k): str(v)[:200] for k, v in image.info.items() if geographic_key.search(str(k))}
                png_metadata.append({'domain': domain, 'identity': identity, 'shape': list(image.size), 'geographic_fields': fields})
    workbook_path = Path('/home/ailab/Desktop/data/emf_data/antenna_configuration.xlsx')
    ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with ZipFile(workbook_path) as workbook:
        strings = []
        if 'xl/sharedStrings.xml' in workbook.namelist():
            for item in ET.fromstring(workbook.read('xl/sharedStrings.xml')).findall('m:si', ns):
                strings.append(''.join(x.text or '' for x in item.findall('.//m:t', ns)))
        sheet = ET.fromstring(workbook.read('xl/worksheets/sheet1.xml'))
        first = sheet.find('m:sheetData/m:row', ns)
        headers = []
        for cell in first.findall('m:c', ns):
            value = cell.find('m:v', ns)
            if cell.attrib.get('t') == 's' and value is not None:
                headers.append(strings[int(value.text)])
            elif cell.attrib.get('t') == 'inlineStr':
                headers.append(''.join(t.text or '' for t in cell.findall('.//m:t', ns)))
            else: headers.append(value.text if value is not None else '')
    geographic_headers = [h for h in headers if geographic_key.search(h)]
    spatial_assets = []
    raw = Path('/home/ailab/Desktop/data/emf_data')
    for extension in ['*.shp', '*.geojson', '*.kml', '*.osm', '*.prj', '*.tif', '*.tiff', '*.json', '*.xml']:
        spatial_assets.extend(str(p) for p in raw.rglob(extension))
    polygon_path = RESEARCH.parent / 'data/RadioMapSeer/polygon/buildings_complete/0.json'
    polygon = json.loads(polygon_path.read_text())
    polygon_description = {'sample_path': str(polygon_path), 'container': type(polygon).__name__,
                           'sample_polygon_bounds': [float(v) for v in [np.asarray([p for polygon in polygon for p in polygon]).min(),
                                                                        np.asarray([p for polygon in polygon for p in polygon]).max()]],
                           'geographic_reference_attached': isinstance(polygon, dict) and any(geographic_key.search(k) for k in polygon)}
    target_ids = sorted(target_images)
    source_ids = sorted(source_images)
    targets = []
    for gid in target_ids:
        image = cv2.imread(str(target_images[gid]), cv2.IMREAD_COLOR)
        proxy = np.all(image[:, :, :2] == 0, axis=-1).astype(np.uint8)
        targets.append(cv2.resize(proxy, (64, 64), interpolation=cv2.INTER_NEAREST).ravel())
    candidates = []
    transformed = []
    for mid in source_ids:
        image = (cv2.imread(str(source_images[mid]), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_NEAREST)
        for flip in [False, True]:
            base = np.fliplr(image) if flip else image
            for rotation in range(4):
                transformed.append(np.rot90(base, rotation).ravel())
                candidates.append({'source_map_id': mid, 'reflection': flip, 'rotation_degrees': rotation * 90})
    targets = np.asarray(targets, dtype=np.float32)
    transformed = np.asarray(transformed, dtype=np.float32)
    intersection = targets @ transformed.T
    union = targets.sum(1)[:, None] + transformed.sum(1)[None, :] - intersection
    iou = intersection / np.maximum(union, 1)
    correspondences = []
    for index, gid in enumerate(target_ids):
        best = np.argsort(iou[index])[-5:][::-1]
        correspondences.append({'target_geometry_group': gid,
                                'target_split': next(r['split'] for r in target_rows if int(r['geometry_group_id']) == gid),
                                'top_candidates': [{**candidates[int(c)], 'proxy_iou': float(iou[index, c])} for c in best]})
    strong = [r for r in correspondences if r['top_candidates'][0]['proxy_iou'] >= .75]
    record = {'scope': 'available metadata and64x64 D4 whole-frame operational-proxy matching; not physical/GIS overlap certification',
              'target_images_checked': len(target_images), 'source_images_checked': len(source_images),
              'png_geographic_metadata_count': sum(bool(r['geographic_fields']) for r in png_metadata),
              'spreadsheet_headers': headers, 'spreadsheet_geographic_headers': geographic_headers,
              'target_spatial_vector_or_gis_assets': spatial_assets, 'source_polygon_example': polygon_description,
              'normalization': 'nearest-neighbor64x64; target raw B/G bothzero versus source binary nonzero;8 D4 transforms',
              'pair_transform_comparisons': len(target_ids) * len(candidates),
              'strong_proxy_correspondences_threshold': .75, 'strong_proxy_correspondences': strong,
              'maximum_proxy_iou': float(iou.max()), 'correspondences': correspondences,
              'unresolved': ['independent physical scene identity', 'tile coordinates/extents', 'source-target geographic overlap',
                             'target physical footprint/outdoor/nodata semantics', 'absolute EMF inverse normalization'],
              'interpretation': 'No raster match cannot establish geographic non-overlap; scale/crop/renderer/physical semantics differ. High matches are candidates requiring original scene metadata. No label-derived mask used.',
              'new_physical_generalization_claim_authorized': False}
    write_json(out / 'audit.json', record)
    write_json(out / 'png_metadata.json', png_metadata)
    lines = ['# Available provenance and layout-correspondence audit', '',
             f"Inspected{len(target_images)} target and{len(source_images)} source building PNGs, workbook headers and available target spatial assets. Geographic PNG metadata fields:{record['png_geographic_metadata_count']}; geographic workbook headers:{geographic_headers}; target GIS/vector assets:{len(spatial_assets)}.", '',
             f"Whole-frame normalized operational-proxy comparisons:{record['pair_transform_comparisons']} (8 D4 transforms); maximum IoU:{record['maximum_proxy_iou']:.4f}; candidates atIoU≥0.75:{len(strong)}.", '',
             'This comparison cannot certify geographic separation. Images differ in renderer/scale/physical semantics, and partial crops/overlapping tiles are not ruled out. The source polygon sample contains local pixel coordinates without an attached geographic reference. Target original vector/rendering assets were previously confirmed unavailable by the user.', '',
             'Physical scene identity, tile extent, geographic source-target overlap, independent outdoor/nodata masks and absolute-unit inversion remain unknown. Empirical image-group claims are supported; physical/geographic scene-generalization claims are withheld. Complete metadata/candidate lists are preserved in audit.json and png_metadata.json.']
    (out / 'results.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()

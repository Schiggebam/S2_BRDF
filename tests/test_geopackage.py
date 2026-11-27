import sqlite3
import struct
from contextlib import closing
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
import io
import json
import yaml

from pyproj import Transformer
from shapely.geometry import MultiPolygon, box
from shapely.ops import transform
from shapely import to_wkb

from brdf.discovery import geopackage_neighbors, read_geopackage
from brdf.__main__ import main
from test_core import scene


def create_gpkg(path, features, field='Name'):
    with closing(sqlite3.connect(path)) as db:
        db.executescript('''
          CREATE TABLE gpkg_contents (table_name TEXT, data_type TEXT);
          CREATE TABLE gpkg_geometry_columns (table_name TEXT, column_name TEXT, srs_id INTEGER);
          CREATE TABLE gpkg_spatial_ref_sys (srs_id INTEGER, organization TEXT, organization_coordsys_id INTEGER, definition TEXT);
          INSERT INTO gpkg_contents VALUES ('tiles', 'features');
          INSERT INTO gpkg_geometry_columns VALUES ('tiles', 'UTM_WKT', 4326);
          INSERT INTO gpkg_spatial_ref_sys VALUES (4326, 'EPSG', 4326, '');
        ''')
        db.execute(f'CREATE TABLE tiles (fid INTEGER PRIMARY KEY, "{field}" TEXT, UTM_EPSG INTEGER, UTM_WKT BLOB)')
        for name, geometry in features:
            payload = b'GP' + bytes([0, 1]) + struct.pack('<i', 4326) + to_wkb(geometry)
            db.execute(f'INSERT INTO tiles ("{field}", UTM_EPSG, UTM_WKT) VALUES (?, ?, ?)', (name, 32632, payload))
        db.commit()


class GeoPackageTests(unittest.TestCase):
    def test_horizontal_neighbors_from_polygons_and_zone_boundaries(self):
        to_geo = Transformer.from_crs(32632, 4326, always_xy=True).transform
        # Full 100 km UTM polygons: east, north and diagonal grid cells.
        cells = [('32UPU', box(600000, 5300000, 700000, 5400000)),
                 ('32UQU', box(700000, 5300000, 800000, 5400000)),
                 ('32UPV', box(600000, 5400000, 700000, 5500000)),
                 ('32UQV', box(700000, 5400000, 800000, 5500000))]
        features = [(name, MultiPolygon([transform(to_geo, polygon)])) for name, polygon in cells]
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            path = Path(tmp)/'tiles.gpkg'
            create_gpkg(path, features, field='Tile')
            geometries, metadata = read_geopackage(path)
            self.assertEqual(metadata['id_field'], 'Tile')
            self.assertEqual(metadata['layer'], 'tiles')
            self.assertEqual(geopackage_neighbors(geometries), [('32UPU', '32UQU'), ('32UPV', '32UQV')])
            # Cross-zone IDs are not considered neighbors, even for coincident geometries.
            geometries['31UCP'] = geometries['32UPU']
            self.assertFalse(any('31UCP' in pair for pair in geopackage_neighbors(geometries)))

    def test_duplicate_and_invalid_ids_fail(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            path = Path(tmp)/'tiles.gpkg'
            create_gpkg(path, [('32UPU', box(9, 47, 10, 48)), ('32UPU', box(10, 47, 11, 48))])
            with self.assertRaisesRegex(ValueError, 'Duplicate tile'):
                read_geopackage(path, id_field='Name')
            with self.assertRaisesRegex(ValueError, 'not found'):
                read_geopackage(path, id_field='wrong')

    def test_cli_uses_geopackage_neighbors_for_four_scene_group(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            to_geo = Transformer.from_crs(32632, 4326, always_xy=True).transform
            tiles = root/'tiles.gpkg'
            create_gpkg(tiles, [('32UPU', transform(to_geo, box(600000, 5300000, 700000, 5400000))),
                                ('32UQU', transform(to_geo, box(700000, 5300000, 800000, 5400000)))], field='Tile')
            for tile in ('32UPU', '32UQU'):
                scene(tile, 'S2A', '20250409T105701', 'R094', root)
                scene(tile, 'S2B', '20250409T104619', 'R051', root)
            wc = root/'wc.vrt'; wc.touch()
            config = {'paths': {'l2a_archive':str(root), 'worldcover_2021':str(wc),
                                'tile_geopackage':str(tiles), 'output':str(root/'out')},
                      'tiles': {'tile_id_field':'Tile'},
                      'selection': {'max_cloud_percent_exclusive':20, 'max_sun_zenith_deg_exclusive':70,
                                    'near_nadir_deg_exclusive':4, 'valid_scl':[4,5],
                                    'landcover_classes':[10,30,40], 'max_pair_minutes':30,
                                    'max_neighbor_sensing_minutes':3},
                      'sampling': {'samples_per_class_per_tile':200, 'window_pixels':512},
                      'processing': {'workers':1, 'geometry_band':'B04'}}
            cfg = root/'config.yaml'; cfg.write_text(yaml.safe_dump(config))
            with patch('brdf.process.process_group', return_value={'group_id':'test', 'status':'complete', 'samples':1}) as process, redirect_stdout(io.StringIO()):
                result = main(['--config', str(cfg), '--start', '2025-04-09', '--end', '2025-04-10'])
            self.assertEqual(result, 0)
            self.assertEqual(process.call_count, 1)
            report = json.loads(next((root/'out'/'runs').glob('*/run.json')).read_text())
            self.assertEqual(report['tile_geopackage']['id_field'], 'Tile')
            self.assertEqual(report['neighbor_pairs'], [['32UPU', '32UQU']])
            self.assertEqual(report['group_count'], 1)


if __name__ == '__main__':
    unittest.main()

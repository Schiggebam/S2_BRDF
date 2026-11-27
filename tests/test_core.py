import contextlib
from datetime import datetime
import json
from pathlib import Path
import tempfile
import types
import sys
import unittest
from unittest.mock import patch

import numpy as np

from brdf.discovery import Scene, discover, make_groups, read_tiles
from brdf.geometry import AngleGrid, angular_distance
from brdf.process import process_group, radiometry, sample_tile


def scene(tile, platform, timestamp, orbit, root, baseline='N0511', produced='20250409T173716', cloud='5'):
    name = f'{platform}_MSIL2A_{timestamp}_{baseline}_{orbit}_T{tile}_{produced}.SAFE'
    path = root/tile[:2]/tile[2]/tile[3:]/'2025'/'04'/name
    granule = path/'GRANULE'/'example'
    granule.mkdir(parents=True)
    (path/'MTD_MSIL2A.xml').write_text('''<root><BOA_QUANTIFICATION_VALUE>10000</BOA_QUANTIFICATION_VALUE>
        <BOA_ADD_OFFSET band_id="3">-1000</BOA_ADD_OFFSET><BOA_ADD_OFFSET band_id="7">-1000</BOA_ADD_OFFSET>
        <BOA_ADD_OFFSET band_id="11">-1000</BOA_ADD_OFFSET></root>''')
    (granule/'MTD_TL.xml').write_text(f'<root><CLOUDY_PIXEL_PERCENTAGE>{cloud}</CLOUDY_PIXEL_PERCENTAGE></root>')
    return path


class TestMatchups(unittest.TestCase):
    def test_four_scenes_and_scene_cloud_limit(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for tile in ('31UCP', '31UDP'):
                scene(tile, 'S2A', '20250409T105701', 'R094', root)
                scene(tile, 'S2B', '20250409T104619', 'R051', root)
            # Different pass on the same day must not replace the first one.
            scene('31UCP', 'S2A', '20250409T120000', 'R099', root, cloud='20')
            scenes, issues = discover(root, ['31UCP', '31UDP'], datetime(2025,4,1), datetime(2025,5,1), 20)
            self.assertFalse(issues)
            self.assertEqual(len(scenes), 4)
            groups = make_groups(scenes, [('31UCP','31UDP')], 30, 3)
            self.assertEqual(len(groups), 1)
            self.assertEqual({s.tile for p in groups[0] for s in p}, {'31UCP','31UDP'})
            quant, offsets = radiometry(scenes[0])
            self.assertEqual((2000 + offsets['B04']) / quant, 0.1)
            self.assertEqual(len(make_groups(scenes[:-1], [('31UCP','31UDP')], 30, 3)), 0)

    def test_multiple_passes_do_not_cross_match(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for tile in ('31UCP','31UDP'):
                for hours, oa, ob in [('10','R094','R051'), ('12','R095','R052')]:
                    scene(tile, 'S2A', f'20250409T{hours}0000', oa, root)
                    scene(tile, 'S2B', f'20250409T{hours}1000', ob, root)
            scenes, _ = discover(root, ['31UCP','31UDP'], datetime(2025,4,1), datetime(2025,5,1), 20)
            self.assertEqual(len(make_groups(scenes, [('31UCP','31UDP')], 30, 3)), 2)

    def test_latest_processing_is_selected_before_cloud_filter(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            scene('31UCP', 'S2A', '20250409T105701', 'R094', root, produced='20250409T173716', cloud='5')
            scene('31UCP', 'S2A', '20250409T105701', 'R094', root, produced='20250410T173716', cloud='20')
            scenes, issues = discover(root, ['31UCP'], datetime(2025,4,1), datetime(2025,5,1), 20)
            self.assertFalse(issues)
            self.assertFalse(scenes)

    def test_tile_list_yaml_and_plain_lines(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'tiles.txt'
            for value in ('- 32UQU\n- 32UPU\n', '32UQU\n32UPU\n'):
                path.write_text(value)
                self.assertEqual(read_tiles(path), ['32UPU','32UQU'])


class TestGeometry(unittest.TestCase):
    def test_angles_and_wraparound_interpolation(self):
        self.assertAlmostEqual(float(angular_distance(10, 90, 10, 270)), 20)
        self.assertAlmostEqual(float(angular_distance(10, 90, 10, 90)), 0, places=5)
        with tempfile.TemporaryDirectory() as td:
            xml = Path(td)/'tile.xml'
            def element(name, row):
                return f'<{name}><COL_STEP>10</COL_STEP><ROW_STEP>10</ROW_STEP><Values_List><VALUES>{row}</VALUES><VALUES>{row}</VALUES></Values_List></{name}>'
            xml.write_text('<root><Geoposition resolution="20"><ULX>0</ULX><ULY>10</ULY></Geoposition>'
                + '<Sun_Angles_Grid>' + element('Zenith', '20 20') + element('Azimuth', '359 1') + '</Sun_Angles_Grid>'
                + '<Viewing_Incidence_Angles_Grids bandId="3">' + element('Zenith', '3 3') + element('Azimuth', '359 1')
                + '</Viewing_Incidence_Angles_Grids></root>')
            values = AngleGrid(xml, 'B04').at(np.array([5.,20.]), np.array([5.,5.]))
            self.assertLess(min(values['vaa'][0], 360-values['vaa'][0]), 0.01)
            self.assertTrue(np.isnan(values['vza'][1]))


class TestResume(unittest.TestCase):
    def test_reuse_and_corrupt_result_recompute(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for tile in ('31UCP','31UDP'):
                scene(tile, 'S2A', '20250409T105701', 'R094', root)
                scene(tile, 'S2B', '20250409T104619', 'R051', root)
            scenes, _ = discover(root, ['31UCP','31UDP'], datetime(2025,4,1), datetime(2025,5,1), 20)
            group = make_groups(scenes, [('31UCP','31UDP')], 30, 3)[0]
            wc = root/'wc.vrt'
            wc.write_text('test')
            cfg = {'paths': {'output': str(root/'out'), 'worldcover_2021':str(wc)},
                   'selection': {}, 'sampling': {}, 'processing': {'geometry_band':'B04'}}
            with patch('brdf.process.group_lock', lambda _: contextlib.nullcontext()), \
                 patch('brdf.process.sample_tile', return_value=([{'group_id':'a', 'test':1}], {})) as sample:
                first = process_group(group, cfg)
                self.assertEqual(first['status'], 'complete')
                self.assertEqual(process_group(group, cfg)['status'], 'reused')
                self.assertEqual(sample.call_count, 2)
                (root/'out'/first['group_id']/'samples.parquet').write_bytes(b'corrupt')
                self.assertEqual(process_group(group, cfg)['status'], 'complete')
                self.assertEqual(sample.call_count, 4)


class TestSampling(unittest.TestCase):
    def test_both_scl_and_per_pixel_angles_and_reflectance(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as td:
            root = Path(td)
            for platform, timestamp, orbit in [('S2A','20250409T105701','R094'), ('S2B','20250409T104619','R051')]:
                path = scene('31UCP', platform, timestamp, orbit, root)
                for band, res in [('B04',10), ('B08',10), ('B12',20), ('SCL',20)]:
                    folder = path/'GRANULE'/'example'/'IMG_DATA'/f'R{res}m'
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder/f'example_{band}_{res}m.jp2').touch()
            scenes, _ = discover(root, ['31UCP'], datetime(2025,4,1), datetime(2025,5,1), 20)
            pair = tuple(sorted(scenes, key=lambda s:s.platform))
            wc = root/'wc.vrt'
            wc.touch()
            config = {'paths': {'worldcover_2021': str(wc)},
                      'selection': {'landcover_classes':[40], 'valid_scl':[4,5],
                                    'max_sun_zenith_deg_exclusive':70, 'near_nadir_deg_exclusive':4},
                      'sampling': {'seed':42, 'samples_per_class_per_tile':10, 'window_pixels':2},
                      'processing': {'geometry_band':'B04'}}
            class Dataset:
                width = height = 2
                crs = 'EPSG:32631'
                transform = types.SimpleNamespace(a=20, b=0, c=0, d=0, e=-20, f=40)
                def __init__(self, path):
                    self.name = str(path)
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def close(self): pass
                def read(self, *args, **kwargs):
                    if self.name.endswith('wc.vrt'):
                        return np.full((2,2), 40)
                    if '_SCL_' in self.name:
                        return np.array([[4,3],[5,4]]) if 'S2B_' in self.name else np.full((2,2), 4)
                    if '_B04_' in self.name: return np.full((2,2), 2000)
                    if '_B08_' in self.name: return np.full((2,2), 4000)
                    return np.full((2,2), 3000)
            class VRT(Dataset):
                def __init__(self, src, **kwargs): super().__init__(src.name)
            class Transformer:
                @staticmethod
                def from_crs(*args, **kwargs): return Transformer()
                def transform(self, x, y): return x, y
            class FakeAngles:
                def __init__(self, path, band): self.a = 'S2A_' in str(path)
                def at(self, x, y):
                    n = len(x)
                    return {'sza': np.full(n, 30), 'saa':np.full(n, 0),
                            'vza': np.where(np.asarray(x)<20, 3, 5) if self.a else np.full(n,10),
                            'vaa': np.full(n, 90 if self.a else 270)}
            rasterio = types.ModuleType('rasterio')
            rasterio.open = Dataset
            vrt = types.ModuleType('rasterio.vrt'); vrt.WarpedVRT = VRT
            enums = types.ModuleType('rasterio.enums'); enums.Resampling = types.SimpleNamespace(nearest=0)
            windows = types.ModuleType('rasterio.windows'); windows.Window = lambda *args: args
            pyproj = types.ModuleType('pyproj'); pyproj.Transformer = Transformer
            modules = {'rasterio':rasterio, 'rasterio.vrt':vrt, 'rasterio.enums':enums,
                       'rasterio.windows':windows, 'pyproj':pyproj}
            # A string-like CRS with rasterio's to_string method.
            Dataset.crs = type('CRS', (str,), {'to_string': lambda self: str(self)})('EPSG:32631')
            with patch.dict(sys.modules, modules), patch('brdf.process.AngleGrid', FakeAngles):
                rows, qa = sample_tile(pair, config, 'group')
            self.assertEqual(len(rows), 2)
            self.assertEqual(qa['valid_scl_landcover'], 3)
            self.assertEqual(qa['valid_geometry'], 2)
            self.assertTrue(all(row['near_nadir_platform'] == 'S2A' for row in rows))
            self.assertAlmostEqual(rows[0]['B04_reflectance_a'], 0.1)
            self.assertAlmostEqual(rows[0]['B12_reflectance_b'], 0.2)
            self.assertAlmostEqual(rows[0]['ndvi_a'], 0.5)
            config['sampling'].update(split_cropland_ndvi=True, bare_ndvi_threshold=0.6)
            with patch.dict(sys.modules, modules), patch('brdf.process.AngleGrid', FakeAngles):
                bare, qa = sample_tile(pair, config, 'group')
            self.assertEqual(qa['sampled_by_stratum']['40_bare'], 2)
            self.assertEqual(qa['sampled_by_stratum']['40_other'], 0)
            self.assertTrue(all(row['sampling_stratum'] == '40_bare' for row in bare))
            config['sampling']['bare_ndvi_threshold'] = 0.49
            with patch.dict(sys.modules, modules), patch('brdf.process.AngleGrid', FakeAngles):
                other, qa = sample_tile(pair, config, 'group')
            self.assertEqual(qa['sampled_by_stratum']['40_other'], 2)


if __name__ == '__main__':
    unittest.main()

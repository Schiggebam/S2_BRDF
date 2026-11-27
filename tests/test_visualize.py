import json
from collections import defaultdict
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
import visualize_results as vis


class VisualizeTests(unittest.TestCase):
    def test_variance_is_calculated_for_each_scene_pair_not_pooled(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            data = root/'samples.parquet'
            rows = []
            for tile, differences in [('32UPU', (0, 10, 20)), ('32UQU', (0, 20, 40))]:
                for difference in differences:
                    row = {'tile_id':tile, 'scene_a':'A_'+tile, 'scene_b':'B_'+tile,
                           'sensing_a_utc':'2025-03-18T10:17:51Z', 'sensing_b_utc':'2025-03-18T10:06:39Z',
                           'worldcover_class':30, 'near_nadir_platform':'S2A',
                           'solar_phase_a_deg':50, 'solar_phase_b_deg':40, 'ndvi_a':0.4, 'ndvi_b':0.4}
                    for band in vis.BANDS:
                        row[f'{band}_reflectance_a'] = 0.1 * (1 + difference/100)
                        row[f'{band}_reflectance_b'] = 0.1
                    rows.append(row)
            pq.write_table(pa.Table.from_pylist(rows), data)
            manifest = {'status':'complete', 'group_id':'example', 'completed_utc':'2025-01-01',
                        'samples_sha256':vis.checksum(data),
                        'identity':{'inputs':[{'id':str(i)} for i in range(4)]}}
            (root/'manifest.json').write_text(json.dumps(manifest))
            results, _ = vis.find_results(root)
            pair_values = defaultdict(list)
            vis.collect(results, 0.01, definition='a-b', pair_values=pair_values)
            stats = [row for row in vis.summarize_pairs(pair_values, 2) if row['band'] == 'B04']
            self.assertEqual(len(stats), 2)
            self.assertEqual([row['n_samples'] for row in stats], [3, 3])
            self.assertAlmostEqual(stats[0]['variance_pct_points_squared'], 100)
            self.assertAlmostEqual(stats[1]['variance_pct_points_squared'], 400)
            self.assertTrue(all(row['status'] == 'too_few_samples' for row in vis.summarize_pairs(pair_values, 4)))
            first_key = next(iter(pair_values))
            duplicate_key = ('other_group', *first_key[1:])
            pair_values[duplicate_key] = pair_values[first_key]
            group_results = [('2025-01-01', '', data, {'group_id':'example'}),
                             ('2025-02-01', '', data, {'group_id':'other_group'})]
            deduplicated, skipped = vis.deduplicate_matchups(pair_values, group_results)
            self.assertNotIn(first_key, deduplicated)
            self.assertIn(duplicate_key, deduplicated)
            self.assertEqual(skipped[0]['skipped_groups'], ['example'])

    def test_sensor_swap_phase_and_denominator_filter(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            data = root/'samples.parquet'
            rows = []
            for near_sensor, a, b, phase_a, phase_b in [
                ('S2A', 0.1, 0.12, 40, 30),
                ('S2B', 0.12, 0.1, 30, 40),
                ('S2A', 0.001, 0.1, 40, 30),
            ]:
                row = {'worldcover_class':40, 'near_nadir_platform':near_sensor,
                       'solar_phase_a_deg':phase_a, 'solar_phase_b_deg':phase_b,
                       'ndvi_a': 0.05, 'ndvi_b': 0.05}
                for band in vis.BANDS:
                    row[f'{band}_reflectance_a'] = a
                    row[f'{band}_reflectance_b'] = b
                rows.append(row)
            pq.write_table(pa.Table.from_pylist(rows), data)
            manifest = {'status':'complete', 'group_id':'example', 'completed_utc':'2025-01-01',
                        'samples_sha256':vis.checksum(data),
                        'identity':{'inputs':[{'id':str(i)} for i in range(4)]}}
            (root/'manifest.json').write_text(json.dumps(manifest))
            results, skipped = vis.find_results(root)
            self.assertFalse(skipped)
            values, groups, counts, _ = vis.collect(results, 0.01)
            self.assertEqual(counts['B04']['low_near'], 1)
            self.assertEqual(counts['B04']['valid'], 2)
            self.assertEqual(len(groups[('40_bare', 'B04', 'S2A near · lower phase')]), 1)
            for number in values[('40_bare', 'B04', 'S2A near · lower phase')]:
                self.assertAlmostEqual(number, 20)
            self.assertAlmostEqual(values[('40_bare', 'B04', 'S2B near · lower phase')][0], 20)
            summary = vis.summarize(values, groups)
            self.assertAlmostEqual(summary[0]['mean_pct'], 20)
            values2, _, _, _ = vis.collect(results, 0.01, ndvi_threshold=0.05)
            self.assertIn(('40_vegetated', 'B04', 'S2A near · lower phase'), values2)
            fixed, _, _, _ = vis.collect(results, 0.01, definition='a-b')
            self.assertAlmostEqual(fixed[('40_bare', 'B04', 'S2A near · lower phase')][0], -100 / 6)
            self.assertAlmostEqual(fixed[('40_bare', 'B04', 'S2B near · lower phase')][0], 20)


if __name__ == '__main__':
    unittest.main()

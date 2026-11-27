from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sys
import uuid

import yaml

from . import __version__
from .discovery import discover, horizontal_neighbors, make_groups, read_tiles, tile_geometry


def date(value):
    try:
        return datetime.strptime(value, '%Y-%m-%d')
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Expected YYYY-MM-DD') from exc


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError('Configuration must be a YAML mapping')
    paths = config['paths']
    for key in ('l2a_archive', 'worldcover_2021', 'tile_list', 'output'):
        if not isinstance(paths.get(key), str) or not paths[key].strip():
            raise ValueError(f'paths.{key} must be set')
        paths[key] = str(Path(paths[key]).expanduser().resolve())
    if not Path(paths['l2a_archive']).is_dir():
        raise ValueError('L2A archive directory does not exist')
    for key in ('worldcover_2021', 'tile_list'):
        if not Path(paths[key]).is_file():
            raise ValueError(f'paths.{key} does not exist: {paths[key]}')
    s, p, run = config['selection'], config['sampling'], config['processing']
    if not 0 < float(s['max_cloud_percent_exclusive']) <= 100:
        raise ValueError('Cloud limit must be in (0, 100]')
    if not 0 < float(s['max_sun_zenith_deg_exclusive']) <= 90 or not 0 < float(s['near_nadir_deg_exclusive']) <= 90:
        raise ValueError('Angle limits must be in (0, 90]')
    if not s['valid_scl'] or not s['landcover_classes']:
        raise ValueError('At least one SCL and land-cover class is required')
    if float(s['max_pair_minutes']) <= 0 or float(s['max_neighbor_sensing_minutes']) < 0:
        raise ValueError('Invalid time limits')
    if int(p['samples_per_class_per_tile']) < 1 or int(p['window_pixels']) < 1:
        raise ValueError('Sample count and window size must be positive')
    if int(run['workers']) < 1 or run['geometry_band'] not in ('B04', 'B08', 'B12'):
        raise ValueError('Invalid worker count or geometry band')
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description='Sample adjacent-tile Sentinel-2A/B L2A overpasses')
    parser.add_argument('--config', required=True, help='YAML configuration file')
    parser.add_argument('--start', type=date, required=True, help='Inclusive UTC date, YYYY-MM-DD')
    parser.add_argument('--end', type=date, required=True, help='Exclusive UTC date, YYYY-MM-DD')
    args = parser.parse_args(argv)
    if args.end <= args.start:
        parser.error('--end must be later than --start')
    config = load_config(args.config)
    paths, select = config['paths'], config['selection']
    tiles = read_tiles(paths['tile_list'])
    out = Path(paths['output'])
    out.mkdir(parents=True, exist_ok=True)
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:10]}"
    run_dir = out/'runs'/run_id
    run_dir.mkdir(parents=True)
    report = {'run_id': run_id, 'software_version': __version__, 'config': config,
              'start_inclusive_utc': args.start.isoformat()+'Z', 'end_exclusive_utc': args.end.isoformat()+'Z',
              'tile_count': len(tiles), 'issues': [], 'groups': []}
    from .process import atomic_json, process_group
    try:
        print(f'Discovering scenes across {len(tiles)} tiles', flush=True)
        # Search a margin across period boundaries, then assign each group to
        # the interval containing the earliest of its four sensing times.
        margin = timedelta(minutes=float(select['max_pair_minutes']) + float(select['max_neighbor_sensing_minutes']))
        scenes, issues = discover(paths['l2a_archive'], tiles, args.start-margin, args.end+margin,
                                  float(select['max_cloud_percent_exclusive']))
        report['discovery_margin_minutes'] = margin.total_seconds()/60
        report['issues'].extend(issues)
        report['eligible_scene_count'] = len(scenes)
        report['eligible_scenes'] = [{'id': s.id, 'tile': s.tile, 'path': str(s.path),
                                      'cloud_percent': s.cloud} for s in scenes]
        report['tiles_without_eligible_scenes'] = sorted(set(tiles) - {s.tile for s in scenes})
        geometries = {}
        for scene in scenes:
            if scene.tile not in geometries:
                try:
                    geometries[scene.tile] = tile_geometry(scene)
                except Exception as exc:
                    report['issues'].append(f'{scene.tile}: cannot read tile geometry: {exc}')
        neighbors = []
        available = sorted(geometries)
        for i, a in enumerate(available):
            (lon1, lat1), _ = geometries[a]
            for b in available[i+1:]:
                (lon2, lat2), _ = geometries[b]
                lon_limit = min(180, 2.0/max(0.01, math.cos(math.radians((lat1+lat2)/2))))
                if abs(lat1-lat2) < 0.6 and abs(lon1-lon2) < lon_limit:
                    if horizontal_neighbors(geometries[a], geometries[b]):
                        neighbors.append((a, b))
        report['neighbor_pairs'] = neighbors
        groups = make_groups(scenes, neighbors, float(select['max_pair_minutes']),
                             float(select['max_neighbor_sensing_minutes']))
        groups = [g for g in groups if args.start <= min(s.sensing for pair in g for s in pair) < args.end]
        report['group_count'] = len(groups)
        print(f'Found {len(scenes)} eligible scenes, {len(neighbors)} neighboring tile pairs, {len(groups)} groups', flush=True)
        workers = int(config['processing']['workers'])
        if workers == 1:
            for group in groups:
                try:
                    result = process_group(group, config)
                    report['groups'].append(result)
                    print(json.dumps(result), flush=True)
                except Exception as exc:
                    report['issues'].append(f"Group {[s.id for pair in group for s in pair]}: {type(exc).__name__}: {exc}")
                    print(report['issues'][-1], file=sys.stderr, flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(process_group, g, config): g for g in groups}
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        report['groups'].append(result)
                        print(json.dumps(result), flush=True)
                    except Exception as exc:
                        group = futures[future]
                        report['issues'].append(f"Group {[s.id for pair in group for s in pair]}: {type(exc).__name__}: {exc}")
                        print(report['issues'][-1], file=sys.stderr, flush=True)
        report['status'] = 'completed_with_errors' if report['issues'] else 'complete'
    except Exception as exc:
        report['status'] = 'failed'
        report['issues'].append(f'{type(exc).__name__}: {exc}')
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    atomic_json(run_dir/'run.json', report)
    print(f'Run report: {run_dir / "run.json"}', flush=True)
    if report['issues']:
        for issue in report['issues']:
            print(issue, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

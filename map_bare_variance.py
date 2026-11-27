#!/usr/bin/env python3
"""Map bare-crop variability and make detailed plots for top tile matchups."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from investigate_landcover import analyze_matchup, load_matchups
from visualize_results import collect, find_results, make_figure, summarize


def location_for_sector(matchup, band, configuration, threshold):
    """Median coordinates of the same valid bare-crop samples used in variance."""
    d = matchup['data']
    near_a = d['near_nadir_platform'] == 'S2A'
    ndvi = np.where(near_a, d['ndvi_a'].astype(float), d['ndvi_b'].astype(float))
    a, b = d[f'{band}_reflectance_a'].astype(float), d[f'{band}_reflectance_b'].astype(float)
    near = np.where(near_a, a, b)
    phase_a, phase_b = d['solar_phase_a_deg'].astype(float), d['solar_phase_b_deg'].astype(float)
    delta = np.where(near_a, phase_b-phase_a, phase_a-phase_b)
    sza = np.where(near_a, d['sza_a_deg'].astype(float), d['sza_b_deg'].astype(float))
    direction = np.array([f'{s} near · {p} phase' for s, p in zip(d['near_nadir_platform'],
        np.where(delta < -1e-8, 'lower', np.where(delta > 1e-8, 'higher', 'equal')))])
    longitude, latitude = d['longitude'].astype(float), d['latitude'].astype(float)
    eligible = ((d['worldcover_class'].astype(int) == 40) & np.isfinite(ndvi) & (ndvi < 0.1) &
                np.isfinite(a) & np.isfinite(b) & (near >= threshold) & (b >= threshold) &
                np.isfinite(delta) & np.isfinite(sza) & (direction == configuration))
    located = eligible & np.isfinite(longitude) & np.isfinite(latitude) & (abs(longitude) <= 180) & (abs(latitude) <= 90)
    if not located.any():
        return None
    return float(np.median(longitude[located])), float(np.median(latitude[located])), int(located.sum())


def rank_matchups(matchups, band, threshold, minimum):
    """Rank each tile pair by its largest qualifying within-sector variance."""
    ranked, skipped = [], []
    for key, matchup in matchups.items():
        classes, _, _ = analyze_matchup(matchup, threshold, minimum)
        candidates = [r for r in classes if r['band'] == band and r['category'] == 'Cropland: NDVI < 0.1'
                      and r['status'] == 'ok' and r['variance_pct_points_squared'] is not None]
        if not candidates:
            skipped.append({'tile_id':key[0], 'scene_a':key[1], 'scene_b':key[2], 'reason':'too few valid bare-cropland samples'})
            continue
        row = max(candidates, key=lambda r: (r['variance_pct_points_squared'], r['configuration']))
        location = location_for_sector(matchup, band, row['configuration'], threshold)
        if location is None:
            skipped.append({'tile_id':key[0], 'scene_a':key[1], 'scene_b':key[2], 'reason':'no valid geographic sample coordinates'})
            continue
        lon, lat, n_located = location
        ranked.append({**row, 'longitude':lon, 'latitude':lat, 'n_located':n_located})
    ranked.sort(key=lambda r: (-r['variance_pct_points_squared'], r['tile_id'], r['scene_a'], r['scene_b']))
    for i, row in enumerate(ranked, start=1): row['rank'] = i
    return ranked, skipped


def draw_outlines(ax, gpkg, layer, id_field, matchups):
    from brdf.discovery import read_geopackage, parse_scene, tile_geometry
    if gpkg:
        geometries, _ = read_geopackage(gpkg, layer=layer, id_field=id_field)
        source = 'tile GeoPackage'
    else:
        # The original archive need not be mounted for analysis. When it is,
        # use its scene tile grids as optional geographic context.
        geometries = {}
        for matchup in matchups.values():
            try:
                inputs = json.loads(Path(matchup['manifest']).read_text(encoding='utf-8'))['identity']['inputs']
                tile = str(matchup['data']['tile_id'][0])
                if tile in geometries: continue
                record = next(r for r in inputs if r['id'] == str(matchup['data']['scene_a'][0]))
                _, geometries[tile] = tile_geometry(parse_scene(Path(record['path'])))
            except (OSError, ValueError, KeyError, StopIteration, ET.ParseError):
                continue
        source = 'available Sentinel-2 tile metadata' if geometries else None
    for geometry in geometries.values():
        for polygon in ([geometry] if geometry.geom_type == 'Polygon' else geometry.geoms):
            x, y = polygon.exterior.xy
            ax.plot(x, y, color='#b0b9bd', linewidth=.35, zorder=1)
    return len(geometries), source


def make_map(rows, output, band, selected_count, tile_gpkg, layer, id_field, matchups):
    fig, ax = plt.subplots(figsize=(10,8), constrained_layout=True)
    n_outlines, source = draw_outlines(ax, tile_gpkg, layer, id_field, matchups)
    if rows:
        x = np.array([r['longitude'] for r in rows]); y = np.array([r['latitude'] for r in rows])
        v = np.array([r['variance_pct_points_squared'] for r in rows])
        dots = ax.scatter(x, y, c=v, cmap='viridis', s=75, edgecolors='white', linewidths=.7, alpha=.9, zorder=3)
        cbar = fig.colorbar(dots, ax=ax, shrink=.75, pad=.02)
        cbar.set_label(f'{band} within-matchup variance (percentage points²)')
        for row in rows[:selected_count]:
            ax.annotate(f"{row['rank']}: {row['tile_id']}", (row['longitude'], row['latitude']),
                        xytext=(6,6), textcoords='offset points', fontsize=8, fontweight='bold', zorder=4)
        xmin, ymin, width, height = ax.dataLim.bounds
        margin_x = max(.25, width*.06)
        margin_y = max(.20, height*.06)
        ax.set_xlim(xmin-margin_x,xmin+width+margin_x)
        ax.set_ylim(ymin-margin_y,ymin+height+margin_y)
        ax.set_aspect(1 / max(.1, math.cos(math.radians(float(np.median(y))))))
    else:
        ax.text(.5,.5,'No tile matchups meet the bare-crop sample criteria', transform=ax.transAxes,ha='center')
    ax.set_xlabel('Longitude (degrees)'); ax.set_ylabel('Latitude (degrees)')
    ax.grid(color='#d0d9da', alpha=.7, linewidth=.6)
    ax.set_title(f'Bare-cropland variance by S2A–S2B tile matchup — {band}',fontweight='bold')
    subtitle = f'Tile outlines from {source}; no coastline or political basemap' if n_outlines else 'Longitude/latitude overview; no coastline or political basemap'
    fig.text(.5,.015,subtitle + '; labels mark the highest-ranked matchups. Multiple dates at one tile can overlap.', ha='center',fontsize=8)
    fig.savefig(output,dpi=170,bbox_inches='tight'); plt.close(fig)
    return n_outlines, source


def write_csv(path, rows):
    fields = ['rank','tile_id','sensing_a_utc','sensing_b_utc','scene_a','scene_b','group_id','band','configuration',
              'n','n_located','variance_pct_points_squared','std_pct_points','iqr_pct_points','median_pct',
              'median_phase_change_deg','longitude','latitude']
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def make_details(rows, result_by_manifest, output, threshold):
    produced = []
    for row in rows:
        key = (row['tile_id'], row['scene_a'], row['scene_b'])
        result = result_by_manifest[row['_manifest']]
        values, groups, counts, _ = collect([result], threshold, definition='a-b', matchup_filter=key)
        if not values:
            raise ValueError(f'No samples after filtering selected matchup {key}')
        suffix = hashlib.sha256('|'.join(key).encode()).hexdigest()[:8]
        target = output/'high_variance'/f"{row['rank']:03d}_{row['sensing_a_utc'][:10]}_{row['tile_id']}_{suffix}"
        target.mkdir(parents=True,exist_ok=True)
        stats = summarize(values,groups)
        with (target/'summary.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=stats[0].keys());writer.writeheader();writer.writerows(stats)
        make_figure(values,target/'relative_difference.png',threshold,0.1,definition='a-b',
                    title=f"{row['tile_id']} — {row['sensing_a_utc'][:10]} — S2A/S2B matchup #{row['rank']}")
        # The four-scene source group includes the horizontally adjacent tile.
        # Preserve both the single-tile drill-down and the two-tile comparison.
        adjacent_values, adjacent_groups, _, _ = collect([result], threshold, definition='a-b')
        adjacent_stats = summarize(adjacent_values, adjacent_groups)
        with (target/'adjacent_tiles_summary.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=adjacent_stats[0].keys());writer.writeheader();writer.writerows(adjacent_stats)
        make_figure(adjacent_values,target/'adjacent_tiles_overview.png',threshold,0.1,definition='a-b',
                    title=f"Adjacent tile matchups — {row['sensing_a_utc'][:10]} — selected #{row['rank']}")
        metadata={k:v for k,v in row.items() if k != '_manifest'}
        metadata.update({'source_manifest':row['_manifest'],'counts_by_band':counts,
                         'adjacent_tile_scene_ids':[record['id'] for record in result[3]['identity']['inputs']],
                         'formula':'100 * (S2A - S2B) / S2B','sample_filter':'near-nadir and S2B reflectance above configured threshold'})
        (target/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
        produced.append(str(target))
    return produced


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results_dir',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--band',choices=['B04','B08','B12'],default='B08',help='Band used for ranking (default B08)')
    parser.add_argument('--top',type=int,default=5,help='Number of highest-variance tile matchups to detail (default 5)')
    parser.add_argument('--min-samples',type=int,default=10,help='Minimum valid bare-crop samples per geometry category (default 10)')
    parser.add_argument('--min-reflectance',type=float,default=.01,help='Minimum S2B and near-nadir reflectance (default .01)')
    parser.add_argument('--tiles-geopackage',type=Path,help='Optional tile GeoPackage to draw geographic outlines')
    parser.add_argument('--tile-layer',help='Tile GeoPackage layer, if multiple feature layers exist')
    parser.add_argument('--tile-id-field',help='Tile ID attribute, if not unambiguous')
    args=parser.parse_args(argv)
    if not args.results_dir.is_dir():parser.error('results_dir must exist')
    if args.top < 1 or args.min_samples < 2:parser.error('--top must be positive and --min-samples at least 2')
    if not np.isfinite(args.min_reflectance) or args.min_reflectance <= 0:parser.error('--min-reflectance must be positive')
    if args.tiles_geopackage and not args.tiles_geopackage.is_file():parser.error('--tiles-geopackage must exist')
    results, skipped_groups=find_results(args.results_dir)
    if not results:raise SystemExit('No complete, checksum-valid results found')
    matchups, duplicates=load_matchups(results,extra_columns=('longitude','latitude'))
    ranked, skipped_matchups=rank_matchups(matchups,args.band,args.min_reflectance,args.min_samples)
    for row in ranked:
        row['_manifest'] = matchups[(row['tile_id'],row['scene_a'],row['scene_b'])]['manifest']
    args.output.mkdir(parents=True,exist_ok=True)
    write_csv(args.output/'bare_variance_ranking.csv',ranked)
    n_outlines, outline_source=make_map(ranked,args.output/'bare_variance_map.png',args.band,min(args.top,len(ranked)),
                                        args.tiles_geopackage,args.tile_layer,args.tile_id_field,matchups)
    result_by_manifest={entry[1]:entry for entry in results}
    detail_dirs=make_details(ranked[:args.top],result_by_manifest,args.output,args.min_reflectance)
    report={'formula':'100 * (S2A - S2B) / S2B', 'bare_definition':'near-nadir NDVI < 0.1',
            'band':args.band,'top_requested':args.top,'minimum_samples_per_sector':args.min_samples,
            'min_reflectance':args.min_reflectance,'matchups':len(matchups),'ranked_matchups':len(ranked),
            'ranking':'largest sample variance of bare cropland among qualifying sensor/phase categories within each tile matchup',
            'map_location':'median longitude and latitude of valid bare-cropland samples in the selected category',
            'tile_outlines':n_outlines,'tile_outline_source':outline_source,'details':detail_dirs,'skipped_groups':skipped_groups,
            'duplicate_matchups':duplicates,'skipped_matchups':skipped_matchups,
            'note':'High means highest in the selected dataset, not an absolute threshold; repeated dates at one location may overlap on the map.'}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f'Ranked {len(ranked)} tile matchups; created {len(detail_dirs)} detailed plots in {args.output}')
    if not ranked: print('No matchups passed bare-crop and coordinate criteria.',file=sys.stderr)


if __name__ == '__main__': main()

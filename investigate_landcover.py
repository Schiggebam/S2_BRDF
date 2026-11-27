#!/usr/bin/env python3
"""Investigate within-matchup spread, geometry contrasts and crop NDVI."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pyarrow.parquet as pq

# Reuse the existing checksum-verified results discovery; no analysis pipeline import.
from visualize_results import find_results, COLORS, DIRECTIONS, legend_label

BANDS = ('B04', 'B08', 'B12')
CLASSES = ('Tree cover', 'Grassland', 'Cropland: NDVI < 0.1', 'Cropland: NDVI ≥ 0.1')
COMPARE = ('Cropland: NDVI ≥ 0.1', 'Grassland', 'Tree cover')
NDVI_EDGES = (0.0, 0.1, 0.2, 0.4, 0.6)
NDVI_LABELS = ('<0', '0–0.1', '0.1–0.2', '0.2–0.4', '0.4–0.6', '≥0.6')
REQUIRED = ['tile_id', 'scene_a', 'scene_b', 'sensing_a_utc', 'sensing_b_utc',
            'worldcover_class', 'near_nadir_platform', 'ndvi_a', 'ndvi_b',
            'solar_phase_a_deg', 'solar_phase_b_deg', 'sza_a_deg', 'sza_b_deg']
REQUIRED += [f'{band}_reflectance_{sensor}' for band in BANDS for sensor in ('a', 'b')]


def load_matchups(results):
    """Keep the newest complete sample set for each specific tile and scene pair."""
    matchups = {}
    skipped = []
    for stamp, manifest_path, path, manifest in results:
        table = pq.read_table(path)
        if table.num_rows == 0:
            continue
        missing = sorted(set(REQUIRED) - set(table.column_names))
        if missing:
            raise ValueError(f'{path}: missing columns: {missing}')
        data = {name: np.asarray(table[name].to_pylist()) for name in REQUIRED}
        keys = list(zip(*(data[k] for k in ('tile_id', 'scene_a', 'scene_b'))))
        if any(any(v is None or not str(v).strip() for v in key) for key in keys):
            raise ValueError(f'{path}: missing scene or tile identifier')
        for key in sorted(set(keys)):
            mask = np.array([row == key for row in keys])
            candidate = {'group_id': manifest['group_id'], 'manifest': manifest_path,
                         'completed_utc': stamp, 'data': {k: v[mask] for k, v in data.items()}}
            rank = (stamp, manifest_path)
            if key in matchups:
                old = matchups[key]
                if rank > (old['completed_utc'], old['manifest']):
                    skipped.append({'matchup': key, 'skipped_manifest': old['manifest'], 'kept_manifest': manifest_path})
                    matchups[key] = candidate
                else:
                    skipped.append({'matchup': key, 'skipped_manifest': manifest_path, 'kept_manifest': old['manifest']})
            else:
                matchups[key] = candidate
    return matchups, skipped


def analyze_matchup(matchup, threshold, minimum):
    """Return class and NDVI-bin summaries without pooling distinct matchups."""
    d = matchup['data']
    n = len(d['tile_id'])
    if not np.isin(d['near_nadir_platform'], ['S2A', 'S2B']).all():
        raise ValueError('Unknown near-nadir platform')
    for key in ('sensing_a_utc', 'sensing_b_utc'):
        if len(set(d[key])) != 1:
            raise ValueError(f'Inconsistent {key} within scene pair')
    near_a = d['near_nadir_platform'] == 'S2A'
    near_ndvi = np.where(near_a, d['ndvi_a'].astype(float), d['ndvi_b'].astype(float))
    phase_a, phase_b = d['solar_phase_a_deg'].astype(float), d['solar_phase_b_deg'].astype(float)
    phase_delta = np.where(near_a, phase_b-phase_a, phase_a-phase_b)
    sun_a, sun_b = d['sza_a_deg'].astype(float), d['sza_b_deg'].astype(float)
    sun_near = np.where(near_a, sun_a, sun_b)
    category = np.array([f'{sensor} near · {direction} phase' for sensor, direction in zip(
        d['near_nadir_platform'], np.where(phase_delta < -1e-8, 'lower', np.where(phase_delta > 1e-8, 'higher', 'equal')))])
    classes = d['worldcover_class'].astype(int)
    class_masks = {'Tree cover': classes == 10, 'Grassland': classes == 30,
                   'Cropland: NDVI < 0.1': (classes == 40) & np.isfinite(near_ndvi) & (near_ndvi < 0.1),
                   'Cropland: NDVI ≥ 0.1': (classes == 40) & np.isfinite(near_ndvi) & (near_ndvi >= 0.1)}
    ndvi_bin = np.searchsorted(NDVI_EDGES, near_ndvi, side='right')
    base = {'group_id': matchup['group_id'], 'tile_id': str(d['tile_id'][0]),
            'scene_a': str(d['scene_a'][0]), 'scene_b': str(d['scene_b'][0]),
            'sensing_a_utc': str(d['sensing_a_utc'][0]), 'sensing_b_utc': str(d['sensing_b_utc'][0])}
    class_rows, bin_rows, exclusions = [], [], {}
    def add_stats(mask, label, band, direction, is_bin):
        count = int(mask.sum())
        if not count:
            return
        values = differences[mask]
        row = {**base, 'band':band, 'configuration':direction, 'category':label, 'n':count,
               'status':'ok' if count >= minimum else 'too_few_samples',
               'median_pct':float(np.median(values)), 'mean_pct':float(np.mean(values)),
               'iqr_pct_points':float(np.percentile(values,75)-np.percentile(values,25)),
               'variance_pct_points_squared':float(np.var(values,ddof=1)) if count >= 2 else None,
               'std_pct_points':float(np.std(values,ddof=1)) if count >= 2 else None,
               'median_phase_change_deg':float(np.median(phase_delta[mask])),
               'median_near_sza_deg':float(np.median(sun_near[mask])),
               'median_near_ndvi':float(np.median(near_ndvi[mask & np.isfinite(near_ndvi)])) if np.any(mask & np.isfinite(near_ndvi)) else None}
        (bin_rows if is_bin else class_rows).append(row)
    for band in BANDS:
        a, b = d[f'{band}_reflectance_a'].astype(float), d[f'{band}_reflectance_b'].astype(float)
        near = np.where(near_a, a, b)
        valid = (np.isfinite(a) & np.isfinite(b) & np.isfinite(phase_delta) & np.isfinite(sun_near) &
                 (near >= threshold) & (b >= threshold))
        exclusions[band] = {'total':n, 'excluded_invalid_or_low_reflectance_or_geometry':int((~valid).sum()),
                            'excluded_cropland_ndvi_missing':int(np.count_nonzero(valid & (classes == 40) & ~np.isfinite(near_ndvi)))}
        differences = np.divide(100*(a-b), b, out=np.full(n, np.nan), where=valid)
        for direction in DIRECTIONS:
            sector = valid & (category == direction)
            for label, class_mask in class_masks.items():
                add_stats(sector & class_mask, label, band, direction, False)
            for index, label in enumerate(NDVI_LABELS):
                add_stats(sector & (classes == 40) & np.isfinite(near_ndvi) & (ndvi_bin == index), label, band, direction, True)
    return class_rows, bin_rows, exclusions


def make_contrasts(class_rows):
    lookup = {(r['group_id'], r['tile_id'], r['scene_a'], r['scene_b'], r['band'], r['configuration'], r['category']): r
              for r in class_rows if r['status'] == 'ok'}
    contrasts = []
    for row in class_rows:
        if row['category'] != CLASSES[2] or row['status'] != 'ok':
            continue
        key = tuple(row[k] for k in ('group_id', 'tile_id', 'scene_a', 'scene_b', 'band', 'configuration'))
        for category in COMPARE:
            other = lookup.get((*key, category))
            if other is None:
                continue
            contrasts.append({**{k:row[k] for k in ('group_id','tile_id','scene_a','scene_b','sensing_a_utc','sensing_b_utc','band','configuration')},
                'comparison_class':category, 'n_bare':row['n'], 'n_comparison':other['n'],
                'bare_median_pct':row['median_pct'], 'comparison_median_pct':other['median_pct'],
                'contrast_pct_points':row['median_pct']-other['median_pct'],
                'bare_median_phase_change_deg':row['median_phase_change_deg'],
                'comparison_median_phase_change_deg':other['median_phase_change_deg'],
                'bare_median_near_sza_deg':row['median_near_sza_deg'],
                'comparison_median_near_sza_deg':other['median_near_sza_deg']})
    return contrasts


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


SUMMARY_FIELDS = ['group_id','tile_id','scene_a','scene_b','sensing_a_utc','sensing_b_utc','band','configuration',
                  'category','n','status','median_pct','mean_pct','iqr_pct_points','variance_pct_points_squared',
                  'std_pct_points','median_phase_change_deg','median_near_sza_deg','median_near_ndvi']
CONTRAST_FIELDS = ['group_id','tile_id','scene_a','scene_b','sensing_a_utc','sensing_b_utc','band','configuration',
                   'comparison_class','n_bare','n_comparison','bare_median_pct','comparison_median_pct',
                   'contrast_pct_points','bare_median_phase_change_deg','comparison_median_phase_change_deg',
                   'bare_median_near_sza_deg','comparison_median_near_sza_deg']


def legend(fig, configurations):
    handles = [Line2D([0],[0], marker='o', color='none', markerfacecolor=COLORS[c], markeredgecolor='none',
                      markersize=7, label=legend_label(c)) for c in configurations]
    if handles:
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5,0.94),
                   ncol=min(len(handles),2), frameon=False, fontsize=8)


def scatter_spread(rows, output):
    ok = [r for r in rows if r['status'] == 'ok']
    configurations = [c for c in DIRECTIONS if any(r['configuration'] == c for r in ok)]
    fig, axes = plt.subplots(2,3, figsize=(15,9), sharex=True)
    fig.subplots_adjust(left=.07,right=.98,bottom=.16,top=.80,hspace=.24,wspace=.28)
    for j, band in enumerate(BANDS):
        for i, (field, label) in enumerate([('iqr_pct_points','IQR (percentage points)'),
                                             ('variance_pct_points_squared','Variance (percentage points²; symlog)')]):
            ax = axes[i,j]
            for row in (r for r in ok if r['band'] == band and r[field] is not None):
                # Stable, small offset makes multiple overpasses visible without aggregating.
                import hashlib
                token = (row['group_id']+'|'+row['tile_id']+'|'+row['configuration']).encode()
                jitter = (int(hashlib.sha256(token).hexdigest()[:8],16)/0xffffffff-.5)*.13
                direction_offset = (configurations.index(row['configuration'])-(len(configurations)-1)/2)*.16
                ax.scatter(CLASSES.index(row['category'])+direction_offset+jitter, row[field],
                           c=COLORS[row['configuration']], s=26, alpha=.7, edgecolors='white', linewidths=.3)
            if i == 1: ax.set_yscale('symlog',linthresh=1)
            if j == 0: ax.set_ylabel(label)
            if i == 0: ax.set_title(band,fontweight='bold')
            ax.set_xticks(range(4), ['Tree','Grass','Crop\nNDVI <0.1','Crop\nNDVI ≥0.1'], fontsize=8)
            ax.grid(axis='y',alpha=.2)
            if not any(r['band'] == band for r in ok): ax.text(.5,.5,'No qualifying data',transform=ax.transAxes,ha='center')
    legend(fig,configurations)
    fig.suptitle('Within-matchup variability: IQR and variance',y=.98,fontsize=15,fontweight='bold')
    fig.text(.5,.03,'One point per tile matchup, class, band and sensor/phase category; no pooled pixel averages.',ha='center',fontsize=9)
    fig.savefig(output,dpi=160,bbox_inches='tight'); plt.close(fig)


def scatter_geometry(rows, output):
    configs = [c for c in DIRECTIONS if any(r['configuration'] == c for r in rows)]
    fig, axes = plt.subplots(3,3,figsize=(15,10),sharex=False,sharey=False)
    fig.subplots_adjust(left=.08,right=.98,bottom=.12,top=.81,wspace=.30,hspace=.36)
    labels = ('Bare − higher-NDVI crop','Bare − grassland','Bare − tree cover')
    for i, comp in enumerate(COMPARE):
        for j, band in enumerate(BANDS):
            ax=axes[i,j]
            subset=[r for r in rows if r['band']==band and r['comparison_class']==comp]
            for r in subset:
                ax.scatter(r['bare_median_phase_change_deg'],r['contrast_pct_points'],c=COLORS[r['configuration']],s=30,alpha=.7,edgecolors='white',linewidths=.3)
            ax.axhline(0,color='gray',ls='--',lw=.7); ax.axvline(0,color='gray',ls=':',lw=.7)
            ax.grid(alpha=.15)
            if i==0: ax.set_title(band,fontweight='bold')
            if j==0: ax.set_ylabel(labels[i]+'\nMedian contrast (pp)',fontsize=9)
            if i==2: ax.set_xlabel('Off − near solar phase (degrees)')
            if not subset: ax.text(.5,.5,'No qualifying pairs',transform=ax.transAxes,ha='center',fontsize=9)
    legend(fig,configs)
    fig.suptitle('Within-matchup class contrast versus geometry',y=.98,fontsize=15,fontweight='bold')
    fig.text(.5,.025,'Each point: median D of bare crop minus median D of the comparison class in the same tile matchup and sensor/phase category. X: bare crop median phase change.',ha='center',fontsize=8)
    fig.savefig(output,dpi=160,bbox_inches='tight'); plt.close(fig)


def ndvi_curves(rows, output):
    ok=[r for r in rows if r['status']=='ok']
    configs=[c for c in DIRECTIONS if any(r['configuration']==c for r in ok)]
    curves=defaultdict(dict)
    for row in ok:
        key=tuple(row[k] for k in ('group_id','tile_id','scene_a','scene_b','band','configuration'))
        curves[key][row['category']]=row['median_pct']
    fig,axes=plt.subplots(1,3,figsize=(15,6),sharey=False)
    fig.subplots_adjust(left=.07,right=.98,bottom=.20,top=.78,wspace=.30)
    for ax,band in zip(axes,BANDS):
        has_data=False
        for key, bins in curves.items():
            if key[4]!=band: continue
            has_data=True
            values=[bins.get(label,np.nan) for label in NDVI_LABELS]
            ax.plot(range(len(NDVI_LABELS)),values,marker='o',markersize=3,lw=.8,alpha=.45,color=COLORS[key[5]])
        ax.axhline(0,color='gray',ls='--',lw=.7)
        ax.set_xticks(range(len(NDVI_LABELS)),NDVI_LABELS,rotation=30,ha='right',fontsize=8)
        ax.set_xlabel('Near-nadir cropland NDVI bin'); ax.set_ylabel('Median S2A − S2B relative difference (%)')
        ax.set_title(band,fontweight='bold'); ax.grid(axis='y',alpha=.2)
        if not has_data: ax.text(.5,.5,'No qualifying bins',transform=ax.transAxes,ha='center')
    legend(fig,configs)
    fig.suptitle('Cropland NDVI response by tile matchup',y=.98,fontsize=15,fontweight='bold')
    fig.text(.5,.03,'One line per tile matchup and sensor/phase category; gaps mean too few samples. Sample counts are not area fractions.',ha='center',fontsize=9)
    fig.savefig(output,dpi=160,bbox_inches='tight'); plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results_dir',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--min-reflectance',type=float,default=.01,help='Minimum S2B and near-nadir reflectance (default 0.01)')
    parser.add_argument('--min-samples',type=int,default=10,help='Minimum samples per matchup/category/bin (default 10)')
    args=parser.parse_args(argv)
    if not args.results_dir.is_dir(): parser.error('results_dir must exist')
    if not np.isfinite(args.min_reflectance) or args.min_reflectance <= 0: parser.error('--min-reflectance must be positive')
    if args.min_samples < 2: parser.error('--min-samples must be at least 2')
    results, skipped=find_results(args.results_dir)
    if not results: raise SystemExit('No complete, checksum-valid results found')
    matchups, duplicates=load_matchups(results)
    class_rows,bin_rows, exclusions=[],[],[]
    for matchup in matchups.values():
        classes,bins,counts=analyze_matchup(matchup,args.min_reflectance,args.min_samples)
        class_rows.extend(classes); bin_rows.extend(bins)
        exclusions.append({'group_id':matchup['group_id'],'tile_id':str(matchup['data']['tile_id'][0]),'counts_by_band':counts})
    contrasts=make_contrasts(class_rows)
    args.output.mkdir(parents=True,exist_ok=True)
    write_csv(args.output/'class_summary.csv',class_rows,SUMMARY_FIELDS)
    write_csv(args.output/'ndvi_bin_summary.csv',bin_rows,SUMMARY_FIELDS)
    write_csv(args.output/'geometry_contrasts.csv',contrasts,CONTRAST_FIELDS)
    scatter_spread(class_rows,args.output/'01_within_matchup_spread.png')
    scatter_geometry(contrasts,args.output/'02_contrast_vs_geometry.png')
    ndvi_curves(bin_rows,args.output/'03_cropland_ndvi.png')
    report={'formula':'100 * (S2A - S2B) / S2B', 'minimum_reflectance':args.min_reflectance,
            'minimum_samples':args.min_samples,'matchup_count':len(matchups),'class_rows':len(class_rows),
            'ndvi_bin_rows':len(bin_rows),'contrast_rows':len(contrasts),'ndvi_edges':NDVI_EDGES,
            'ndvi_bins':'left-inclusive, right-exclusive; tails include values outside [0, 1]',
            'geometry_x':'median of off-nadir minus near-nadir solar phase for bare-cropland samples',
            'exclusions':exclusions,'duplicate_matchups':duplicates,'skipped_groups':skipped,
            'interpretation':'Classes are different locations in each matchup; variance is within-sample spread, not uncertainty. NDVI sample counts are not area fractions.'}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f'Analyzed {len(matchups)} tile matchups; wrote 3 figures and 3 CSV files to {args.output}')
    if skipped or duplicates: print(f'Skipped {len(skipped)} duplicate/invalid groups and {len(duplicates)} duplicate tile matchups; see report.json',file=sys.stderr)


if __name__ == '__main__': main()

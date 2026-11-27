#!/usr/bin/env python3
"""Visualize completed BRDF sample outputs without importing the processing code."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pyarrow.parquet as pq

BANDS = ('B04', 'B08', 'B12')
CLASS_NAMES = {10: 'Tree cover', 20: 'Shrubland', 30: 'Grassland', 40: 'Cropland',
               50: 'Built-up', 60: 'Bare / sparse', 70: 'Snow / ice', 80: 'Water',
               90: 'Wetland', 95: 'Mangroves', 100: 'Moss / lichen'}
DIRECTIONS = tuple(f'{sensor} near · {phase} phase' for sensor in ('S2A', 'S2B')
                   for phase in ('lower', 'higher', 'equal'))
COLORS = {'S2A near · lower phase': '#2166ac', 'S2A near · higher phase': '#8bb7dc',
          'S2B near · lower phase': '#ed9a91', 'S2B near · higher phase': '#b2182b',
          'S2A near · equal phase': '#777777', 'S2B near · equal phase': '#bbbbbb'}


def legend_label(direction):
    sensor, phase = direction.split(' near · ')
    off_sensor = 'S2B' if sensor == 'S2A' else 'S2A'
    scattering = {'lower phase': 'toward backscatter', 'higher phase': 'toward forward scatter',
                  'equal phase': 'equal solar phase'}[phase]
    return f'{sensor} near; {off_sensor} off-nadir {scattering}'


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def find_results(root):
    candidates, skipped = {}, []
    for path in sorted(root.rglob('manifest.json')):
        try:
            manifest = json.loads(path.read_text(encoding='utf-8'))
            data = path.parent / 'samples.parquet'
            if manifest.get('status') != 'complete' or not data.is_file():
                skipped.append(f'{path}: incomplete')
                continue
            if checksum(data) != manifest.get('samples_sha256'):
                skipped.append(f'{path}: checksum mismatch')
                continue
            ids = tuple(sorted(x['id'] for x in manifest['identity']['inputs']))
            if len(ids) != 4:
                raise ValueError('expected four scene IDs')
            candidate = (manifest.get('completed_utc', ''), str(path), data, manifest)
            if ids in candidates:
                previous = candidates[ids]
                if candidate[:2] > previous[:2]:
                    candidates[ids] = candidate
                    skipped.append(f'{previous[1]}: superseded by newer output for the same scenes')
                else:
                    skipped.append(f'{path}: superseded by newer output for the same scenes')
            else:
                candidates[ids] = candidate
        except (OSError, ValueError, KeyError, TypeError) as exc:
            skipped.append(f'{path}: {exc}')
    return sorted(candidates.values(), key=lambda x: x[1]), skipped


def collect(results, threshold, ndvi_threshold=0.1, definition='off-near', pair_values=None, matchup_filter=None):
    values = defaultdict(list)
    groups = defaultdict(set)
    counts = defaultdict(lambda: {'input': 0, 'nonfinite': 0, 'low_near': 0, 'valid': 0})
    inputs = []
    for _, manifest_path, data, manifest in results:
        table = pq.read_table(data)
        if matchup_filter is not None and table.num_rows:
            import pyarrow.compute as pc
            filter_columns = ('tile_id', 'scene_a', 'scene_b')
            missing_filter = set(filter_columns) - set(table.column_names)
            if missing_filter:
                raise ValueError(f'{data}: missing matchup filter columns: {sorted(missing_filter)}')
            mask = pc.equal(table[filter_columns[0]], matchup_filter[0])
            for column, value in zip(filter_columns[1:], matchup_filter[1:]):
                mask = pc.and_(mask, pc.equal(table[column], value))
            table = table.filter(mask)
        if table.num_rows == 0:
            inputs.append({'manifest': manifest_path, 'rows': 0})
            continue
        required = ['worldcover_class', 'near_nadir_platform', 'solar_phase_a_deg', 'solar_phase_b_deg', 'ndvi_a', 'ndvi_b']
        if pair_values is not None:
            required += ['tile_id', 'scene_a', 'scene_b', 'sensing_a_utc', 'sensing_b_utc']
        required += [f'{band}_reflectance_{sensor}' for band in BANDS for sensor in ('a', 'b')]
        missing = sorted(set(required) - set(table.column_names))
        if missing:
            raise ValueError(f'{data}: missing columns: {missing}')
        col = lambda name: np.asarray(table[name].to_pylist())
        classes = col('worldcover_class')
        sensor = col('near_nadir_platform')
        if pair_values is not None:
            tile_ids, scene_a, scene_b = (col(name) for name in ('tile_id', 'scene_a', 'scene_b'))
            time_a, time_b = (col(name) for name in ('sensing_a_utc', 'sensing_b_utc'))
            matchup_keys = list(zip(tile_ids, scene_a, scene_b, time_a, time_b))
            if any(any(value is None or str(value).strip() == '' for value in key) for key in matchup_keys):
                raise ValueError(f'{data}: missing matchup scene, time or tile identifiers')
            distinct_matchups = sorted(set(matchup_keys))
            matchup_masks = {key: np.array([row == key for row in matchup_keys]) for key in distinct_matchups}
        if not np.isin(sensor, ['S2A', 'S2B']).all():
            raise ValueError(f'{data}: unknown near-nadir platform')
        a_near = sensor == 'S2A'
        ndvi_a = col('ndvi_a').astype(float)
        ndvi_b = col('ndvi_b').astype(float)
        near_ndvi = np.where(a_near, ndvi_a, ndvi_b)
        crop_valid = (classes != 40) | np.isfinite(near_ndvi)
        class_keys = np.array([str(x) for x in classes], dtype=object)
        class_keys[(classes == 40) & (near_ndvi < ndvi_threshold)] = '40_bare'
        class_keys[(classes == 40) & (near_ndvi >= ndvi_threshold)] = '40_vegetated'
        phase_a = col('solar_phase_a_deg').astype(float)
        phase_b = col('solar_phase_b_deg').astype(float)
        phase_delta = np.where(a_near, phase_b-phase_a, phase_a-phase_b)
        direction = np.array([f'{s} near · {p} phase' for s, p in zip(sensor,
            np.where(phase_delta < -1e-8, 'lower', np.where(phase_delta > 1e-8, 'higher', 'equal')))])
        for band in BANDS:
            a = col(f'{band}_reflectance_a').astype(float)
            b = col(f'{band}_reflectance_b').astype(float)
            near, off = np.where(a_near, a, b), np.where(a_near, b, a)
            finite = np.isfinite(near) & np.isfinite(off) & np.isfinite(phase_delta)
            low = finite & (near < threshold)
            valid = finite & ~low
            counts[band]['input'] += len(near)
            counts[band]['nonfinite'] += int((~finite).sum())
            counts[band]['low_near'] += int(low.sum())
            counts[band].setdefault('low_s2b_denominator', 0)
            if definition == 'a-b':
                low_b = valid & (b < threshold)
                counts[band]['low_s2b_denominator'] += int(low_b.sum())
                valid &= ~low_b
            counts[band].setdefault('missing_crop_ndvi', 0)
            counts[band]['missing_crop_ndvi'] += int((valid & ~crop_valid).sum())
            valid &= crop_valid
            counts[band]['valid'] += int(valid.sum())
            for cls in np.unique(class_keys[valid]):
                for category in DIRECTIONS:
                    mask = valid & (class_keys == cls) & (direction == category)
                    if mask.any():
                        key = (str(cls), band, category)
                        values[key].append(100 * ((a[mask] - b[mask]) / b[mask] if definition == 'a-b'
                                                  else (off[mask] - near[mask]) / near[mask]))
                        groups[key].add(manifest['group_id'])
                        if pair_values is not None:
                            for matchup, matchup_mask in matchup_masks.items():
                                pair_mask = mask & matchup_mask
                                if not pair_mask.any():
                                    continue
                                delta = 100 * ((a[pair_mask] - b[pair_mask]) / b[pair_mask] if definition == 'a-b'
                                               else (off[pair_mask] - near[pair_mask]) / near[pair_mask])
                                pair_key = (manifest['group_id'], *matchup, str(cls), band, category)
                                pair_values[pair_key].append(delta)
        inputs.append({'manifest': manifest_path, 'rows': table.num_rows, 'group_id': manifest['group_id']})
    return {k: np.concatenate(v) for k, v in values.items()}, groups, dict(counts), inputs


def summarize(values, groups):
    rows = []
    for (cls, band, direction), data in sorted(values.items()):
        class_id = int(cls.split('_')[0])
        condition = cls.split('_', 1)[1] if '_' in cls else ''
        rows.append({'worldcover_class': class_id, 'landcover': CLASS_NAMES.get(class_id, f'Class {class_id}'),
                     'crop_condition': condition,
                     'band': band, 'phase_direction': direction, 'n_samples': len(data),
                     'n_overpass_groups': len(groups[(cls, band, direction)]),
                     'mean_pct': float(np.mean(data)), 'std_pct': float(np.std(data, ddof=1)) if len(data)>1 else None,
                     'variance_pct_squared': float(np.var(data, ddof=1)) if len(data)>1 else None,
                     'median_pct': float(np.median(data)), 'p25_pct': float(np.percentile(data, 25)),
                     'p75_pct': float(np.percentile(data, 75)), 'p05_pct': float(np.percentile(data, 5)),
                     'p95_pct': float(np.percentile(data, 95))})
    return rows


def summarize_pairs(pair_values, min_samples):
    """Summarize pixels separately for every tile, acquisition pair and class."""
    rows = []
    for key, chunks in sorted(pair_values.items()):
        group_id, tile, scene_a, scene_b, time_a, time_b, cls, band, direction = key
        data = np.concatenate(chunks)
        class_id = int(cls.split('_')[0])
        sufficient = len(data) >= min_samples
        rows.append({'group_id': group_id, 'tile_id': tile, 'scene_a': scene_a, 'scene_b': scene_b,
                     'sensing_a_utc': time_a, 'sensing_b_utc': time_b,
                     'worldcover_class': class_id, 'landcover': CLASS_NAMES.get(class_id, f'Class {class_id}'),
                     'crop_condition': cls.split('_', 1)[1] if '_' in cls else '',
                     'band': band, 'phase_direction': direction, 'n_samples': len(data),
                     'status': 'ok' if sufficient else 'too_few_samples',
                     'mean_pct': float(np.mean(data)), 'median_pct': float(np.median(data)),
                     'p25_pct': float(np.percentile(data, 25)), 'p75_pct': float(np.percentile(data, 75)),
                     'iqr_pct_points': float(np.percentile(data, 75)-np.percentile(data, 25)),
                     'std_pct_points': float(np.std(data, ddof=1)) if sufficient else None,
                     'variance_pct_points_squared': float(np.var(data, ddof=1)) if sufficient else None})
    return rows


def deduplicate_matchups(pair_values, results):
    """Keep one sampled instance when a tile pair occurs in multiple groups."""
    completed = {manifest['group_id']: (stamp, manifest['group_id']) for stamp, _, _, manifest in results}
    choices = defaultdict(set)
    for group_id, tile, scene_a, scene_b, *_ in pair_values:
        choices[(tile, scene_a, scene_b)].add(group_id)
    chosen = {key: max(group_ids, key=lambda group: completed[group]) for key, group_ids in choices.items()}
    ignored = [{'tile_id': tile, 'scene_a': a, 'scene_b': b, 'kept_group': chosen[(tile, a, b)],
                'skipped_groups': sorted(group_ids - {chosen[(tile, a, b)]})}
               for (tile, a, b), group_ids in sorted(choices.items()) if len(group_ids) > 1]
    return {key: value for key, value in pair_values.items() if key[0] == chosen[key[1:4]]}, ignored


def make_pair_variance_figure(rows, output, ndvi_threshold, definition, scale):
    plotted = [row for row in rows if row['status'] == 'ok']
    if not plotted:
        return False
    def class_key(row):
        return str(row['worldcover_class']) + ('_' + row['crop_condition'] if row['crop_condition'] else '')
    classes = sorted({class_key(row) for row in rows}, key=lambda x: (int(x.split('_')[0]), x))
    directions = [d for d in DIRECTIONS if any(row['phase_direction'] == d for row in plotted)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.23, top=0.80, wspace=0.12)
    for ax, band in zip(axes, BANDS):
        for row in (r for r in plotted if r['band'] == band):
            cls, direction = class_key(row), row['phase_direction']
            j = directions.index(direction)
            offset = (j-(len(directions)-1)/2) * min(0.25, 0.7/max(len(directions), 1))
            identifier = '|'.join(str(row[k]) for k in ('group_id', 'tile_id', 'scene_a', 'scene_b', 'phase_direction'))
            jitter = (int(hashlib.sha256(identifier.encode()).hexdigest()[:8], 16)/0xffffffff - 0.5)*0.08
            ax.scatter(classes.index(cls)+offset+jitter, row['variance_pct_points_squared'],
                       color=COLORS[direction], alpha=0.75, edgecolors='white', linewidths=0.4, s=35)
        labels = [f'Cropland\nbare (<{ndvi_threshold:g})' if c == '40_bare' else
                  f'Cropland\nNDVI ≥{ndvi_threshold:g}' if c == '40_vegetated' else
                  CLASS_NAMES.get(int(c), f'Class {c}') for c in classes]
        ax.set_xticks(range(len(classes)), labels, fontsize=8)
        ax.set_title(band, fontweight='bold')
        if scale == 'symlog':
            ax.set_yscale('symlog', linthresh=1)
        ax.grid(axis='y', alpha=0.2)
    axes[0].set_ylabel('Within-matchup sample variance (percentage points²)' +
                       ('\nSymmetric log scale' if scale == 'symlog' else ''))
    legend = [Line2D([0], [0], marker='o', color='none', markerfacecolor=COLORS[d],
                     markeredgecolor='none', label=legend_label(d)) for d in directions]
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.5, 0.91), ncol=max(1, min(3, len(legend))), frameon=False)
    label = '(S2A − S2B) / S2B' if definition == 'a-b' else '(off-nadir − near-nadir) / near-nadir'
    fig.suptitle('Reflectance-difference variance in individual tile matchups', y=0.98, fontsize=15, fontweight='bold')
    fig.text(0.5, 0.025, f'One point = one tile matchup and class; sample variance (ddof=1) of 100 × {label}. No pooling across matchups.', ha='center', fontsize=9)
    fig.savefig(output, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return True


def make_figure(values, output, threshold, ndvi_threshold, definition='off-near', title=None):
    class_set = {k[0] for k in values}
    if any(c.startswith('40_') for c in class_set):
        class_set.update(('40_bare', '40_vegetated'))
    classes = sorted(class_set, key=lambda x: (int(x.split('_')[0]), x))
    directions = [d for d in DIRECTIONS if any(k[2] == d for k in values)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharey=False)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.23, top=0.80, wspace=0.25)
    for ax, band in zip(axes, BANDS):
        positions, arrays, categories, labels = [], [], [], defaultdict(dict)
        for i, cls in enumerate(classes):
            for j, direction in enumerate(directions):
                data = values.get((cls, band, direction))
                if data is None:
                    continue
                offset = (j - (len(directions)-1)/2) * 0.25
                positions.append(i + offset)
                arrays.append(data)
                categories.append(direction)
                labels[i][direction] = len(data)
        if arrays:
            boxes = ax.boxplot(arrays, positions=positions, widths=0.20, patch_artist=True,
                               showfliers=False, showmeans=True,
                               meanprops={'marker':'D', 'markerfacecolor':'white', 'markeredgecolor':'black', 'markersize':4},
                               medianprops={'color':'black', 'linewidth':1.4})
            for box, direction in zip(boxes['boxes'], categories):
                box.set_facecolor(COLORS[direction]); box.set_alpha(0.7)
            for i in range(len(classes)):
                text = ' / '.join(str(labels[i].get(direction, 0)) for direction in directions)
                ax.text(i, -0.12, f'n={text}', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=7)
        ax.axhline(0, color='#777777', lw=0.8, ls='--')
        labels = [f'Cropland\nbare (<{ndvi_threshold:g})' if c == '40_bare' else
                  f'Cropland\nNDVI ≥{ndvi_threshold:g}' if c == '40_vegetated' else
                  CLASS_NAMES.get(int(c), f'Class {c}') for c in classes]
        ax.set_xticks(range(len(classes)), labels, rotation=0, ha='center', fontsize=8)
        ax.set_title(band, fontweight='bold')
        ax.set_ylabel('(S2A − S2B) / S2B (%)' if definition == 'a-b' else 'Off-nadir vs near-nadir reflectance (%)')
        ax.grid(axis='y', alpha=0.2)
    legend = [Line2D([0],[0], color=COLORS[d], lw=8, alpha=.7, label=legend_label(d)) for d in directions]
    legend.append(Line2D([0],[0], marker='D', markerfacecolor='white', markeredgecolor='black', linestyle='None', label='Mean'))
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.5, 0.91), ncol=len(legend), frameon=False)
    fig.suptitle(title or 'L2A reflectance differences by land cover', y=0.98, fontsize=16, fontweight='bold')
    filter_description = (f'Near-nadir and S2B reflectance ≥ {threshold:g}' if definition == 'a-b'
                          else f'Near-nadir reflectance ≥ {threshold:g}')
    fig.text(0.5, 0.025, f'Boxes: 25th–75th percentiles; line: median; whiskers: 1.5 × IQR; outliers hidden. {filter_description}.', ha='center', fontsize=9)
    fig.savefig(output, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results_dir', type=Path, help='Root directory containing group manifest.json files')
    parser.add_argument('--output', type=Path, required=True, help='Directory for plots, summary CSV and report')
    parser.add_argument('--min-near-reflectance', type=float, default=0.01,
                        help='Exclude near-nadir reflectances below this positive threshold (default: 0.01)')
    parser.add_argument('--bare-ndvi-threshold', type=float, default=0.1,
                        help='Classify cropland with near-nadir NDVI below this value as bare (default: 0.1)')
    parser.add_argument('--definition', choices=('off-near', 'a-b'), default='off-near',
                        help='Relative difference: off-nadir vs near-nadir, or consistent (S2A-S2B)/S2B')
    parser.add_argument('--min-pair-samples', type=int, default=2,
                        help='Minimum valid samples per matchup and class to plot variance (default: 2)')
    parser.add_argument('--variance-scale', choices=('linear', 'symlog'), default='symlog',
                        help='Variance plot y-axis scale (default: symlog; use linear for a linear axis)')
    parser.add_argument('--include-pooled', action='store_true',
                        help='Also produce the original pooled box plot and summary')
    args = parser.parse_args(argv)
    if not args.results_dir.is_dir(): parser.error('results_dir must exist')
    if not np.isfinite(args.min_near_reflectance) or args.min_near_reflectance <= 0:
        parser.error('--min-near-reflectance must be positive')
    if not np.isfinite(args.bare_ndvi_threshold): parser.error('--bare-ndvi-threshold must be finite')
    if args.min_pair_samples < 2: parser.error('--min-pair-samples must be at least 2')
    results, skipped = find_results(args.results_dir)
    if not results:
        raise SystemExit('No complete, checksum-valid result groups found.\n' + '\n'.join(skipped))
    pair_values = defaultdict(list)
    values, groups, counts, inputs = collect(results, args.min_near_reflectance, args.bare_ndvi_threshold, args.definition, pair_values)
    if not values: raise SystemExit('No samples passed the reflectance and geometry filters')
    pair_values, duplicate_matchups = deduplicate_matchups(pair_values, results)
    pair_rows = summarize_pairs(pair_values, args.min_pair_samples)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'pair_summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=pair_rows[0].keys()); writer.writeheader(); writer.writerows(pair_rows)
    has_plot = make_pair_variance_figure(pair_rows, args.output/'variance_per_pair.png', args.bare_ndvi_threshold, args.definition, args.variance_scale)
    if args.include_pooled:
        rows = summarize(values, groups)
        with (args.output/'summary.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
        make_figure(values, args.output/'relative_difference.png', args.min_near_reflectance, args.bare_ndvi_threshold, args.definition)
    report = {'formula': ('100 * (S2A - S2B) / S2B' if args.definition == 'a-b' else '100 * (off_nadir - near_nadir) / near_nadir'),
              'phase_direction': 'off-nadir solar phase minus near-nadir solar phase',
              'min_near_reflectance': args.min_near_reflectance, 'bare_ndvi_threshold': args.bare_ndvi_threshold,
              'crop_ndvi_source': 'near-nadir observation', 'counts_by_band': counts,
              'min_pair_samples': args.min_pair_samples, 'variance_scale': args.variance_scale,
              'pair_summary_rows': len(pair_rows), 'too_few_samples': sum(row['status'] != 'ok' for row in pair_rows),
              'duplicate_matchups': duplicate_matchups,
              'pooled_outputs_requested': args.include_pooled,
              'inputs': inputs, 'skipped': skipped,
              'note': 'Each variance describes pixels within one tile matchup and class, not uncertainty across independent overpasses.'}
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(f'Analyzed {len(inputs)} overpass groups; wrote {args.output / "pair_summary.csv"}')
    if has_plot:
        print(f'Variance plot: {args.output / "variance_per_pair.png"}')
    else:
        print('No variance plot: all matchup/class subsets have fewer than the required number of samples.', file=sys.stderr)
    for message in skipped: print(f'Skipped: {message}', file=sys.stderr)


if __name__ == '__main__':
    main()

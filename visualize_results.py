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


def collect(results, threshold, ndvi_threshold=0.1, definition='off-near'):
    values = defaultdict(list)
    groups = defaultdict(set)
    counts = defaultdict(lambda: {'input': 0, 'nonfinite': 0, 'low_near': 0, 'valid': 0})
    inputs = []
    for _, manifest_path, data, manifest in results:
        table = pq.read_table(data)
        if table.num_rows == 0:
            inputs.append({'manifest': manifest_path, 'rows': 0})
            continue
        required = ['worldcover_class', 'near_nadir_platform', 'solar_phase_a_deg', 'solar_phase_b_deg', 'ndvi_a', 'ndvi_b']
        required += [f'{band}_reflectance_{sensor}' for band in BANDS for sensor in ('a', 'b')]
        missing = sorted(set(required) - set(table.column_names))
        if missing:
            raise ValueError(f'{data}: missing columns: {missing}')
        col = lambda name: np.asarray(table[name].to_pylist())
        classes = col('worldcover_class')
        sensor = col('near_nadir_platform')
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


def make_figure(values, output, threshold, ndvi_threshold, definition='off-near'):
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
    def legend_label(direction):
        sensor, phase = direction.split(' near · ')
        off_sensor = 'S2B' if sensor == 'S2A' else 'S2A'
        scattering = {'lower phase': 'toward backscatter', 'higher phase': 'toward forward scatter',
                      'equal phase': 'equal solar phase'}[phase]
        return f'{sensor} near; {off_sensor} off-nadir {scattering}'
    legend = [Line2D([0],[0], color=COLORS[d], lw=8, alpha=.7, label=legend_label(d)) for d in directions]
    legend.append(Line2D([0],[0], marker='D', markerfacecolor='white', markeredgecolor='black', linestyle='None', label='Mean'))
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.5, 0.91), ncol=len(legend), frameon=False)
    fig.suptitle('L2A reflectance differences by land cover', y=0.98, fontsize=16, fontweight='bold')
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
    args = parser.parse_args(argv)
    if not args.results_dir.is_dir(): parser.error('results_dir must exist')
    if not np.isfinite(args.min_near_reflectance) or args.min_near_reflectance <= 0:
        parser.error('--min-near-reflectance must be positive')
    if not np.isfinite(args.bare_ndvi_threshold): parser.error('--bare-ndvi-threshold must be finite')
    results, skipped = find_results(args.results_dir)
    if not results:
        raise SystemExit('No complete, checksum-valid result groups found.\n' + '\n'.join(skipped))
    values, groups, counts, inputs = collect(results, args.min_near_reflectance, args.bare_ndvi_threshold, args.definition)
    if not values: raise SystemExit('No samples passed the reflectance and geometry filters')
    rows = summarize(values, groups)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    make_figure(values, args.output/'relative_difference.png', args.min_near_reflectance, args.bare_ndvi_threshold, args.definition)
    report = {'formula': ('100 * (S2A - S2B) / S2B' if args.definition == 'a-b' else '100 * (off_nadir - near_nadir) / near_nadir'),
              'phase_direction': 'off-nadir solar phase minus near-nadir solar phase',
              'min_near_reflectance': args.min_near_reflectance, 'bare_ndvi_threshold': args.bare_ndvi_threshold,
              'crop_ndvi_source': 'near-nadir observation', 'counts_by_band': counts,
              'inputs': inputs, 'skipped': skipped,
              'note': 'Samples are pooled; variability is sample SD, not uncertainty across independent overpasses.'}
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(f'Analyzed {len(inputs)} groups; wrote {args.output / "relative_difference.png"} and summary.csv')
    for message in skipped: print(f'Skipped: {message}', file=sys.stderr)


if __name__ == '__main__':
    main()

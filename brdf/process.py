"""Windowed, reproducible sampling and atomic per-overpass results."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import hashlib
import heapq
import json
import os
from pathlib import Path
import traceback

import numpy as np

from . import __version__
from .discovery import one_path, xml_root
from .geometry import AngleGrid, BAND_IDS, angular_distance


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def band_path(scene, band):
    res = 10 if band in ('B04', 'B08') else 20
    return one_path(scene.path.glob(f'GRANULE/*/IMG_DATA/R{res}m/*_{band}_{res}m.jp2'), f'{band} raster')


def scl_path(scene):
    return one_path(scene.path.glob('GRANULE/*/IMG_DATA/R20m/*_SCL_20m.jp2'), '20 m SCL raster')


def radiometry(scene):
    root = xml_root(scene.product_xml)
    quant = root.findtext('.//BOA_QUANTIFICATION_VALUE')
    if quant is None or float(quant) <= 0:
        raise ValueError(f'Missing or invalid BOA_QUANTIFICATION_VALUE: {scene.product_xml}')
    offsets = {}
    for band, band_id in BAND_IDS.items():
        node = root.find(f'.//BOA_ADD_OFFSET[@band_id="{band_id}"]')
        if node is None:
            # Older processing baselines have no offset. A modern missing offset is an error.
            if int(scene.baseline[1:]) < 400:
                offsets[band] = 0.0
                continue
            raise ValueError(f'Missing {band} BOA_ADD_OFFSET: {scene.product_xml}')
        offsets[band] = float(node.text)
    return float(quant), offsets


@contextmanager
def group_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a+b') as lock:
        if os.name == 'nt':
            import msvcrt
            import time
            if path.stat().st_size == 0:
                lock.write(b'0')
                lock.flush()
            while True:
                try:
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == 'nt':
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


def atomic_json(path, value):
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def sample_tile(pair, config, group_id):
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling
    from rasterio.windows import Window
    from pyproj import Transformer

    a, b = pair
    setting, sampling = config['selection'], config['sampling']
    seed = int(digest([sampling['seed'], group_id, a.tile])[:16], 16)
    rng = np.random.default_rng(seed)
    limit = int(sampling['samples_per_class_per_tile'])
    classes = [int(v) for v in setting['landcover_classes']]
    crop_split = sampling.get('split_cropland_ndvi', False) and 40 in classes
    crop_threshold = float(sampling.get('bare_ndvi_threshold', 0.1))
    strata = [str(c) for c in classes if c != 40 or not crop_split]
    if crop_split:
        strata.extend(['40_bare', '40_other'])
    heaps = {c: [] for c in strata}
    eligible = {str(c): 0 for c in classes}
    eligible_strata = {c: 0 for c in strata}
    serial = 0
    qa = {'pixels_examined': 0, 'valid_scl_landcover': 0, 'valid_radiometry': 0,
          'valid_geometry': 0, 'eligible_by_class': eligible,
          'eligible_by_stratum': eligible_strata, 'crop_undefined_ndvi': 0,
          'eligible_closest_to_nadir': {'S2A': 0, 'S2B': 0}}
    quant_a, offsets_a = radiometry(a)
    quant_b, offsets_b = radiometry(b)
    angle_a = AngleGrid(a.tile_xml, config['processing']['geometry_band'])
    angle_b = AngleGrid(b.tile_xml, config['processing']['geometry_band'])

    with ExitStack() as stack:
        reference = stack.enter_context(rasterio.open(band_path(a, 'B12')))
        if reference.crs is None:
            raise ValueError(f'Missing CRS: {reference.name}')
        def aligned(path):
            src = stack.enter_context(rasterio.open(path))
            if src.crs is None:
                raise ValueError(f'Missing CRS: {path}')
            return stack.enter_context(WarpedVRT(src, crs=reference.crs, transform=reference.transform,
                width=reference.width, height=reference.height, resampling=Resampling.nearest,
                src_nodata=0, nodata=0))
        layers = {}
        for label, scene in [('a', a), ('b', b)]:
            layers[f'scl_{label}'] = aligned(scl_path(scene))
            for band in BAND_IDS:
                layers[f'{band}_{label}'] = aligned(band_path(scene, band))
        # WorldCover class 0 is nodata; nearest preserves categorical values.
        layers['lc'] = aligned(config['paths']['worldcover_2021'])
        b_crs = rasterio.open(band_path(b, 'B12'))
        stack.callback(b_crs.close)
        to_b = Transformer.from_crs(reference.crs, b_crs.crs, always_xy=True)
        to_lonlat = Transformer.from_crs(reference.crs, 'EPSG:4326', always_xy=True)
        size = int(sampling['window_pixels'])
        for row0 in range(0, reference.height, size):
            for col0 in range(0, reference.width, size):
                window = Window(col0, row0, min(size, reference.width-col0), min(size, reference.height-row0))
                data = {name: src.read(1, window=window, masked=False) for name, src in layers.items()}
                qa['pixels_examined'] += data['lc'].size
                mask = (np.isin(data['lc'], classes) & np.isin(data['scl_a'], setting['valid_scl']) &
                        np.isin(data['scl_b'], setting['valid_scl']))
                qa['valid_scl_landcover'] += int(mask.sum())
                for label in ('a', 'b'):
                    for band in BAND_IDS:
                        mask &= data[f'{band}_{label}'] != 0
                qa['valid_radiometry'] += int(mask.sum())
                rr, cc = np.where(mask)
                if not len(rr):
                    continue
                rows, cols = rr + row0, cc + col0
                # Pixel centers; the 20 m B12 grid is the common spatial support.
                tr = reference.transform
                x = tr.c + (cols+0.5)*tr.a + (rows+0.5)*tr.b
                y = tr.f + (cols+0.5)*tr.d + (rows+0.5)*tr.e
                bx, by = to_b.transform(x, y)
                ga, gb = angle_a.at(x, y), angle_b.at(np.asarray(bx), np.asarray(by))
                valid = np.ones(len(rr), dtype=bool)
                for geom in (ga, gb):
                    for values in geom.values():
                        valid &= np.isfinite(values)
                    valid &= geom['sza'] < setting['max_sun_zenith_deg_exclusive']
                valid &= np.minimum(ga['vza'], gb['vza']) < setting['near_nadir_deg_exclusive']
                qa['valid_geometry'] += int(valid.sum())
                indices = np.flatnonzero(valid)
                if not len(indices):
                    continue
                closest_a = int(np.count_nonzero(ga['vza'][indices] <= gb['vza'][indices]))
                qa['eligible_closest_to_nadir']['S2A'] += closest_a
                qa['eligible_closest_to_nadir']['S2B'] += len(indices)-closest_a
                # Keep only the largest random priorities in this window; others
                # cannot enter the global reservoir for their class.
                candidates = []
                lc_values = data['lc'][rr[indices], cc[indices]]
                for lc in classes:
                    eligible[str(lc)] += int(np.count_nonzero(lc_values == lc))
                stratum_values = lc_values.astype(str).astype(object)
                if crop_split:
                    use_a = ga['vza'][indices] <= gb['vza'][indices]
                    red_a = (data['B04_a'][rr[indices], cc[indices]].astype(float) + offsets_a['B04']) / quant_a
                    nir_a = (data['B08_a'][rr[indices], cc[indices]].astype(float) + offsets_a['B08']) / quant_a
                    red_b = (data['B04_b'][rr[indices], cc[indices]].astype(float) + offsets_b['B04']) / quant_b
                    nir_b = (data['B08_b'][rr[indices], cc[indices]].astype(float) + offsets_b['B08']) / quant_b
                    red, nir = np.where(use_a, red_a, red_b), np.where(use_a, nir_a, nir_b)
                    denom = nir + red
                    ndvi = np.divide(nir-red, denom, out=np.full(len(indices), np.nan), where=denom != 0)
                    crop = lc_values == 40
                    stratum_values[crop & (ndvi < crop_threshold)] = '40_bare'
                    stratum_values[crop & (ndvi >= crop_threshold)] = '40_other'
                    qa['crop_undefined_ndvi'] += int(np.count_nonzero(crop & ~np.isfinite(ndvi)))
                for stratum in strata:
                    class_indices = indices[stratum_values == stratum]
                    eligible_strata[stratum] += len(class_indices)
                    if not len(class_indices):
                        continue
                    priorities = rng.random(len(class_indices))
                    n = min(limit, len(class_indices))
                    best = np.argpartition(priorities, len(priorities)-n)[-n:]
                    candidates.extend((int(class_indices[k]), float(priorities[k]), stratum) for k in best)
                if not candidates:
                    continue
                lon, lat = to_lonlat.transform(x[[i for i, _, _ in candidates]], y[[i for i, _, _ in candidates]])
                for j, (idx, priority, stratum) in enumerate(candidates):
                    if len(heaps[stratum]) >= limit and priority <= heaps[stratum][0][0]:
                        continue
                    lc = int(data['lc'][rr[idx], cc[idx]])
                    serial += 1
                    g1 = {k: float(v[idx]) for k, v in ga.items()}
                    g2 = {k: float(v[idx]) for k, v in gb.items()}
                    record = {'group_id': group_id, 'tile_id': a.tile, 'row': int(rows[idx]),
                              'col': int(cols[idx]), 'x': float(x[idx]), 'y': float(y[idx]),
                              'crs': reference.crs.to_string(), 'longitude': float(lon[j]), 'latitude': float(lat[j]),
                              'worldcover_class': lc, 'sampling_stratum': stratum, 'scene_a': a.id, 'scene_b': b.id,
                              'sensing_a_utc': a.sensing.isoformat()+'Z', 'sensing_b_utc': b.sensing.isoformat()+'Z',
                              'time_difference_seconds_b_minus_a': (b.sensing-a.sensing).total_seconds(),
                              'scl_a': int(data['scl_a'][rr[idx], cc[idx]]),
                              'scl_b': int(data['scl_b'][rr[idx], cc[idx]]),
                              'near_nadir_platform': 'S2A' if g1['vza'] <= g2['vza'] else 'S2B',
                              'view_separation_deg': float(angular_distance(g1['vza'], g1['vaa'], g2['vza'], g2['vaa']))}
                    for label, geom, quant, offsets in [('a', g1, quant_a, offsets_a), ('b', g2, quant_b, offsets_b)]:
                        record.update({f'{k}_{label}_deg': v for k, v in geom.items()})
                        record[f'relative_azimuth_{label}_deg'] = (geom['vaa']-geom['saa']+180) % 360 - 180
                        record[f'solar_phase_{label}_deg'] = float(angular_distance(geom['sza'], geom['saa'], geom['vza'], geom['vaa']))
                        for band in BAND_IDS:
                            dn = int(data[f'{band}_{label}'][rr[idx], cc[idx]])
                            record[f'{band}_dn_{label}'] = dn
                            record[f'{band}_reflectance_{label}'] = (dn + offsets[band]) / quant
                        red, nir = record[f'B04_reflectance_{label}'], record[f'B08_reflectance_{label}']
                        record[f'ndvi_{label}'] = (nir-red)/(nir+red) if nir+red != 0 else None
                    for band in BAND_IDS:
                        record[f'{band}_difference_b_minus_a'] = record[f'{band}_reflectance_b']-record[f'{band}_reflectance_a']
                    item = (priority, serial, record)
                    if len(heaps[stratum]) < limit:
                        heapq.heappush(heaps[stratum], item)
                    else:
                        heapq.heapreplace(heaps[stratum], item)
    records = [item[2] for cls in strata for item in sorted(heaps[cls], reverse=True)]
    qa['sampled_by_class'] = {str(cls): sum(len(heaps[s]) for s in strata if s == str(cls) or s.startswith(f'{cls}_')) for cls in classes}
    qa['sampled_by_stratum'] = {cls: len(heaps[cls]) for cls in strata}
    qa['radiometry'] = {'a': {'quantification': quant_a, 'offsets': offsets_a},
                         'b': {'quantification': quant_b, 'offsets': offsets_b}}
    return records, qa


def process_group(group, config):
    import pyarrow as pa
    import pyarrow.parquet as pq
    scenes = [s for pair in group for s in pair]
    # Metadata checksums cause changed inputs or settings to produce a different output identity.
    inputs = [{'id': s.id, 'path': str(s.path), 'cloud_percent': s.cloud,
               'product_metadata_sha256': sha256_file(s.product_xml),
               'tile_metadata_sha256': sha256_file(s.tile_xml)} for s in scenes]
    wc_stat = Path(config['paths']['worldcover_2021']).stat()
    source_hashes = {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    identity = {'inputs': inputs, 'selection': config['selection'], 'sampling': config['sampling'],
                'geometry_band': config['processing']['geometry_band'],
                'worldcover_2021': config['paths']['worldcover_2021'],
                'worldcover_size': wc_stat.st_size, 'worldcover_mtime_ns': wc_stat.st_mtime_ns,
                'worldcover_revision': config['processing'].get('worldcover_revision', ''),
                'software_version': __version__, 'source_sha256': source_hashes}
    fingerprint = digest(identity)
    group_id = f"{scenes[0].sensing:%Y%m%d}_{group[0][0].tile}_{group[1][0].tile}_{fingerprint[:16]}"
    out = Path(config['paths']['output']) / group_id
    with group_lock(Path(config['paths']['output']) / '.locks' / f'{group_id}.lock'):
        out.mkdir(parents=True, exist_ok=True)
        manifest_path, data_path = out/'manifest.json', out/'samples.parquet'
        if manifest_path.exists() and data_path.exists():
            try:
                existing = json.loads(manifest_path.read_text(encoding='utf-8'))
                if (existing.get('status') == 'complete' and existing.get('fingerprint') == fingerprint
                        and existing.get('samples_sha256') == sha256_file(data_path)):
                    return {'group_id': group_id, 'status': 'reused', 'samples': existing['sample_count']}
            except (ValueError, OSError, KeyError):
                pass
        try:
            records, quality = [], {}
            for pair in group:
                rows, qa = sample_tile(pair, config, group_id)
                records.extend(rows)
                quality[pair[0].tile] = qa
            table = pa.Table.from_pylist(records) if records else pa.table({'group_id': pa.array([], type=pa.string())})
            tmp = out/f'samples.{os.getpid()}.tmp.parquet'
            pq.write_table(table, tmp, compression='zstd')
            os.replace(tmp, data_path)
            manifest = {'status': 'complete', 'group_id': group_id, 'fingerprint': fingerprint,
                        'completed_utc': datetime.now(timezone.utc).isoformat(), 'identity': identity,
                        'quality': quality, 'sample_count': len(records), 'samples_sha256': sha256_file(data_path)}
            atomic_json(manifest_path, manifest)
            (out/'failure.json').unlink(missing_ok=True)
            return {'group_id': group_id, 'status': 'complete', 'samples': len(records)}
        except Exception:
            atomic_json(out/'failure.json', {'group_id': group_id, 'status': 'failed',
                'time_utc': datetime.now(timezone.utc).isoformat(), 'traceback': traceback.format_exc(), 'identity': identity})
            raise

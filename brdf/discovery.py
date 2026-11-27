"""Discover SAFE products and form unambiguous neighboring-tile matchups."""
from __future__ import annotations

from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import itertools
import re
import sqlite3
import struct
import xml.etree.ElementTree as ET

SCENE_RE = re.compile(r"^(S2[AB])_MSIL2A_(\d{8}T\d{6})_(N\d{4})_(R\d{3})_T(\d{2}[A-Z]{3})_(\d{8}T\d{6})\.SAFE$")


def xml_root(path):
    root = ET.parse(path).getroot()
    for element in root.iter():
        element.tag = element.tag.split('}')[-1]
    return root


def one_path(paths, description):
    paths = sorted(paths)
    if len(paths) != 1:
        raise ValueError(f"Expected one {description}, found {len(paths)}: {paths}")
    return paths[0]


@dataclass(frozen=True)
class Scene:
    path: Path
    tile: str
    platform: str
    sensing: datetime
    orbit: str
    baseline: str
    produced: datetime
    cloud: float
    tile_xml: Path
    product_xml: Path

    @property
    def id(self):
        return self.path.name.removesuffix('.SAFE')


def parse_scene(path):
    match = SCENE_RE.match(path.name)
    if not match:
        raise ValueError(f"Invalid SAFE product name: {path}")
    platform, sensing, baseline, orbit, tile, produced = match.groups()
    product_xml = one_path(path.glob('MTD_MSIL2A.xml'), 'product metadata')
    tile_xml = one_path(path.glob('GRANULE/*/MTD_TL.xml'), 'tile metadata')
    # CLOUDY_PIXEL_PERCENTAGE in tile metadata describes this granule.
    root = xml_root(tile_xml)
    cloud_node = root.find('.//CLOUDY_PIXEL_PERCENTAGE')
    if cloud_node is None:
        raise ValueError(f"Missing tile CLOUDY_PIXEL_PERCENTAGE: {tile_xml}")
    cloud = float(cloud_node.text)
    if not 0 <= cloud <= 100:
        raise ValueError(f"Invalid cloud percentage in {tile_xml}: {cloud}")
    return Scene(path, tile, platform, datetime.strptime(sensing, '%Y%m%dT%H%M%S'), orbit,
                 baseline, datetime.strptime(produced, '%Y%m%dT%H%M%S'), cloud,
                 tile_xml, product_xml)


def read_tiles(path):
    import yaml
    content = Path(path).read_text(encoding='utf-8-sig')
    try:
        parsed = yaml.safe_load(content) if content.lstrip().startswith(('-', '[')) else content.splitlines()
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid tile list {path}: {exc}") from exc
    if not isinstance(parsed, list) or not parsed:
        raise ValueError('Tile list must be a non-empty YAML list or newline-separated IDs')
    tiles = [str(x).strip().upper().removeprefix('T') for x in parsed if str(x).strip()]
    invalid = [t for t in tiles if not re.fullmatch(r'\d{2}[C-HJ-NP-X][A-HJ-NP-Z]{2}', t)]
    if invalid:
        raise ValueError(f'Invalid MGRS tile IDs: {invalid}')
    return sorted(set(tiles))


def _sql_identifier(value):
    return '"' + value.replace('"', '""') + '"'


def _normalize_tile(value):
    tile = str(value).strip().upper().removeprefix('T') if value is not None else ''
    return tile if re.fullmatch(r'\d{2}[C-HJ-NP-X][A-HJ-NP-Z]{2}', tile) else None


def _gpkg_geometry(blob):
    """Decode a GeoPackage geometry binary header and its embedded WKB."""
    from shapely import from_wkb
    if blob is None or len(blob) < 8 or blob[:2] != b'GP' or blob[2] != 0:
        raise ValueError('Invalid or missing GeoPackage geometry')
    flags = blob[3]
    envelope = (flags >> 1) & 7
    if envelope > 4 or flags & 0x20:
        raise ValueError('Unsupported GeoPackage geometry encoding')
    lengths = (0, 4, 6, 6, 8)
    offset = 8 + 8 * lengths[envelope]
    if len(blob) <= offset or flags & 0x10:
        raise ValueError('Empty or truncated GeoPackage geometry')
    byte_order = '<' if flags & 1 else '>'
    srs_id = struct.unpack(byte_order + 'i', blob[4:8])[0]
    return from_wkb(blob[offset:]), srs_id


def read_geopackage(path, layer=None, id_field=None):
    """Read polygon tiles from a GeoPackage, returning IDs and lon/lat geometries."""
    from pyproj import CRS, Transformer
    from shapely.ops import transform as transform_geometry

    uri = Path(path).resolve().as_uri() + '?mode=ro'
    with closing(sqlite3.connect(uri, uri=True)) as db:
        layers = db.execute("SELECT table_name FROM gpkg_contents WHERE data_type='features' ORDER BY table_name").fetchall()
        layers = [row[0] for row in layers]
        if layer is None:
            if len(layers) != 1:
                raise ValueError(f'GeoPackage has {len(layers)} feature layers {layers}; set tiles.geopackage_layer')
            layer = layers[0]
        if layer not in layers:
            raise ValueError(f'GeoPackage layer {layer!r} not found; available feature layers: {layers}')
        metadata = db.execute('SELECT column_name, srs_id FROM gpkg_geometry_columns WHERE table_name=?', (layer,)).fetchall()
        if len(metadata) != 1:
            raise ValueError(f'Expected one geometry column in GeoPackage layer {layer!r}')
        geom_col, srs_id = metadata[0]
        srs = db.execute('SELECT organization, organization_coordsys_id, definition FROM gpkg_spatial_ref_sys WHERE srs_id=?', (srs_id,)).fetchone()
        if srs is None or srs_id in (-1, 0):
            raise ValueError(f'GeoPackage layer {layer!r} has no defined CRS')
        try:
            crs = CRS.from_user_input(f'{srs[0]}:{srs[1]}') if str(srs[0]).upper() == 'EPSG' else CRS.from_wkt(srs[2])
        except Exception as exc:
            raise ValueError(f'Cannot interpret CRS for GeoPackage layer {layer!r}: {exc}') from exc
        columns = [r[1] for r in db.execute(f'PRAGMA table_info({_sql_identifier(layer)})').fetchall() if r[1] != geom_col]
        rows = db.execute(f'SELECT * FROM {_sql_identifier(layer)}').fetchall()
        names = [r[1] for r in db.execute(f'PRAGMA table_info({_sql_identifier(layer)})').fetchall()]
        if not rows:
            raise ValueError(f'GeoPackage layer {layer!r} is empty')
        if id_field is None:
            candidates = [c for c in columns if all(_normalize_tile(row[names.index(c)]) for row in rows)]
            if len(candidates) != 1:
                raise ValueError(f'Could not uniquely identify tile ID field in {layer!r}; candidates: {candidates}; set tiles.tile_id_field')
            id_field = candidates[0]
        if id_field not in columns:
            raise ValueError(f'Tile ID field {id_field!r} not found in {layer!r}; fields: {columns}')
        converter = Transformer.from_crs(crs, 4326, always_xy=True)
        geometries = {}
        for row in rows:
            tile = _normalize_tile(row[names.index(id_field)])
            if tile is None:
                raise ValueError(f'Invalid tile ID {row[names.index(id_field)]!r} in {layer}.{id_field}')
            if tile in geometries:
                raise ValueError(f'Duplicate tile ID {tile} in {layer}.{id_field}')
            geom, geometry_srs = _gpkg_geometry(row[names.index(geom_col)])
            if geometry_srs != srs_id:
                raise ValueError(f'Inconsistent CRS for tile {tile}: {geometry_srs} != {srs_id}')
            if geom.geom_type not in ('Polygon', 'MultiPolygon') or geom.is_empty or not geom.is_valid:
                raise ValueError(f'Tile {tile} must have a nonempty valid polygon geometry')
            geom = transform_geometry(converter.transform, geom)
            if not all(map(lambda x: abs(x) < float('inf'), geom.bounds)):
                raise ValueError(f'Invalid geographic coordinates for {tile}')
            geometries[tile] = geom
    return geometries, {'layer': layer, 'id_field': id_field, 'source_crs': crs.to_string()}


def geopackage_neighbors(geometries, tolerance_m=2500):
    """Find horizontal MGRS neighbors using polygon centers in their common UTM CRS.

    Full grid polygons are expected: their projected centers are 100 km apart
    east/west and have matching northings, whether grid cells touch or overlap.
    """
    from pyproj import Transformer
    from shapely.ops import transform as transform_geometry
    if not 0 < tolerance_m < 25000:
        raise ValueError('tiles.neighbor_tolerance_m must be positive and less than 25000')
    transformed = {}
    for tile, geom in geometries.items():
        zone = int(tile[:2])
        if not 1 <= zone <= 60:
            raise ValueError(f'Invalid UTM zone for {tile}')
        north = tile[2] >= 'N'
        transformer = Transformer.from_crs(4326, (32600 if north else 32700) + zone, always_xy=True)
        projected = transform_geometry(transformer.transform, geom)
        center = projected.centroid
        if not center.is_valid or not (-1000000 < center.x < 2000000):
            raise ValueError(f'Tile {tile} geometry does not appear to be in its stated UTM zone')
        transformed[tile] = (zone, north, center.x, center.y)
    pairs = []
    tiles = sorted(transformed)
    for i, a in enumerate(tiles):
        zone_a, north_a, x_a, y_a = transformed[a]
        for b in tiles[i+1:]:
            zone_b, north_b, x_b, y_b = transformed[b]
            if (zone_a, north_a) == (zone_b, north_b) and abs(abs(x_a-x_b)-100000) <= tolerance_m and abs(y_a-y_b) <= tolerance_m:
                pairs.append((a, b))
    return pairs


def discover(archive, tiles, start, end, cloud_limit):
    scenes, issues = [], []
    months = []
    month = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while month < end:
        months.append(month)
        month = (month.replace(day=28) + timedelta(days=4)).replace(day=1)
    for tile in tiles:
        for month in months:
            folder = Path(archive) / tile[:2] / tile[2] / tile[3:] / str(month.year) / f'{month.month:02d}'
            paths = set(folder.glob('S2[AB]_MSIL2A_*.SAFE'))
            paths.update(Path(archive).glob(f'S2[AB]_MSIL2A_*_T{tile}_*.SAFE'))
            for path in sorted(paths):
                match = SCENE_RE.match(path.name)
                if not match:
                    issues.append(f'Invalid product name: {path}')
                    continue
                sensing = datetime.strptime(match.group(2), '%Y%m%dT%H%M%S')
                if not start <= sensing < end or match.group(5) != tile:
                    continue
                try:
                    scene = parse_scene(path)
                    scenes.append(scene)
                except (ValueError, OSError, ET.ParseError) as exc:
                    issues.append(f'{path}: {exc}')
    # Prefer the newest processing for the same acquisition; do not drop distinct overpasses.
    latest = {}
    for scene in scenes:
        key = (scene.tile, scene.platform, scene.sensing, scene.orbit)
        if key not in latest or (scene.baseline, scene.produced) > (latest[key].baseline, latest[key].produced):
            latest[key] = scene
    return sorted((s for s in latest.values() if s.cloud < cloud_limit), key=lambda s: s.id), issues


def tile_geometry(scene):
    """Return scene tile center and footprint from the 20 m metadata grid."""
    from pyproj import CRS, Transformer
    from shapely.geometry import Polygon
    root = xml_root(scene.tile_xml)
    crs = CRS.from_user_input(root.findtext('.//HORIZONTAL_CS_CODE'))
    pos = root.find('.//Geoposition[@resolution="20"]')
    size = root.find('.//Size[@resolution="20"]')
    if pos is None or size is None:
        raise ValueError(f'Missing 20 m georeferencing in {scene.tile_xml}')
    x, y = float(pos.findtext('ULX')), float(pos.findtext('ULY'))
    dx, dy = float(pos.findtext('XDIM')), float(pos.findtext('YDIM'))
    width, height = int(size.findtext('NCOLS')) * dx, int(size.findtext('NROWS')) * dy
    transform = Transformer.from_crs(crs, 4326, always_xy=True)
    corners = [transform.transform(px, py) for px, py in ((x,y), (x+width,y), (x+width,y+height), (x,y+height))]
    center = transform.transform(x+width/2, y+height/2)
    return center, Polygon(corners)


def horizontal_neighbors(first, second):
    """Use local metric positions and footprint overlap, including across UTM zones."""
    from pyproj import CRS, Transformer
    from shapely.ops import transform as transform_geom
    (lon1, lat1), poly1 = first
    (lon2, lat2), poly2 = second
    if abs(lon1-lon2) > 180:  # avoid unhandled antimeridian topology
        return False
    local = CRS.from_proj4(f'+proj=aeqd +lat_0={(lat1+lat2)/2} +lon_0={(lon1+lon2)/2} +datum=WGS84')
    transformer = Transformer.from_crs(4326, local, always_xy=True)
    x1, y1 = transformer.transform(lon1, lat1)
    x2, y2 = transformer.transform(lon2, lat2)
    dx, dy = abs(x1-x2), abs(y1-y2)
    # MGRS squares are nominally 100 km apart with 10 km tile overlaps.
    return (70000 <= dx <= 120000 and dy < 40000 and
            transform_geom(transformer.transform, poly1).intersects(transform_geom(transformer.transform, poly2)))


def make_groups(scenes, neighbor_pairs, pair_minutes, neighbor_minutes):
    by_tile = defaultdict(list)
    for scene in scenes:
        by_tile[scene.tile].append(scene)
    groups = []
    for left, right in sorted(neighbor_pairs):
        possible = []
        for tile in (left, right):
            a = [s for s in by_tile[tile] if s.platform == 'S2A']
            b = [s for s in by_tile[tile] if s.platform == 'S2B']
            possible.append([(x, y) for x, y in itertools.product(a, b)
                             if abs((x.sensing-y.sensing).total_seconds()) <= pair_minutes*60])
        for first, second in itertools.product(*possible):
            if any(first[i].orbit != second[i].orbit or
                   abs((first[i].sensing-second[i].sensing).total_seconds()) > neighbor_minutes*60
                   for i in (0, 1)):
                continue
            groups.append((first, second))
    return sorted(groups, key=lambda g: tuple(s.id for pair in g for s in pair))

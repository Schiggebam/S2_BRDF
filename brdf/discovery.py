"""Discover SAFE products and form unambiguous neighboring-tile matchups."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import itertools
import re
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

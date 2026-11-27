"""Interpolate Sentinel tile angle grids and compute vector angles in degrees."""
from __future__ import annotations

import numpy as np
from .discovery import xml_root

BAND_IDS = {'B04': 3, 'B08': 7, 'B12': 11}


def _grid(node):
    return np.array([[float(v) for v in row.text.split()] for row in node.findall('./Values_List/VALUES')], dtype=float)


def _circular_mean(stack):
    valid = np.isfinite(stack)
    rad = np.radians(np.where(valid, stack, 0))
    count = valid.sum(axis=0)
    return np.where(count, np.degrees(np.arctan2((np.sin(rad)*valid).sum(axis=0),
                                                (np.cos(rad)*valid).sum(axis=0))) % 360, np.nan)


class AngleGrid:
    def __init__(self, tile_xml, band):
        root = xml_root(tile_xml)
        pos = root.find('.//Geoposition[@resolution="20"]')
        self.x0, self.y0 = float(pos.findtext('ULX')), float(pos.findtext('ULY'))
        sun = root.find('.//Sun_Angles_Grid')
        self.dx = float(sun.findtext('Zenith/COL_STEP'))
        self.dy = float(sun.findtext('Zenith/ROW_STEP'))
        blocks = [b for b in root.findall('.//Viewing_Incidence_Angles_Grids')
                  if b.get('bandId') == str(BAND_IDS[band])]
        if not blocks:
            raise ValueError(f'Missing {band} viewing angles: {tile_xml}')
        z = np.stack([_grid(b.find('Zenith')) for b in blocks])
        az = np.stack([_grid(b.find('Azimuth')) for b in blocks])
        count = np.isfinite(z).sum(axis=0)
        vza = np.divide(np.nansum(z, axis=0), count, out=np.full(count.shape, np.nan), where=count > 0)
        self.values = {'sza': _grid(sun.find('Zenith')), 'saa': _grid(sun.find('Azimuth')),
                       'vza': vza, 'vaa': _circular_mean(az)}

    def at(self, x, y):
        row, col = (self.y0-y)/self.dy, (x-self.x0)/self.dx
        output = {}
        for name, grid in self.values.items():
            valid = np.isfinite(grid)
            def interpolate(values):
                return _bilinear(np.where(valid, values, 0.0), row, col)
            weight = _bilinear(valid.astype(float), row, col)
            if name.endswith('aa'):
                rad = np.radians(grid)
                result = np.degrees(np.arctan2(interpolate(np.sin(rad)), interpolate(np.cos(rad)))) % 360
            else:
                result = np.divide(interpolate(grid), weight, out=np.full(weight.shape, np.nan), where=weight > 0)
            output[name] = np.where(weight >= 0.5, result, np.nan)
        return output


def _bilinear(grid, row, col):
    inside = np.isfinite(row) & np.isfinite(col) & (row >= 0) & (col >= 0) & (row <= grid.shape[0]-1) & (col <= grid.shape[1]-1)
    r = np.where(inside, row, 0)
    c = np.where(inside, col, 0)
    r0, c0 = np.floor(r).astype(int), np.floor(c).astype(int)
    r1, c1 = np.minimum(r0+1, grid.shape[0]-1), np.minimum(c0+1, grid.shape[1]-1)
    dr, dc = r-r0, c-c0
    result = (grid[r0,c0]*(1-dr)*(1-dc) + grid[r1,c0]*dr*(1-dc) +
              grid[r0,c1]*(1-dr)*dc + grid[r1,c1]*dr*dc)
    return np.where(inside, result, 0)


def angular_distance(zenith1, azimuth1, zenith2, azimuth2):
    z1, z2, diff = np.radians(zenith1), np.radians(zenith2), np.radians(azimuth1-azimuth2)
    cosine = np.cos(z1)*np.cos(z2) + np.sin(z1)*np.sin(z2)*np.cos(diff)
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))

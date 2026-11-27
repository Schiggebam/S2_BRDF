from collections import defaultdict
import xml.etree.ElementTree as ET
import numpy as np
import xarray as xr
from pathlib import Path
from pyproj import Transformer
from datetime import datetime

from rts import RTS_S2, setup_rts_logger, L2A_processor

from shared.helper_s2 import get_datetime_from_s2ids, _get_fn_prod

from typing import List

import numpy as np

def circular_mean_deg(stack, axis=0):
    """
    Compute circular mean (degrees) along given axis.
    """
    radians = np.deg2rad(stack)
    sin_mean = np.nanmean(np.sin(radians), axis=axis)
    cos_mean = np.nanmean(np.cos(radians), axis=axis)
    return np.rad2deg(np.arctan2(sin_mean, cos_mean))


def circular_diff_deg(a, b):
    """
    Compute minimal angular difference (degrees) between two azimuth grids.
    Result is in range [-180, 180].
    """
    return (a - b + 180) % 360 - 180

def compute_relative_azimuth(saa, vaa):
    """
    Compute relative azimuth = sun azimuth - view azimuth
    with circular wrapping.
    """
    return (saa - vaa + 180) % 360 - 180

def find_max_relative_azimuth_location(ds, var_name="delta_relaz"):
    """
    Find the grid cell where absolute relative azimuth difference is largest.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing delta relative azimuth grid.
    var_name : str
        Name of the variable storing delta relative azimuth.

    Returns
    -------
    dict with:
        - max_value
        - row_index
        - col_index
        - x_coord
        - y_coord
    """

    data = ds[var_name].values

    # Absolute difference
    abs_data = np.abs(data)

    # Ignore NaNs
    if np.all(np.isnan(abs_data)):
        raise ValueError("All values are NaN.")

    # Index of maximum value
    flat_index = np.nanargmax(abs_data)
    row, col = np.unravel_index(flat_index, abs_data.shape)

    # Get coordinate center
    x_coord = float(ds["x"].values[col])
    y_coord = float(ds["y"].values[row])

    return {
        "max_value_deg": float(abs_data[row, col]),
        "row_index": int(row),
        "col_index": int(col),
        "x_coord": x_coord,
        "y_coord": y_coord,
    }

def extract_full_viewing_angles(root, band_id: int):
    """
    Extract full-tile viewing zenith and azimuth grids
    for a specific bandId by combining all detector grids.

    Parameters
    ----------
    root : XML root (namespace already stripped)
    band_id : int
        Sentinel-2 internal bandId (e.g., 3 for B04 red band)

    Returns
    -------
    grid_vza_full : np.ndarray
    grid_vaa_full : np.ndarray
    """

    # Find all viewing grids for the selected band
    view_blocks = [
        block for block in root.findall(".//Viewing_Incidence_Angles_Grids")
        if block.attrib.get("bandId") == str(band_id)
    ]

    if not view_blocks:
        raise ValueError(f"No Viewing_Incidence_Angles_Grids found for bandId={band_id}")

    vza_list = []
    vaa_list = []

    for block in view_blocks:

        zenith_node = block.find("Zenith")
        azimuth_node = block.find("Azimuth")

        # --- Zenith ---
        vals_zenith = zenith_node.find("Values_List").findall("VALUES")
        grid_vza = np.array([
            [float(v) if v != "NaN" else np.nan for v in row.text.split()]
            for row in vals_zenith
        ])

        # --- Azimuth ---
        vals_azimuth = azimuth_node.find("Values_List").findall("VALUES")
        grid_vaa = np.array([
            [float(v) if v != "NaN" else np.nan for v in row.text.split()]
            for row in vals_azimuth
        ])

        vza_list.append(grid_vza)
        vaa_list.append(grid_vaa)

    # Stack detector grids
    vza_stack = np.stack(vza_list)
    vaa_stack = np.stack(vaa_list)

    # Combine detectors via nanmean
    # grid_vza_full = np.nanmean(vza_stack, axis=0)
    # grid_vaa_full = np.nanmean(vaa_stack, axis=0)
    grid_vza_full = np.nanmean(vza_stack, axis=0)
    grid_vaa_full = circular_mean_deg(vaa_stack, axis=0)

    return grid_vza_full, grid_vaa_full



def get_angle_grid_from_s2id(
    scene_id,
    d_base_s2=Path("/dss/dsstbyfs01/pn56su/pn56su-dss-0008/Sentinel-2-Col-1/L2A/tiles/"),
    date=None, tile_id=None
):
    """
    Returns full SZA and SAA grids as an xarray Dataset with CRS and coordinates.
    """

    if date is None:
        date = datetime.strptime(scene_id.split('_')[2], "%Y%m%dT%H%M%S")

    if tile_id is None:
        tile_id = scene_id.split('_')[5][1:]

    fn_xml = d_base_s2 / tile_id[:2] / tile_id[2] / tile_id[3:]
    fn_xml = fn_xml / str(date.year) / str(date.month).zfill(2)
    fn_xml = fn_xml / (scene_id + ".SAFE")

    try:
        fn_xml = next(fn_xml.glob("GRANULE/*/MTD_TL.xml"))
    except StopIteration:
        return None

    root = ET.parse(fn_xml).getroot()

    epsg_code = root.find('.//HORIZONTAL_CS_CODE').text.strip()

    gp = root.find('.//Geoposition[@resolution="20"]')
    ULX = float(gp.find('ULX').text)
    ULY = float(gp.find('ULY').text)
    XDIM = float(gp.find('XDIM').text)
    YDIM = float(gp.find('YDIM').text)

    # Navigate explicitly
    root = ET.parse(fn_xml).getroot()

    # --- Strip namespace from all tags ---
    for elem in root.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]

    sun_grid = root.find(".//Sun_Angles_Grid")

    if sun_grid is None:
        raise ValueError("Sun_Angles_Grid not found")

    zenith_node = sun_grid.find("Zenith")
    azimuth_node = sun_grid.find("Azimuth")

    col_step = float(zenith_node.find("COL_STEP").text)
    row_step = float(zenith_node.find("ROW_STEP").text)

    vals_zenith = zenith_node.find("Values_List").findall("VALUES")
    grid_zenith = np.array([
        [float(v) for v in row.text.split()]
        for row in vals_zenith
    ])

    vals_azimuth = azimuth_node.find("Values_List").findall("VALUES")
    grid_azimuth = np.array([
        [float(v) for v in row.text.split()]
        for row in vals_azimuth
    ])

    # --- Extract viewing angles ---
    grid_vza, grid_vaa = extract_full_viewing_angles(root, band_id=3)


    nrows, ncols = grid_zenith.shape

    # --- Build coordinate arrays in projected CRS ---
    x_coords = ULX + np.arange(ncols) * col_step
    y_coords = ULY - np.arange(nrows) * row_step

    # --- Create xarray Dataset ---
    ds = xr.Dataset(
        {
            "SZA": (("y", "x"), grid_zenith),
            "SAA": (("y", "x"), grid_azimuth),
            "VZA": (("y", "x"), grid_vza),
            "VAA": (("y", "x"), grid_vaa),
        },
        coords={
            "x": x_coords,
            "y": y_coords,
        },
        attrs={
            "crs": epsg_code,
            "scene_id": scene_id,
        },
    )


    return ds

def find_s2abc_pairs(tile_id: str, years:List[str], months: List[str] | None, day=None):
    setup_rts_logger()

    rts = RTS_S2(
        tile_id,
        L2A_processor.Sen2Cor,
        years,
        months,
        ["B04", "B08", "B12"],
        max_cloud_cover=10.0,
        max_sun_zenith_angle=70,
        max_nodata_percentage=100,
        min_processor_version="N0500"
    )

    rts.crawl_files()

    scene_strings = list(rts.d_files.keys())
    datetimes = get_datetime_from_s2ids(scene_strings)

    day_dict = defaultdict(dict)

    for scene, dt in zip(scene_strings, datetimes):
        date_str = dt.strftime("%Y%m%d")
        if day is not None:
            if dt.day != int(day):
                continue
            
        if scene.startswith("S2A"):
            day_dict[date_str]["S2A"] = {"id": scene, "datetime": dt, "rts": rts.d_files[scene], "rts_obj": rts}
        elif scene.startswith("S2B"):
            day_dict[date_str]["S2B"] = {"id": scene, "datetime": dt, "rts": rts.d_files[scene], "rts_obj": rts}
        elif scene.startswith("S2C"):
            day_dict[date_str]["S2C"] = {"id": scene, "datetime": dt, "rts": rts.d_files[scene], "rts_obj": rts}

    # Keep only days where at least two satellites are present
    result = {
        date: platforms
        for date, platforms in day_dict.items()
        if len(platforms) >= 2
    }

    return result

def flip_azimuth_180(vaa):
    """
    Flip viewing azimuth by 180 degrees with circular wrapping.
    """
    return (vaa + 180) % 360


def get_s2_pair_geometry_tiles(tile_id: str, years, months):

    results = []

    pairs = find_s2abc_pairs(tile_id, years, months)

    for date, data in pairs.items():

        if "S2A" not in data or "S2B" not in data:
            continue

        d0 = data["S2A"]   # always S2A
        d1 = data["S2B"]   # always S2B

        print(f"{tile_id} {date} -> S2A: {d0['id']}")
        print(f"{tile_id} {date} -> S2B: {d1['id']}")

        # --- First observation ---
        # d0 = data[list(data.keys())[0]]
        ds0 = get_angle_grid_from_s2id(
            d0["id"],
            date=d0["datetime"],
            tile_id=tile_id
        )

        # --- Second observation ---
        # d1 = data[list(data.keys())[1]]
        ds1 = get_angle_grid_from_s2id(
            d1["id"],
            date=d1["datetime"],
            tile_id=tile_id
        )

        if ds0 is None or ds1 is None:
            continue

        # --- Raw angles ---
        SZA_0 = ds0["SZA"].values
        SZA_1 = ds1["SZA"].values
        VZA_0 = ds0["VZA"].values
        VZA_1 = ds1["VZA"].values
        SAA_0 = ds0["SAA"].values
        SAA_1 = ds1["SAA"].values
        VAA_0 = ds0["VAA"].values
        VAA_1 = ds1["VAA"].values

        # --- Flip second VAA ---
        VAA_1_flipped = flip_azimuth_180(VAA_1)

        # --- Relative azimuth ---
        relaz_0 = compute_relative_azimuth(SAA_0, VAA_0)
        relaz_1 = compute_relative_azimuth(SAA_1, VAA_1_flipped)

        delta_relaz = circular_diff_deg(relaz_1, relaz_0)

        # --- (Still simple difference, but keep for now)
        relzen_0 = SZA_0 - VZA_0
        relzen_1 = SZA_1 - VZA_1

        # --- Local azimuth
        local_az_0 = circular_diff_deg(SAA_0, VAA_0)
        local_az_1 = circular_diff_deg(SAA_1, VAA_1_flipped)

        # --- Compute scattering angle ---
        theta_s0 = np.deg2rad(SZA_0)
        theta_v0 = np.deg2rad(VZA_0)
        phi0 = np.deg2rad(relaz_0)

        cos_scatter_0 = (
            np.cos(theta_s0) * np.cos(theta_v0)
            + np.sin(theta_s0) * np.sin(theta_v0) * np.cos(phi0)
        )

        scatter_0 = np.rad2deg(np.arccos(np.clip(cos_scatter_0, -1, 1)))

        # --- Scattering angle obs 1 ---
        theta_s1 = np.deg2rad(SZA_1)
        theta_v1 = np.deg2rad(VZA_1)
        phi1 = np.deg2rad(relaz_1)

        cos_scatter_1 = (
            np.cos(theta_s1) * np.cos(theta_v1)
            + np.sin(theta_s1) * np.sin(theta_v1) * np.cos(phi1)
        )

        scatter_1 = np.rad2deg(np.arccos(np.clip(cos_scatter_1, -1, 1)))

        delta_sza = SZA_1 - SZA_0
        delta_vza = VZA_1 - VZA_0
        delta_relaz = circular_diff_deg(relaz_1, relaz_0)
        delta_scatter = scatter_1 - scatter_0

        abs_delta_sza = np.abs(delta_sza)
        abs_delta_vza = np.abs(delta_vza)
        abs_delta_relaz = np.abs(delta_relaz)
        abs_delta_scatter = np.abs(delta_scatter)

        ds_geom = xr.Dataset(
            {
                # Raw angles
                "SZA_0": (("y", "x"), SZA_0),
                "SZA_1": (("y", "x"), SZA_1),
                "VZA_0": (("y", "x"), VZA_0),
                "VZA_1": (("y", "x"), VZA_1),
                "SAA_0": (("y", "x"), SAA_0),
                "SAA_1": (("y", "x"), SAA_1),
                "VAA_0": (("y", "x"), VAA_0),
                "VAA_1": (("y", "x"), VAA_1),

                # Derived
                "scatter_0": (("y","x"), scatter_0),
                "scatter_1": (("y","x"), scatter_1),
                "relaz_0": (("y", "x"), relaz_0),
                "relaz_1": (("y", "x"), relaz_1),
                # "delta_relaz": (("y", "x"), delta_relaz),
                "relzen_0": (("y", "x"), relzen_0),
                "relzen_1": (("y", "x"), relzen_1),
                "local_az_0": (("y", "x"), local_az_0),
                "local_az_1": (("y", "x"), local_az_1),

                "delta_SZA": (("y","x"), delta_sza),
                "delta_VZA": (("y","x"), delta_vza),
                "delta_relaz": (("y","x"), delta_relaz),
                "delta_scatter": (("y","x"), delta_scatter),

                "abs_delta_SZA": (("y","x"), abs_delta_sza),
                "abs_delta_VZA": (("y","x"), abs_delta_vza),
                "abs_delta_relaz": (("y","x"), abs_delta_relaz),
                "abs_delta_scatter": (("y","x"), abs_delta_scatter),
            },
            coords={
                "x": ds0["x"],
                "y": ds0["y"],
            },
            attrs={
                "tile_id": tile_id,
                "date": date,
                "scene_0": d0["id"],
                "scene_1": d1["id"],
            }
        )

        results.append(ds_geom)

    return results

##
# def test_tile(tile_id):
if __name__ == "__main__":
    tile_id = "31UDP"
    # tile_id = "33KWP"
    years = ["2025", "2026"]
    months = None
    r = find_s2abc_pairs(tile_id, years, months)
    print(r.keys())
    
    for date, data  in r.items():
        print(date)
        # print(data)
        print()
        if "S2A" not in data or "S2B" not in data:
            continue

        d0 = data["S2A"]
        d1 = data["S2B"]
        # first obs
        # d = data[list(data.keys())[0]]
        ds0 = get_angle_grid_from_s2id(d0["id"], date=d0["datetime"], tile_id=tile_id)
        # second obs
        # d = data[list(data.keys())[1]]
        ds1 = get_angle_grid_from_s2id(d1["id"], date=d1["datetime"], tile_id=tile_id)

        vaa0 = ds0["VAA"].values
        vaa1 = ds1["VAA"].values
        vaa1 = flip_azimuth_180(vaa1)

        print(np.nanmean(ds0["VAA"]))
        print(np.nanmean(ds1["VAA"]))

        # --- Compute differences ---
        delta_sza = ds1["SZA"] - ds0["SZA"]

        delta_saa = circular_diff_deg(
            ds1["SAA"].values,
            ds0["SAA"].values
        )

        # --- Statistics SZA ---
        print("SZA difference statistics:")
        print(f"  Mean: {float(delta_sza.mean().values):.4f}°")
        print(f"  Min : {float(delta_sza.min().values):.4f}°")
        print(f"  Max : {float(delta_sza.max().values):.4f}°")

        # --- Statistics SAA ---
        print("SAA difference statistics:")
        print(f"  Mean: {float(np.mean(delta_saa)):.4f}°")
        print(f"  Min : {float(np.min(delta_saa)):.4f}°")
        print(f"  Max : {float(np.max(delta_saa)):.4f}°")

        # --- Compute relative azimuth for both scenes ---
        relaz_0 = compute_relative_azimuth(ds0["SAA"].values, vaa0)
        relaz_1 = compute_relative_azimuth(ds1["SAA"].values, vaa1)

        # --- Δ relative azimuth ---
        delta_relaz = circular_diff_deg(relaz_1, relaz_0)

        ds_delta = xr.Dataset(
            {
                "delta_relaz": (("y", "x"), delta_relaz)
            },
            coords={
                "x": ds0["x"],
                "y": ds0["y"],
            }
        )

        result = find_max_relative_azimuth_location(ds_delta)
        print(result)

        print("Relative Azimuth difference statistics:")
        print(f"  Mean: {float(np.nanmean(delta_relaz)):.4f}°")
        print(f"  Min : {float(np.nanmin(delta_relaz)):.4f}°")
        print(f"  Max : {float(np.nanmax(delta_relaz)):.4f}°")

        print("\n" + "="*40 + "\n")
        exit(0)


    #python3 -m f_BRDF_measurement.find_S2AC_pairs
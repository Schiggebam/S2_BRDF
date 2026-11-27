from pathlib import Path

import numpy as np
import xarray as xr
import rioxarray
import rasterio
import geopandas as gpd
from rasterio.warp import reproject, Resampling
from rasterio.transform import rowcol
from pyproj import Transformer
from datetime import datetime

from shapely.geometry import Point

from scipy.interpolate import RegularGridInterpolator

from f_BRDF_measurement.main_worldwide import get_mdt_msixml_L2A, compute_geometry
from shared.helper_S2_L1C import get_angle_grid_from_s2id

from f_BRDF_measurement.main import prepare_lc

from f_BRDF_measurement.S2_L1C_atmos_corrected import L1C_AtmosCorr
from f_BRDF_measurement.S2_L1C_firstOrderCorr import L1C_IlluminationCorrection

# P6S profiles
from Py6S import AtmosProfile, AeroProfile
from shared.helper_satellite import CWL_S2

# ============================================================
# USER CONFIGURATION
# ============================================================


dir_base_l1c = Path("/dss/dsstbyfs02/pn49cu/pn49cu-dss-0021/phd_paul/datasets/S2_L1C")


# #32UPU
# level = "1C"
# s2a_safe = "S2A_MSIL1C_20250318T101751_N0511_R065_T32UPU_20250318T140212.SAFE"
# s2b_safe = "S2B_MSIL1C_20250318T100639_N0511_R022_T32UPU_20250318T134421.SAFE"
# s2a_ndvi = dir_base_l1c / "masks/S2A_MSIL1C_20250318T101751_N0511_R065_T32UPU_20250318T140212__NDVI.tif"
# s2b_ndvi = dir_base_l1c / "masks/S2B_MSIL1C_20250318T100639_N0511_R022_T32UPU_20250318T134421__NDVI.tif"
# LIMIT_A = True

# 32UQU

# s2a_safe = "S2A_MSIL1C_20250318T101751_N0511_R065_T32UQU_20250318T140212.SAFE"
# s2b_safe = "S2B_MSIL1C_20250318T100639_N0511_R022_T32UQU_20250318T134421.SAFE"
# s2a_ndvi = dir_base_l1c / "masks/S2A_MSIL1C_20250318T101751_N0511_R065_T32UQU_20250318T140212__NDVI.tif"
# s2b_ndvi = dir_base_l1c / "masks/S2B_MSIL1C_20250318T100639_N0511_R022_T32UQU_20250318T134421__NDVI.tif"
# level = "1C"
# LIMIT_A = False


# # 32UPU L2A
# dir_base_l2a = Path("/dss/dsstbyfs01/pn56su/pn56su-dss-0008/Sentinel-2-Col-1/L2A/tiles/32/U/PU/2025/03")
# s2a_safe = "S2A_MSIL2A_20250318T101751_N0511_R065_T32UPU_20250318T160400.SAFE"
# s2b_safe = "S2B_MSIL2A_20250318T100639_N0511_R022_T32UPU_20250318T141314.SAFE"
# s2a_ndvi = dir_base_l1c / "masks/S2A_MSIL1C_20250318T101751_N0511_R065_T32UPU_20250318T140212__NDVI.tif"
# s2b_ndvi = dir_base_l1c / "masks/S2B_MSIL1C_20250318T100639_N0511_R022_T32UPU_20250318T134421__NDVI.tif"
# level = "2A"
# LIMIT_A = True

# # 32UQU L2A
# dir_base_l2a = Path("/dss/dsstbyfs01/pn56su/pn56su-dss-0008/Sentinel-2-Col-1/L2A/tiles/32/U/QU/2025/03")
# s2a_safe = "S2A_MSIL2A_20250318T101751_N0511_R065_T32UQU_20250318T160400.SAFE"
# s2b_safe = "S2B_MSIL2A_20250318T100639_N0511_R022_T32UQU_20250318T141314.SAFE"
# s2a_ndvi = dir_base_l1c / "masks/S2A_MSIL1C_20250318T101751_N0511_R065_T32UQU_20250318T140212__NDVI.tif"
# s2b_ndvi = dir_base_l1c / "masks/S2B_MSIL1C_20250318T100639_N0511_R022_T32UQU_20250318T134421__NDVI.tif"
# level = "2A"
# LIMIT_A = False


# 32ULC L2A
# dir_base_l2a = Path("/dss/dsstbyfs01/pn56su/pn56su-dss-0008/Sentinel-2-Col-1/L2A/tiles/32/U/LC/2025/06")
# s2a_safe = "S2A_MSIL2A_20250612T103701_N0511_R008_T32ULC_20250612T124517.SAFE"
# s2b_safe = "S2B_MSIL2A_20250612T102559_N0511_R108_T32ULC_20250612T131505.SAFE"
# # s2a_ndvi = dir_base_l1c / "masks/S2A_MSIL1C_20250318T101751_N0511_R065_T32UPU_20250318T140212__NDVI.tif"
# # s2b_ndvi = dir_base_l1c / "masks/S2B_MSIL1C_20250318T100639_N0511_R022_T32UPU_20250318T134421__NDVI.tif"
# level = "2A"
# LIMIT_A = True

tile_id = "32UMC"
dir_base_l2a = Path("/dss/dsstbyfs01/pn56su/pn56su-dss-0008/Sentinel-2-Col-1/L2A/tiles/32/U/MC/2025/06")
s2a_safe = "S2A_MSIL2A_20250612T103701_N0511_R008_T32UMC_20250612T124517.SAFE"
s2b_safe = "S2B_MSIL2A_20250612T102559_N0511_R108_T32UMC_20250612T131505.SAFE"
# s2a_ndvi = dir_base_l1c / "masks/S2A_MSIL1C_20250318T101751_N0511_R065_T32UPU_20250318T140212__NDVI.tif"
# s2b_ndvi = dir_base_l1c / "masks/S2B_MSIL1C_20250318T100639_N0511_R022_T32UPU_20250318T134421__NDVI.tif"
level = "2A"
LIMIT_A = False


BANDS = [
    "B05",
    "B12",
]



N_SAMPLES = 50
PATCH_SIZE = 1
SEED = 42

# Operational definition of near-nadir
NADIR_LIMIT = 4.0



# WorldCover classes
VALID_LC = [
    10,   # Tree cover
    30,   # Grassland
    40,   # Cropland
]

WORLD_COVER_VRT = (
    "/dss/dsstbyfs02/pn49cu/pn49cu-dss-0021/worldsoils/processing/dias_mundi/ws-obs/datalake/landcover/ESA_WorldCover_2020/Europe_extended.vrt"
)

LANDCOVER_TMP = Path("/dss/dsstbyfs02/scratch/04/di35qiw/s2_brdf/landcover_tmp/")

LANDCOVER_TMP.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# P6S ATMOSPHERIC PROFILE
# ============================================================

ATMOS_DEFAULT = {
    "atmos_profile": AtmosProfile.MidlatitudeSummer,
    "aero_profile": AeroProfile.Continental,
    "aot550": 0.1,
    "altitude_km": 0.5,
}


# ============================================================
# SCENE IDS
# ============================================================

s2a_id = s2a_safe.replace(".SAFE", "")
s2b_id = s2b_safe.replace(".SAFE", "")

tile_id = s2a_safe.split("_")[5][1:]


# ============================================================
# L2A METADATA
# ============================================================

fn_mtd_s2a = get_mdt_msixml_L2A(
    s2a_safe,
    sensor="S2A"
)

fn_mtd_s2b = get_mdt_msixml_L2A(
    s2b_safe,
    sensor="S2B"
)


def sample_array_at_xy(
    array,
    transform,
    xs,
    ys,
):

    rows, cols = rowcol(
        transform,
        xs,
        ys
    )

    rows = np.asarray(rows)
    cols = np.asarray(cols)

    values = np.full(
        len(xs),
        np.nan,
        dtype=np.float32
    )

    valid = (
        (rows >= 0) &
        (rows < array.shape[0]) &
        (cols >= 0) &
        (cols < array.shape[1])
    )

    values[valid] = array[
        rows[valid],
        cols[valid]
    ]

    return values

def resample_ndvi_to_reference(fn_ndvi, fn_reference):

    with rasterio.open(fn_ndvi) as src, \
         rasterio.open(fn_reference) as ref:

        dst = np.full(
            (ref.height, ref.width),
            np.nan,
            dtype=np.float32
        )

        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=ref.transform,
            dst_crs=ref.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    return dst, ref.transform, ref.crs



# ============================================================
# ANGLE INTERPOLATION
# ============================================================

def interpolate_vza(
    angles,
    x,
    y,
):

    xa = angles.x.values
    ya = angles.y.values
    vza = angles.VZA.values

    # scipy requires increasing coordinates
    if xa[0] > xa[-1]:
        xa = xa[::-1]
        vza = vza[:, ::-1]

    if ya[0] > ya[-1]:
        ya = ya[::-1]
        vza = vza[::-1, :]

    interpolator = RegularGridInterpolator(
        (ya, xa),
        vza,
        bounds_error=False,
        fill_value=np.nan,
    )

    xx, yy = np.meshgrid(
        x,
        y
    )

    return interpolator(
        np.column_stack(
            (
                yy.ravel(),
                xx.ravel(),
            )
        )
    ).reshape(
        len(y),
        len(x)
    )

def calculate_ndvi_at_location(b04, b08, x, y):
    """
    Calculate NDVI at a single projected map coordinate.

    B04 and B08 must both be at 10 m resolution and use
    the same CRS.
    """
    # Check that the coordinate lies within both raster extents.
    for band_name, da in [("B04", b04), ("B08", b08)]:
        left, bottom, right, top = da.rio.bounds()

        if not (left <= x <= right and bottom <= y <= top):
            return np.nan

    red = float(b04.sel(x=x, y=y, method="nearest").values)
    nir = float(b08.sel(x=x, y=y, method="nearest").values)

    if not np.isfinite(red) or not np.isfinite(nir):
        return np.nan

    denominator = nir + red

    if denominator == 0:
        return np.nan

    return (nir - red) / denominator

def get_l1c_band(scene_safe, band,):
    fn = list((dir_base_l1c / scene_safe).glob(f"GRANULE/*/IMG_DATA/*_{band}.jp2"))[0]

    return rioxarray.open_rasterio(fn, masked=True,).squeeze()

def get_l2a_band(scene_safe, band, resolution):
    pattern = (
        f"GRANULE/*/IMG_DATA/R{resolution}m/"
        f"*_{band}_{resolution}m.jp2"
    )

    files = list((dir_base_l2a / scene_safe).glob(pattern))

    if len(files) != 1:
        raise FileNotFoundError(
            f"Expected one file for {scene_safe}, {band}, "
            f"{resolution} m; found {len(files)}."
        )

    return rioxarray.open_rasterio(files[0], masked=True, ).squeeze()

def get_ndvi_band_path(scene_safe, band):
    """
    Find a native 10 m B04 or B08 raster for an L2A scene.
    """
    if level == "2A":
        pattern = f"GRANULE/*/IMG_DATA/R10m/*_{band}_10m.jp2"
        base_dir = dir_base_l2a
    elif level == "1C":
        pattern = f"GRANULE/*/IMG_DATA/*_{band}.jp2"
        base_dir = dir_base_l1c
    else:
        raise ValueError(f"Unsupported processing level: {level}")

    files = list((base_dir / scene_safe).glob(pattern))

    if len(files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {band} raster for {scene_safe}; "
            f"found {len(files)} using pattern {pattern}"
        )

    return files[0]


def ndvi_at_points(red_raster, nir_raster, coords):
    """
    Calculate NDVI at map coordinates using native B04 and B08 rasters.

    Parameters
    ----------
    red_raster : rasterio.DatasetReader
        Native 10 m B04 raster.
    nir_raster : rasterio.DatasetReader
        Native 10 m B08 raster.
    coords : list of (x, y)
        Coordinates in the raster CRS.

    Returns
    -------
    ndvi : numpy.ndarray
        NDVI at each coordinate. Invalid observations are NaN.
    valid : numpy.ndarray
        Boolean mask indicating valid NDVI observations.
    """
    if len(coords) == 0:
        return np.array([], dtype=float), np.array([], dtype=bool)

    # Read the nearest native raster pixel at each coordinate.
    red = np.ma.vstack(
        list(red_raster.sample(coords, masked=True))
    ).astype(float).filled(np.nan)[:, 0]

    nir = np.ma.vstack(
        list(nir_raster.sample(coords, masked=True))
    ).astype(float).filled(np.nan)[:, 0]

    denominator = nir + red

    valid = (
        np.isfinite(red)
        & np.isfinite(nir)
        & (denominator != 0)
    )

    ndvi = np.full(len(coords), np.nan, dtype=float)

    ndvi[valid] = (
        (nir[valid] - red[valid])
        / denominator[valid]
    )

    return ndvi, valid

# ============================================================
# BUILD COMMON SAMPLE LOCATIONS

def get_l2a_band_path(scene_safe, band, resolution):
    """Return the path to a native-resolution L2A band."""
    files = list(
        (dir_base_l2a / scene_safe).glob(
            f"GRANULE/*/IMG_DATA/R{resolution}m/*_{band}_{resolution}m.jp2"
        )
    )

    if len(files) != 1:
        raise FileNotFoundError(
            f"Expected one {band} file at {resolution} m; found {len(files)}."
        )

    return files[0]


def ndvi_at_points(b04, b08, coords):
    """Calculate NDVI at map coordinates using native 10 m B04/B08."""
    if not coords:
        return np.array([]), np.array([])

    red = np.ma.vstack(
        list(b04.sample(coords, masked=True))
    ).astype(float).filled(np.nan)[:, 0]

    nir = np.ma.vstack(
        list(b08.sample(coords, masked=True))
    ).astype(float).filled(np.nan)[:, 0]

    denominator = nir + red
    valid = (
        np.isfinite(red)
        & np.isfinite(nir)
        & (denominator != 0)
    )

    ndvi = np.full(len(coords), np.nan)
    ndvi[valid] = (
        (nir[valid] - red[valid]) / denominator[valid]
    )

    return ndvi, valid

def build_sample_locations():

    print("\n")
    print("=" * 70)
    print("BUILDING SAMPLE LOCATIONS")
    print("=" * 70)

    # --------------------------------------------------------
    # Use B12 / 20 m as reference grid
    # --------------------------------------------------------

    ref_band = "B12"

    if level == "1C":

        ref_band_fn_a = list(
            (dir_base_l1c / s2a_safe).glob(
                f"GRANULE/*/IMG_DATA/*_{ref_band}.jp2"
            )
        )[0]

        ref_band_fn_b = list(
            (dir_base_l1c / s2b_safe).glob(
                f"GRANULE/*/IMG_DATA/*_{ref_band}.jp2"
            )
        )[0]

    elif level == "2A":

        print(dir_base_l2a / s2a_safe)

        ref_band_fn_a = list(
            (dir_base_l2a / s2a_safe).glob(
                f"GRANULE/*/IMG_DATA/R20m/*_{ref_band}_20m.jp2"
            )
        )[0]

        ref_band_fn_b = list(
            (dir_base_l2a / s2b_safe).glob(
                f"GRANULE/*/IMG_DATA/R20m/*_{ref_band}_20m.jp2"
            )
        )[0]

    else:
        raise ValueError(f"Unsupported processing level: {level}")

    # --------------------------------------------------------
    # Check that both scenes use the same 20 m UTM grid
    # --------------------------------------------------------

    with (
        rasterio.open(ref_band_fn_a) as src_a,
        rasterio.open(ref_band_fn_b) as src_b,
    ):

        if src_a.crs != src_b.crs:
            raise RuntimeError(
                "S2A and S2B B12 have different CRS."
            )

        if src_a.transform != src_b.transform:
            raise RuntimeError(
                "S2A and S2B B12 have different transforms."
            )

        if (
            src_a.width != src_b.width
            or src_a.height != src_b.height
        ):
            raise RuntimeError(
                "S2A and S2B B12 have different dimensions."
            )

    # --------------------------------------------------------
    # WorldCover aligned to S2A B12
    # --------------------------------------------------------

    fn_wc_local = prepare_lc(
        fn_lc_vrt=WORLD_COVER_VRT,
        fn_mincomp=ref_band_fn_a,
        aligned_landcover=(
            str(
                LANDCOVER_TMP
                / f"worldcover_{tile_id}_{ref_band}_{s2a_id}.tif"
            )
        ),
        res="20m",
    )

    wc = rioxarray.open_rasterio(
        fn_wc_local,
        masked=True,
    ).squeeze()

    # --------------------------------------------------------
    # Load viewing-angle grids for B12
    # --------------------------------------------------------

    angles_a = get_angle_grid_from_s2id(
        s2a_id,
        fn_mtd_s2a,
        int(ref_band[1:]),
    )

    angles_b = get_angle_grid_from_s2id(
        s2b_id,
        fn_mtd_s2b,
        int(ref_band[1:]),
        flip_grid=True,
    )

    # --------------------------------------------------------
    # Load reference image grids
    # --------------------------------------------------------

    if level == "1C":

        da_a = get_l1c_band(
            s2a_safe,
            ref_band,
        )

        da_b = get_l1c_band(
            s2b_safe,
            ref_band,
        )

    elif level == "2A":

        da_a = get_l2a_band(
            s2a_safe,
            ref_band,
            resolution=20,
        )

        da_b = get_l2a_band(
            s2b_safe,
            ref_band,
            resolution=20,
        )

    # --------------------------------------------------------
    # Spatial intersection
    # --------------------------------------------------------

    xmin = max(
        float(da_a.x.min()),
        float(da_b.x.min()),
    )

    xmax = min(
        float(da_a.x.max()),
        float(da_b.x.max()),
    )

    ymin = max(
        float(da_a.y.min()),
        float(da_b.y.min()),
    )

    ymax = min(
        float(da_a.y.max()),
        float(da_b.y.max()),
    )

    if xmin >= xmax or ymin >= ymax:
        raise RuntimeError(
            "S2A and S2B have no spatial overlap."
        )

    # --------------------------------------------------------
    # Common B12 grid
    # --------------------------------------------------------

    x_mask = (
        (da_a.x.values >= xmin)
        & (da_a.x.values <= xmax)
    )

    y_mask = (
        (da_a.y.values >= ymin)
        & (da_a.y.values <= ymax)
    )

    x = da_a.x.values[x_mask]
    y = da_a.y.values[y_mask]

    # --------------------------------------------------------
    # Make sure the B12 grids are actually identical
    # --------------------------------------------------------

    if not np.array_equal(
        da_a.x.values,
        da_b.x.values,
    ):
        raise RuntimeError(
            "S2A/S2B B12 x coordinates differ."
        )

    if not np.array_equal(
        da_a.y.values,
        da_b.y.values,
    ):
        raise RuntimeError(
            "S2A/S2B B12 y coordinates differ."
        )

    # --------------------------------------------------------
    # Viewing geometry
    # --------------------------------------------------------

    vza_a = interpolate_vza(
        angles_a,
        x,
        y,
    )

    vza_b = interpolate_vza(
        angles_b,
        x,
        y,
    )

    # --------------------------------------------------------
    # WorldCover on the same grid
    # --------------------------------------------------------

    wc = wc.sel(
        x=x,
        y=y,
        method="nearest",
    )

    wc_values = wc.values

    # --------------------------------------------------------
    # Base valid mask
    #
    # A must be near nadir if LIMIT_A is True.
    # Otherwise B must satisfy the nadir constraint.
    #
    # Both viewing angles must be valid, and the WorldCover
    # class must be one of VALID_LC.
    # --------------------------------------------------------

    vza_limiter = vza_a if LIMIT_A else vza_b

    base_mask = (
        np.isfinite(vza_a)
        & np.isfinite(vza_b)
        & (vza_limiter < NADIR_LIMIT)
        & np.isin(
            wc_values,
            VALID_LC,
        )
    )

    print(
        f"\nBase valid pixels "
        f"({'VZA_A' if LIMIT_A else 'VZA_B'} < "
        f"{NADIR_LIMIT}° + WorldCover {VALID_LC}): "
        f"{base_mask.sum():,}"
    )

    # --------------------------------------------------------
    # Open native 10 m B04 and B08 rasters
    #
    # NDVI is calculated directly at the sample coordinates.
    # No precomputed NDVI masks or resampled NDVI arrays
    # are required.
    # --------------------------------------------------------

    print("\nOpening native 10 m B04/B08 rasters...")

    b04_fn_a = get_ndvi_band_path(
        s2a_safe,
        "B04",
    )

    b08_fn_a = get_ndvi_band_path(
        s2a_safe,
        "B08",
    )

    b04_fn_b = get_ndvi_band_path(
        s2b_safe,
        "B04",
    )

    b08_fn_b = get_ndvi_band_path(
        s2b_safe,
        "B08",
    )

    # --------------------------------------------------------
    # Random sampling, separately for each land-cover class
    # --------------------------------------------------------

    rng = np.random.default_rng(SEED)

    samples = []

    # Batch size used when searching for cropland pixels
    # satisfying the NDVI threshold in both observations.
    batch_size = 512

    with (
        rasterio.open(b04_fn_a) as red_a,
        rasterio.open(b08_fn_a) as nir_a,
        rasterio.open(b04_fn_b) as red_b,
        rasterio.open(b08_fn_b) as nir_b,
    ):

        # ----------------------------------------------------
        # Check that the native B04/B08 grids are compatible
        # ----------------------------------------------------

        for name, red, nir in (
            ("S2A", red_a, nir_a),
            ("S2B", red_b, nir_b),
        ):

            if red.crs != nir.crs:
                raise RuntimeError(
                    f"{name} B04 and B08 have different CRS."
                )

            if (
                red.width != nir.width
                or red.height != nir.height
                or red.transform != nir.transform
            ):
                raise RuntimeError(
                    f"{name} B04 and B08 do not use the same grid."
                )

        # ----------------------------------------------------
        # Process each land-cover class
        # ----------------------------------------------------

        for lc in VALID_LC:

            # Start with geometry and land-cover conditions.
            lc_mask = (
                base_mask
                & (wc_values == lc)
            )

            # Candidate pixel indices on the common B12 grid.
            yy_lc, xx_lc = np.where(lc_mask)

            n_available = len(yy_lc)

            # Store selected pixels as:
            # (row index, column index, NDVI_A, NDVI_B)
            chosen = []

            # =================================================
            # CROPLAND: NDVI MUST BE < 0.15 IN BOTH OBSERVATIONS
            # =================================================

            if lc == 40:

                print(
                    f"\nLC {lc}: searching {n_available:,} "
                    f"cropland candidates for NDVI < 0.15 "
                    f"in both S2A and S2B..."
                )

                # Randomize the order so the selected pixels
                # remain a random sample of valid candidates.
                order = rng.permutation(n_available)

                # Search in batches rather than calculating
                # NDVI for every pixel in the full scene.
                for start in range(
                    0,
                    n_available,
                    batch_size,
                ):

                    idx = order[
                        start:start + batch_size
                    ]

                    coords = [
                        (
                            float(x[xx_lc[j]]),
                            float(y[yy_lc[j]]),
                        )
                        for j in idx
                    ]

                    # Calculate NDVI directly from B04/B08
                    # at the candidate coordinates.
                    ndvi_a, valid_a = ndvi_at_points(
                        red_a,
                        nir_a,
                        coords,
                    )

                    ndvi_b, valid_b = ndvi_at_points(
                        red_b,
                        nir_b,
                        coords,
                    )

                    # Retain only pixels with valid NDVI
                    # below the threshold in BOTH scenes.
                    keep = (
                        valid_a
                        & valid_b
                        & np.isfinite(ndvi_a)
                        & np.isfinite(ndvi_b)
                        & (ndvi_a < 0.15)
                        & (ndvi_b < 0.15)
                    )

                    for k in np.flatnonzero(keep):

                        j = idx[k]

                        chosen.append(
                            (
                                yy_lc[j],
                                xx_lc[j],
                                ndvi_a[k],
                                ndvi_b[k],
                            )
                        )

                        # Stop once enough samples have been
                        # selected for this land-cover class.
                        if len(chosen) >= N_SAMPLES:
                            break

                    if len(chosen) >= N_SAMPLES:
                        break

            # =================================================
            # OTHER LAND-COVER CLASSES
            # =================================================

            else:

                n = min(
                    N_SAMPLES,
                    n_available,
                )

                if n > 0:

                    # Randomly select candidate pixels.
                    selected = rng.choice(
                        n_available,
                        size=n,
                        replace=False,
                    )

                    coords = [
                        (
                            float(x[xx_lc[j]]),
                            float(y[yy_lc[j]]),
                        )
                        for j in selected
                    ]

                    # Calculate NDVI at these locations too,
                    # so every sample has directly calculated
                    # NDVI values for both observations.
                    ndvi_a, _ = ndvi_at_points(
                        red_a,
                        nir_a,
                        coords,
                    )

                    ndvi_b, _ = ndvi_at_points(
                        red_b,
                        nir_b,
                        coords,
                    )

                    for k, j in enumerate(selected):

                        chosen.append(
                            (
                                yy_lc[j],
                                xx_lc[j],
                                ndvi_a[k],
                                ndvi_b[k],
                            )
                        )

            # -------------------------------------------------
            # Report sampling results
            # -------------------------------------------------

            print(
                f"LC {lc}: "
                f"{n_available:,} candidates -> "
                f"{len(chosen):,} samples"
            )

            if len(chosen) == 0:

                print(
                    f"WARNING: no valid samples for LC {lc}"
                )

                if lc == 40:
                    print(
                        "  No cropland candidates satisfied "
                        "the NDVI < 0.15 condition in both scenes."
                    )

                continue

            # -------------------------------------------------
            # Store sample locations and associated values
            # -------------------------------------------------

            for py, px, ndvi_a, ndvi_b in chosen:

                samples.append(
                    {
                        "x": float(x[px]),
                        "y": float(y[py]),
                        "vza_a": float(vza_a[py, px]),
                        "vza_b": float(vza_b[py, px]),
                        "wc": int(wc_values[py, px]),
                        "ndvi_a": float(ndvi_a),
                        "ndvi_b": float(ndvi_b),
                    }
                )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        f"\nTotal selected samples: {len(samples)}"
    )

    for lc in VALID_LC:

        n_lc = sum(
            s["wc"] == lc
            for s in samples
        )

        print(
            f"  LC {lc}: {n_lc}"
        )

    if len(samples) == 0:
        raise RuntimeError(
            "No valid sampling locations."
        )

    # --------------------------------------------------------
    # Viewing-angle summary
    # --------------------------------------------------------

    print(
        f"\nA VZA: "
        f"{np.min([s['vza_a'] for s in samples]):.2f} -> "
        f"{np.max([s['vza_a'] for s in samples]):.2f}°"
    )

    print(
        f"B VZA: "
        f"{np.min([s['vza_b'] for s in samples]):.2f} -> "
        f"{np.max([s['vza_b'] for s in samples]):.2f}°"
    )

    # --------------------------------------------------------
    # Diagnostic specifically for cropland
    # --------------------------------------------------------

    crop_samples = [
        s for s in samples
        if s["wc"] == 40
    ]

    if crop_samples:

        print("\nCropland NDVI:")

        print(
            f"  S2A: "
            f"{np.min([s['ndvi_a'] for s in crop_samples]):.3f} -> "
            f"{np.max([s['ndvi_a'] for s in crop_samples]):.3f}"
        )

        print(
            f"  S2B: "
            f"{np.min([s['ndvi_b'] for s in crop_samples]):.3f} -> "
            f"{np.max([s['ndvi_b'] for s in crop_samples]):.3f}"
        )

    return samples


# ============================================================
# ANALYSE ONE BAND: L2A DIRECT REFLECTANCE
# ============================================================


BAND_RESOLUTION = {"B04": 10, "B05": 20, "B08": 10}

LC_NAMES = {
    10: "Tree cover",
    30: "Grassland",
    40: "Cropland",
}


def sample_pixel(da, x, y):
    """Return nearest pixel value, or NaN if outside the raster."""
    left, bottom, right, top = da.rio.bounds()

    if not (left <= x <= right and bottom <= y <= top):
        return np.nan

    return float(da.sel(x=x, y=y, method="nearest").values)


def calculate_ndvi(b04, b08, x, y):
    """Calculate NDVI from 10 m red and NIR bands at one location."""
    red = sample_pixel(b04, x, y)
    nir = sample_pixel(b08, x, y)
    denominator = nir + red

    if not np.isfinite(red) or not np.isfinite(nir) or denominator == 0:
        return np.nan

    return (nir - red) / denominator


def analyse_band(band, samples):
    """Compare S2A and S2B L2A reflectance at sample locations."""

    print(f"\n{'=' * 60}\nBAND: {band}\n{'=' * 60}")

    # Load the comparison band at its native resolution.
    res = BAND_RESOLUTION[band]
    da_a = get_l2a_band(s2a_safe, band, res)
    da_b = get_l2a_band(s2b_safe, band, res)

    # Load 10 m red/NIR bands for on-demand NDVI calculation.
    b04_a, b08_a = get_l2a_band(s2a_safe, "B04", 10), get_l2a_band(s2a_safe, "B08", 10)
    b04_b, b08_b = get_l2a_band(s2b_safe, "B04", 10), get_l2a_band(s2b_safe, "B08", 10)

    if da_a.rio.crs != da_b.rio.crs:
        raise ValueError(f"{band}: S2A and S2B CRS differ.")

    if not (
        np.array_equal(da_a.x.values, da_b.x.values)
        and np.array_equal(da_a.y.values, da_b.y.values)
    ):
        raise ValueError(f"{band}: S2A and S2B grids differ.")

    crs = da_a.rio.crs
    valid = []

    try:
        for i, s in enumerate(samples):
            if i % 25 == 0:
                print(f"  Sample {i}/{len(samples)}")

            x, y = s["x"], s["y"]

            # Comparison-band reflectance.
            a = sample_pixel(da_a, x, y)
            b = sample_pixel(da_b, x, y)

            if not np.isfinite(a) or not np.isfinite(b):
                continue

            # Calculate NDVI directly at the sampling location.
            s["ndvi_a"] = calculate_ndvi(b04_a, b08_a, x, y)
            s["ndvi_b"] = calculate_ndvi(b04_b, b08_b, x, y)

            # Store reflectance and differences.
            diff = b - a
            mean_refl = (a + b) / 2

            s.update({
                "refl_a": a,
                "refl_b": b,
                "diff_ba": diff,
                "rel_diff": diff / mean_refl if mean_refl != 0 else np.nan,
            })

            valid.append(s)

    finally:
        for da in (da_a, da_b, b04_a, b08_a, b04_b, b08_b):
            da.close()

    if not valid:
        print("No valid reflectance samples.")
        return []

    # ---------------------------------------------------------
    # Statistics by land-cover class
    # ---------------------------------------------------------
    print(f"\nValid samples: {len(valid)}")

    for lc, name in LC_NAMES.items():
        subset = [s for s in valid if s["wc"] == lc]

        if not subset:
            print(f"\n{name}: no samples")
            continue

        a = np.array([s["refl_a"] for s in subset])
        b = np.array([s["refl_b"] for s in subset])
        d = b - a
        rd = np.array([s["rel_diff"] for s in subset])
        rd = rd[np.isfinite(rd)]

        print(f"\n{name} (N={len(subset)})")
        print(f"  S2A reflectance:       {a.mean():.6f}")
        print(f"  S2B reflectance:       {b.mean():.6f}")
        print(f"  Mean difference B-A:   {d.mean():+.6f}")
        print(f"  Difference std:        {d.std(ddof=1):.6f}" if len(d) > 1
              else "  Difference std:        NaN (N < 2)")
        print(f"  Mean relative diff.:   {100 * rd.mean():+.3f}%"
              if len(rd) else "  Mean relative diff.:   NaN")

        vza = np.array([s["vza_b"] for s in subset])
        print(f"  S2B VZA: {vza.mean():.2f}° "
              f"(range {vza.min():.2f}–{vza.max():.2f}°)")

    # ---------------------------------------------------------
    # Export samples to GeoPackage
    # ---------------------------------------------------------
    records = [{
        "lc": s["wc"],
        "ndvi_a": s["ndvi_a"],
        "ndvi_b": s["ndvi_b"],
        "vza_a": s["vza_a"],
        "vza_b": s["vza_b"],
        "refl_a": s["refl_a"],
        "refl_b": s["refl_b"],
        "diff_ba": s["diff_ba"],
        "rel_diff": s["rel_diff"],
        "geometry": Point(s["x"], s["y"]),
    } for s in valid]

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)

    output_dir = Path("/dss/dsstbyfs02/scratch/04/di35qiw/S2_l2a")
    output_dir.mkdir(parents=True, exist_ok=True)

    out = output_dir / f"{tile_id}_samples_{band}.gpkg"
    gdf.to_file(out, layer="samples", driver="GPKG")

    print(f"\nGeoPackage written to: {out}")

    return valid

# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Important:
    # sample locations are generated ONCE.
    #
    # Therefore B04 and B12 use exactly the same physical
    # locations.
    # --------------------------------------------------------

    samples = build_sample_locations()

    # --------------------------------------------------------
    # Analyse bands
    # --------------------------------------------------------

    results = {}

    for band in ["B04", "B05", "B08"]:
        results[band] = analyse_band(band, samples)


if __name__ == "__main__":
    main()

# python3 -m f_BRDF_measurement.check_nadir_d_ref_l2a
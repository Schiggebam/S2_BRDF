from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rioxarray
import matplotlib.pyplot as plt

from scipy.stats import pointbiserialr
from rasterio.enums import Resampling


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

dir_base_l2a = Path(
    "/dss/dsstbyfs01/pn56su/pn56su-dss-0008/"
    "Sentinel-2-Col-1/L2A/tiles/32/U/PU/2025/08"
)

s2a_scene = (
    "S2A_MSIL2A_20250815T102041_N0511_R065_"
    "T32UPU_20250815T140757.SAFE"
)
s2b_scene = (
    "S2B_MSIL2A_20250815T100559_N0511_R022_"
    "T32UPU_20250815T125703.SAFE"
)

WORLD_COVER_VRT = (
    "/dss/dsstbyfs02/pn49cu/pn49cu-dss-0021/"
    "worldsoils/processing/dias_mundi/ws-obs/datalake/"
    "landcover/ESA_WorldCover_2020/Europe_extended.vrt"
)

FN_POLYGON = (
    "/dss/dsshome1/04/di35qiw/"
    "look_here_for_stereo_test_1.gpkg"
)

ENMAP_MASK_DIR = Path("/dss/dsstbyfs02/pn49cu/pn49cu-dss-0021/playground/EnMAP_composite/mask/DT0000148819/DT0000148819_20250812T105345Z_004_V010502_20250813T202806Z")

N_SAMPLES = 1000
RANDOM_SEED = 43

NCAI_THRESHOLD = 0.014
NDVI_THRESHOLD = 0.25
NCAI_THRESHOLD_2 = 0.014



# ---------------------------------------------------------------------
# Read only the portion inside the polygon
# ---------------------------------------------------------------------

polygon = gpd.read_file(FN_POLYGON, layer="a")

if polygon.empty or polygon.crs is None:
    raise ValueError("The polygon is empty or has no defined CRS.")


def open_clipped(fn, polygon, masked=True):
    """Read a raster clipped to a polygon, with CRS and overlap checks."""

    with rasterio.open(fn) as src:
        raster_crs = src.crs

        if raster_crs is None:
            raise ValueError(f"Raster has no CRS: {fn}")

        raster_bounds = src.bounds
        raster_box = gpd.GeoSeries(
            [__import__("shapely").geometry.box(*raster_bounds)],
            crs=raster_crs,
        )

    if polygon.crs is None:
        raise ValueError("Polygon has no CRS.")

    poly = polygon.to_crs(raster_crs).copy()
    poly.geometry = poly.geometry.make_valid()

    if not poly.geometry.intersects(raster_box.iloc[0]).any():
        raise ValueError(
            f"Polygon does not overlap raster.\n"
            f"Raster: {fn}\n"
            f"Raster CRS: {raster_crs}\n"
            f"Raster bounds: {raster_bounds}\n"
            f"Polygon bounds in raster CRS: {poly.total_bounds}"
        )

    with rioxarray.open_rasterio(fn, masked=masked) as src:
        out = src.rio.clip(
            poly.geometry,
            poly.crs,
            drop=True,
            from_disk=True,
        ).squeeze(drop=True).load()

    return out


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

    data = open_clipped(files[0], polygon)

    # Preserve the original DN offset correction.
    if band != "SCL":
        data = data.where(data != -10000) - 1000

    return data


def align_to_reference(data, reference):
    """Align a clipped raster to the reference grid."""
    out = data.rio.reproject_match(
        reference,
        resampling=Resampling.nearest,
    ).load()

    data.close()
    return out


# ---------------------------------------------------------------------
# Sentinel-2: crop first, then align
# ---------------------------------------------------------------------

s2a_b04 = get_l2a_band(s2a_scene, "B04", 10)
s2a_b08 = get_l2a_band(s2a_scene, "B08", 10)

s2b_b04 = align_to_reference(
    get_l2a_band(s2b_scene, "B04", 10),
    s2a_b04,
)

s2a_scl = align_to_reference(
    get_l2a_band(s2a_scene, "SCL", 20),
    s2a_b04,
)

s2b_scl = align_to_reference(
    get_l2a_band(s2b_scene, "SCL", 20),
    s2a_b04,
)

wc = align_to_reference(
    open_clipped(WORLD_COVER_VRT, polygon),
    s2a_b04,
)

def print_raster_stats(name, da):
    values = da.values
    valid = np.isfinite(values)

    print(f"\n{name}")
    print("  Shape:", values.shape)
    print("  CRS:", da.rio.crs)
    print("  Bounds:", da.rio.bounds())
    print("  Finite pixels:", valid.sum(), "/", values.size)

    if valid.any():
        print("  Min:", np.nanmin(values))
        print("  Max:", np.nanmax(values))
        print(
            "  Unique values:",
            np.unique(values[valid])[:30],
        )


print_raster_stats("S2A SCL", s2a_scl)
print_raster_stats("S2B SCL", s2b_scl)

print("Done loading data (1)")


# ---------------------------------------------------------------------
# Build the valid Sentinel-2 mask
# ---------------------------------------------------------------------

rel_diff_s2 = (
    (s2a_b04 - s2b_b04)
    / s2b_b04.where(s2b_b04 != 0)
)

rel_diff_s2 = rel_diff_s2.where(np.isfinite(rel_diff_s2))

valid_s2 = (
    s2a_scl.isin([4, 5, 6])
    & s2b_scl.isin([4, 5, 6])
    & (wc == 40)
    & rel_diff_s2.notnull()
    & np.isfinite(s2a_b04)
    & np.isfinite(s2b_b04)
)

# Convert to NumPy once processing on the raster grid is complete.
valid_s2 = valid_s2.values
rel_diff = rel_diff_s2.values


from scipy.ndimage import gaussian_filter

# Smooth the relative reflectance difference
rel_diff_smooth = gaussian_filter(
    np.nan_to_num(rel_diff, nan=0.0),
    sigma=1.0,
    radius=2,  # 5 x 5 kernel
)

# Avoid smoothing across invalid pixels: normalize by a filtered validity mask
valid_rel_diff = np.isfinite(rel_diff)

weights = gaussian_filter(
    valid_rel_diff.astype(float),
    sigma=1.0,
    radius=2,
)

rel_diff_smooth = np.divide(
    rel_diff_smooth,
    weights,
    out=np.full_like(rel_diff_smooth, np.nan, dtype=float),
    where=weights > 1e-6,
)

rel_diff = rel_diff_smooth

# Release rasters no longer needed.
for da in (
    s2a_b04, s2a_b08, s2b_b04,
    s2a_scl, s2b_scl, wc, rel_diff_s2,
):
    da.close()

del s2a_b04, s2a_b08, s2b_b04
del s2a_scl, s2b_scl, wc, rel_diff_s2


# ---------------------------------------------------------------------
# EnMAP: process one paired acquisition at a time
# ---------------------------------------------------------------------

ncai_files = {
    f.parent: f
    for f in ENMAP_MASK_DIR.glob("SOIL_MASK_ncai_narrow.tif")
}

ndvi_files = {
    f.parent: f
    for f in ENMAP_MASK_DIR.glob("SOIL_MASK_ndvi_narrow.tif")
}

paired_dirs = sorted(set(ncai_files) & set(ndvi_files))

if not paired_dirs:
    raise FileNotFoundError("No paired EnMAP NCAI/NDVI masks found.")

# These arrays are the only EnMAP results retained in memory.
enmap_valid = np.zeros(valid_s2.shape, dtype=bool)
enmap_soil = np.zeros(valid_s2.shape, dtype=bool)
enmap_non_soil = np.zeros(valid_s2.shape, dtype=bool)

# Reference grid metadata.
ref = get_l2a_band(s2a_scene, "B04", 10)

for i, folder in enumerate(paired_dirs, start=1):
    ncai = open_clipped(ncai_files[folder], polygon)
    ndvi = open_clipped(ndvi_files[folder], polygon)

    ncai = align_to_reference(ncai, ref)
    ndvi = align_to_reference(ndvi, ref)

    ncai_values = ncai.values
    ndvi_values = ndvi.values

    valid = np.isfinite(ncai_values) & np.isfinite(ndvi_values)

    # Soil: NCAI < threshold and NDVI < threshold
    soil = (
        (ncai_values < NCAI_THRESHOLD)
        & (ndvi_values < NDVI_THRESHOLD)
    )

    # Non-soil: NCAI > threshold and NDVI < threshold
    non_soil = (
        (ncai_values > NCAI_THRESHOLD_2)
        & (ndvi_values < NDVI_THRESHOLD)
    )

    enmap_valid |= valid
    enmap_soil |= valid & soil
    enmap_non_soil |= valid & non_soil

    ncai.close()
    ndvi.close()

    del ncai_values, ndvi_values, valid, soil, non_soil

    if i % 10 == 0 or i == len(paired_dirs):
        print(f"Processed {i}/{len(paired_dirs)} EnMAP pairs")

ref.close()
del ref

# ---------------------------------------------------------------------
# Define eligible soil and non-soil pixels
# ---------------------------------------------------------------------

eligible_soil = valid_s2 & enmap_soil
eligible_non_soil = valid_s2 & enmap_non_soil

# Exclude pixels classified as both soil and non-soil
ambiguous = enmap_soil & enmap_non_soil
eligible_soil &= ~ambiguous
eligible_non_soil &= ~ambiguous

print(f"Eligible soil pixels:     {eligible_soil.sum():,}")
print(f"Eligible non-soil pixels: {eligible_non_soil.sum():,}")
print(f"Ambiguous pixels excluded: {ambiguous.sum():,}")


# ---------------------------------------------------------------------
# Raster diagnostics
# ---------------------------------------------------------------------

def report_mask(name, mask):
    mask = np.asarray(mask, dtype=bool)
    print(
        f"{name:30s}: {mask.sum():>12,} / {mask.size:,} "
        f"({100 * mask.mean():.4f}%)"
    )


print("\n--- Raster diagnostics ---")
print("S2 grid shape:", valid_s2.shape)
print("Relative difference shape:", rel_diff.shape)
print("EnMAP validity shape:", enmap_valid.shape)
print("EnMAP soil shape:", enmap_soil.shape)
print("EnMAP non-soil shape:", enmap_non_soil.shape)

report_mask("S2 valid", valid_s2)
report_mask("S2 finite relative diff", np.isfinite(rel_diff))
report_mask("EnMAP valid", enmap_valid)
report_mask("Eligible soil", eligible_soil)
report_mask("Eligible non-soil", eligible_non_soil)

# Sample from the union of the two mutually exclusive classes
eligible = eligible_soil | eligible_non_soil
report_mask("Final eligible", eligible)


# ---------------------------------------------------------------------
# Sample eligible pixels
# ---------------------------------------------------------------------

rows, cols = np.where(eligible)

if len(rows) == 0:
    raise ValueError("No pixels satisfy all sampling criteria.")

rng = np.random.default_rng(RANDOM_SEED)
n = min(N_SAMPLES, len(rows))

idx = rng.choice(len(rows), size=n, replace=False)
rows, cols = rows[idx], cols[idx]

# Class label: 1 = soil, 0 = non-soil
soil_labels = eligible_soil[rows, cols].astype(int)

df = pd.DataFrame({
    "row": rows,
    "col": cols,
    "rel_diff_s2": rel_diff[rows, cols],
    "enmap_soil": soil_labels,
})

df["enmap_class"] = df["enmap_soil"].map({
    0: "Non-soil",
    1: "Soil",
})

print(f"\nEligible S2 pixels: {valid_s2.sum():,}")
print(f"Eligible pixels for analysis: {eligible.sum():,}")
print(f"Sample size: {len(df):,}")
print(df["enmap_class"].value_counts())


# ---------------------------------------------------------------------
# Correlation test
# ---------------------------------------------------------------------

x = df["enmap_soil"].to_numpy()
y = df["rel_diff_s2"].to_numpy()

if len(np.unique(x)) < 2:
    raise ValueError("Both EnMAP classes are needed for correlation.")

r, p = pointbiserialr(x, y)

print(f"\nPoint-biserial correlation: r = {r:.4f}")
print(f"P-value: {p:.4g}")
print(f"N = {len(df)}")


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

jitter = rng.normal(0, 0.035, size=len(df))

axes[0].scatter(
    df["rel_diff_s2"],
    df["enmap_soil"] + jitter,
    s=18,
    alpha=0.45,
)

axes[0].set_yticks([0, 1])
axes[0].set_yticklabels(["Non-soil", "Soil"])
axes[0].set_xlabel("(S2A B04 - S2B B04) / S2B B04")
axes[0].set_ylabel("EnMAP classification")
axes[0].set_title(f"Point-biserial r = {r:.3f}, p = {p:.3g}")
axes[0].grid(alpha=0.25)

axes[1].boxplot(
    [
        df.loc[df.enmap_soil == 0, "rel_diff_s2"],
        df.loc[df.enmap_soil == 1, "rel_diff_s2"],
    ],
    tick_labels=["Non-soil", "Soil"],
    showfliers=False,
)

axes[1].set_ylabel("(S2A B04 - S2B B04) / S2B B04")
axes[1].set_title("S2 relative reflectance difference")
axes[1].grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.savefig("/dss/dsshome1/04/di35qiw/code_phd_paul/results/f_stereo/32UPU_test_1.png")


#python3 -m f_BRDF_measurement.stereo_npv_detection
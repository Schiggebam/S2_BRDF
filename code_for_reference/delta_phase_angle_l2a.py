from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from f_BRDF_measurement.main_worldwide import get_mdt_msixml_L2A
from shared.helper_S2_L1C import get_angle_grid_from_s2id


# 32UPU L2A
dir_base_l2a = Path("/dss/dsstbyfs01/pn56su/pn56su-dss-0008/Sentinel-2-Col-1/L2A/tiles/32/U/PU/2025/03")
s2a_safe = "S2A_MSIL2A_20250318T101751_N0511_R065_T32UPU_20250318T160400.SAFE"
s2b_safe = "S2B_MSIL2A_20250318T100639_N0511_R022_T32UPU_20250318T141314.SAFE"

fn_mtd_s2a = get_mdt_msixml_L2A(s2a_safe, sensor="S2A")
fn_mtd_s2b = get_mdt_msixml_L2A(s2b_safe, sensor="S2B")     # returns xml

print(fn_mtd_s2b)

s2a_id = s2a_safe.removesuffix(".SAFE")
s2b_id = s2b_safe.removesuffix(".SAFE")

band_id = "B04"
band_number = int(band_id[1:])

angles_s2a = get_angle_grid_from_s2id(s2a_id, fn_mtd_s2a, band_number)  
angles_s2b = get_angle_grid_from_s2id(s2b_id, fn_mtd_s2b, band_number, flip_grid=False)




# ============================================================
# CALCULATE PHASE ANGLE
# ============================================================

def calculate_phase_angle(ds):
    """
    Calculate the phase angle between the directions from the
    target to the Sun and from the target to the sensor.

    Phase angle:
        0 degrees   = backscatter geometry
        90 degrees  = perpendicular geometry
        180 degrees = forward-scatter geometry

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing SZA, SAA, VZA and VAA in degrees.

    Returns
    -------
    xarray.DataArray
        Phase angle in degrees.
    """

    # Convert to radians
    sza = np.deg2rad(ds["SZA"])
    saa = np.deg2rad(ds["SAA"])
    vza = np.deg2rad(ds["VZA"])
    vaa = np.deg2rad(ds["VAA"])

    # Relative azimuth difference
    delta_azimuth = saa - vaa

    # Cosine of phase angle
    cos_phase = (
        np.cos(sza) * np.cos(vza)
        + np.sin(sza) * np.sin(vza)
        * np.cos(delta_azimuth)
    )

    # Avoid numerical errors outside [-1, 1]
    cos_phase = cos_phase.clip(min=-1.0, max=1.0)

    phase_angle = np.rad2deg(np.arccos(cos_phase))

    phase_angle.name = "PHASE"
    phase_angle.attrs["units"] = "degrees"
    phase_angle.attrs["description"] = (
        "Phase angle: 0=backscatter, 180=forward scatter"
    )

    return phase_angle


phase_s2a = calculate_phase_angle(angles_s2a)
phase_s2b = calculate_phase_angle(angles_s2b)



# ============================================================
# ALIGN GRIDS
# ============================================================

print("S2A grid:", phase_s2a.shape)
print("S2B grid:", phase_s2b.shape)

# Exact alignment is preferable. This checks whether the
# coordinates match rather than silently interpolating them.
try:
    phase_s2a, phase_s2b = xr.align(
        phase_s2a,
        phase_s2b,
        join="exact",
    )
except ValueError as exc:
    raise ValueError(
        "The angle grids do not have identical coordinates. "
        "Verify their CRS, spatial transforms and orientation, "
        "then reproject/resample one grid to the other before "
        "calculating pixelwise differences."
    ) from exc


# ============================================================
# DIFFERENCE: S2A - S2B
# ============================================================

# Positive difference means S2B has a smaller phase angle.

phase_s2a, phase_s2b = xr.align(
    phase_s2a,
    phase_s2b,
    join="exact",
)

# A pixel belongs to the overlap if both phase angles are valid.
overlap = (
    np.isfinite(phase_s2a)
    & np.isfinite(phase_s2b)
)


delta_phase = (phase_s2a - phase_s2b).where(overlap)

# Extract only overlapping pixels.
values = delta_phase.values
values = values[np.isfinite(values)]

print("\n" + "=" * 60)
print("PHASE ANGLE DIFFERENCE: S2A - S2B")
print("COMMON SWATH ONLY")
print("=" * 60)

print(f"Overlapping pixels: {values.size:,}")

if values.size > 0:
    print(f"Mean:         {np.mean(values):.3f} degrees")
    print(f"Median:       {np.median(values):.3f} degrees")
    print(f"Std:          {np.std(values):.3f} degrees")
    print(f"5th percentile:  {np.percentile(values, 5):.3f} degrees")
    print(f"95th percentile: {np.percentile(values, 95):.3f} degrees")

    print(
        "\nPixels where S2B is more backscatter-oriented: "
        f"{np.mean(values > 0) * 100:.2f}%"
    )

    print(
        "Pixels where S2A is more backscatter-oriented: "
        f"{np.mean(values < 0) * 100:.2f}%"
    )

    print(
        "Pixels with approximately equal phase angles "
        f"(|difference| < 0.1 degrees): "
        f"{np.mean(np.abs(values) < 0.1) * 100:.2f}%"
    )
else:
    print("No overlapping valid pixels found.")


# ============================================================
# PLOTS
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

phase_s2a.plot(
    ax=axes[0],
    cmap="viridis",
    vmin=0,
    vmax=180,
    cbar_kwargs={"label": "Phase angle (degrees)"},
)
axes[0].set_title("S2A phase angle")

phase_s2b.plot(
    ax=axes[1],
    cmap="viridis",
    vmin=0,
    vmax=180,
    cbar_kwargs={"label": "Phase angle (degrees)"},
)
axes[1].set_title("S2B phase angle")

limit = np.nanpercentile(np.abs(values), 98)

delta_phase.plot(
    ax=axes[2],
    cmap="RdBu_r",
    vmin=-limit,
    vmax=limit,
    cbar_kwargs={"label": "S2A - S2B (degrees)"},
)
axes[2].set_title("Phase-angle difference")

plt.tight_layout()
plt.savefig("/dss/dsshome1/04/di35qiw/code_phd_paul/results/f_stereo/delta_phase_angle_plt.png")


# ============================================================
# HISTOGRAM
# ============================================================

plt.figure(figsize=(9, 5))

plt.hist(values, bins=100)

plt.axvline(
    0,
    linestyle="--",
    linewidth=2,
    label="Equal phase angles",
)

plt.axvline(
    np.median(values),
    linestyle="-",
    linewidth=2,
    label=f"Median = {np.median(values):.2f}°",
)

plt.xlabel("Phase angle difference: S2A - S2B (degrees)")
plt.ylabel("Number of pixels")
plt.title("Sentinel-2 phase-angle difference")
plt.legend()
plt.tight_layout()
plt.savefig("/dss/dsshome1/04/di35qiw/code_phd_paul/results/f_stereo/delta_phase_angle_hist.png")

# python3 -m f_BRDF_measurement.delta_phase_angle_l2a
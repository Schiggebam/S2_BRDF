
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
from pyproj import Transformer
import geopandas as gpd
from shapely.geometry import Point
from scipy.ndimage import uniform_filter
from scipy.spatial import ConvexHull

from datetime import datetime

from Py6S import AtmosProfile, AeroProfile

# from f_BRDF_measurement.find_S2AC_pairs import get_angle_grid_from_s2id
from f_BRDF_measurement.S2_L1C_atmos_corrected import L1C_AtmosCorr
from f_BRDF_measurement.S2_L1C_firstOrderCorr import L1C_IlluminationCorrection
from f_BRDF_measurement.vis_ross_li import *
from f_BRDF_measurement.main import prepare_lc


from shared.helper_S2_L1C import get_angle_grid_from_s2id

from shared.helper_satellite import CWL_S2


def angular_difference(a, b):
    return (b - a + 180) % 360 - 180

def viewing_direction_difference(
    vza_a,
    vaa_a,
    vza_b,
    vaa_b
):
    """
    Angular separation between two viewing directions [deg].
    """

    vza_a = np.deg2rad(vza_a)
    vaa_a = np.deg2rad(vaa_a)

    vza_b = np.deg2rad(vza_b)
    vaa_b = np.deg2rad(vaa_b)

    xa = np.sin(vza_a) * np.sin(vaa_a)
    ya = np.sin(vza_a) * np.cos(vaa_a)
    za = np.cos(vza_a)

    xb = np.sin(vza_b) * np.sin(vaa_b)
    yb = np.sin(vza_b) * np.cos(vaa_b)
    zb = np.cos(vza_b)

    dot = xa * xb + ya * yb + za * zb
    dot = np.clip(dot, -1.0, 1.0)

    return np.rad2deg(np.arccos(dot))

def plot_theoretical_sampling_space(
    angles_s2a,
    angles_s2b,
    fn_out=None,
    verbose=True
):

    # =====================================================
    # LOAD
    # =====================================================

    sza_a, sza_b = angles_s2a.SZA.values, angles_s2b.SZA.values
    vza_a, vza_b = angles_s2a.VZA.values, angles_s2b.VZA.values
    saa_a, saa_b = angles_s2a.SAA.values, angles_s2b.SAA.values
    vaa_a, vaa_b = angles_s2a.VAA.values, angles_s2b.VAA.values

    valid = (
        np.isfinite(sza_a) &
        np.isfinite(sza_b) &
        np.isfinite(vza_a) &
        np.isfinite(vza_b) &
        np.isfinite(saa_a) &
        np.isfinite(saa_b) &
        np.isfinite(vaa_a) &
        np.isfinite(vaa_b)
    )

    dlook = viewing_direction_difference(
        vza_a,
        vaa_a,
        vza_b,
        vaa_b
    )

    print("viewing_direction-difference")
    print(
        np.nanpercentile(
            dlook[valid],
            [0, 5, 50, 95, 100]
        )
    )

    # =====================================================
    # GEOMETRY
    # =====================================================

    raa_a = relative_azimuth(saa_a, vaa_a)
    raa_b = relative_azimuth(saa_b, vaa_b)

    dsza = angular_difference(sza_b, sza_a)
    dvza = angular_difference(vza_b, vza_a)
    dvaa = angular_difference(vaa_b, vaa_a)
    draa = angular_difference(raa_b, raa_a)

    # =====================================================
    # KERNELS
    # =====================================================

    kvol_a = ross_thick(sza_a, vza_a, raa_a)
    kvol_b = ross_thick(sza_b, vza_b, raa_b)

    kgeo_a = li_sparse(sza_a, vza_a, raa_a)
    kgeo_b = li_sparse(sza_b, vza_b, raa_b)

    dkvol = kvol_b - kvol_a
    dkgeo = kgeo_b - kgeo_a

    # =====================================================
    # DIAGNOSTICS
    # =====================================================

    if verbose:

        def p(x):
            return np.nanpercentile(x[valid], [5, 50, 95])

        print("\nGeometry diagnostics")
        print("--------------------")
        print(f"ΔSZA  [5,50,95] = {p(dsza)}")
        print(f"ΔVZA  [5,50,95] = {p(dvza)}")
        print(f"ΔVAA  [5,50,95] = {p(dvaa)}")
        print(f"ΔRAA  [5,50,95] = {p(draa)}")

        med_dvaa = np.nanmedian(np.abs(dvaa[valid]))
        med_draa = np.nanmedian(np.abs(draa[valid]))

        if med_dvaa > 10:
            print("WARNING: unusually large VAA differences.")

        if med_draa > 15:
            print("WARNING: unusually large RAA differences.")

        if np.nanmax(np.abs(draa[valid])) > 90:
            print("WARNING: extreme RAA perturbations detected.")

    # =====================================================
    # FLATTEN
    # =====================================================

    dsza = dsza[valid].ravel()
    dlook = dlook[valid].ravel()

    draa = draa[valid].ravel()
    dvza = dvza[valid].ravel()

    dkgeo = dkgeo[valid].ravel()
    dkvol = dkvol[valid].ravel()

    vza_a_plot = vza_a[valid].ravel()
    vaa_a_plot = vaa_a[valid].ravel()

    vza_b_plot = vza_b[valid].ravel()
    vaa_b_plot = vaa_b[valid].ravel()

    # =====================================================
    # OPTIONAL DOWNSAMPLE
    # =====================================================

    nmax = 50000

    if len(dsza) > nmax:

        idx = np.random.choice(
            len(dsza),
            nmax,
            replace=False
        )

        dsza = dsza[idx]
        dlook = dlook[idx]

        dkgeo = dkgeo[idx]
        dkvol = dkvol[idx]

        vza_a_plot = vza_a_plot[idx]
        vaa_a_plot = vaa_a_plot[idx]

        vza_b_plot = vza_b_plot[idx]
        vaa_b_plot = vaa_b_plot[idx]

    # =====================================================
    # PLOT
    # =====================================================

    fig = plt.figure(
        figsize=(18, 6),
        constrained_layout=True
    )

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133, projection="polar")

    # -----------------------------------------------------
    # Angular perturbation space
    # -----------------------------------------------------
    pts = np.column_stack([dsza, dlook])

    ax1.scatter(
        dsza,
        dlook,
        s=5,
        alpha=0.5
    )

    if len(pts) > 3:

        hull = ConvexHull(pts)

        for simplex in hull.simplices:
            ax1.plot(
                pts[simplex, 0],
                pts[simplex, 1],
                "k--",
                lw=1
            )

    ax1.set_xlabel("ΔSZA [deg]")
    ax1.set_ylabel("Viewing Direction Separation Δψ [deg]")

    ax1.set_title(
        "Angular Perturbation Space"
    )

    # -----------------------------------------------------
    # Kernel perturbation space
    # -----------------------------------------------------

    pts = np.column_stack([dkgeo, dkvol])

    mask = (
        np.isfinite(dkgeo) &
        np.isfinite(dkvol)
    )

    pts = pts[mask]

    ax2.scatter(
        dkgeo,
        dkvol,
        s=5,
        alpha=0.5
    )

    if len(pts) > 3:

        hull = ConvexHull(pts)

        for simplex in hull.simplices:
            ax2.plot(
                pts[simplex, 0],
                pts[simplex, 1],
                "k--",
                lw=1
            )

    ax2.set_xlabel("ΔKgeo")
    ax2.set_ylabel("ΔKvol")

    ax2.set_title(
        "Kernel Perturbation Space"
    )

    theta_a = np.deg2rad(vaa_a_plot)
    theta_b = np.deg2rad(vaa_b_plot)

    r_a = vza_a_plot
    r_b = vza_b_plot

    # ax3.scatter(
    #     theta_a,
    #     r_a,
    #     s=1,
    #     alpha=0.1,
    #     label="S2A"
    # )
# 
    # ax3.scatter(
    #     theta_b,
    #     r_b,
    #     s=1,
    #     alpha=0.1,
    #     label="S2B"
    # )
    sc = ax3.scatter(
        theta_a,
        vza_a_plot,
        c=dlook,
        s=3,
        cmap="viridis"
    )

    plt.colorbar(sc, ax=ax3, label="Δψ [deg]")

    ax3.set_theta_zero_location("N")
    ax3.set_theta_direction(-1)

    ax3.set_rlabel_position(90)

    ax3.set_title(
        "Viewing Geometry Hemisphere"
    )

    ax3.legend(
        loc="upper right"
    )

    plt.suptitle(
        "Theoretical Sentinel-2A/B BRDF Sampling Space"
    )

    if fn_out:
        plt.savefig(fn_out, dpi=300)

    plt.close()

def plot_full_geometry_fields(angles_s2a, angles_s2b, fn_out):

    import matplotlib.pyplot as plt
    import numpy as np

    # =====================================================
    # LOAD ANGLES
    # =====================================================

    sza_a = angles_s2a.SZA.values
    sza_b = angles_s2b.SZA.values

    vza_a = angles_s2a.VZA.values
    vza_b = angles_s2b.VZA.values

    saa_a = angles_s2a.SAA.values
    saa_b = angles_s2b.SAA.values

    vaa_a = angles_s2a.VAA.values
    vaa_b = angles_s2b.VAA.values

    # =====================================================
    # VALID OVERLAP
    # =====================================================

    valid = (
        ~np.isnan(sza_a) &
        ~np.isnan(sza_b) &
        ~np.isnan(vza_a) &
        ~np.isnan(vza_b)
    )

    # =====================================================
    # RELATIVE AZIMUTH
    # =====================================================

    raa_a = relative_azimuth(
        saa_a,
        vaa_a
    )

    raa_b = relative_azimuth(
        saa_b,
        vaa_b
    )

    # =====================================================
    # KERNELS
    # =====================================================

    kvol_a = ross_thick(
        sza_a,
        vza_a,
        raa_a
    )

    kvol_b = ross_thick(
        sza_b,
        vza_b,
        raa_b
    )

    kgeo_a = li_sparse(
        sza_a,
        vza_a,
        raa_a
    )

    kgeo_b = li_sparse(
        sza_b,
        vza_b,
        raa_b
    )

    # =====================================================
    # DIFFERENCES
    # =====================================================

    fields = {
        "ΔSZA [deg]": sza_b - sza_a,
        "ΔVZA [deg]": vza_b - vza_a,
        "ΔRAA [deg]": raa_b - raa_a,
        "ΔKvol": kvol_b - kvol_a,
        "ΔKgeo": kgeo_b - kgeo_a,
    }

    # =====================================================
    # APPLY VALID MASK
    # =====================================================

    for key in fields:
        fields[key] = np.where(
            valid,
            fields[key],
            np.nan
        )

    # =====================================================
    # PLOTTING CONFIG
    # =====================================================

    plot_cfg = {
        "ΔSZA [deg]": "viridis",
        "ΔVZA [deg]": "cividis",
        "ΔRAA [deg]": "plasma",
        "ΔKvol": "magma",
        "ΔKgeo": "inferno",
    }

    # =====================================================
    # PLOTTING
    # =====================================================

    fig, axes = plt.subplots(
        1,
        len(fields),
        figsize=(24, 5),
        constrained_layout=True
    )

    for ax, (title, arr) in zip(axes, fields.items()):

        # ---------------------------------------------
        # robust limits
        # ---------------------------------------------

        vmin = np.nanpercentile(
            arr,
            2
        )

        vmax = np.nanpercentile(
            arr,
            98
        )

        # ---------------------------------------------
        # colormap
        # ---------------------------------------------

        cmap = plot_cfg[title]

        im = ax.imshow(
            arr,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )

        ax.set_title(title)

        ax.set_xticks([])
        ax.set_yticks([])

        plt.colorbar(
            im,
            ax=ax,
            fraction=0.046
        )

    plt.suptitle(
        "Sentinel-2 Differential Geometry Manifold",
        fontsize=18
    )

    plt.savefig(fn_out)

# =========================================================
# CONFIG
# =========================================================

ATMOS_DEFAULT = {
    "atmos_profile": AtmosProfile.MidlatitudeSummer,
    "aero_profile": AeroProfile.Continental,
    "aot550": 0.1,
    "altitude_km": 0.5,
}

ATMOS_LIBYA = {
    "atmos_profile": AtmosProfile.Tropical,
    "aero_profile": AeroProfile.Desert,
    "aot550": 0.03,
    "altitude_km": 0.3,
}

ATMOS_GREENLAND = {
    "atmos_profile": AtmosProfile.SubarcticWinter,
    "aero_profile": AeroProfile.Maritime,
    "aot550": 0.03,
    "altitude_km": 0.1,
}

BANDS = ["B04", "B08", "B12"]
BANDS = ["B12"]

N_SAMPLES = 200
PATCH_SIZE = 1

SEED = 42

dir_base_l1c = Path("/dss/dsstbyfs02/pn49cu/pn49cu-dss-0021/phd_paul/datasets/S2_L1C")

d_scenes = {
    "Libya": {
        "20250725": {
            "S2A": "S2A_MSIL1C_20250725T091041_N0511_R050_T34RFS_20250725T101000.SAFE",
            "S2B": "S2B_MSIL1C_20250725T085559_N0511_R007_T34RFS_20250725T105553.SAFE",
        },
        # "20251102": {
        #     "S2A": "S2A_MSIL1C_20251102T091151_N0511_R050_T34RFS_20251102T094902.SAFE",
        #     "S2B": "S2B_MSIL1C_20251102T090029_N0511_R007_T34RFS_20251102T093737.SAFE",
        # },
        "bbox_epsg4326": None,
        "atmos_profile": ATMOS_LIBYA,
        #"bbox_epsg4326": {"N": 0, "W": 0, "S": 0, "E": 0}
        "skip": True,

    },
    "32UPU": {
        "20250318": {
            "S2A": "S2A_MSIL1C_20250318T101751_N0511_R065_T32UPU_20250318T140212.SAFE",
            "S2B": "S2B_MSIL1C_20250318T100639_N0511_R022_T32UPU_20250318T134421.SAFE",
        },
        "bbox_epsg4326": None,
        "atmos_profile": ATMOS_DEFAULT,
        "skip": False,
        "wc_class": 40,
        "ndvi": "/dss/dsstbyfs02/pn49cu/pn49cu-dss-0021/phd_paul/datasets/S2_L1C/masks/S2A_MSIL1C_20250318T101751_N0511_R065_T32UPU_20250318T140212__NDVI.tif",
        "ndvi_max": 0.15
    },
    "32VML_Greenland": {
        "20250912": {
            "S2A": "S2A_MSIL1C_20250711T142801_N0511_R139_T23VNL_20250711T180513.SAFE",
            "S2B": "S2B_MSIL1C_20250711T141739_N0511_R096_T23VNL_20250711T161424.SAFE",
        },
        "bbox_epsg4326": None,
        "atmos_profile": ATMOS_GREENLAND,
        "skip": True,
    },
    "40VEP_Grasslands": {
        "20250912": {
            "S2A": "",
            "S2B": "",
        },
        "bbox_epsg4326": None,
        "atmos_profile": ATMOS_DEFAULT,
        "skip": True,
    },
}

# =========================================================
# HELPERS
# =========================================================
def build_worldcover_class_mask(wc, class_id, kernel_size=5):

    # =====================================================
    # TARGET CLASS
    # =====================================================

    class_mask = (
        wc.values == class_id
    ).astype(np.float32)

    # =====================================================
    # LOCAL HOMOGENEITY
    # =====================================================

    local_fraction = uniform_filter(
        class_mask,
        size=kernel_size,
        mode="constant",
        cval=0
    )

    # =====================================================
    # FULLY HOMOGENEOUS WINDOW
    # =====================================================

    homogeneous_mask = (
        local_fraction == 1.0
    )

    return homogeneous_mask

def get_mtd_msixml(scene_id, fn_scene: Path | str):
    return next(Path(fn_scene).glob(f"{scene_id}*/GRANMTD_MSIL1C.xml"))

def get_mdt_msixml_L2A(scene_id, sensor="S2A", d_base_s2=Path("/dss/dsstbyfs01/pn56su/pn56su-dss-0008/Sentinel-2-Col-1/L2A/tiles/"), date=None, tile_id=None):
    if date is None:
        date = datetime.strptime(scene_id.split('_')[2], "%Y%m%dT%H%M%S")

    if tile_id is None:
        tile_id = scene_id.split('_')[5][1:]

    fn_xml = d_base_s2 / tile_id[:2] / tile_id[2] / tile_id[3:]
    fn_xml = fn_xml / str(date.year) / str(date.month).zfill(2)
    # fn_xml = fn_xml / (scene_id + ".SAFE")

    try:
        fns_xml = list(fn_xml.glob(f"{sensor}_MSIL2A_{str(date.year)}{str(date.month).zfill(2)}{str(date.day).zfill(2)}*/GRANULE/*/MTD_TL.xml"))
        if len(fns_xml) == 1:
            return fns_xml[0]
        else:
            raise ValueError(f"L2A xml not found or too many {len(fns_xml)}.")
    except StopIteration:
        return None

def sample_valid_pixels(mask, n_samples, seed=42):

    rng = np.random.default_rng(seed)

    py, px = np.where(mask)

    if len(py) == 0:
        return []

    idx = rng.choice(len(py), size=min(n_samples, len(py)), replace=False)

    return list(zip(py[idx], px[idx]))


def build_bbox_mask(da, bbox_epsg4326):

    if bbox_epsg4326 is None:
        return np.ones((da.sizes["y"], da.sizes["x"]), dtype=bool)

    transformer = Transformer.from_crs("EPSG:4326", da.rio.crs, always_xy=True)

    xmin, ymin = transformer.transform(
        bbox_epsg4326["W"],
        bbox_epsg4326["S"]
    )

    xmax, ymax = transformer.transform(
        bbox_epsg4326["E"],
        bbox_epsg4326["N"]
    )

    xx, yy = np.meshgrid(da.x.values, da.y.values)

    return (
        (xx >= xmin) &
        (xx <= xmax) &
        (yy >= ymin) &
        (yy <= ymax)
    )


def compute_overlap_mask(da_a, da_b):

    arr_a = da_a.values
    arr_b = da_b.values

    return (
        np.isfinite(arr_a) &
        np.isfinite(arr_b) &
        (arr_a > 0) &
        (arr_b > 0)
    )


def compute_geometry(angles, x_proj, y_proj):

    return {
        "sza": float(angles.SZA.sel(x=x_proj, y=y_proj, method="nearest")),
        "saa": float(angles.SAA.sel(x=x_proj, y=y_proj, method="nearest")),
        "vza": float(angles.VZA.sel(x=x_proj, y=y_proj, method="nearest")),
        "vaa": float(angles.VAA.sel(x=x_proj, y=y_proj, method="nearest")),
    }


def compute_kernel_features(geom):

    relaz = relative_azimuth(geom["saa"], geom["vaa"])

    return {
        "relaz": relaz,
        "kvol": ross_thick(geom["sza"], geom["vza"], relaz),
        "kgeo": li_sparse(geom["sza"], geom["vza"], relaz),
    }



# =========================================================
# MAIN
# =========================================================

def main():

    

    for scene_name, scene_cfg in d_scenes.items():
        
        if scene_cfg.get("skip", False):
            continue

        print("\n================================================")
        print("SCENE:", scene_name)
        print("================================================")
        rows = []
        bbox_epsg4326 = scene_cfg.get("bbox_epsg4326")
        atmos_profile = scene_cfg.get("atmos_profile")

        for date, pair in scene_cfg.items():

            if not date.startswith("20"):
                continue

            print("\nDATE:", date)

            s2a_safe = pair["S2A"]
            s2b_safe = pair["S2B"]

            s2a_id = s2a_safe.replace(".SAFE", "")
            s2b_id = s2b_safe.replace(".SAFE", "")

            tile_id = s2a_safe.split("_")[5][1:]

            # =====================================================
            # LOAD ANGLES ONCE
            # =====================================================

            fn_mtd_s2a_l2a = get_mdt_msixml_L2A(s2a_safe, sensor="S2A")
            fn_mtd_s2b_l2a = get_mdt_msixml_L2A(s2b_safe, sensor="S2B")


            # =====================================================
            # BUILD MASTER SAMPLE LOCATIONS
            # =====================================================

            ref_band = BANDS[-1]        # use band with coarsest

            ref_band_fn = list((dir_base_l1c / s2a_safe).glob(f"GRANULE/*/IMG_DATA/*_{ref_band}.jp2"))[0]
            fn_wc_local = prepare_lc(
                fn_lc_vrt="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0021/worldsoils/processing/dias_mundi/ws-obs/datalake/landcover/ESA_WorldCover_2020/Europe_extended.vrt",
                fn_mincomp=ref_band_fn,
                aligned_landcover=f"/dss/dsstbyfs02/scratch/04/di35qiw/s2_brdf/landcover_tmp/worldcover_{tile_id}_{ref_band}_{scene_name}.tif",
                res="20m"
            )


            ref_atmos = {
                **atmos_profile,
                "wavelength": CWL_S2[ref_band] / 1000.0
            }

            l1c_ref = L1C_IlluminationCorrection(
                fn_s2a_safe=dir_base_l1c / s2a_safe,
                fn_s2b_safe=dir_base_l1c / s2b_safe,
                band_id=ref_band,
                atmos_settings=ref_atmos
            )

            da_a_ref = l1c_ref.da_s2a
            da_b_ref = l1c_ref.da_s2b

            wc = rioxarray.open_rasterio(fn_wc_local, masked=True).squeeze()

            ndvi_mask = None

            if scene_cfg.get("ndvi") is not None:

                ndvi = rioxarray.open_rasterio(
                    scene_cfg["ndvi"],
                    masked=True
                ).squeeze()
                if (ndvi.shape[0] == da_a_ref.shape[0] * 2 and ndvi.shape[1] == da_a_ref.shape[1] * 2):

                    print("Downsampling NDVI from 10m -> 20m")

                    arr = ndvi.values.astype(np.float32)

                    arr = arr.reshape(da_a_ref.shape[0],  2, da_a_ref.shape[1], 2)

                    arr = np.nanmean(arr, axis=(1, 3) )

                    ndvi = xr.DataArray(arr, coords=da_a_ref.coords, dims=da_a_ref.dims )

                print("NDVI:", ndvi.shape, "REF:", da_a_ref.shape )
                assert ndvi.shape == da_a_ref.shape

            assert da_a_ref.shape == da_b_ref.shape
            assert da_a_ref.rio.transform() == da_b_ref.rio.transform()

            overlap_mask = compute_overlap_mask(
                da_a_ref,
                da_b_ref
            )

            bbox_mask = build_bbox_mask(
                da_a_ref,
                bbox_epsg4326
            )

            if ndvi_mask is None and scene_cfg.get("ndvi") is not None:
                ndvi_mask = np.ones(da_a_ref.shape, dtype=bool)

                if scene_cfg.get("ndvi_min") is not None:
                    ndvi_mask &= (ndvi.values >= scene_cfg["ndvi_min"])

                if scene_cfg.get("ndvi_max") is not None:
                    ndvi_mask &= (ndvi.values <= scene_cfg["ndvi_max"])

                ndvi_mask &= np.isfinite(ndvi.values)

            valid_mask = overlap_mask & bbox_mask
            if scene_cfg.get("wc_class") is not None:

                lc_mask = build_worldcover_class_mask(
                    wc,
                    class_id=scene_cfg["wc_class"]
                )

                valid_mask &= lc_mask
            if ndvi_mask is not None:
                valid_mask &= ndvi_mask

            pts = sample_valid_pixels(
                valid_mask,
                N_SAMPLES,
                seed=SEED
            )

            print("    valid samples:", len(pts))

            transformer_to_ll = Transformer.from_crs(
                da_a_ref.rio.crs,
                "EPSG:4326",
                always_xy=True
            )

            # =====================================================
            # GEOMETRY + LOCATION ONCE
            # =====================================================

            sample_info = []

            for sample_id, (py, px) in enumerate(pts):

                x_proj = float(da_a_ref.x.values[px])
                y_proj = float(da_a_ref.y.values[py])

                lon, lat = transformer_to_ll.transform(
                    x_proj,
                    y_proj
                )

                # wc_class = int(wc.values[py, px])
                # print(wc_class)

                sample_info.append({
                    "sample_id": sample_id,
                    "lat": lat,
                    "lon": lon,
                    "x_proj": x_proj,
                    "y_proj": y_proj,
                })

            # =====================================================
            # LOOP BANDS
            # =====================================================

            for band_id in BANDS:

                print("  BAND:", band_id)

                angles_s2a = get_angle_grid_from_s2id(s2a_id, fn_mtd_s2a_l2a, int(band_id[1:]))  
                angles_s2b = get_angle_grid_from_s2id(s2b_id, fn_mtd_s2b_l2a, int(band_id[1:]), flip_grid=True)

                if angles_s2a is None:
                    raise RuntimeError(f"Failed loading angles for {s2a_id}")

                if angles_s2b is None:
                    raise RuntimeError(f"Failed loading angles for {s2b_id}")

                # if band_id == "B12":
                #     plot_theoretical_sampling_space(angles_s2a, angles_s2b, f"/dss/dsshome1/04/di35qiw/code_phd_paul/results/f_worldwide/theoretical_sampling_space_{scene_name}_{band_id}.png")
                #     plot_full_geometry_fields(angles_s2a, angles_s2b, f"/dss/dsshome1/04/di35qiw/code_phd_paul/results/f_worldwide/full_geom_grid_{scene_name}_{band_id}.png")
                #     print("Plotting done")
                #     exit(0)

                atmos_settings = {
                    **atmos_profile,
                    "wavelength": CWL_S2[band_id] / 1000.0
                }

                l1c = L1C_IlluminationCorrection(
                    fn_s2a_safe=dir_base_l1c / s2a_safe,
                    fn_s2b_safe=dir_base_l1c / s2b_safe,
                    band_id=band_id,
                    atmos_settings=atmos_settings
                )

                for sample in sample_info:

                    sample_id = sample["sample_id"]

                    geom_a = compute_geometry(
                        angles_s2a,
                        sample["x_proj"],
                        sample["y_proj"]
                    )

                    geom_b = compute_geometry(
                        angles_s2b,
                        sample["x_proj"],
                        sample["y_proj"]
                    )

                    kernels_a = compute_kernel_features(geom_a)
                    kernels_b = compute_kernel_features(geom_b)

                    if sample_id % 10 == 0:
                        print("    sample:", sample_id)

                    lat = sample["lat"]
                    lon = sample["lon"]

                    patch_a, patch_b = l1c.get_patch(
                        lat=lat,
                        lon=lon,
                        size=PATCH_SIZE,
                        atmos_corr=True,
                        geometry_s2a=geom_a,
                        geometry_s2b=geom_b
                    )

                    if patch_a is None or patch_b is None:
                        print("ERROR - one of the 2 patches does not exists", patch_a, patch_b)
                        continue

                    refl_a = float(np.nanmean(patch_a))
                    refl_b = float(np.nanmean(patch_b))

                    mean_refl = 0.5 * (refl_a + refl_b)

                    if mean_refl < 0.01:
                        print("REF too low!", mean_refl)
                        continue

                    delta = (refl_b - refl_a) / mean_refl

                    rows.append({
                        "scene": scene_name,
                        "date": date,
                        "band": band_id,
                        "sample_id": sample_id,

                        "lat": lat,
                        "lon": lon,

                        "x_proj": sample["x_proj"],
                        "y_proj": sample["y_proj"],

                        "refl_s2a": refl_a,
                        "refl_s2b": refl_b,
                        "delta": delta,

                        "kvol_a": kernels_a["kvol"],
                        "kgeo_a": kernels_a["kgeo"],

                        "kvol_b": kernels_b["kvol"],
                        "kgeo_b": kernels_b["kgeo"],

                        "dkvol": kernels_b["kvol"] - kernels_a["kvol"],
                        "dkgeo": kernels_b["kgeo"] - kernels_a["kgeo"],

                        "sza_a": geom_a["sza"],
                        "vza_a": geom_a["vza"],

                        "sza_b": geom_b["sza"],
                        "vza_b": geom_b["vza"],

                        "vaa_a": geom_a["vaa"],
                        "vaa_b": geom_b["vaa"],

                        "saa_a": geom_a["saa"],
                        "saa_b": geom_b["saa"],
                    })

        df = pd.DataFrame(rows)
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(
                df["lon"],
                df["lat"]
            ),
            crs="EPSG:4326"
        )
        
        out_fn = f"results/f_worldwide/brdf_measurements_{scene_name}_wc{scene_cfg.get('wc_class')}"
        if scene_cfg.get('wc_class') == 40 and scene_cfg.get("ndvi") is not None:
            out_fn += f"_ndvi{scene_cfg.get('ndvi_max')}"
        out_fn_parquet = out_fn + ".parquet"
        out_fn_gpkg = out_fn + ".gpkg"

        # out_fn_parquet = f"results/f_worldwide/brdf_measurements_{scene_name}_wc{scene_cfg.get('wc_class')}.parquet"
        # out_fn_gpkg = f"results/f_worldwide/brdf_measurements_{scene_name}_wc{scene_cfg.get('wc_class')}.gpkg"
        gdf.to_parquet(out_fn_parquet)

        gdf.to_file(out_fn_gpkg, driver="GPKG")
        print("\nWritten:")
        print(out_fn_parquet)
        print(out_fn_gpkg)
    print("\nDONE")
    


if __name__ == "__main__":
    main()

# python3 -m f_BRDF_measurement.main_worldwide
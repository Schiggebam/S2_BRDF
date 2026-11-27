from pathlib import Path

import numpy as np
import rioxarray

from pyproj import Transformer
from Py6S import *


class L1C_IlluminationCorrection:

    def __init__(
            self,
            fn_s2a_safe: str | Path,
            fn_s2b_safe: str | Path,
            band_id: str,
            atmos_settings: dict | None = None):

        self.fn_s2a_safe = Path(fn_s2a_safe)
        self.fn_s2b_safe = Path(fn_s2b_safe)

        self.band_id = band_id

        self.fn_s2a_l1c = list(
            self.fn_s2a_safe.glob(
                f"GRANULE/*/IMG_DATA/*_{band_id}.jp2"
            )
        )[0]

        self.fn_s2b_l1c = list(
            self.fn_s2b_safe.glob(
                f"GRANULE/*/IMG_DATA/*_{band_id}.jp2"
            )
        )[0]

        # =====================================================
        # LOAD TOA REFLECTANCE
        # =====================================================

        self.da_s2a = (
            rioxarray.open_rasterio(self.fn_s2a_l1c)
            .squeeze()
            .astype("float32") / 10000.0
        )

        self.da_s2b = (
            rioxarray.open_rasterio(self.fn_s2b_l1c)
            .squeeze()
            .astype("float32") / 10000.0
        )

        self.crs = self.da_s2a.rio.crs

        self.transformer = Transformer.from_crs(
            "EPSG:4326",
            self.crs,
            always_xy=True
        )

        # =====================================================
        # ATMOS SETTINGS
        # =====================================================

        self.atmos_settings = atmos_settings or {
            "atmos_profile": AtmosProfile.MidlatitudeSummer,
            "aero_profile": AeroProfile.Continental,
            "aot550": 0.05,
            "altitude_km": 0.5,
            "wavelength": 0.665,
        }

    # =====================================================
    # FIRST-ORDER CORRECTION
    # =====================================================

    def _correct(
            self,
            patch,
            sza,
            saa,
            vza,
            vaa):

        # -------------------------------------------------
        # 1) ILLUMINATION NORMALIZATION
        # -------------------------------------------------

        mu0 = np.cos(np.deg2rad(sza))

        patch_corr = patch / mu0

        # -------------------------------------------------
        # 2) OPTIONAL SIMPLE ATMOSPHERIC TRANSMITTANCE
        # -------------------------------------------------

        s = SixS()

        s.geometry = Geometry.User()

        s.geometry.solar_z = sza
        s.geometry.solar_a = saa

        s.geometry.view_z = vza
        s.geometry.view_a = vaa

        s.atmos_profile = AtmosProfile.PredefinedType(
            self.atmos_settings["atmos_profile"]
        )

        s.aero_profile = AeroProfile.PredefinedType(
            self.atmos_settings["aero_profile"]
        )

        s.aot550 = self.atmos_settings["aot550"]

        s.altitudes.set_sensor_satellite_level()

        s.altitudes.set_target_custom_altitude(
            self.atmos_settings["altitude_km"]
        )

        s.wavelength = Wavelength(
            self.atmos_settings["wavelength"]
        )

        s.run()

        # -------------------------------------------------
        # TOTAL TRANSMITTANCE
        # -------------------------------------------------

        trans_down = (
            s.outputs.transmittance_total_scattering.total *
            s.outputs.transmittance_global_gas.total
        )

        trans_up = (
            s.outputs.transmittance_total_scattering.total *
            s.outputs.transmittance_global_gas.total
        )

        total_trans = trans_down * trans_up

        # -------------------------------------------------
        # APPLY ONLY MULTIPLICATIVE ATMOSPHERIC CORRECTION
        # -------------------------------------------------

        if total_trans > 1e-6:
            patch_corr = patch_corr / total_trans

        # -------------------------------------------------
        # NUMERICAL SAFETY
        # -------------------------------------------------

        patch_corr = np.clip(
            patch_corr,
            0,
            2
        )

        return patch_corr.astype("float32")

    # =====================================================
    # PATCH EXTRACTION
    # =====================================================

    def get_patch(
            self,
            lat: float,
            lon: float,
            size: int,
            atmos_corr: bool = False,
            geometry_s2a: dict | None = None,
            geometry_s2b: dict | None = None):

        if size % 2 == 0:
            raise ValueError("Patch size must be odd.")

        x_proj, y_proj = self.transformer.transform(
            lon,
            lat
        )

        def extract_patch(da):

            px = np.argmin(
                np.abs(da.x.values - x_proj)
            )

            py = np.argmin(
                np.abs(da.y.values - y_proj)
            )

            half = size // 2

            return da.isel(
                x=slice(px - half, px + half + 1),
                y=slice(py - half, py + half + 1)
            ).values

        patch_s2a = extract_patch(self.da_s2a)
        patch_s2b = extract_patch(self.da_s2b)
        # =====================================================
        # OPTIONAL CORRECTION
        # =====================================================

        if atmos_corr:

            if geometry_s2a is None or geometry_s2b is None:
                raise ValueError(
                    "Geometry required for correction."
                )

            patch_s2a = self._correct(
                patch_s2a,
                geometry_s2a["sza"],
                geometry_s2a["saa"],
                geometry_s2a["vza"],
                geometry_s2a["vaa"]
            )

            patch_s2b = self._correct(
                patch_s2b,
                geometry_s2b["sza"],
                geometry_s2b["saa"],
                geometry_s2b["vza"],
                geometry_s2b["vaa"]
            )
            
        return patch_s2a, patch_s2b
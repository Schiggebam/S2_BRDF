from datetime import datetime
from pathlib import Path

import numpy as np
import rioxarray
import xarray as xr
from pyproj import Transformer
from Py6S import *

from shared.helper_satellite import CWL_S2

class L1C_AtmosCorr:
    def __init__(self, fn_s2a_safe: str | Path, fn_s2b_safe: str | Path, band_id: str, atmos_settings: dict | None = None):
        self.fn_s2a_safe = Path(fn_s2a_safe)
        self.fn_s2b_safe = Path(fn_s2b_safe)
        self.band_id = band_id

        self.fn_s2a_l1c = list(self.fn_s2a_safe.glob(f"GRANULE/*/IMG_DATA/*_{band_id}.jp2"))[0]
        self.fn_s2b_l1c = list(self.fn_s2b_safe.glob(f"GRANULE/*/IMG_DATA/*_{band_id}.jp2"))[0]

        # Load data
        self.da_s2a = rioxarray.open_rasterio(self.fn_s2a_l1c).squeeze()
        self.da_s2b = rioxarray.open_rasterio(self.fn_s2b_l1c).squeeze()

        # Convert to reflectance
        self.da_s2a = self.da_s2a.astype("float32") / 10000.0
        self.da_s2b = self.da_s2b.astype("float32") / 10000.0

        self.crs = self.da_s2a.rio.crs

        # Coordinate transformer
        self.transformer = Transformer.from_crs(
            "EPSG:4326",
            self.crs,
            always_xy=True
        )

        # Atmospheric defaults
        self.atmos_settings = atmos_settings or {
            "atmos_profile": AtmosProfile.MidlatitudeSummer,
            "aero_profile": AeroProfile.Continental,
            "aot550": 0.1,
            "altitude_km": 0.5,
            "wavelength": CWL_S2[band_id] / 1000.0
        }

    # --------------------------------------------------
    # Internal 6S correction
    # --------------------------------------------------

    def _atmospheric_correct(self, patch, sza, saa, vza, vaa):

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
        s.atmos_corr = AtmosCorr.AtmosCorrLambertianFromReflectance(0.2)        # (paul) actual init value does not matter here, i think
        # s,atmos_corr = AtmosCorr.AtmosCorrBRDFFromRadiance()
        s.run()

        xa = s.outputs.coef_xa
        xb = s.outputs.coef_xb
        xc = s.outputs.coef_xc

        print("XA_B_C", xa, xb, xc)

        y = xa * patch - xb
        rho = y / (1 + xc * y)

        return rho

    # --------------------------------------------------
    # Public patch extraction
    # --------------------------------------------------

    def get_patch(self,
              lat: float,
              lon: float,
              size: int,
              atmos_corr: bool = False,
              geometry_s2a: dict | None = None,
              geometry_s2b: dict | None = None):
        """
        Returns:
            patch_s2a, patch_s2b

        geometry dict must contain:
            {
                "sza": float,
                "saa": float,
                "vza": float,
                "vaa": float
            }
        """

        if size % 2 == 0:
            raise ValueError("Patch size must be odd.")

        # Convert lat/lon to projected coordinates
        x_proj, y_proj = self.transformer.transform(lon, lat)

        def extract_patch(da):

            px = np.argmin(np.abs(da.x.values - x_proj))
            py = np.argmin(np.abs(da.y.values - y_proj))

            half = size // 2

            return da.isel(
                x=slice(px - half, px + half + 1),
                y=slice(py - half, py + half + 1)
            ).values

        # Extract both patches
        patch_s2a = extract_patch(self.da_s2a)
        patch_s2b = extract_patch(self.da_s2b)

        print(lat, lon, patch_s2a, patch_s2b)

        # --------------------------------------------------
        # Atmospheric correction (optional)
        # --------------------------------------------------

        if atmos_corr:

            if geometry_s2a is None or geometry_s2b is None:
                raise ValueError("Geometry must be provided for both S2A and S2B.")

            patch_s2a = self._atmospheric_correct(
                patch_s2a,
                geometry_s2a["sza"],
                geometry_s2a["saa"],
                geometry_s2a["vza"],
                geometry_s2a["vaa"]
            )

            patch_s2b = self._atmospheric_correct(
                patch_s2b,
                geometry_s2b["sza"],
                geometry_s2b["saa"],
                geometry_s2b["vza"],
                geometry_s2b["vaa"]
            )
        print("After: ", patch_s2a, patch_s2b)
        return patch_s2a, patch_s2b

if __name__ == "__main__":
    date_str = "20250318T101751"
    dt_obj = datetime.strptime(date_str, "%Y%m%dT%H%M%S")

    l1c_atmos_corr = L1C_AtmosCorr(dt_obj, "32UPU", "B04")

    print(l1c_atmos_corr.fn_s2a_l1c)


# python3 -m f_BRDF_measurement.S2_L1C_atmos_corrected
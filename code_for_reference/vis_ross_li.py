import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------
# Helper geometry functions
# ---------------------------------------------------

def relative_azimuth(saa, vaa):
    return (vaa - saa + 180) % 360 - 180


def ross_thick(sza, vza, relaz):
    sza = np.deg2rad(sza)
    vza = np.deg2rad(vza)
    relaz = np.deg2rad(relaz)

    cos_xi = (
        np.cos(sza) * np.cos(vza)
        + np.sin(sza) * np.sin(vza) * np.cos(relaz)
    )
    cos_xi = np.clip(cos_xi, -1, 1)
    xi = np.arccos(cos_xi)

    k_vol = (
        ((np.pi/2 - xi) * cos_xi + np.sin(xi))
        / (np.cos(sza) + np.cos(vza))
        - np.pi/4
    )

    return k_vol


def li_sparse(sza, vza, relaz):
    sza = np.deg2rad(sza)
    vza = np.deg2rad(vza)
    relaz = np.deg2rad(relaz)

    tan_s = np.tan(sza)
    tan_v = np.tan(vza)

    sec_s = 1 / np.cos(sza)
    sec_v = 1 / np.cos(vza)

    D = np.sqrt(
        tan_s**2 +
        tan_v**2 -
        2 * tan_s * tan_v * np.cos(relaz)
    )

    t = np.arccos(
        np.clip(
            (2 * D) / (sec_s + sec_v),
            -1, 1
        )
    )

    O = (1/np.pi) * (t - np.sin(t) * np.cos(t)) * (sec_s + sec_v)

    cos_xi = (
        np.cos(sza)*np.cos(vza)
        + np.sin(sza)*np.sin(vza)*np.cos(relaz)
    )

    k_geo = (
        O
        - sec_s
        - sec_v
        + 0.5 * (1 + cos_xi) * sec_s * sec_v
    )

    return k_geo


# ---------------------------------------------------
# Main plotting function
# ---------------------------------------------------

def plot_delta_kernel_space(ds, fn_out=None):
    """
    Plot ΔKvol vs ΔKgeo.
    Color = ΔReflectance.
    One subplot per landcover.
    """

    # ---- reflectance difference
    rho_a = ds["patch_S2A"].mean(dim=("y", "x")).values
    rho_b = ds["patch_S2B"].mean(dim=("y", "x")).values
    delta_rho = rho_b - rho_a

    # ---- geometry
    geom_a = ds["geom_S2A"].values
    geom_b = ds["geom_S2B"].values

    sza_a, saa_a, vza_a, vaa_a = geom_a.T
    sza_b, saa_b, vza_b, vaa_b = geom_b.T

    relaz_a = relative_azimuth(saa_a, vaa_a)
    relaz_b = relative_azimuth(saa_b, vaa_b)

    # ---- compute kernels
    kvol_a = ross_thick(sza_a, vza_a, relaz_a)
    kvol_b = ross_thick(sza_b, vza_b, relaz_b)

    kgeo_a = li_sparse(sza_a, vza_a, relaz_a)
    kgeo_b = li_sparse(sza_b, vza_b, relaz_b)

    delta_kvol = kvol_b - kvol_a
    delta_kgeo = kgeo_b - kgeo_a

    classes = ds["class"].values
    unique_classes = np.unique(classes)

    fig, axes = plt.subplots(
        1,
        len(unique_classes),
        figsize=(6 * len(unique_classes), 5),
        sharey=True
    )

    if len(unique_classes) == 1:
        axes = [axes]

    vmin = np.nanmin(delta_rho)
    vmax = np.nanmax(delta_rho)

    for ax, lc in zip(axes, unique_classes):

        mask = classes == lc

        sc = ax.scatter(
            delta_kvol[mask],
            delta_kgeo[mask],
            c=delta_rho[mask],
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            s=20,
            alpha=0.7,
            edgecolors="none"
        )

        ax.set_title(lc)
        ax.set_xlabel("Δ Kvol (Ross-Thick)")
        ax.set_ylabel("Δ Kgeo (Li-Sparse)")
        ax.grid(True)

    cbar = fig.colorbar(
        sc,
        ax=axes,
        orientation="vertical",
        fraction=0.04,
        pad=0.04
    )
    cbar.set_label("Δ Reflectance (S2B - S2A)")

    plt.suptitle("BRDF Difference in Ross–Li Kernel Space", y=1.02)
    plt.tight_layout()

    if fn_out:
        plt.savefig(fn_out, dpi=300, bbox_inches="tight")
    else:
        plt.show()
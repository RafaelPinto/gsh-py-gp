from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.collections import QuadMesh
from segysak import open_seisnc

from src.data import make_sua_surfaces
from src.data.make_sua_surfaces import SEISNC_PATH
from src.data.make_summary import SUMMARY_DIR
from src.definitions import ROOT_DIR

FIGURES_DIR = ROOT_DIR / "reports/figures"


def index_z(df: pd.DataFrame, z_name: str = "depth") -> Callable:
    """Build vectorized function that returns depth given (northing, easting).

    Parameters
    ----------
    df : pandas.DataFrame
        Gridded surface as a Dataframe. It should contain the coordinate
        columns (easting, northing) and depth or time.
    z_name : {"depth", "time"}, optional
        Column name in df for the vertical dimension.

    Returns
    -------
    get_depth_array : np.ufunc
        Vectorized function that returns depth given (nothing, easting).

    """
    indexed_z = {}
    for row in df.itertuples():
        indexed_z[(row.easting, row.northing)] = getattr(row, z_name)

    def get_z(easting: float, northing: float) -> float:
        return indexed_z.get((easting, northing), np.nan)

    get_depth_array = np.frompyfunc(get_z, nin=2, nout=1)

    return get_depth_array


def get_meshes_from_gridded_surface_pointset(
    df: pd.DataFrame,
    z_name: str = "depth",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create coordinates meshes from a gridded surface stored as a point set.

    Parameters
    ----------
    df : pandas.DataFrame
        Gridded surface as a Dataframe. It should contain the coordinate
        columns (easting, northing) and depth.
    z_name : {"depth", "time"}, optional
        Column name in df for the vertical dimension.

    Returns
    -------
    X : ndarray
        Easting coordinates mesh.
    Y : ndarray
        Northing coordinates mesh.
    Z : ndarray
        Depth values mesh.

    """
    unique_northing = np.sort(df.northing.unique())
    unique_easting = np.sort(df.easting.unique())

    X, Y = np.meshgrid(unique_easting, unique_northing)

    get_vert_array = index_z(df, z_name)
    Z = get_vert_array(X, Y)
    Z = Z.astype(float)

    return X, Y, Z


def plot_cartesian_gridded_surface(
    df: pd.DataFrame,
    ax: plt.Axes,
    z_name: str = "depth",
    title: str | None = None,
    cmap: str = "viridis_r",
    vmin: float | None = None,
    vmax: float | None = None,
    alpha: float | None = None,
) -> QuadMesh:
    """Plot points in cartesian grid.

    Parameters
    ----------
    df : pandas.DataFrame
        Gridded surface as a Dataframe. It should contain the coordinate
        columns (easting, northing) and depth.
    z_name : {"depth", "time"}, optional
        Column name in df for the vertical dimension.
    ax : plt.Axes
        A single matplotlib `Axes` object.
    title : str or None, optional
        Title to be placed on top of the figure. Defaults to `None`, i.e.,
        no title.
    cmap : str
        Colormap.
    vmin, vmax : float
        Define the data range that the colormap covers.
    alpha : float
        The alpha blending value, between 0 (transparent) and 1 (opaque).

    Returns
    -------
    ax : plt.Axes
        A single matplotlib `Axes` object.

    """
    X, Y, Z = get_meshes_from_gridded_surface_pointset(df, z_name)

    im = ax.pcolormesh(
        X,
        Y,
        Z,
        cmap=cmap,
        shading="nearest",
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
    )
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.axis("equal")

    if title:
        ax.set_title(title)

    return im


def save_fig_inline_with_regional_and_summary_surfaces(
    filename: str | None = None,
    inl_sel: int = 9100,
    xline_min: int = 7570,
    xline_max: int = 9600,
    depth_min: int = 0,
    depth_max: int = 4000,
    regional_surfaces: List[str] = ["ns_b", "ck_b", "rnro1_t", "ze_t", "ro_t"],
    quantile_surfaces: List[float] = [0.10, 0.25, 0.50, 0.75, 0.90],
):
    """Save inline with regional and summary surfaces.

    Parameters
    ----------
    filename: str, optional
        The file name that will be given to the saved figure.
    inl_sel: int, optional
        The inline number to slice the seismic data. Defaults to 9100.
    xline_min: int, optional
        The minimun xline to show. Defaults to 7570.
    xline_max: int, optional
        The maximun xline to show. Defaults to 9600.
    depth_min: int, optional
        The minimum depth to show. Defaults to 0 m.
    depth_max: int, optional
        The maximum depth to show. Defaults to 4000 m.
    regional_surfaces: List[str], optional
        A list with the namse of the reginal surfaces to plot. Defaults to
        ["ns_b", "ck_b", "rnro1_t", "ze_t", "ro_t"].
    quantile_surfaces: List[float], optional
        A list with the values of the quantile surfaces to plot. Deafults to
        [0.10, 0.25, 0.50, 0.75, 0.90]

    """
    if filename is None:
        filename = f"Inline_{inl_sel}_with_regional_and_summary_surfaces.png"
    # Load seismic
    seisnc = open_seisnc(SEISNC_PATH, chunks={"inline": 100})

    # Load regional surfaces
    surfaces = {
        surface_name: make_sua_surfaces.load_mapped_horizon(surface_name)
        for surface_name in regional_surfaces
    }

    # Load summary surfaces
    quantiles = xr.open_dataarray(SUMMARY_DIR / "quantiles.nc")

    # Plot
    xline_range = slice(xline_min, xline_max)
    depth_range = slice(depth_min, depth_max)

    opt = dict(
        x="xline",
        y="depth",
        add_colorbar=True,
        interpolation="spline16",
        robust=True,
        yincrease=False,
        cmap="Greys",
    )

    f, ax = plt.subplots(figsize=(16, 10), constrained_layout=True)

    seisnc.data.sel(
        iline=inl_sel, xline=xline_range, depth=depth_range
    ).plot.imshow(ax=ax, **opt)

    for surface_name, surface in surfaces.items():
        trace = surface.sel(iline=inl_sel, xline=xline_range)
        ax.plot(trace.xline, trace.values, label=surface_name)

    for q in quantiles["quantile"]:
        q = q.values
        if q not in quantile_surfaces:
            continue
        quantile_trace = quantiles.sel(
            iline=inl_sel, xline=xline_range, quantile=q
        )
        ax.plot(
            quantile_trace.xline,
            quantile_trace.values,
            label=f"RO T P{q*100:.0f}",
        )

    ax.set_xlim(xline_min, xline_max)
    ax.set_ylim(depth_max, depth_min)

    ax.invert_xaxis()

    if not FIGURES_DIR.exists():
        FIGURES_DIR.mkdir(parents=True)
    dst = FIGURES_DIR / filename
    f.savefig(str(dst))


def main():
    save_fig_inline_with_regional_and_summary_surfaces()

    save_fig_inline_with_regional_and_summary_surfaces(
        filename="Zoomed_inline_9100.png",
        regional_surfaces=["rnro1_t", "ro_t"],
    )

from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.collections import QuadMesh
from matplotlib.patches import Polygon
from segysak import open_seisnc

from src.data import make_sua_surfaces
from src.data.make_sua_surfaces import (
    HORIZON_PATH,
    PROXY_SURFACES_DIR,
    SEISNC_PATH,
    load_horizon,
)
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
    show_legend: bool = False,
) -> None:
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
    show_legend: bool, optional
        Answers: Should we show the legend? Defaults to False (do not show).

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

    if show_legend:
        f.legend(loc="lower center", ncol=9, bbox_to_anchor=(0.5, 0.05))

    if not FIGURES_DIR.exists():
        FIGURES_DIR.mkdir(parents=True)
    dst = FIGURES_DIR / filename
    f.savefig(str(dst))


def save_fig_inline_with_proxy_surfaces(
    filename: str | None = None,
    inl_sel: int = 9100,
    xline_min: int = 7625,
    xline_max: int = 7900,
    depth_min: int = 2700,
    depth_max: int = 3200,
) -> None:
    """Save inline with SUA proxy surfaces.

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

    """
    if filename is None:
        filename = f"Inline_{inl_sel}_with_proxy_surfaces.png"

    # Load seismic
    seisnc = open_seisnc(SEISNC_PATH, chunks={"inline": 100})

    # Load proxy surfaces
    surfaces = xr.open_mfdataset(
        str(PROXY_SURFACES_DIR / "*.nc"),
        combine="nested",
        concat_dim="anhydrite_perc",
        parallel=True,
    )

    surfaces = surfaces.set_xindex(coord_names="perc")

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

    for perc in surfaces.perc.values:
        trace = surfaces.sel(perc=perc, iline=inl_sel, xline=xline_range)
        label = f"{perc:.2f}"
        ax.plot(trace.xline, trace.depth, label=label)

    ax.set_xlim(xline_min, xline_max)
    ax.set_ylim(depth_max, depth_min)

    ax.invert_xaxis()
    f.legend(loc="lower center", ncol=9, bbox_to_anchor=(0.5, 0.05))

    if not FIGURES_DIR.exists():
        FIGURES_DIR.mkdir(parents=True)
    dst = FIGURES_DIR / filename
    f.savefig(str(dst))


def save_seismic_basemap(filename: str | None = None, inl_sel: int = 9100):
    """Save seismic basemap with given inline and RO Top posted.

    Parameters
    ----------
    filename: str, optional
        The file name that will be given to the saved figure.
    inl_sel: int, optional
        The inline number to slice the seismic data. Defaults to 9100.

    """

    if filename is None:
        filename = f"basemap_iline_{inl_sel}.png"

    # Load seismic
    seisnc = open_seisnc(SEISNC_PATH, chunks={"inline": 100})
    seisnc.seis.calc_corner_points()

    # Load RO Top horizon
    ro_t = load_horizon(HORIZON_PATH["ro_t"])

    # Coordinates text label positions in basemap
    text_aligment_kwargs = [
        {"horizontalalignment": "left", "verticalalignment": "bottom"},
        {"horizontalalignment": "left", "verticalalignment": "top"},
        {"horizontalalignment": "right", "verticalalignment": "top"},
        {"horizontalalignment": "right", "verticalalignment": "bottom"},
    ]

    corners = np.array(seisnc.attrs["corner_points_xy"])

    # Build seismic grid area (rectangle)
    survey_limits = Polygon(
        corners,
        fill=False,
        edgecolor="r",
        linewidth=2,
        label="3D survey extent",
    )

    # Plot Top Rotliegend
    title = "Basemap: Top Rotliegend"
    f, ax = plt.subplots(figsize=(14, 14))

    im = plot_cartesian_gridded_surface(
        ro_t,
        ax=ax,
        title=title,
        cmap="viridis_r",
        vmax=3000,
        vmin=2500,
    )
    f.colorbar(im, ax=ax, label="Depth (m)")

    # Plot seismic grid area
    ax.add_patch(survey_limits)

    # Plot selected inline
    selected_inline = seisnc.data.sel(iline=inl_sel)
    ax.plot(
        selected_inline.cdp_x,
        selected_inline.cdp_y,
        color="blue",
        label=f"Inline: {inl_sel}",
    )
    ax.axis("equal")
    ax.legend()

    # Add (inline, xline) labels to the seimic grid corners
    for corner_point, corner_point_xy, kwargs in zip(
        seisnc.attrs["corner_points"],
        seisnc.attrs["corner_points_xy"],
        text_aligment_kwargs,
    ):
        x, y = corner_point_xy
        ax.text(x, y, str(corner_point), kwargs)
    plt.show()

    if not FIGURES_DIR.exists():
        FIGURES_DIR.mkdir(parents=True)
    dst = FIGURES_DIR / filename
    f.savefig(str(dst), bbox_inches="tight")


def main():
    # Inline with regional surfaces and quantiles
    save_fig_inline_with_regional_and_summary_surfaces()

    # Inline zoomed in with Top of Salt and RO
    save_fig_inline_with_regional_and_summary_surfaces(
        filename="Zoomed_inline_9100.png",
        regional_surfaces=["rnro1_t", "ro_t"],
    )

    # Inline with all SUA proxy surfaces
    save_fig_inline_with_proxy_surfaces()

    # Inline with summary surfaces
    inl_sel = 9100
    xline_min = 7625
    xline_max = 7900
    depth_min = 2700
    depth_max = 3200
    quantile_surfaces = [0.10, 0.25, 0.50, 0.75, 0.90]
    save_fig_inline_with_regional_and_summary_surfaces(
        filename=f"Inline_{inl_sel}_with_summary_surfaces.png",
        inl_sel=inl_sel,
        xline_min=xline_min,
        xline_max=xline_max,
        depth_min=depth_min,
        depth_max=depth_max,
        regional_surfaces=[],
        quantile_surfaces=quantile_surfaces,
        show_legend=True,
    )

    # Basemap
    save_seismic_basemap()

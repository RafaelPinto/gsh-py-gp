from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import QuadMesh


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

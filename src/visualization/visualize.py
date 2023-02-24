from typing import Tuple, Callable

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def index_depth(df : pd.DataFrame) -> Callable:
    """Build vectorized function that returns depth given (northing, easting).

    Parameters
    ----------
    df : pandas.DataFrame
        Gridded surface as a Dataframe. It should contain the coordinate columns
        (easting, northing) and depth.

    Returns
    -------
    get_depth_array : np.ufunc
        Vectorized function that returns depth given (nothing, easting).
    
    """
    indexed_depth = {}
    for row in df.itertuples():
        indexed_depth[(row.easting, row.northing)] = row.depth

    def get_depth(easting : float, northing : float) -> float:
        return indexed_depth.get((easting, northing), np.nan)
    
    get_depth_array = np.frompyfunc(get_depth, nin=2, nout=1)
    
    return get_depth_array


def get_meshes_from_gridded_surface_pointset(df : pd.DataFrame) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray
    ]:
    """Create coordinates meshes from a gridded surface stored as a point set.

    Parameters
    ----------
    df : pandas.DataFrame
        Gridded surface as a Dataframe. It should contain the coordinate columns
        (easting, northing) and depth.

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

    get_depth_array = index_depth(df)
    Z = get_depth_array(X, Y)
    Z = Z.astype(float)

    return X, Y, Z


def plot_cartesian_gridded_surface(
    df: pd.DataFrame,
    title: str | None = None,
    figsize: Tuple[int, int] = (8, 8)
    ) -> Tuple[plt.Figure, plt.Axes]: # type: ignore
    """Plot points in cartesian grid.

    Parameters
    ----------
    df : pandas.DataFrame
        Gridded surface as a Dataframe. It should contain the coordinate columns
        (easting, northing) and depth.
    title : str or None, optional
        Title to be placed on top of the figure. Defaults to `None`, i.e.,
        no title.
    figsize: tuple[int, int], optional
        Figure width, height in inches. Defaults to (8, 8).

    Returns
    -------
    fig : plt.figure.Figure
        Matplotlib `Figure`.
    ax : plt.Axes
        A single matplotlib `Axes` object.
        
    """
    X, Y, Z = get_meshes_from_gridded_surface_pointset(df)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('equal')
    im = ax.pcolormesh(X, Y, Z, cmap="viridis_r", shading="nearest")
    fig.colorbar(im, ax=ax, label="Depth (m)")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    
    if title:
        ax.set_title(title)

    return fig, ax
import matplotlib.pyplot as plt
import pandas as pd


def plot_cartesian_depth_points(
    df: pd.DataFrame,
    title: str | None = None,
    figsize: tuple[int, int] = (8, 8)
    ) -> tuple[plt.Figure, plt.Axes]: # type: ignore
    """Plot points in cartesian grid.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with point values to be plotted. It should contain the
        coordinate columns (easting, northing) and depth.
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
    fig, ax = plt.subplots(figsize=figsize)
    plt.scatter(df.easting, df.northing, c=df.depth, cmap="viridis")
    plt.colorbar(label="Depth (m)")
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    
    if title:
        plt.title(title)

    return fig, ax
import numpy as np
import pandas as pd


def check_grid_spacing(coordinate: pd.Series) -> np.ndarray:
    """Print grid size value count and return unique coordinates.

    Parameters
    ----------
    coordinate : pd.Series
        The coordinate values, for example, the northing column.

    Returns
    -------
    unique_coordinate : np.ndarray
        The unique coordinates sorted.

    """
    unique_coordinate = np.sort(coordinate.unique())
    grid_size_count = pd.Series(unique_coordinate).diff().value_counts()
    df = grid_size_count.to_frame(name="Count")
    df.index.name = "Grid size"
    print(df)
    return unique_coordinate

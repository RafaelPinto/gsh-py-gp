from typing import Tuple

import numpy as np
import pandas as pd

from src.data.utils import read_petrel_exported_pointset


def check_grid_spacing(coordinate : pd.Series) -> np.ndarray:
    unique_coordinate = np.sort(coordinate.unique())
    grid_size_count = pd.Series(unique_coordinate).diff().value_counts()
    df = grid_size_count.to_frame(name="Count")
    df.index.name = "Grid size"
    print(df)
    return unique_coordinate

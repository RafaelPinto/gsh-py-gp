from pathlib import Path

import pandas as pd


def line_count(filepath: Path) -> None:
    """Print number of lines (count) in file.

    Parameters
    ----------
        filepath : pathlib.Path
            Path to the file that will be read.

    """
    count = 0
    with open(filepath, "r") as fhandle:
        for line in fhandle:
            count += 1
    print(f"Line count: {count:,}")


def head(filepath: Path, max_line_count: int = 10) -> None:
    """Print lines from file.

    Parameters
    ----------
        filepath : pathlib.Path
            Path to the file that will be read.
        max_line_count : int, optional
            Limit of lines that will be read. Defaults to 10.

    """
    with open(filepath, "r") as fhandle:
        for _, line in zip(range(max_line_count), fhandle):
            print(line, end="")


def read_petrel_exported_pointset(filepath: Path) -> pd.DataFrame:
    """Read Petrel exported point set into a dataframe.

    Parameters
    ----------
    filename : Path
        Path to the Petrel exported file.

    Returns
    -------
    df : pd.DataFrame
        Point set as a dataframe.

    Notes
    -----
    Point set format assumptions:
        - The file has no header.
        - Column order is: Inline, Xline, Easting, Northing, Depth.
        - Values are space separated.

    """
    col_names = ["inline", "xline", "easting", "northing", "depth"]
    df = pd.read_csv(filepath, sep=r"\s+", header=None, names=col_names)
    return df

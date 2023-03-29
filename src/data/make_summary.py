from pathlib import Path
from typing import List

import xarray as xr

from src.definitions import ROOT_DIR

SURFACES_DIR = ROOT_DIR / "data/processed/surfaces"

SUMMARY_DIR = ROOT_DIR / "data/processed/summary"


def calculate_quantiles(
    surfaces: xr.Dataset,
    dst_dir: Path | None = None,
    overwrite: bool = False,
    q: List[float] | None = None,
) -> xr.DataArray:
    """Save and return surfaces depth quantiles.

    Parameters
    ----------
    surfaces: xr.Dataset
        Xarray representation of the surfaces to be summarized.
    dst_dir: Path, optional
        Destination directory to save the summary surfaces.
        Defaults to "data/processed/summary"
    overwrite: bool, optional
        Answers the question: Should we overwrite any previous summary
        surfaces? Defaults to False.
    q: List[float], optional
        List with quantile values (floats) to be calculated.
        Defaults to [0.1, 0.25, 0.5, 0.75, 0.9].

    Returns
    -------
    quantiles: xr.DataArray
        Summary surfaces.
    """
    print("Calculating quantiles.")
    if not dst_dir:
        dst_dir = SUMMARY_DIR

    if not q:
        q = [0.1, 0.25, 0.5, 0.75, 0.9]

    dst = dst_dir / "quantiles.nc"

    if not dst.exists() or overwrite:
        dst_dir.mkdir(parents=True, exist_ok=True)
        surfaces = surfaces.chunk(dict(anhydrite_perc=-1))
        quantiles = surfaces.depth.quantile(q, dim="anhydrite_perc")
        quantiles.to_netcdf(dst)

    quantiles = xr.open_dataarray(dst)

    return quantiles


def main(overwrite):
    """Calculate quantiles."""
    surfaces = xr.open_mfdataset(
        str(SURFACES_DIR / "*.nc"),
        combine="nested",
        concat_dim="anhydrite_perc",
        parallel=True,
        chunks={"anhydrite_perc": -1},
    )
    surfaces = surfaces.set_xindex(coord_names="perc")

    calculate_quantiles(surfaces, overwrite=overwrite)

from pathlib import Path
from typing import Dict

import pandas as pd
import xarray as xr
from segysak import open_seisnc, segy

from src.definitions import ROOT_DIR

# Downloaded files directory
DST_DIR = ROOT_DIR / "data"

SEISMIC_DIR = DST_DIR / "external/groningen/Seismic_Volume"

# Path to seismic data in SEGY format
SEGY_PATH = SEISMIC_DIR / "R3136_15UnrPrDMkD_Full_D_Rzn_RMO_Shp_vG.SEGY"

SEGY_INLINE_BYTE = 5
SEGY_XLINE_BYTE = 21
SEGY_CDPX_BYTE = 73
SEGY_CDPY_BYTE = 77

# Path to seismic data in SEISNC format
SEISNC_PATH = DST_DIR / "interim/R3136_15UnrPrDMkD_Full_D_Rzn_RMO_Shp_vG.seisnc"

HORIZON_DIR = DST_DIR / "external/groningen/Horizon_Interpretation"

HORIZON_PATH = {
    "RNRO1_T": HORIZON_DIR / "DCAT201605_R3136_RNRO1_T_pk_depth",
    "RO_T": HORIZON_DIR / "RO____T",
}


def load_horizons(horizon_paths: Dict[str, Path]) -> Dict[str, pd.DataFrame]:
    """Load horizons in 5 column format.

    Parameters
    ----------
    horizon_paths: Dict[str, Path]
        A mapping from horizons names to horizon file path.

    Returns
    -------
    horizons: Dict[str, Path]
        A mapping from horizons names to horizons loaded as dataframes.
    """
    col_names = ["inline", "xline", "easting", "northing", "depth"]

    horizons = {}
    for horizon_name, horizon_path in horizon_paths.items():
        horizons[horizon_name] = pd.read_csv(
            horizon_path, sep=r"\s+", header=None, names=col_names
        )
    return horizons


def convert_horizon_to_xarray(
    horizon: pd.DataFrame, seisnc: xr.Dataset
) -> xr.DataArray:
    """Convert horizon from dataframe to xarray and map it to the seismic grid.

    Parameters
    ----------
    horizon: pd.Dataframe
        The horizon to be converted to xarray format.

    seisnic: xr.Dataset
        The opened seismic data in SEISNC format.

    Returns
    -------
    horizon_mapped: xr.DataArray
    """
    # Map horizon to seismic grid and fill in missing data (convex hull)
    horizon_mapped = seisnc.seis.surface_from_points(
        horizon,
        "depth",
        left=("cdp_x", "cdp_y"),
        right=("easting", "northing"),
    )

    # Seismic is no longer the data, but the horizon depth is
    horizon_mapped = horizon_mapped.depth
    horizon_mapped = horizon_mapped.drop_vars("depth")

    return horizon_mapped


def calc_mixed_salt_vp(
    anhydrite_perc: float, anhydrite_vp: float = 5900, halite_vp: float = 4400
) -> float:
    """Calculate the mixed salt velocity given anhydrite composition percent.

    Parameters
    ----------
    anhydrite_perc: float
        Percent of anhydrite in the mixed salt. Must be between 0 (no
        anhydrite) and 1 (no halite).
    anhydrite_vp: float, optional
        Anhydrite P-wave velocity. Defaults to 5900 m/s.
    halite_vp: float, optional
        Halite P-wave velocity. Defaults to 4400 m/s.

    Returns
    -------
    float
        Mixed salt P-wave velocity.
    """
    if anhydrite_perc < 0:
        raise ValueError(
            f"The anhydrite percent cannot be less than zero: {anhydrite_perc}"
        )
    if anhydrite_perc > 1:
        raise ValueError(
            f"The anhydrite percent cannot be more than one: {anhydrite_perc}"
        )
    return halite_vp * (1 - anhydrite_perc) + anhydrite_vp * anhydrite_perc


def update_horizon(
    reference_top_horizon: xr.DataArray,
    reference_isochron: xr.DataArray,
    anhydrite_perc: float,
) -> xr.DataArray:
    """Update the target horizon based on the P-wave velocity change.

    Parameter
    ---------
    reference_top_horizon: xr.DataArray
        Horizon that corresponds to the top of the isochrone.
    reference_isochron: xr.DataArray
        Isochrone (vertical time thickness) from the reference top horizon to
        the target horizon.
    anhydrite_perc: float
        Percent of anhydrite in the mixed salt. Must be between 0 (no
        anhydrite) and 1 (no halite). Used to calculate the updated P-wave
        velocity.

    Returns
    -------
    horizon_update: xr.DataArray
        Target horizon with updated depth.
    """

    salt_vp_update = calc_mixed_salt_vp(anhydrite_perc)
    isochore_update = reference_isochron * salt_vp_update
    horizon_update = reference_top_horizon + isochore_update

    # Format xarray
    horizon_update = horizon_update.expand_dims(dim="anhydrite_perc", axis=0)
    horizon_update = horizon_update.assign_coords(
        perc=("anhydrite_perc", [anhydrite_perc])
    )
    horizon_update = horizon_update.set_xindex(coord_names="perc")

    return horizon_update


def main():
    """Make SUA proxy surfaces."""
    # Convert downloaded seismic data from SEGY to SEISNC format
    # This will take 2-3 hours
    if not SEISNC_PATH.exists():
        print("Seismic data in SEISNC format was not found.")
        print("Converting seismic data from SEGY to SEISNC.")
        segy.segy_converter(
            SEGY_PATH,
            SEISNC_PATH,
            iline=SEGY_INLINE_BYTE,
            xline=SEGY_XLINE_BYTE,
            cdpx=SEGY_CDPX_BYTE,
            cdpy=SEGY_CDPY_BYTE,
            vert_domain="DEPTH",
        )

    print("Loading seismic data in SEISNC format.")
    seisnc = open_seisnc(SEISNC_PATH, chunks={"inline": 100})

    print("Loading horizons.")
    horizons = load_horizons(HORIZON_PATH)

    print("Converting horizons to Xarray.")
    rnr01_t = convert_horizon_to_xarray(horizons["RNRO1_T"], seisnc)
    ro_t = convert_horizon_to_xarray(horizons["RO_T"], seisnc)

    print("Calculating isochore.")
    salt_isochore = ro_t - rnr01_t

    reference_anhydrite_perc = 0.2
    reference_salt_vp = calc_mixed_salt_vp(reference_anhydrite_perc)
    # t = d / v
    reference_salt_isochrone = salt_isochore / reference_salt_vp

    dst_dir = DST_DIR / "processed/surfaces"
    dst_dir.mkdir(parents=True, exist_ok=True)

    anhydrite_percs = [perc / 100 for perc in range(5, 33, 1)]
    for anhydrite_perc in anhydrite_percs:
        print(f"Updating target structure with anhydrite percent: {anhydrite_perc}.")
        target_update = update_horizon(
            rnr01_t, reference_salt_isochrone, anhydrite_perc=anhydrite_perc
        )
        dst = dst_dir / f"ro_t_anhydrite_perc_{anhydrite_perc:.2f}.nc"
        target_update.to_netcdf(dst)
        print(f"Saved target surface to: {dst}")


if __name__ == "__main__":
    main()

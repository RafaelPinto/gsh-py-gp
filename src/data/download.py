from pathlib import Path

import requests

from src.definitions import ROOT_DIR

URL_BASE = "https://swung-hosted.s3.ca-central-1.amazonaws.com/"

FILES_TO_DOWNLOAD = [
    "groningen/README.txt",
    "groningen/FILENAMES.txt",
    "groningen/3DGrid/3D_Grid_Export_settings.PNG",
    "groningen/3DGrid/3D_Grid_Horizon_order.png",
    "groningen/Formation_tops/Groningen__Formation_tops__EPSG_28992.csv",
    "groningen/Horizon_Interpretation/DCAT201605_R3136_CK_B_pk_depth",
    "groningen/Horizon_Interpretation/DCAT201605_R3136_NS_B_tr_depth",
    "groningen/Horizon_Interpretation/DCAT201605_R3136_RNRO1_T_pk_depth",
    "groningen/Horizon_Interpretation/DCAT201605_R3136_RNRO1_T_pk_t",
    "groningen/Horizon_Interpretation/DCAT201605_R3136_ZE_T_na_depth",
    "groningen/Horizon_Interpretation/RO____T",
    "groningen/Seismic_Volume/R3136_15UnrPrDMkD_Full_D_Rzn_RMO_Shp_vG.SEGY",
]

# Local place to save the downloaded files
DST_DIR = ROOT_DIR / "data/external/"


def download_from_groningen(
    files: list, dst_dir: Path, overwrite: bool = False
) -> None:
    """Dowload files from Groningen's open data fork.

    Parameters
    ----------
        files : List[str]
            List of file paths relative to the open data fork S3 bucket
            location, for example, "groningen/README.txt".
        dst_dir : pathlib.Path
            Destination directory for the dowloaded files.
        overwrite: Bool, optional
            What should we do if the file already exists.
            Default to False, i.e., don't overwrite.

    Notes
    -----
        Based on the code available at:
        https://github.com/agilescientific/groningen/blob/main/notebooks/Read_data_from_cloud.ipynb

    """
    for filepath in files:
        fullpath = dst_dir / filepath

        if fullpath.exists() and not overwrite:
            print(f"Will not overwrite file: {filepath}")
            continue

        parent_dir = fullpath.parent
        if not parent_dir.is_dir():
            parent_dir.mkdir(parents=True)

        url = URL_BASE + filepath

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(fullpath, "wb") as f:
                # Bytes in chunk.
                for chunk in r.iter_content(chunk_size=2_097_152):
                    f.write(chunk)


if __name__ == "__main__":
    download_from_groningen(FILES_TO_DOWNLOAD, DST_DIR, overwrite=False)

from pathlib import Path
import requests


URL_BASE = "https://swung-hosted.s3.ca-central-1.amazonaws.com/"


def line_count(filepath : Path) -> None:
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


def head(filepath : Path, max_line_count : int = 10) -> None:
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
            print(line)


def download_from_groningen(
    files : list,
    dst_dir : Path,
    overwrite : bool = False
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
                for chunk in r.iter_content(chunk_size=2_097_152):  # Bytes in chunk.
                    f.write(chunk) 
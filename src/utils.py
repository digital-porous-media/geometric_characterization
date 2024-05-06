import subprocess
import numpy as np
import tifffile
import csv
import os
import pandas as pd
import wget

def runcmd(cmd, verbose = True, *args, **kwargs):

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass

def download_file(url, filename):
    """
    Run a wget command to download a file from an HTTP or FTP server
    :param url: URL of the file to download
    :param filename: Downloaded file name
    :return: None
    """
    wget.download(url, out=f"{filename}.ubc")
    # runcmd(f'wget -O {filename}.ubc {url}')

def get_datafiles(datapath='data'):
    """
    Download the data files contained in data/data_links.csv
    :return: None
    """
    with open(f'{datapath}/data_links.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        # Skip the header row
        next(csv_reader, None)
        for row in csv_reader:
            # Only download the data files if it does not already exist
            if not os.path.exists(f'{datapath}/{row[0]}.ubc'):
                # Download the file from Digital Rocks Portal
                download_file(row[1], f'{datapath}/{row[0]}')

            # Convert to tiff and invert pore and solid labels
            img = np.fromfile(f"{datapath}/{row[0]}.ubc", dtype=np.uint8).reshape((512, 512, 512))
            img = -1 * img + 1
            tifffile.imwrite(f"{datapath}/{row[0]}.tif", img.astype(np.uint8))

            # Remove the ubc file. Only keep the tiff file
            runcmd(f"rm -f {datapath}/{row[0]}.ubc")


def read_tiff(datapath: str) -> np.ndarray:
    """
    Reads a 3D numpy array from a tif file at the given filepath.
    :param datapath: The filepath of the tif file to read.
    :returns: np.ndarray: A 3D numpy array containing the data from the tif file.
    :raises: ValueError: If the given filepath does not point to a valid tif file.
    """

    if not os.path.exists(datapath):
        raise ValueError(f"File {datapath} does not exist.")

    extension = os.path.splitext(datapath)[1]
    if not (extension == ".tif" or extension == ".tiff"):
        raise ValueError(f"File {datapath} is not a tif file.")

    return tifffile.imread(datapath)


def create_results_directory(directory_path: str = '.'):
    """
    Create a new results directory
    :param directory_path: Path in which a directory named 'image_characterization_results'
    :return: None
    """
    # Create a parent results directory if it does not already exist at the specified path
    os.makedirs(os.path.join(directory_path, 'image_characterization_results'), exist_ok=True)


def write_results(results_df: pd.DataFrame, results_type: str,  directory_path: str = '.', filetype: str = 'csv', **kwargs) -> None:
    """
    Write a csv file containing the results of the analysis
    :results_dict: Dictionary containing the results of the analysis
    :results_type: Type of the analysis. Options are 'minkowski', 'heterogeneity', and 'subsets'
    :data_name: Filename of the data. This will be used as the filename of the csv file (<data_name>.csv)
    :filetype: Type of the file to write. Options are 'parquet', 'csv', 'pickle', 'feather'. Default is
    :return: None
    """

    # Check that results_type is valid
    assert results_type.lower() in ['minkowski', 'heterogeneity', 'subsets'], \
        "Results type must be 'minkowski', 'heterogeneity', or 'subsets'"

    filetype = filetype.lower()
    assert filetype in ['parquet', 'csv', 'pickle', 'feather', 'json'], \
        "Filetype must be 'parquet', 'csv', 'pickle', 'feather', 'json'"

    read_filetype_dict = {'parquet': pd.read_parquet,
                          'csv': pd.read_csv,
                          'feather': pd.read_feather,
                          'pickle': pd.read_pickle,
                          'json': pd.read_json}

    # First create the results directory structure
    create_results_directory(directory_path)

    path_tmp = os.path.join(directory_path, "image_characterization_results", f"{results_type}.{filetype}")
    # Check if the results file already exists
    # TODO: Should we use actual append functions? Should we rethink our algorithm for writing these files?
    if os.path.exists(path_tmp):
        # If so, read the results file, append a row and write it back out
        df_tmp = read_filetype_dict[filetype](path_tmp)
        # Concatenate the results dataframe with the existing dataframe
        results_df = pd.concat([df_tmp, results_df.copy()], ignore_index=True)
        # Drop any rows with duplicated data names. keep only the last occurrence (assuming to be most recently added)
        results_df = results_df.drop_duplicates(subset=['Name'], keep='last', ignore_index=True)

    filetype_write_dict = {'parquet': results_df.to_parquet,
                           'csv': results_df.to_csv,
                           'feather': results_df.to_feather,
                           'pickle': results_df.to_pickle,
                           'json': results_df.to_json}
    # Write the results to a file
    filetype_write_dict[filetype](path_tmp, **kwargs)


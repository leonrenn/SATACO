"""The functions in this file are used to open parquet files
and read the correlation matrix from file.
"""
import os
import pathlib
from typing import List, Tuple

import numpy as np
import pandas as pd

from exceptions.exceptions import CorrelationMatrixFormatError

SR_PATH_FILE: str = str(pathlib.Path(
    __file__).parent.resolve()) + "/../../results/signal_regions.txt"

SR_NZ_PATH_FILE: str = str(pathlib.Path(
    __file__).parent.resolve()) + "/../../results/signal_regions_non_zero.txt"


def read_parquet_event_SR() -> pd.DataFrame:
    """Read compressed parquet file into pandas
    dataframe.

    Returns:
        pd.DataFrame: Dataframe of events of corresponding
        signal regions.
    """
    with open(SR_PATH_FILE, "r") as file:
        sr_names: List[str] = file.readlines()
    df_event_SR: pd.DataFrame = pd.read_parquet(
        path=str(pathlib.Path(__file__).parent.resolve()) +
        "/../../results/event_SR.parquet.gzip",
        columns=sr_names)
    return df_event_SR


def read_parquet_corr() -> pd.DataFrame:
    """Read compressed parquet file into pandas
    dataframe.

    Returns:
        pd.DataFrame: Dataframe of events of corresponding
        signal regions.
    """
    with open(SR_NZ_PATH_FILE, "r") as file:
        sr_names: List[str] = file.readlines()
    df_corr: pd.DataFrame = pd.read_parquet(
        path=str(pathlib.Path(__file__).parent.resolve()) +
        "/../../results/correltations.parquet.gzip",
        columns=sr_names)
    return df_corr


def read_corr_matrix(corr_matrix_file_path: str) -> Tuple:
    """Read correlation matrix from txt file.

    Args:
        corr_matrix_file_path (str): File path to the
        correlation matrix in comma seperated format.

    Returns:
        Tuple: Correlation matrix, SR names.
    """
    if os.path.isfile(path=corr_matrix_file_path):
        try:
            df: pd.DataFrame = pd.read_csv(
                filepath_or_buffer=corr_matrix_file_path)
            SR_names: List[str] = list(df.columns)
            corr_matrix: np.array = df.to_numpy(dtype=np.float32)
        except Exception:
            raise CorrelationMatrixFormatError
    else:
        raise FileNotFoundError
    return corr_matrix, SR_names

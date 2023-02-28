"""Saving functions for the main program
in a very intuitive way to their normal saving locations.
"""

import os
import pathlib
from typing import List

import pandas as pd


def save_df_SR_event(df_event_SR: pd.DataFrame,
                     compression: str = "gzip",
                     index: bool = False) -> None:
    """Saves the pandas dataframe into compressed 'gzip'
    parquet file. Function extracted from main in order
    to use multiprocessing in a clear format.

    Args:
        df_event_SR (pd.DataFrame): Dataframe with events fo the
        corresponding Signal Region.
        compression (str, optional): Compression style.
        Defaults to "gzip".
        index (bool, optional): Including indexes.
        Defaults to False.
    """
    df_event_SR.to_parquet(
        str(pathlib.Path(__file__).parent.resolve()) +
        "/../../results/event_SR.parquet.gzip",
        index=index,
        compression=compression)
    return


def save_df_corr(df_corr: pd.DataFrame,
                 compression: str = "gzip",
                 index: bool = False) -> None:
    """Saves the pandas dataframe into compressed 'gzip'
    parquet file. Function extracted from main in order
    to use multiprocessing in a clear format.

    Args:
        df_corr (pd.DataFrame): Dataframe with events fo the
        corresponding Signal Region.
        compression (str, optional): Compression style.
        Defaults to "gzip".
        index (bool, optional): Including indexes.
        Defaults to False.
    """
    df_corr.to_parquet(
        str(pathlib.Path(__file__).parent.resolve()) +
        "/../../results/correlations.parquet.gzip",
        index=index,
        compression=compression)
    return


def save_SR_names(SR_names: List[str],
                  zero_cols: bool) -> None:
    """Saving SR line per line from current analysis
    in a txt file.

    Args:
        SR_names (List[str]): Signal regions that are
        analyzed during the run.
        zero_cols (bool): Include zero columns.
    """
    sr_file_path: str = str(pathlib.Path(
        __file__).parent.resolve()) + "/../../results/signal_regions.txt"
    if zero_cols:
        sr_file_path = sr_file_path[:-4] + "_non_zero.txt"
    with open(sr_file_path, "w") as file:
        for sr_idx, sr in enumerate(SR_names):
            if sr_idx != len(SR_names):
                file.write(sr+"\n")
            else:
                file.write(sr+"\n")
    return


def clear_result_dir() -> None:
    """Remove everything from the result directory.
    """
    result_dir: str = str(pathlib.Path(
        __file__).parent.resolve()) + "/../../results"
    for file in os.listdir(result_dir):
        os.remove(result_dir + "/"+file)
    return

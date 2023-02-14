"""The functions in this file are not used in the main program (yet?),
but were created to load the parquet files from the result
directory with the correct headers.
"""
import pathlib
import typing as List

import pandas as pd

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


def read_parquet_SR_SR() -> pd.DataFrame:
    """Read compressed parquet file into pandas
    dataframe.

    Returns:
        pd.DataFrame: Dataframe of events of corresponding
        signal regions.
    """
    with open(SR_NZ_PATH_FILE, "r") as file:
        sr_names: List[str] = file.readlines()
    df_SR_SR: pd.DataFrame = pd.read_parquet(
        path=str(pathlib.Path(__file__).parent.resolve()) +
        "/../../results/SR_SR.parquet.gzip",
        columns=sr_names)
    return df_SR_SR


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

import pathlib

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


def save_df_SR_SR(df_SR_SR: pd.DataFrame,
                  compression: str = "gzip",
                  index: bool = False) -> None:
    """Saves the pandas dataframe into compressed 'gzip'
    parquet file. Function extracted from main in order 
    to use multiprocessing in a clear format.

    Args:
        df_SR_SR (pd.DataFrame): Dataframe with events fo the 
        corresponding Signal Region.
        compression (str, optional): Compression style.
        Defaults to "gzip".
        index (bool, optional): Including indexes.
        Defaults to False.
    """
    df_SR_SR.to_parquet(
        str(pathlib.Path(__file__).parent.resolve()) +
        "/../../results/SR_SR.parquet.gzip",
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

"""The functions in this file are not used in the main program yet,
but were created to load the parquet files from the result
directory with the correct headers.
"""
import pathlib
from typing import List

import pandas as pd


def read_parquet_event_SR(signal_regions: List[str] = None) -> pd.DataFrame:
    pass


def read_parquet_SR_SR(signal_regions: List[str] = None) -> pd.DataFrame:
    pass


def read_parquet_corr(signal_regions: List[str] = None) -> pd.DataFrame:
    pass

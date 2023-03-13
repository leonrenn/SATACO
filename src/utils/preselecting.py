"""This file contains functions
that preselect signal regions.
"""
import os
import pathlib
from typing import List, Tuple

import pandas as pd

from exceptions.exceptions import NonPreselectionInfoFound
from utils.functions import calc_SR_sensitivity


def filter_SRs(analysis_name: str,
               path: str) -> Tuple:
    """Filters signal regions of given analysis
    when given preselection flag and correct path to
    pMSSM analysis info files.

    Args:
        analysis_name (str): Name of current analysis.
        path (str): Path to the info directory.

    Returns:
        Tuple: SR regions that can be further analyzed,
        Weights as significances.
    """
    path: str = str(pathlib.Path(__file__).parent.resolve()) + \
        "/../../" + \
        f"{path}/{analysis_name}.info"
    if os.path.isfile(path=path) is True:
        df_info = pd.read_csv(filepath_or_buffer=path)
        approved_SR_names: List[str] = list(df_info["SR"])
        approved_weights: List[float] = calc_SR_sensitivity(
            df_event_SR=df_info,
            method="from info files",
            calculate=True)
    else:
        print(f"For analysis {analysis_name}, there is no info file"
              f" in {path}, that specifies the analysis.")
        raise NonPreselectionInfoFound
    return approved_SR_names, approved_weights

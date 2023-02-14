import time
from typing import Dict, List

import numpy as np
import pandas as pd
from networkx.classes import DiGraph

from exceptions.exceptions import NotEnoughStatistcis


def df_mapping_dict(SR_names: List[str],
                    inv: bool = False) -> Dict[int, str]:
    """Generate mapping dictionary for dataframe
    renaming and getting key rapidly

    Args:
        SR_names (List[str]):
        inv (bool): Inverts the dictionary. Defaults to False.

    Returns:
        Dict[int, str]: Returns mapping dictionary.
    """
    mapping_dict: Dict[int, str] = {}
    for sr_idx, sr in enumerate(SR_names):
        mapping_dict[sr_idx] = sr
    if inv is True:
        mapping_dict = {v: k for k, v in mapping_dict.items()}
    return mapping_dict


def calc_num_combs(len_sr_names: int) -> int:
    """Formula to calculate the number of
    combinations with replacement for two
    signal regions.

    Args:
        len_sr_names (int): Length of list of signal regions.

    Returns:
        int: Number of combinations.
    """
    return int((len_sr_names + 1) * len_sr_names/2)


def check_sufficient_statistics(SR_SR_matrix: np.array,
                                event_num: int,
                                confidence: float = 0.95) -> None:
    # TODO: Algorithm like in TACO that checks if enough data has
    # been gathered for further staistic analysis
    # https://gitlab.com/t-a-c-o/taco_code/-/blob/master/codes/accepter_v2.py
    if False:
        raise NotEnoughStatistcis
    return


def calc_pearson_corr(SR_SR_matrix: np.array) -> np.array:
    """Calculation of the Pearson correlation coefficient
    of the signal region matrix.

    Args:
        SR_SR_matrix (np.array): Signal region matrix that
        have entries that share events.

    Returns:
        np.array: Coefficient matrix.
    """
    return np.corrcoef(SR_SR_matrix, dtype=np.float32)


def threshold_corr_matrix(correlation_matrix: np.array,
                          threshold: float = 0.2) -> np.array:
    """Compute correlation matrix below certain threshold.

    Args:
        correlation_matrix (np.array): Pearson correlations coefficients
        threshold (float, optional): Threshold for allowed correlation.
        Defaults to 0.2.

    Returns:
        np.array: Allowed correlation in binary format.
    """
    return correlation_matrix > threshold


def transform_overlap_digraph(SR_SR_matrix: np.array) -> DiGraph:
    pass


def find_best_combination(SR_digraph: DiGraph) -> List[str]:
    pass

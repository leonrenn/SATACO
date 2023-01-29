from typing import Dict, List

import numpy as np


def df_mapping_dict(sr_names: List[str], inv: bool = False) -> Dict[int, str]:
    """Generate mapping dictionary for dataframe
    renaming and getting key rapidly

    Args:
        sr_names (List[str]):
        inv (bool): Inverts the dictionary. Defaults to False.

    Returns:
        Dict[int, str]: Returns mapping dictionary.
    """
    mapping_dict: Dict[int, str] = {}
    for sr_idx, sr in enumerate(sr_names):
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


def calc_pearson_corr(SR_SR_matrix: np.array) -> np.array:
    """Calculation of the Pearson correlation coefficient
    of the signal region matrix.

    Args:
        SR_SR_matrix (np.array): Signal region matrix that
        have entries that share events.

    Returns:
        np.array: Coefficient matrix.
    """
    return np.corrcoef(SR_SR_matrix)

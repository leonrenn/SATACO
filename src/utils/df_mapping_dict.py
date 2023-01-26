from typing import Dict, List


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

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

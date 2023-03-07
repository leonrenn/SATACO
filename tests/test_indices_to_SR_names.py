from typing import Dict, List

from utils.functions import indices_to_SR_names


def test_indices_to_SR_names() -> None:
    """This test tests the functionality
    of assigning names instead of indices.
    """
    SR_names: List[str] = ["SR_0",
                           "SR_1",
                           "SR_2",
                           "SR_3"]
    path_dictionary: Dict = {0: {"path": [2, 1],
                                 "weight": 2.0},
                             1: {"path": [1, 0],
                                 "weight": 1.5}}
    path_dictionary_names: Dict = indices_to_SR_names(
        SR_names=SR_names,
        path_dictionary=path_dictionary)

    assert path_dictionary_names == {0: {"path": ["SR_2", "SR_1"],
                                         "weight": 2.0},
                                     1: {"path": ["SR_1", "SR_0"],
                                         "weight": 1.5}}

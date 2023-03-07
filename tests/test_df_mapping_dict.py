from typing import Dict, List

import pytest

from utils.functions import df_mapping_dict


@pytest.mark.parametrize("inv", [True, False])
def test_mapping_dict(inv: bool) -> None:
    """This test tests the functionality
    of generating a dictionary out of SR names.
    """
    SR_names: List[str] = ["SR_0",
                           "SR_1",
                           "SR_2",
                           "SR_3"]
    mapping_dict: Dict = df_mapping_dict(
        SR_names=SR_names,
        inv=inv)
    if inv is False:
        assert mapping_dict == {0: "SR_0",
                                1: "SR_1",
                                2: "SR_2",
                                3: "SR_3"}
    else:
        assert mapping_dict == {"SR_0": 0,
                                "SR_1": 1,
                                "SR_2": 2,
                                "SR_3": 3}

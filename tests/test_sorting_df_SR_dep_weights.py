from typing import Any, List

import numpy as np
import pandas as pd

from utils.functions import sort_df_SR_dep_weights


def test_sorting_df_SR_dep_weights() -> None:
    """This test tests the functionality
    of the sorting dataframe with signal regions
    dependent on weights from highest to lowest.
    """
    unsorted_weights: List[Any] = [0.0,
                                   5.2,
                                   2,
                                   0.0,
                                   1]
    unsorted_SR: List[str] = ["SR_0",
                              "SR_1",
                              "SR_2",
                              "SR_3",
                              "SR_4"]
    df: pd.DataFrame = pd.DataFrame(data=np.empty(shape=(5, 5)),
                                    columns=unsorted_SR)
    df_output, weights_output = sort_df_SR_dep_weights(
        df_event_SR=df,
        weights=unsorted_weights)
    print(weights_output)
    assert weights_output == {'SR_1': 5.2,
                              'SR_2': 2,
                              'SR_4': 1,
                              'SR_0': 0.0,
                              'SR_3': 0.0}
    assert list(df_output.columns) == ["SR_1",
                                       "SR_2",
                                       "SR_4",
                                       "SR_0",
                                       "SR_3"]

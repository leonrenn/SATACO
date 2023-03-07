from typing import Dict, List, Optional

import numpy as np
import pytest

from utils.path_finder import PathFinder


@pytest.mark.parametrize("weights",
                         [None, [1, 2, 4]])
@pytest.mark.parametrize("correlations",
                         [np.array([[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]]),
                          np.array([[1, 1, 1],
                                    [1, 1, 1],
                                    [1, 1, 1]]),
                          np.array([[1, 0, 0],
                                    [0.2, 1, 0],
                                    [0, 0, 1]]),
                          np.array([[1, 0, 0],
                                    [0.02, 1, 0],
                                    [0, 0, 1]])])
def test_path_finder(correlations: np.array,
                     weights: Optional[List]) -> None:
    """This test tests the methods of the
    pathfinder class that are used in the
    SATACO project.
    """
    pf: PathFinder = PathFinder(correlations=correlations,
                                weights=weights)
    proposed_paths: Dict = pf.find_path()

    if (correlations == np.eye(3)).all():
        assert proposed_paths[0]["path"] == [0, 1, 2]
        if weights is None:
            assert proposed_paths[0]["weight"] == 3.0
        else:
            assert proposed_paths[0]["weight"] == 7.0
    elif (correlations == np.ones(3)).all():
        if weights is None:
            assert proposed_paths[0]["path"] == [0]
            assert proposed_paths[0]["weight"] == 1.0
        else:
            assert proposed_paths[0]["path"] == [2]
            assert proposed_paths[0]["weight"] == 4.0
    elif (correlations == np.array([[1, 0.2, 0],
                                   [0.2, 1, 0],
                                   [0, 0, 1]])).all():
        if weights is None:
            assert proposed_paths[0]["path"] == [0, 2]
            assert proposed_paths[0]["weight"] == 2.0
        else:
            assert proposed_paths[0]["path"] == [1, 2]
            assert proposed_paths[0]["weight"] == 6.0
    elif (correlations == np.array([[1, 0.02, 0],
                                   [0.02, 1, 0],
                                   [0, 0, 1]])).all():
        assert proposed_paths[0]["path"] == [0, 1, 2]
        if weights is None:

            assert proposed_paths[0]["weight"] == 3.0
        else:
            assert proposed_paths[0]["weight"] == 7.0

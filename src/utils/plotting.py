import pathlib
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def SR_matrix_plotting(SR_SR_matrix: np.array,
                       column_names: List[str],
                       show: bool = False) -> None:
    """Generates a figure of the SR matrix showing
    which SR regions share the same events.

    Args:
        SR_SR_matrix (np.array): SR_SR - matrix in numpy format.
        column_names (List[str]): Names of the SR regions.
        show (bool, optional): Shows figure while 
        running the analysis run. Defaults to False.
    """
    fig, ax = plt.subplots()
    ax.pcolor(SR_SR_matrix, cmap=plt.cm.Blues)
    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(len(column_names)) + 0.5, minor=False)
    ax.set_yticks(np.arange(len(column_names)) + 0.5, minor=False)
    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.xaxis.set_ticks_position('both')
    ax.set_xticklabels(column_names, minor=False)
    ax.set_yticklabels(column_names, minor=False)
    fig.savefig(str(pathlib.Path(__file__).parent.resolve()) +
                "/../../results/SR_SR.png")
    if show is True:
        plt.show()
    return

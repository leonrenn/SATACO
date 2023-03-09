"""Plotting functions for the main program provide a very
handy way to plot different matrices.
"""

import pathlib
from itertools import combinations
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def corr_matrix_plotting(correlation_matrix: np.array,
                         column_names: List[str],
                         threshold: float) -> None:
    """Generates a figure of the correlation matrix showing
    which SR regions correlate.

    Args:
        correlation_matrix (np.array): Pearson Correlation Coeff.
        column_names (List[str]): Names of the SR regions.
    """

    # diagonal elements are painted grey (self correlating SRs),
    # white for SR correlations below threshold and black for above
    correlation_matrix = np.array(correlation_matrix > threshold,
                                  dtype=np.float32)
    # indices of lower triangle and set them to 1
    indices_lower = np.tril_indices_from(correlation_matrix)
    correlation_matrix[indices_lower] = 1
    # set diagonal to 0.5 for better visualization
    np.fill_diagonal(correlation_matrix, 0.5)
    # transpose matrix as in TACO paper
    correlation_matrix = correlation_matrix.T

    font = {'size': 6}

    # using rc function
    plt.rc('font', **font)
    fig, ax = plt.subplots(figsize=(14, 14))

    ax.pcolor(correlation_matrix,
              cmap=plt.cm.Greys)

    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(len(column_names)) + 0.5,
                  minor=False)
    ax.set_yticks(np.arange(len(column_names)) + 0.5,
                  minor=False)
    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.xaxis.set_ticks_position('both')
    ax.set_xticklabels(column_names,
                       minor=False,
                       rotation=45)
    ax.set_yticklabels(column_names,
                       minor=False)

    fig.savefig(str(pathlib.Path(__file__).parent.resolve()) +
                "/../../results/correlations_threshold.png")
    return


def correlation_free_entries_marking(correlation_matrix: np.array,
                                     proposed_paths: Dict,
                                     column_names: List[str],
                                     threshold: float):
    """Generates a figure that holds the binary correlation
    matrix as well as it colors the path of the uncorrolated
    SR in light green color.

    Args:
        Correlation_matrix (np.array): Correlation matrix.
        proposed_paths (Dict): Proposed paths py TACO.
        column_names (List[str]): SR names.
    """
    # diagonal elements are painted grey (self correlating SRs),
    # white for SR correlations below threshold and black for above
    correlation_matrix = np.array(correlation_matrix > threshold,
                                  dtype=np.float32)
    # indices of lower triangle and set them to 1
    indices_lower = np.tril_indices_from(correlation_matrix)
    correlation_matrix[indices_lower] = 1
    # set diagonal to 0.5 for better visualization
    np.fill_diagonal(correlation_matrix, 0.5)
    # transpose matrix as in TACO paper
    correlation_matrix = correlation_matrix.T

    # just one path
    combs = combinations(proposed_paths[0]["path"], r=2)
    for comb in combs:
        comb = sorted(comb, reverse=True)
        correlation_matrix[comb[0], comb[1]] = -1

    cmap = matplotlib.cm.Greys
    cmap.set_under(color='green',
                   alpha=0.4)

    font = {'size': 6}

    # using rc function
    plt.rc('font', **font)
    fig, ax = plt.subplots(figsize=(14, 14))

    ax.pcolor(correlation_matrix,
              cmap=cmap,
              vmin=-0.001)

    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(len(column_names)) + 0.5,
                  minor=False)
    ax.set_yticks(np.arange(len(column_names)) + 0.5,
                  minor=False)
    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.xaxis.set_ticks_position('both')
    ax.set_xticklabels(column_names,
                       minor=False,
                       rotation=45)
    ax.set_yticklabels(column_names,
                       minor=False)

    fig.savefig(str(pathlib.Path(__file__).parent.resolve()) +
                "/../../results/correlations_path.png")
    return

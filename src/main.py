import os
import pathlib
import re
from argparse import ArgumentParser
from itertools import combinations_with_replacement
from typing import Dict, List

import matplotlib.pyplot as plt  # TODO: Plot matrix overlapping
import numpy as np
import pandas as pd
import uproot as ur
from tqdm import tqdm
from uproot.reading import ReadOnlyFile

from exceptions.exceptions import (NonSimpleAnalysisFormat,
                                   SAFileNotFoundError, SAValueError)
from utils.calc_num_combs import calc_num_combs
from utils.df_mapping_dict import df_mapping_dict
from utils.info import info

# MAIN


def main() -> int:
    # 0) START
    print("\t\tSATACO\t\t")
    info()

    # 1) ARGUMENTS FROM CLI
    # parse for arguments from CLI and provide help
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("-r", "--root", type=str, required=True,
                        help=".root files as input from SimpleAnalysis Tool.")
    # parser.add_argument("-d","--droot", type=str, required=True,
    #                    help="Directory with .root files
    #                    from SimpleAnalysis Tool.")
    parser_dict: Dict[str, str] = vars(parser.parse_args())
    # split files in csv format from CLI
    file_paths: List[str] = parser_dict["root"].split(",")
    analysis_names: List[str] = []

    # 2) CHECK FOR EXISTENCE & FOR SA-FORMAT (BEFORE SCAN)
    # the specific output format is specified under
    # https://simpleanalysis.docs.cern.ch in the OUTPUT section and is
    # dependend on the flags that were set for the analysis
    print("Checking correctness of input:\n")
    try:
        for file_path in tqdm(file_paths):
            analysis_name: str = os.path.basename(file_path).strip(".root")
            analysis_names.append(analysis_name)

            if os.path.exists(file_path) is False:
                raise SAFileNotFoundError
            temp: ReadOnlyFile = ur.open(file_path)
            classnames = temp.classnames()
            temp.close()
            if (str(classnames) != "{'ntuple;1': 'TTree'}" and
                    re.search(".*'ntuple;1'.*", str(classnames)) is not None):
                raise NonSimpleAnalysisFormat
    except SAFileNotFoundError:
        print(f"The file {file_path} could not be found. Exit.")
        return 1

    except SAValueError:
        print(f"The file {file_path} could not be read \n"
              "with the uproot python module. Exit.")
        return 2

    except NonSimpleAnalysisFormat:
        print(f"The file {file_path} is in the format of \n"
              f"{str(temp.classnames())}. This is not in the\n"
              "expected SA format [-n] of {'ntuple;1': 'TTree'}.\n"
              "Do not use option '-o' in simpleAnalysis.\n"
              "Exit.")
        return 3

    # 3) RUN ANALYSIS ON INPUT
    # info about which analyses where provided to SATACO
    print("\nAnalyses SATACO runs on:")
    for analysis_name in analysis_names:
        print(f"- {analysis_name}")
    print("\n")

    # 3.1) CONCATENATION OF EVENT SIGNAL REGION MATRICES
    event_SR_matrix_list: List[np.array] = []
    sr_names: List[str] = []

    print("Files preprocessing:\n")
    for file_idx, file_path in enumerate(tqdm(file_paths)):
        # opening the file
        with ur.open(file_path) as file:
            # access to the ntuple structure
            ttree = file["ntuple"]
            # signal regions are the keys of the ttree
            signal_regions = ttree.keys()
            # signal regions with counts of events
            # that passed
            ttree_arrays = ttree.arrays()
            # empty matrix to store data in numpy style
            events = np.empty((len(ttree_arrays), len(signal_regions)))
            # iterating through signal regions to extract the
            # arrays and store row wise
            for sr_idx, sr in enumerate(signal_regions):
                sr_names.append(sr)
                events[:, sr_idx] = ttree_arrays[sr]
            # list of matrices
            event_SR_matrix_list.append(events)

    # concatenate the matrices
    event_SR_matrix_combined: np.array = np.concatenate(
        event_SR_matrix_list, axis=1)
    print(f"\nNumber of events: {event_SR_matrix_combined.shape[0]}\n"
          f"Number of SRs: {event_SR_matrix_combined.shape[1]}.")

    # convert into dataframe
    df_event_SR_matrix_combined: pd.DataFrame = pd.DataFrame(
        data=event_SR_matrix_combined)
    df_event_SR_matrix_combined = df_event_SR_matrix_combined.rename(
        columns=df_mapping_dict(sr_names))

    # print merge results to the console
    print(
        f"\nHead of merged dataframe:\n\n{df_event_SR_matrix_combined.head()}")

    # 3.2) OVERLAP CALCULATION
    # for overlap calculation the events and the eventWeights
    # are deleted
    for _ in analysis_names:
        df_event_SR = df_event_SR_matrix_combined.drop(
            columns=["Event", "eventWeight"])
    # write to csv as a part of the results
    df_event_SR.to_csv(
        str(pathlib.Path(__file__).parent.resolve()) +
        "/../results/event_SR.csv",
        index=False, header=True)
    # combinatorics through the different columns
    # generate combinations with replacement
    column_names: List[str] = df_event_SR.columns
    combs = combinations_with_replacement(column_names, r=2)
    print("\nTo do are a total of "
          f"{calc_num_combs(len(column_names))} combinations.\n")

    SR_SR_matrix: np.array = np.zeros(
        (len(column_names), len(column_names)))
    # create inverse mapping dict for getting access to the
    # index via the signal region name
    inv_mapping: Dict[str, int] = df_mapping_dict(column_names, inv=True)

    # iterate through the combs
    for comb in tqdm(combs):
        # shared event calculates a vector that tells which
        # events actually are accepted for both signal regions
        shared_events: np.array = np.array(
            df_event_SR[comb[0]]*df_event_SR[comb[-1]], dtype=bool)
        i, j = inv_mapping[comb[0]], inv_mapping[comb[1]]
        SR_SR_matrix[i, j] = np.sum(shared_events)
    # print signal region matrix
    print("Signal regions from the Analyses provided are shown in the "
          f" following matrix:\n\n{SR_SR_matrix}.")

    # save in dataframe and save to csv
    df_SR_SR: pd.DataFrame = pd.DataFrame(SR_SR_matrix, columns=column_names)
    df_SR_SR.to_csv(str(pathlib.Path(__file__).parent.resolve()) +
                    "/../results/SR_SR.csv", index=False, header=True)

    # 4) VISUALIZATION
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
    plt.show()
    return 0


if __name__ == "__main__":
    main()

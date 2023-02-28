"""Main Program called from the command line.
"""

import os
import time
from argparse import ArgumentParser
from itertools import combinations_with_replacement
from multiprocessing import Process
from typing import Dict, List

import numpy as np
import pandas as pd
import uproot as ur
from tqdm import tqdm
from uproot.reading import ReadOnlyFile

from exceptions.exceptions import (InvalidArgumentError,
                                   NonSimpleAnalysisFormat,
                                   NoParserArgumentsError, NotARootFile,
                                   SADirectoryNotFoundError,
                                   SAFileNotFoundError, SAValueError,
                                   SAWrongArgument)
from utils.functions import (calc_num_combs, calc_SR_sensitivity,
                             df_mapping_dict, indices_to_SR_names,
                             sort_df_SR_dep_weights, threshold_corr_matrix)
from utils.parsing import build_parser
from utils.path_finder import PathFinder
from utils.plotting import (corr_matrix_plotting,
                            correlation_free_entries_marking)
from utils.preprocessing import preprocess_input
from utils.printer import info, result, sataco, summary
from utils.saving import (clear_result_dir, save_df_corr, save_df_SR_event,
                          save_SR_names)

# MAIN


def main() -> int:
    # 0) START
    STARTTIME: float = time.time()

    # delete all files from result folders
    clear_result_dir()

    sataco()
    info()

    # 1) ARGUMENTS FROM CLI
    # parse for arguments from CLI and provide help

    parser: ArgumentParser = build_parser()
    parser_dict: Dict[str, str] = vars(parser.parse_args())

    try:
        file_paths: List[str] = []
        if parser_dict["root"] is not None and parser_dict["droot"] is None:
            try:
                if os.path.isdir(parser_dict["root"]) is True:
                    raise SAWrongArgument
                # split files in csv format from CLI
                file_paths = parser_dict["root"].split(",")
            except SAWrongArgument:
                print("The argument is a directory '"
                      f"{parser_dict['root']}',\n"
                      "but a file is needed for this flag.")
                return 1
        elif parser_dict["root"] is None and parser_dict["droot"] is not None:
            print(f"Entering given directory: {parser_dict['droot']}")
            try:
                if os.path.exists(parser_dict["droot"]) is True:
                    file_paths = os.listdir(parser_dict["droot"])
                    file_paths = [
                        f"{parser_dict['droot']}/{path}"
                        for path in file_paths]
                else:
                    raise SADirectoryNotFoundError
            except SADirectoryNotFoundError:
                print(f"The directory '{parser_dict['droot']}' "
                      "was not found. Exit.")
                return 2
        elif parser_dict["root"] is None and parser_dict["droot"] is None:
            raise NoParserArgumentsError
    except NoParserArgumentsError:
        print("No arguments were given via the command line, although required"
              "at least one of [-r] or [-d].")
        return 3

    # 2) CHECK FOR EXISTENCE & FOR SA-FORMAT (BEFORE SCAN)
    # the specific output format is specified under
    # https://simpleanalysis.docs.cern.ch in the OUTPUT section and is
    # dependend on the flags that were set for the analysis
    print("Checking correctness of input:\n")
    analysis_names: List[str] = []
    try:
        for file_path in tqdm(file_paths):
            analysis_name: str = os.path.basename(file_path).strip(".root")
            analysis_names.append(analysis_name)

            if os.path.exists(file_path) is False:
                raise SAFileNotFoundError
            try:
                temp: ReadOnlyFile = ur.open(file_path)
            except NotARootFile:
                print(f"The file {file_path} is not in the root format. Exit.")
                return 4
            classnames = temp.classnames()
            temp.close()
            if str(classnames).find("ntuple") == -1:
                raise NonSimpleAnalysisFormat
    except SAFileNotFoundError:
        print(f"The file {file_path} could not be found. Exit.")
        return 5

    except SAValueError:
        print(f"The file {file_path} could not be read \n"
              "with the uproot python module. Exit.")
        return 6

    except NonSimpleAnalysisFormat:
        print(f"The file {file_path} is in the format of \n"
              f"{str(temp.classnames())}. This is not in the\n"
              "expected SA format [-n] of {...'ntuple;_num_': 'TTree'...}.\n"
              "Do not use option '-o' in simpleAnalysis.\n"
              "Exit.")
        return 7

    # 3) RUN ANALYSIS ON INPUT
    # info about which analyses where provided to SATACO
    print("\nAnalyses SATACO runs on:")
    for analysis_name in analysis_names:
        print(f"- {analysis_name}")
    print("\n")

    # 3.1) CONCATENATION OF EVENT SIGNAL REGION MATRICES
    SR_names: List[str]
    event_SR_matrix_combined: np.array

    # preprocess input
    event_SR_matrix_combined, SR_names = preprocess_input(
        analysis_names=analysis_names,
        file_paths=file_paths)

    # convert into dataframe
    df_event_SR_matrix_combined: pd.DataFrame = pd.DataFrame(
        data=event_SR_matrix_combined,
        dtype=np.float32)
    df_event_SR_matrix_combined = df_event_SR_matrix_combined.rename(
        columns=df_mapping_dict(SR_names))

    # 3.2) OVERLAP CALCULATION
    # for overlap calculation the events and the eventWeights
    # are deleted
    df_event_SR = df_event_SR_matrix_combined
    for _ in analysis_names:
        df_event_SR = df_event_SR_matrix_combined.drop(
            columns=["Event", "eventWeight"])

    # write to parquet as a part of the results
    process_save_df_SR_event = Process(
        target=save_df_SR_event,
        args=(df_event_SR,))
    process_save_df_SR_event.start()

    # save signal regions in txt file
    save_SR_names(SR_names=SR_names,
                  zero_cols=False)

    # combinatorics through the different columns
    # generate combinations with replacement
    column_names: List[str] = df_event_SR.columns
    # more efficient combining -> leave out all SR where
    # no events are accepted at all
    print("\nZero columns are removed:\n")
    non_zero_column_names: List[str] = []

    for name in column_names:
        if df_event_SR[name].any():
            non_zero_column_names.append(name)
        else:
            print(f"\t - Removed Signal Region: {name}")
    if len(non_zero_column_names) == 0:
        print("\nNo columns since no columns accepted a single event.")
        return 8
    df_event_SR = df_event_SR[non_zero_column_names]
    # save non zero column names into txt file
    # not necessary to multiprocess because this is fast
    save_SR_names(SR_names=non_zero_column_names,
                  zero_cols=True)
    print(
        "\nRemoved in total: "
        f"{len(column_names) - len(non_zero_column_names)}")

    correlation_matrix: np.array = np.zeros(
        shape=(len(non_zero_column_names), len(non_zero_column_names)),
        dtype=np.float32)

    # before saving rearange matrix with weights
    # generate the weights
    print("\nGenerating weights for hereditary search:\n")
    calculate: bool
    if parser_dict["no_weights"] is True:
        calculate = False
    else:
        calculate = True

    weights_SR: List[float] = calc_SR_sensitivity(df_event_SR=df_event_SR,
                                                  method="simple",
                                                  calculate=calculate)
    # sort dataframe if weights are calculated
    sorted_dict_SR_weigths: Dict = dict(
        zip(non_zero_column_names, [1]*len(non_zero_column_names)))

    if weights_SR is not None:
        df_event_SR, sorted_dict_SR_weigths = sort_df_SR_dep_weights(
            df_event_SR=df_event_SR,
            weights=weights_SR)
        weights_SR = list(sorted_dict_SR_weigths.values())
        # change to sorted version
        non_zero_column_names = df_event_SR.columns.to_list()

    combs = combinations_with_replacement(non_zero_column_names, r=2)
    print("\nTo do are a total of "
          f"{calc_num_combs(len(non_zero_column_names))} combinations.\n")

    # create inverse mapping dict for getting access to the
    # index via the signal region name
    inv_mapping: Dict[str, int] = df_mapping_dict(
        SR_names=non_zero_column_names,
        inv=True)

    # iterate through the combs
    combs = [comb for comb in combs]
    for comb in tqdm(combs):
        i, j = inv_mapping[comb[0]], inv_mapping[comb[1]]
        correlation_matrix[i, j] = np.correlate(
            a=df_event_SR[comb[0]],
            v=df_event_SR[comb[1]])/(np.linalg.norm(df_event_SR[comb[0]]) *
                                     np.linalg.norm(df_event_SR[comb[1]]))
        # fill the full matrix, but the i=j index not twice
        if i != j:
            correlation_matrix[j, i] = correlation_matrix[i, j]

    # save in dataframe and save to parquet
    df_correlation: pd.DataFrame = pd.DataFrame(
        data=correlation_matrix,
        columns=non_zero_column_names)

    process_save_df_corr: Process = Process(
        target=save_df_corr,
        args=(df_correlation,))
    process_save_df_corr.start()

    corr_matrix_binary = threshold_corr_matrix(
        correlation_matrix=correlation_matrix)

    process_corr_matrix_plotting: Process = Process(
        target=corr_matrix_plotting,
        args=(corr_matrix_binary, non_zero_column_names))
    process_corr_matrix_plotting.start()

    # 7) GRAPH ALGORITHM
    # IMPORTANT: These algorithms are taken from the TACO SW.
    # initialize the path finder
    threshold: float
    if parser_dict["threshold"] is None:
        threshold = 0.01
    else:
        try:
            threshold = parser_dict["threshold"]
            if threshold <= 0.00:
                raise InvalidArgumentError

        except InvalidArgumentError:
            print("\nThe argument for the threshold is not valid.")
            return 9

    path_finder: PathFinder = PathFinder(
        correlations=correlation_matrix,
        threshold=threshold,
        source=0,
        weights=weights_SR)

    # start the algorithm to find the best path
    proposed_paths: Dict = path_finder.find_path(top=5)

    # plot the top=1 path into binary matrix
    process_matrix_path_plotting: Process = Process(
        target=correlation_free_entries_marking,
        args=(corr_matrix_binary,
              proposed_paths,
              non_zero_column_names))
    process_matrix_path_plotting.start()

    # change indices to SR names
    proposed_paths_SR_Names: Dict = indices_to_SR_names(
        SR_names=non_zero_column_names,
        path_dictionary=proposed_paths)

    # save the proposed paths to a txt file
    # and print nicely on the command line
    result(proposed_paths=proposed_paths_SR_Names,
           dict_SR_weights=sorted_dict_SR_weigths)

    # 8) JOINING MULTIPROCESSES
    process_save_df_SR_event.join()
    process_save_df_corr.join()
    process_corr_matrix_plotting.join()
    process_matrix_path_plotting.join()
    print("\nAll 4 processes joined.")

    # 9) SUMMARY
    summary(STARTTIME=STARTTIME)
    return 0


if __name__ == "__main__":
    main()

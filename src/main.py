"""
Main Program called from the command line.
"""
import time
from argparse import ArgumentParser
from itertools import combinations_with_replacement
from multiprocessing import Process
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from exceptions.exceptions import (InvalidArgumentError,
                                   NonSimpleAnalysisFormat,
                                   NoParserArgumentsError, NotARootFile,
                                   NotEnoughStatistcis,
                                   SADirectoryNotFoundError,
                                   SAFileNotFoundError, SAWrongArgument)
from utils.functions import (calc_num_combs, calc_SR_sensitivity,
                             check_sufficient_statistics, df_mapping_dict,
                             indices_to_SR_names, sort_df_SR_dep_weights)
from utils.parsing import build_parser, check_parser_input_files
from utils.path_finder import PathFinder
from utils.plotting import (corr_matrix_plotting,
                            correlation_free_entries_marking)
from utils.preprocessing import check_for_input_correctness, preprocess_input
from utils.printer import info, result, sataco, summary
from utils.saving import (clear_result_dir, save_df_corr, save_df_SR_event,
                          save_SR_names)


# MAIN
def main() -> int:
    # 0) START
    STARTTIME: float = time.time()

    # delete all files from result folders
    clear_result_dir()

    # print sataco letters on CLI
    sataco()
    # print info on CLI
    info()

    # 1) ARGUMENTS FROM CLI

    # 1.1) PARSE ARGUMENTS FROM CLI
    parser: ArgumentParser = build_parser()
    parser_dict: Dict[str, str] = vars(parser.parse_args())

    # 1.2) CHECK FOR NECESSARY INPUT FILES IN PARSER
    file_paths: List[str]
    try:
        file_paths = check_parser_input_files(parser_dict=parser_dict)
    except SAWrongArgument:
        print("The argument is a directory '"
              f"{parser_dict['root']}',\n"
              "but a file is needed for this flag.")
        return 1
    except SADirectoryNotFoundError:
        print(f"The directory '{parser_dict['droot']}' "
              "was not found. Exit.")
        return 2
    except NoParserArgumentsError:
        print("No arguments were given via the command line, although"
              " required at least one of [-r] or [-d].")
        return 3

    # 2) CHECK FOR EXISTENCE & FOR SA-FORMAT
    # the specific output format is specified under
    # https://simpleanalysis.docs.cern.ch in the OUTPUT section and is
    # dependend on the flags that were set for the analysis
    print("Checking correctness of input:\n")
    analysis_names: List[str] = []
    try:
        analysis_names = check_for_input_correctness(file_paths=file_paths)
    except NotARootFile:
        return 4
    except SAFileNotFoundError:
        return 5
    except NonSimpleAnalysisFormat:
        return 6

    # 3) RUN ANALYSIS ON INPUT

    # info about which analyses where provided to SATACO
    print("\nAnalyses SATACO runs on:")
    for analysis_name in analysis_names:
        print(f"- {analysis_name}")
    print("\n")

    # 3.1) CONCATENATION OF EVENT SIGNAL REGION MATRICES

    # preprocess input
    SR_names: List[str]
    event_SR_matrix_combined: np.array
    event_SR_matrix_combined, SR_names = preprocess_input(
        analysis_names=analysis_names,
        file_paths=file_paths)

    # convert into dataframe
    df_event_SR_matrix_combined: pd.DataFrame = pd.DataFrame(
        data=event_SR_matrix_combined,
        dtype=np.float32)
    # rename columns of dataframe
    df_event_SR_matrix_combined = df_event_SR_matrix_combined.rename(
        columns=df_mapping_dict(SR_names))

    # 3.2) OVERLAP CALCULATION

    # for overlap calculation the events and the eventWeights
    # are remove from dataframe
    df_event_SR = df_event_SR_matrix_combined
    for _ in analysis_names:
        df_event_SR = df_event_SR_matrix_combined.drop(
            columns=["Event", "eventWeight"])

    # write to parquet as part of the results
    process_save_df_SR_event = Process(
        target=save_df_SR_event,
        args=(df_event_SR,))
    process_save_df_SR_event.start()

    # save opensignal regions in txt file
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

    # alternatively:
    # df_event_SR = df_event_SR.loc[:, (df_event_SR != 0).any(axis=0)]
    # non_zero_column_names = list(df_event_SR.columns)

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

    combs = combinations_with_replacement(iterable=non_zero_column_names,
                                          r=2)
    print("\nTo do are a total of "
          f"{calc_num_combs(len(non_zero_column_names))} combinations.\n")

    # create inverse mapping dict for getting access to the
    # index via the signal region name
    inv_mapping: Dict[str, int] = df_mapping_dict(
        SR_names=non_zero_column_names,
        inv=True)

    # get threshold from command line
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

    # iterate through the combs
    combs = [comb for comb in combs]
    i: int
    j: int
    vec_i: np.array
    vec_j: np.array
    for comb in tqdm(combs):
        i, j = inv_mapping[comb[0]], inv_mapping[comb[1]]
        vec_i, vec_j = df_event_SR[comb[0]], df_event_SR[comb[1]]
        correlation_matrix[i, j] = np.dot(
            a=vec_i,
            b=vec_j)/(np.linalg.norm(vec_i) *
                      np.linalg.norm(vec_j))
        if parser_dict["statistics"] is True:
            try:
                check_sufficient_statistics(vec_i=vec_i,
                                            vec_j=vec_j,
                                            event_num=df_event_SR.shape[0],
                                            threshold=threshold)
            except NotEnoughStatistcis:
                print("Not enough statistics to make overlapping"
                      " calculations. Exit.")
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

    process_corr_matrix_plotting: Process
    if parser_dict["no_plots"] is not True:
        process_corr_matrix_plotting: Process = Process(
            target=corr_matrix_plotting,
            args=(correlation_matrix,
                  non_zero_column_names,
                  threshold))
        process_corr_matrix_plotting.start()

    # 7) GRAPH ALGORITHM
    # IMPORTANT: These algorithms are taken from the TACO SW.
    # initialize the path finder
    path_finder: PathFinder = PathFinder(
        correlations=correlation_matrix,
        threshold=threshold,
        source=0,
        weights=weights_SR)

    # start the algorithm to find the best path
    proposed_paths: Dict
    if parser_dict["top_paths"] is not None:
        proposed_paths: Dict = path_finder.find_path(
            top=parser_dict["top_paths"])
    else:
        proposed_paths: Dict = path_finder.find_path(
            top=1)

    process_matrix_path_plotting: Process
    if parser_dict["no_plots"] is not True:
        # plot the top=1 path into binary matrix
        process_matrix_path_plotting: Process = Process(
            target=correlation_free_entries_marking,
            args=(correlation_matrix,
                  proposed_paths,
                  non_zero_column_names,
                  threshold))
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

    if parser_dict["no_plots"] is not True:
        process_corr_matrix_plotting.join()
        process_matrix_path_plotting.join()
        print("\nAll 4 processes joined.")
    else:
        print("\nAll 2 processes joined.")

    # 9) SUMMARY
    summary(STARTTIME=STARTTIME)
    return 0


if __name__ == "__main__":
    main()

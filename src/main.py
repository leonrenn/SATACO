"""Main Program called from the command line.
"""

import os
import time
from argparse import ArgumentParser
from itertools import combinations_with_replacement
from multiprocessing import Process, cpu_count
from typing import Dict, List

import numpy as np
import pandas as pd
import uproot as ur
from tqdm import tqdm
from uproot.reading import ReadOnlyFile

from exceptions.exceptions import (NonSimpleAnalysisFormat,
                                   NoParserArgumentsError, NotARootFile,
                                   NotEnoughStatistcis,
                                   SADirectoryNotFoundError,
                                   SAFileNotFoundError, SAValueError,
                                   SAWrongArgument)
from utils.functions import (calc_num_combs, calc_pearson_corr,
                             calc_SR_sensitivity, check_sufficient_statistics,
                             df_mapping_dict, indices_to_SR_names,
                             threshold_corr_matrix)
from utils.path_finder import PathFinder
from utils.plotting import (SR_matrix_plotting, corr_matrix_plotting,
                            correlation_free_entries_marking)
from utils.printer import info, result, sataco, summary
from utils.saving import (save_df_corr, save_df_SR_event, save_df_SR_SR,
                          save_sr_names)

# MAIN


def main() -> int:
    # 0) START
    STARTTIME = time.time()
    num_cpu: int = cpu_count()

    sataco()
    info()

    # 1) ARGUMENTS FROM CLI
    # parse for arguments from CLI and provide help

    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("-r",
                        "--root",
                        type=str, required=False,
                        help=".root files as input from "
                        "SimpleAnalysis Tool.")

    parser.add_argument("-d",
                        "--droot",
                        type=str,
                        required=False,
                        help="Directory with .root files"
                        "from SimpleAnalysis Tool.")
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
    event_SR_matrix_list: List[np.array] = []
    sr_names: List[str] = []

    print("Files preprocessing:\n")
    # TODO: Think about muliprocessing
    for _, file_path in enumerate(tqdm(file_paths)):
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
            events: np.array = np.empty(
                shape=(len(ttree_arrays), len(signal_regions)),
                dtype=np.float32)
            # iterating through signal regions to extract the
            # arrays and store row wise
            for sr_idx, sr in enumerate(signal_regions):
                sr_names.append(sr)
                events[:, sr_idx] = np.array(
                    ttree_arrays[sr], dtype=np.float32)
            # list of matrices
            event_SR_matrix_list.append(events)

    # concatenate the matrices
    event_SR_matrix_combined: np.array = np.concatenate(
        event_SR_matrix_list,
        axis=1,
        dtype=np.float32)
    print(f"\nNumber of events: {event_SR_matrix_combined.shape[0]}\n"
          "Number of SRs: "
          f"{event_SR_matrix_combined.shape[1] - len(analysis_names)*2}.")

    # convert into dataframe
    # TODO: This takes a lot of time ... -> Improve
    df_event_SR_matrix_combined: pd.DataFrame = pd.DataFrame(
        data=event_SR_matrix_combined,
        dtype=np.float32)
    df_event_SR_matrix_combined = df_event_SR_matrix_combined.rename(
        columns=df_mapping_dict(sr_names))

    # 3.2) OVERLAP CALCULATION
    # for overlap calculation the events and the eventWeights
    # are deleted
    for _ in analysis_names:
        df_event_SR = df_event_SR_matrix_combined.drop(
            columns=["Event", "eventWeight"])

    # write to parquet as a part of the results
    process_save_df_SR_event = Process(
        target=save_df_SR_event,
        args=(df_event_SR,))
    process_save_df_SR_event.start()

    # save signal regions in txt file
    save_sr_names(sr_names=sr_names,
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
    # save non zero column names into txt file
    # not necessary to multiprocess because this is fast
    save_sr_names(sr_names=non_zero_column_names,
                  zero_cols=True)
    print(
        "\nRemoved in total: "
        f"{len(column_names) - len(non_zero_column_names)}")

    combs = combinations_with_replacement(non_zero_column_names, r=2)
    print("\nTo do are a total of "
          f"{calc_num_combs(len(non_zero_column_names))} combinations.\n")

    SR_SR_matrix: np.array = np.zeros(
        shape=(len(non_zero_column_names), len(non_zero_column_names)),
        dtype=np.float32)

    # create inverse mapping dict for getting access to the
    # index via the signal region name
    inv_mapping: Dict[str, int] = df_mapping_dict(
        SR_names=non_zero_column_names,
        inv=True)

    # iterate through the combs
    combs = [comb for comb in combs]
    for comb in tqdm(combs):
        # shared event calculates a vector that tells which
        # events actually are accepted for both signal regions
        # devide by two because take mean of both events
        shared_events: np.array = np.array(
            object=df_event_SR[comb[0]] * df_event_SR[comb[-1]],
            dtype=bool) *\
            np.array(
                object=df_event_SR[comb[0]] + df_event_SR[comb[-1]],
                dtype=np.float32) * 0.5
        i, j = inv_mapping[comb[0]], inv_mapping[comb[1]]
        SR_SR_matrix[i, j] = np.sum(a=shared_events)
        # fill the full matrix, but the i=j index not twice
        if i != j:
            SR_SR_matrix[j, i] = SR_SR_matrix[i, j]

    # save in dataframe and save to parquet
    df_SR_SR: pd.DataFrame = pd.DataFrame(
        data=SR_SR_matrix,
        columns=non_zero_column_names)

    save_df_SR_SR(df_SR_SR=df_SR_SR)
    process_save_df_SR_SR: Process = Process(
        target=save_df_SR_SR,
        args=(df_SR_SR,))
    process_save_df_SR_SR.start()

    # 4) VISUALIZATION
    process_SR_matrix_plotting: Process = Process(
        target=SR_matrix_plotting,
        args=(SR_SR_matrix, non_zero_column_names))
    process_SR_matrix_plotting.start()

    # 5) SUFFICIENT NUMBER OF EVENTS CHECK
    # -> ACCEPTANCE MATRIX
    try:
        check_sufficient_statistics(
            SR_SR_matrix=SR_SR_matrix,
            event_num=df_event_SR_matrix_combined.shape[0])
    except NotEnoughStatistcis:
        print("The accepatance matrix has not had enough entries \n"
              "for validating an overlap.\n"
              "Exit.")
        return 8

    # 6) CALCULATION OF PEARSON COEFFICIENT AND CUTTING
    pearson_coeff: np.array = calc_pearson_corr(SR_SR_matrix=SR_SR_matrix)
    # save in dataframe and save to csv
    df_corr: pd.DataFrame = pd.DataFrame(
        data=pearson_coeff,
        columns=non_zero_column_names)

    process_save_df_corr: Process = Process(
        target=save_df_corr,
        args=(df_corr,))
    process_save_df_corr.start()

    # -> OVERLAP MATRIX
    corr_matrix_binary = threshold_corr_matrix(
        correlation_matrix=pearson_coeff)

    process_corr_matrix_plotting: Process = Process(
        target=corr_matrix_plotting,
        args=(corr_matrix_binary, non_zero_column_names))
    process_corr_matrix_plotting.start()

    # 7) GRAPH ALGORITHM
    # IMPORTANT: These algorithms are taken from the TACO SW.

    # generate the weights
    print("\nGenerating weights for hererditary search:\n")
    weights_SR: List[float] = calc_SR_sensitivity(df_SR_events=df_event_SR,
                                                  method="simple",
                                                  calculate=True)

    # initialize the path finder
    path_finder: PathFinder = PathFinder(corelations=pearson_coeff,
                                         threshold=0.01,
                                         source=0,
                                         weights=weights_SR)
    # start the algorithm to find the best path
    proposed_paths: Dict = path_finder.find_path(top=3)

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

    result(best_SR_comb=proposed_paths_SR_Names)

    # 8) JOINING MULTIPROCESSES
    process_save_df_SR_event.join()
    process_save_df_SR_SR.join()
    process_save_df_corr.join()
    process_SR_matrix_plotting.join()
    process_corr_matrix_plotting.join()
    process_matrix_path_plotting.join()
    print("\nAll 6 processes joined.")

    # 9) SUMMARY
    summary(STARTTIME=STARTTIME)
    return 0


if __name__ == "__main__":
    main()

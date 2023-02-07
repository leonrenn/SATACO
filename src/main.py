import os
import pathlib
from argparse import ArgumentParser
from itertools import combinations_with_replacement
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
from utils.functools import (calc_num_combs, calc_pearson_corr,
                             check_sufficient_statistics, df_mapping_dict)
from utils.plotting import SR_matrix_plotting
from utils.printer import info, sataco, summary

# MAIN


def main() -> int:
    # 0) START
    sataco()
    info()

    # 1) ARGUMENTS FROM CLI
    # parse for arguments from CLI and provide help

    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("-r", "--root", type=str, required=False,
                        help=".root files as input from SimpleAnalysis Tool.")

    parser.add_argument("-d", "--droot", type=str, required=False,
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
            if str(classnames).find("'ntuple;1': 'TTree'") == -1:
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
              "expected SA format [-n] of {'ntuple;1': 'TTree'}.\n"
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
    # more efficient combining -> leave out all SR where no events are accepted
    # at all
    print("\nZero columns where removed from the analysis:\n")
    non_zero_column_names: List[str] = []

    for name in column_names:
        if df_event_SR[name].any():
            non_zero_column_names.append(name)
        else:
            print(f"\t - Removed Signal Region: {name}")

    combs = combinations_with_replacement(non_zero_column_names, r=2)
    print("\nTo do are a total of "
          f"{calc_num_combs(len(non_zero_column_names))} combinations.\n")

    SR_SR_matrix: np.array = np.zeros(
        (len(non_zero_column_names), len(non_zero_column_names)))
    # create inverse mapping dict for getting access to the
    # index via the signal region name
    inv_mapping: Dict[str, int] = df_mapping_dict(
        non_zero_column_names, inv=True)

    # iterate through the combs
    for comb in tqdm(combs, leave=True):
        # shared event calculates a vector that tells which
        # events actually are accepted for both signal regions
        shared_events: np.array = np.array(
            df_event_SR[comb[0]]*df_event_SR[comb[-1]], dtype=bool) * np.array(
            df_event_SR[comb[0]] + df_event_SR[comb[-1]], dtype=float)
        i, j = inv_mapping[comb[0]], inv_mapping[comb[1]]
        SR_SR_matrix[i, j] = np.sum(shared_events)
        if i != j:
            SR_SR_matrix[j, i] = np.sum(shared_events)
    # print signal region matrix
    print("Signal regions from the Analyses provided are shown in the "
          f" following matrix:\n\n{SR_SR_matrix}.")

    # save in dataframe and save to csv
    df_SR_SR: pd.DataFrame = pd.DataFrame(
        SR_SR_matrix, columns=non_zero_column_names)
    df_SR_SR.to_csv(str(pathlib.Path(__file__).parent.resolve()) +
                    "/../results/SR_SR.csv", index=False, header=True)

    # 4) VISUALIZATION
    SR_matrix_plotting(SR_SR_matrix=SR_SR_matrix,
                       column_names=non_zero_column_names)
    # 5) SUFFICIENT NUMBER OF EVENTS CHECK
    try:
        check_sufficient_statistics(
            SR_SR_matrix=SR_SR_matrix,
            event_num=df_event_SR_matrix_combined.shape[0])
    except NotEnoughStatistcis:
        print("The accepatance matrix has not had enough entries \n"
              "for validating an overlap.\n"
              "Exit.")
        return 8

    # 6) CALCULATION OF PEARSON COEFFICIENT
    pearson_coeff: np.array = calc_pearson_corr(SR_SR_matrix)
    # TODO: Specify with event bins and event weights
    # a later apply cut to the pearson coefficients

    # 7) PATH FINDING AS COMBINATION
    # TODO: Implement path finding algortihm from
    # TACO.

    # 8) SUMMARY
    summary()
    return 0


if __name__ == "__main__":
    main()

"""
Provides functions for the main program that are used for checking
and preprocessing the input.
"""
import os
import pathlib
from typing import Dict, List

import numpy as np
import uproot as ur
from tqdm import tqdm
from uproot.reading import ReadOnlyFile
from yaspin import yaspin
from yaspin.spinners import Spinners

from exceptions.exceptions import (NonPreselectionInfoFound,
                                   NonSimpleAnalysisFormat, NotARootFile,
                                   SAFileNotFoundError)
from utils.preselecting import filter_SRs


def check_for_input_correctness(file_paths: List[str]) -> List[str]:
    """Checks for correct formats of provided input
    and returns a list of analysis names

    Args:
        file_paths (List[str]): Paths to files that should be analyzed.

    Returns:
        List[str]: Analysis names.
    """
    analysis_names: List[str] = []
    for file_path in tqdm(file_paths):
        analysis_name: str = os.path.basename(file_path).strip(".root")

        analysis_names.append(analysis_name)

        if os.path.exists(file_path) is False:
            print(f"The file {file_path} could not be found. Exit.")
            raise SAFileNotFoundError

        try:
            temp: ReadOnlyFile = ur.open(file_path)
        except ValueError:
            print(f"The file {file_path} is not in the root format. Exit.")
            raise NotARootFile

        classnames = temp.classnames()
        temp.close()
        if str(classnames).find("ntuple") == -1:
            print(f"The file {file_path} is in the format of \n"
                  f"{str(temp.classnames())}. This is not in the\n"
                  "expected SA format [-n] of "
                  "{...'ntuple;_num_': 'TTree'...}.\n"
                  "Do not use option '-o' in simpleAnalysis.\n"
                  "Exit.")
            raise NonSimpleAnalysisFormat

    return analysis_names


def preprocess_input(analysis_names: List[str],
                     file_paths: List[str],
                     parser_dict: Dict):
    """Store data from files into vectors that are later
    transformed to dataframes

    Args:
        analysis_names (List[str]): Names of analyses.
        file_paths (List[str]): Paths to the analyzed files.

    Returns:
        _type_: Event SR matrix, signal region names, weights.
    """
    SR_names: List[str] = []
    SR_weights: List[float] = []
    approved_SR_names: List[str] = None
    approved_weights: List[float] = None

    # list for storing events of corresponding SR
    event_SR_matrix_list: List[np.array] = []
    print("Files preprocessing:\n")
    for idx, file_path in enumerate(tqdm(file_paths,
                                         position=0,
                                         leave=True)):

        # if preselecting is wanted
        if parser_dict["preselecting"] is not None:
            if os.path.isdir(str(pathlib.Path(__file__).parent.resolve()) +
                             "/../../" +
                             parser_dict["preselecting"]):

                approved_SR_names, approved_weights = filter_SRs(
                    analysis_name=analysis_names[idx],
                    path=parser_dict["preselecting"])
            else:
                print("There is no preselection info under "
                      f"{str(pathlib.Path(__file__).parent.resolve())}"
                      "/../../"
                      f"{parser_dict['preselecting']}")
                raise NonPreselectionInfoFound
        # opening the file
        with ur.open(file_path) as file:
            # access to the ntuple structure
            ttree = file["ntuple"]
            # signal regions are the keys of the ttree
            signal_regions = list(ttree.keys())

            # signal regions with counts of events
            # that passed
            if approved_SR_names is not None:
                # use only approved SR (by further analysis)
                signal_regions = approved_SR_names
                SR_weights += approved_weights
            else:
                signal_regions.remove("Event")
                signal_regions.remove("eventWeight")
            SR_names += signal_regions

            if signal_regions != []:
                # only accept signal regions if sr list in info files
                # are not []
                ttree_arrays = ttree.arrays(signal_regions)
            else:
                print(f"No signal regions from {file_path} are accepted.")

            # empty matrix to store data in numpy style
            events: np.array = np.empty(
                shape=(len(ttree_arrays), len(signal_regions)),
                dtype=np.float32)
            # iterating through signal regions to extract the
            # arrays and store row wise
            for sr_idx, sr in enumerate(signal_regions):
                events[:, sr_idx] = np.array(
                    ttree_arrays[sr], dtype=np.float32)
            # list of matrices
            event_SR_matrix_list.append(events)

    # concatenate the matrices
    with yaspin(Spinners.earth, text="Concatenate of SRs.") as spinner:
        event_SR_matrix_combined: np.array = np.concatenate(
            event_SR_matrix_list,
            axis=1,
            dtype=np.float32)
        spinner.ok("\nConcatenation of SRs successfully.")
    print(f"\nNumber of events: {event_SR_matrix_combined.shape[0]}\n"
          "Number of SRs: "
          f"{event_SR_matrix_combined.shape[1]}.")

    return event_SR_matrix_combined, SR_names, SR_weights

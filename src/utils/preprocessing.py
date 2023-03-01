"""
Provides functions for the main program that are used for checking
and preprocessing the input.
"""
import os
from typing import List

import numpy as np
import uproot as ur
from tqdm import tqdm
from uproot.reading import ReadOnlyFile

from exceptions.exceptions import (NonSimpleAnalysisFormat, NotARootFile,
                                   SAFileNotFoundError)


def check_for_input_correctness(file_paths: List[str]) -> List[str]:
    """Checks for correct formats of provided input
    and returns a list of analysis names

    Args:
        file_paths (List[str]): Paths to files that should be analyzed.

    Raises:
        SAFileNotFoundError: File was not found.
        NotARootFile: File is not a root file.
        NonSimpleAnalysisFormat: File has not SA format.

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
                     file_paths: List[str]):
    """Store data from files into vectors that are later
    transformed to dataframes

    Args:
        analysis_names (List[str]): Names of analyses.
        file_paths (List[str]): Paths to the analyzed files.

    Returns:
        _type_: Event SR matrix and signal region names.
    """
    # list for storing events of corresponding SR
    event_SR_matrix_list: List[np.array] = []
    # names of the signal regions
    SR_names: List[str] = []
    print("Files preprocessing:\n")
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
                SR_names.append(sr)
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

    return event_SR_matrix_combined, SR_names

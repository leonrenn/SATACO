import os
import re
from argparse import ArgumentParser
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import uproot as ur
from tqdm import tqdm
from uproot.reading import ReadOnlyFile

from exceptions.exceptions import (NonSimpleAnalysisFormat,
                                   SAFileNotFoundError, SAValueError)
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

    # 3.1) For analyses that are not combined
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
                events[:, sr_idx] = ttree_arrays[sr]

    return 0


if __name__ == "__main__":
    main()

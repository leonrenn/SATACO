import os
from argparse import ArgumentParser
from typing import Dict, List

import uproot as ur
from tqdm import tqdm
from uproot.reading import ReadOnlyFile

from exceptions.exceptions import NonSimpleAnalysisFormat
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

    # 2) CHECK FOR EXISTENCE & FOR SA-FORMAT (BEFORE SCAN)
    # the specific output format is specified under
    # https://simpleanalysis.docs.cern.ch in the OUTPUT section and is
    # dependend on the flags that were set for the analysis
    print("Checking correctness of input:\n")
    try:
        for file_path in tqdm(file_paths):
            if os.path.exists(file_path) is False:
                raise FileNotFoundError
            temp: ReadOnlyFile = ur.open(file_path)
            classnames: str = temp.classnames()
            temp.close()
            if str(classnames) != "{'ntuple;1': 'TTree'}":
                raise NonSimpleAnalysisFormat
    except FileNotFoundError:
        print(f"The file {file_path} could not be found. Exit.")
        return 1

    except ValueError:
        print(f"The file {file_path} could not be read "
              "with the uproot python module. Exit.")
        return 2

    except NonSimpleAnalysisFormat:
        print(f"The file {file_path} is in the format of "
              f"{str(temp.classnames())}. This is not in the"
              " expected SA format of {'ntuple;1': 'TTree'}."
              "Exit.")
        return 3

    return 0


if __name__ == "__main__":
    main()

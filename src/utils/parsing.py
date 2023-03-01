"""
Building the parser for the main program.
"""
import os
from argparse import ArgumentParser
from typing import Dict, List

from exceptions.exceptions import (NoParserArgumentsError,
                                   SADirectoryNotFoundError, SAWrongArgument)


def build_parser() -> ArgumentParser:
    """Build parser for the main program.

    Returns:
        ArgumentParser: Parser.
    """
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
    parser.add_argument("-t",
                        "--threshold",
                        type=float,
                        required=False,
                        help="Threshold for "
                        "generating binary "
                        "correlation matrix.")
    parser.add_argument("-nw",
                        "--no_weights",
                        required=False,
                        action="store_true",
                        help="Force that all signal regions"
                        " are treated equally in "
                        "significance.")

    return parser


def check_parser_input_files(parser_dict: Dict[str, str]) -> List[str]:
    file_paths: List[str] = []
    if parser_dict["root"] is not None and parser_dict["droot"] is None:
        if os.path.isdir(parser_dict["root"]) is True:
            raise SAWrongArgument
        # split files in csv format from CLI
        file_paths = parser_dict["root"].split(",")
    elif parser_dict["root"] is None and parser_dict["droot"] is not None:
        if os.path.exists(parser_dict["droot"]) is True:
            print(f"Entering given directory: {parser_dict['droot']}")
            try:
                file_paths = os.listdir(parser_dict["droot"])
            except NotADirectoryError:
                raise SADirectoryNotFoundError
            file_paths = [
                f"{parser_dict['droot']}/{path}"
                for path in file_paths]
        else:
            raise SADirectoryNotFoundError
    elif parser_dict["root"] is None and parser_dict["droot"] is None:
        raise NoParserArgumentsError
    return file_paths

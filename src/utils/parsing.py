"""Building the parserfor the main program.
"""
from argparse import ArgumentParser


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

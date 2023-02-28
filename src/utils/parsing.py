"""Building the parserfor the main program.
"""
from argparse import ArgumentParser


def build_parser() -> ArgumentParser:
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
    return parser

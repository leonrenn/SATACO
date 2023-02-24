"""Printing functions print informations stored in
different files to the console.
"""

import json
import pathlib
from datetime import timedelta
from time import time
from typing import Dict


def sataco() -> None:
    """Prints SATACO letters from sataco.txt file.
    """
    sataco_file_path: str = str(pathlib.Path(
        __file__).parent.resolve()) + "/sataco.txt"
    sataco_file = open(sataco_file_path)
    sataco_content = sataco_file.read()
    sataco_file.close()
    print(sataco_content)
    return


def info() -> None:
    """Prints info on the command line stored
    in the info.txt file.
    """
    info_file_path: str = str(pathlib.Path(
        __file__).parent.resolve()) + "/info.txt"
    info_file = open(info_file_path)
    info_content = info_file.read()
    info_file.close()
    print(info_content)
    return


def result(best_SR_comb: Dict,
           dict_SR_weights: Dict = None,
           top: int = -1) -> None:
    """Prints best combination on the command line
    and writes results into dedicated file.

    Args:
        best_SR_comb (Dict): Best SR combinations from
        HDFS graph algrotihm.
    """
    result_file_path: str = str(pathlib.Path(
        __file__).parent.resolve()) + "/../../results/best_SR_comb.txt"
    print("\n---------------------------------------")
    print("\nThe best combination of SRs is:\n")
    counter: int = 0
    with open(result_file_path, "w") as result_file:
        for paths, summed_weight in best_SR_comb.items():
            print(
                f"Path weight = {summed_weight}: ",
                end="")
            result_file.write(
                str(counter) + " (" +
                str(summed_weight) + "): ")
            # TODO: very ugly reconverting
            SR_from_string = [k.strip(" '").strip("'")
                              for k in paths.strip("[]").split(",")]
            for SR in SR_from_string:
                result_file.write(
                    SR + " (weight: " +
                    str(dict_SR_weights.get(SR)) + "), ")

            print(*paths, sep=", ", end=".\n")
            result_file.write("\n")
            counter += 1
            if counter == top:
                break
    print(f"\nSaved under:\n{result_file_path}.\n ")
    print("---------------------------------------")
    return


def summary(STARTTIME: float) -> None:
    """Prints summary on the command line stores
    in the summary.txt file and program duration.

    Args:
        STARTTIME (float): Unix time the program started.
    """
    summary_file_path: str = str(pathlib.Path(
        __file__).parent.resolve()) + "/summary.txt"
    summary_file = open(summary_file_path)
    summary_content = summary_file.read()
    summary_file.close()
    print(summary_content)
    print("Process finished after: "
          f"{timedelta(seconds=time()-STARTTIME)}.")
    return

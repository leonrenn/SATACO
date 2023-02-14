import pathlib
from datetime import timedelta
from time import time
from typing import List


def sataco() -> None:
    """Prints SATACO letters from sataco.txt file.
    """
    sataco_file_path = str(pathlib.Path(
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
    info_file_path = str(pathlib.Path(
        __file__).parent.resolve()) + "/info.txt"
    info_file = open(info_file_path)
    info_content = info_file.read()
    info_file.close()
    print(info_content)
    return


def result(best_SR_comb: List[str]) -> None:
    """Prints best combination ton the command line
    and writes results into dedicated file.

    Args:
        best_SR_comb (List[str]): Best SR combinations from
        HDFS graph algrotihm.
    """
    result_file_path = str(pathlib.Path(
        __file__).parent.resolve()) + "/../best_SR_comb.txt"
    with open(result_file_path, "w") as result_file:
        for SR in best_SR_comb:
            result_file.write(SR + "\n")

    print("---------------------------------------")
    print("The best combination of SRs is:\n")
    print(*best_SR_comb, sep=" ", end=".\n")
    print(f"Saved under: \n {result_file_path}.")
    print("---------------------------------------")
    return


def summary(STARTTIME: float) -> None:
    """Prints summary on the command line stored
    in the summary.txt file and program duration.

    Args:
        STARTTIME (float): Unix time the program started.
    """
    summary_file_path = str(pathlib.Path(
        __file__).parent.resolve()) + "/summary.txt"
    summary_file = open(summary_file_path)
    summary_content = summary_file.read()
    summary_file.close()
    print(summary_content)
    print("Process finished after: "
          f"{timedelta(seconds=time()-STARTTIME)}.")
    return

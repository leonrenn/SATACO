import pathlib


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


def summary() -> None:
    """Prints summary on the command line stored
    in the summary.txt file.
    """
    summary_file_path = str(pathlib.Path(
        __file__).parent.resolve()) + "/summary.txt"
    summary_file = open(summary_file_path)
    summary_content = summary_file.read()
    summary_file.close()
    print(summary_content)
    return


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

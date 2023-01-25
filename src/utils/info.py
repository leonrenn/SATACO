import pathlib


def info() -> None:
    info_file_path = str(pathlib.Path(__file__).parent.resolve()) + "/info.txt"
    info_file = open(info_file_path)
    info_content = info_file.read()
    info_file.close()
    print(info_content)
    return

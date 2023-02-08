from networkx.exception import NetworkXUnfeasible


class NonSimpleAnalysisFormat(Exception):
    """Raised when the input file is not of the
    output of the SimpleAnalysis tool."""
    pass


class SAFileNotFoundError(FileNotFoundError):
    """Raised when the files are not found 
    at the beginning of the SATACO programm.
    """
    pass


class SAValueError(ValueError):
    """Raised when files cannot be read because
    they are in the wrong format.
    """
    pass


class NoParserArgumentsError(Exception):
    """Raised when no arguments are given through 
    the command line."""
    pass


class SADirectoryNotFoundError(NotADirectoryError):
    """Raised when the directory with the root files
    are not found at the beginning of the SATACO programm.
    """
    pass


class SAWrongArgument(IsADirectoryError):
    """Raised when the argument is a directory but a
    file is needed for the used [-r] flag.
    """
    pass


class NotARootFile(ValueError):
    """Raised when the file that uproot is trying
    to read is not a root file.
    """
    pass


class NotEnoughStatistcis(Exception):
    """Raised when the acceptance matrix has not
    aquired enough statistics. (Maybe delete exception
    because analysis must continue in either way)"""
    pass


class NoGraphSolution(NetworkXUnfeasible):
    """Raised when the algorithm for longest path in DAG
    does not find a solution or graph is incorrect.
    """

"""
Exceptions for the main program.
"""


class NonSimpleAnalysisFormat(Exception):
    """Raised when the input file is not of the
    output of the SimpleAnalysis tool."""
    pass


class SAFileNotFoundError(FileNotFoundError):
    """Raised when the files are not found
    at the beginning of the SATACO programm.
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


class InvalidArgumentError(Exception):
    """Invalid arguments as threshold.
    """


class CorrelationMatrixFormatError(Exception):
    """Raised when the correlation matrix 
    is not in the correct format in the txt file.
    """


class NonPreselectionInfoFound(Exception):
    """Raised when path to preselecting
    is not valid.
    """

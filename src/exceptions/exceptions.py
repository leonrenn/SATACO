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
